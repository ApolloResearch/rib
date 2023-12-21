import json
import time
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import numpy as np
import torch
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from rib.data import (
    HFDatasetConfig,
    ModularArithmeticDatasetConfig,
    VisionDatasetConfig,
)
from rib.hook_fns import rotate_pre_forward_hook_fn
from rib.hook_manager import Hook, HookedModel
from rib.linalg import calc_rotation_matrix
from rib.loader import load_model_and_dataset_from_rib_results
from rib.log import logger
from rib.models import MLP, SequentialTransformer
from rib.rib_builder import RibBuildResults
from rib.types import TORCH_DTYPES, RootPath, StrDtype
from rib.utils import (
    calc_exponential_ablation_schedule,
    check_outfile_overwrite,
    eval_cross_entropy_loss,
    eval_model_accuracy,
    load_config,
    set_seed,
)

BasisVecs = Union[Float[Tensor, "d_hidden d_hidden_trunc"], Float[Tensor, "d_hidden d_hidden"]]
BasisVecsPinv = Union[Float[Tensor, "d_hidden_trunc d_hidden"], Float[Tensor, "d_hidden d_hidden"]]
AblationAccuracies = dict[str, dict[int, float]]


class ScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schedule_type: Literal["exponential", "linear"]
    early_stopping_threshold: Optional[float] = Field(
        None,
        description="The threshold to use for stopping the ablation calculations early. If None,"
        "we don't use early stopping.",
    )
    specific_points: Optional[list[int]] = Field(
        None,
        description="A list of number of vecs remaining to add to the schedule. If None, we use"
        "the default schedule.",
    )


class ExponentialScheduleConfig(ScheduleConfig):
    schedule_type: Literal["exponential"]
    ablate_every_vec_cutoff: Optional[int] = Field(
        None,
        description="The point in the exponential schedule at which we start ablating every"
        "individual vector. If None, always ablate every vector.",
    )
    exp_base: Optional[float] = Field(2.0, description="The base of the exponential schedule.")


class LinearScheduleConfig(ScheduleConfig):
    schedule_type: Literal["linear"]
    n_points: int = Field(
        ...,
        description="The number of points to use in the linear ablation schedule. Must be specified if schedule_type is linear and cannot be specified if schedule_type is exponential.",
    )


class AblationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    exp_name: str
    out_dir: Optional[RootPath] = Field(
        Path(__file__).parent / "out",
        description="Directory for the output files. Defaults to `./out/`. If None, no output "
        "is written. If a relative path, it is relative to the root of the rib repo.",
    )
    ablation_type: Literal["rib", "orthogonal"]
    interaction_graph_path: RootPath
    schedule: Union[ExponentialScheduleConfig, LinearScheduleConfig] = Field(
        ...,
        discriminator="schedule_type",
        description="The schedule to use for ablations.",
    )
    dataset: Union[ModularArithmeticDatasetConfig, HFDatasetConfig, VisionDatasetConfig] = Field(
        ...,
        discriminator="dataset_type",
        description="The dataset to use to build the graph.",
    )
    ablation_node_layers: list[str]
    batch_size: int
    dtype: StrDtype
    seed: int
    eval_type: Literal["accuracy", "ce_loss"] = Field(
        ...,
        description="The type of evaluation to perform on the model before building the graph.",
    )


def get_ablation_schedule(schedule_config: ScheduleConfig, n_vecs: int) -> list[int]:
    """Get the ablation schedule for a given number of basis vectors.

    Args:
        schedule_config: The config for the ablation schedule.
        n_vecs: The number of basis vectors.

    Returns:
        A list of the number of vectors to ablate at each step.
    """
    if isinstance(schedule_config, ExponentialScheduleConfig):
        ablation_schedule = calc_exponential_ablation_schedule(
            n_vecs=n_vecs,
            exp_base=schedule_config.exp_base,
            ablate_every_vec_cutoff=schedule_config.ablate_every_vec_cutoff,
        )
    elif isinstance(schedule_config, LinearScheduleConfig):
        assert schedule_config.n_points >= 2, f"{schedule_config.n_points} must be at least 2."
        assert (
            schedule_config.n_points <= n_vecs
        ), f"{schedule_config.n_points} must be <= {n_vecs}."

        ablation_schedule = [
            int(a) for a in np.linspace(n_vecs, 0, schedule_config.n_points, dtype=int)
        ]
    else:
        raise NotImplementedError(f"Schedule: {schedule_config.schedule_type} not supported.")

    if schedule_config.specific_points is not None:
        # Ignore the specific points that are greater than the number of vecs
        specific_ablated_vecs = [n_vecs - x for x in schedule_config.specific_points if x <= n_vecs]
        # Add our specific points for the number of vecs remaining to the ablation schedule
        ablation_schedule = sorted(
            list(set(ablation_schedule + specific_ablated_vecs)), reverse=True
        )

    return ablation_schedule


@torch.inference_mode()
def ablate_node_layers_and_eval(
    basis_matrices: list[tuple[BasisVecs, BasisVecsPinv]],
    ablation_node_layers: list[str],
    hooked_model: HookedModel,
    data_loader: DataLoader,
    eval_fn: Callable,
    graph_module_names: list[str],
    schedule_config: Union[ExponentialScheduleConfig, LinearScheduleConfig],
    device: str,
    dtype: Optional[torch.dtype] = None,
) -> AblationAccuracies:
    """Rotate to and from a truncated basis and compare ablation accuracies/losses.

    Note that we want our ablation schedules for different bases to match up, even though different
    bases may have different number of basis vectors due to truncation. We therefore create our
    ablation schedule assuming a non-truncated basis (i.e. using the full hidden size
    (basis_vecs.shape[0])).

    Args:
        basis_matrices: List of basis vector matrices and their pseudoinverses. In the orthogonal
            basis case, the pseudoinverse is the transpose.
        ablation_node_layers: The names of the node layers whose (rotated) inputs we want to ablate.
        hooked_model: The hooked model.
        data_loader: The data loader to use for testing.
        eval_fn: The function to use to evaluate the model.
        graph_module_names: The names of the modules we want to build the graph around.
        schedule_config: The config for the ablation schedule.
        device: The device to run the model on.
        dtype: The data type to cast the inputs to. Ignored if int32 or int64.

    Returns:
        A dictionary mapping node layers to ablation accuracies/losses.
    """
    results: AblationAccuracies = {}
    for ablation_node_layer, module_name, (basis_vecs, basis_vecs_pinv) in zip(
        ablation_node_layers, graph_module_names, basis_matrices
    ):
        ablation_schedule = get_ablation_schedule(schedule_config, n_vecs=basis_vecs.shape[0])

        base_result: Optional[float] = None

        # Track the results for the case when there is no ablation. There may be many of these, so we
        # store them to avoid recomputing.
        n_truncated_vecs = basis_vecs.shape[0] - basis_vecs.shape[1]
        no_ablation_result: Optional[float] = None

        results[ablation_node_layer] = {}
        # Iterate through possible number of ablated vectors, starting from no ablated vectors
        for i, n_ablated_vecs in enumerate(
            tqdm(
                ablation_schedule[::-1],
                total=len(ablation_schedule),
                desc=f"Ablating {module_name}",
            )
        ):
            # Note that we may have truncated vectors with small eigenvalues, so we make sure not to
            # ablate more vectors than we have remaining
            n_vecs_remaining = basis_vecs.shape[0] - n_ablated_vecs

            # Count the n_ablated_vecs taking into account the truncation
            n_ablated_vecs_trunc = max(n_ablated_vecs - n_truncated_vecs, 0)

            if n_ablated_vecs_trunc == 0 and no_ablation_result is not None:
                results[ablation_node_layer][n_vecs_remaining] = no_ablation_result
                continue

            basis_vecs = basis_vecs.to(device)
            basis_vecs_pinv = basis_vecs_pinv.to(device)
            rotation_matrix = calc_rotation_matrix(
                vecs=basis_vecs,
                vecs_pinv=basis_vecs_pinv,
                n_ablated_vecs=n_ablated_vecs_trunc,
            )

            rotation_hook = Hook(
                name=ablation_node_layer,
                data_key="rotation",
                fn=rotate_pre_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={"rotation_matrix": rotation_matrix},
            )

            node_layer_score = eval_fn(
                hooked_model, data_loader, hooks=[rotation_hook], dtype=dtype, device=device
            )
            results[ablation_node_layer][n_vecs_remaining] = node_layer_score

            if schedule_config.early_stopping_threshold is not None:
                if i == 0:
                    base_result = node_layer_score
                else:
                    # If the result is more than `early_stopping_threshold` different than the base result,
                    # then we stop ablating vectors.
                    if (
                        abs(node_layer_score - base_result)
                        > schedule_config.early_stopping_threshold
                    ):
                        break

    return results


def load_basis_matrices(
    interaction_graph_info: RibBuildResults,
    ablation_node_layers: list[str],
    ablation_type: Literal["rib", "orthogonal"],
    dtype: torch.dtype,
    device: str,
) -> list[tuple[BasisVecs, BasisVecsPinv]]:
    """Load the basis matrices and their pseudoinverses.

    Supports both rib and orthogonal basis matrices. Converts each matrix to the specified dtype
    and device.
    """
    if ablation_type == "rib":
        basis_matrix_key = "interaction_rotations"
    elif ablation_type == "orthogonal":
        basis_matrix_key = "eigenvectors"
    else:
        raise ValueError(f"ablation_type must be one of ['rib', 'orthogonal']")

    # Get the basis vecs and their pseudoinverses using the module_names as keys
    basis_matrices: list[tuple[BasisVecs, BasisVecsPinv]] = []
    for module_name in ablation_node_layers:
        for basis_info in getattr(interaction_graph_info, basis_matrix_key):
            if basis_info["node_layer_name"] == module_name:
                if ablation_type == "rib":
                    assert basis_info["C"] is not None, f"{module_name} has no C matrix."
                    assert basis_info["C_pinv"] is not None, f"{module_name} has no C_pinv matrix."
                    basis_vecs = basis_info["C"].to(dtype=dtype, device=device)
                    basis_vecs_pinv = basis_info["C_pinv"].to(dtype=dtype, device=device)
                elif ablation_type == "orthogonal":
                    assert basis_info["U"] is not None, f"{module_name} has no U matrix."
                    basis_vecs = basis_info["U"].to(dtype=dtype, device=device)
                    # Pseudoinverse of an orthonormal matrix is its transpose
                    basis_vecs_pinv = basis_vecs.T.detach().clone()
                basis_matrices.append((basis_vecs, basis_vecs_pinv))
                break
    assert len(basis_matrices) == len(
        ablation_node_layers
    ), f"Could not find all node_layer modules in the interaction graph config."
    return basis_matrices


def load_bases_and_ablate(
    config_path_or_obj: Union[str, AblationConfig], force: bool = False
) -> AblationAccuracies:
    """Load basis matrices and run ablation experiments.

    The process is as follows:
        1. Load pre-saved basis matrices (Typcially RIB bases (Cs) or orthogonal bases (Us)).
        2. Load the corresponding model and dataset (the dataset may be non-overlapping with
            that used to create the basis matrices).
        3. For each number of ablated nodes `n`, create a rotation matrix that has the effect of
        rotating to and from the new basis with `n` fewer basis vectors.
        4. Run the test set through the model, applying the above rotations at each node layer, and
        calculate the resulting accuracy/loss.
        5. Repeat steps 3 and 4 for a range of values of n, storing the resulting accuracies/losses.

    Note that we don't create a node layer at the output of the final module, as ablating nodes in
    this layer is not useful.

    Args:
        config_path_or_obj: The path to the config file or the config object itself.
        force: Whether to overwrite existing output files.

    Returns:
        A dictionary mapping node layers to accuracies/losses. If the config has an out_dir, the
        results are also written to a file in that directory.
    """
    start_time = time.time()
    config = load_config(config_path_or_obj, config_model=AblationConfig)

    if config.out_dir is not None:
        config.out_dir.mkdir(parents=True, exist_ok=True)
        out_file = config.out_dir / f"{config.exp_name}_ablation_results.json"
        if not check_outfile_overwrite(out_file, force):
            raise FileExistsError("Not overwriting output file")

    set_seed(config.seed)
    interaction_graph_info = RibBuildResults(**torch.load(config.interaction_graph_path))

    assert set(config.ablation_node_layers) <= set(
        interaction_graph_info.config.node_layers
    ), "The node layers in the config must be a subset of the node layers in the interaction graph."

    assert "output" not in config.ablation_node_layers, "Cannot ablate the output node layer."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = TORCH_DTYPES[config.dtype]

    basis_matrices = load_basis_matrices(
        interaction_graph_info=interaction_graph_info,
        ablation_node_layers=config.ablation_node_layers,
        ablation_type=config.ablation_type,
        dtype=dtype,
        device=device,
    )

    model, dataset = load_model_and_dataset_from_rib_results(
        interaction_graph_info,
        device=device,
        dtype=dtype,
        node_layers=config.ablation_node_layers,
        dataset_config=config.dataset,
        return_set=config.dataset.return_set,
    )
    model.to(device=torch.device(device), dtype=dtype)
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    hooked_model = HookedModel(model)

    # Test model accuracy/loss before running ablations, ta be sure
    eval_fn: Callable = (
        eval_model_accuracy if config.eval_type == "accuracy" else eval_cross_entropy_loss
    )
    eval_results = eval_fn(hooked_model, data_loader, dtype=dtype, device=device)
    logger.info("Model %s on dataset: %.4f", config.eval_type, eval_results)

    if isinstance(model, MLP):
        graph_module_names = config.ablation_node_layers
    else:
        assert isinstance(model, SequentialTransformer)
        graph_module_names = [f"sections.{sec}" for sec in model.sections if sec != "pre"]

    ablation_results: AblationAccuracies = ablate_node_layers_and_eval(
        basis_matrices=basis_matrices,
        ablation_node_layers=config.ablation_node_layers,
        hooked_model=hooked_model,
        data_loader=data_loader,
        eval_fn=eval_fn,
        graph_module_names=graph_module_names,
        schedule_config=config.schedule,
        device=device,
        dtype=dtype,
    )

    time_taken = f"{(time.time() - start_time) / 60:.1f} minutes"
    logger.info("Finished in %s.", time_taken)

    results = {
        "config": json.loads(config.model_dump_json()),
        "results": ablation_results,
        "time_taken": time_taken,
    }
    if config.out_dir is not None:
        with open(out_file, "w") as f:
            json.dump(results, f)
        logger.info("Wrote results to %s", out_file)

    return ablation_results
