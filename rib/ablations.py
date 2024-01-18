import json
import time
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import numpy as np
import torch
from jaxtyping import Bool, Float
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from rib.data import (
    HFDatasetConfig,
    ModularArithmeticDatasetConfig,
    VisionDatasetConfig,
)
from rib.data_accumulator import Edges
from rib.hook_fns import edge_ablation_forward_hook_fn, rotate_pre_forward_hook_fn
from rib.hook_manager import Hook, HookedModel
from rib.interaction_algos import Eigenvectors, InteractionRotation
from rib.linalg import calc_rotation_matrix
from rib.loader import load_model_and_dataset_from_rib_config
from rib.log import logger
from rib.models import MLP, SequentialTransformer
from rib.rib_builder import RibBuildResults
from rib.types import TORCH_DTYPES, RootPath, StrDtype
from rib.utils import (
    check_outfile_overwrite,
    eval_cross_entropy_loss,
    eval_model_accuracy,
    load_config,
    set_seed,
)

BasisVecs = Union[Float[Tensor, "d_hidden d_hidden_trunc"], Float[Tensor, "d_hidden d_hidden"]]
BasisVecsPinv = Union[Float[Tensor, "d_hidden_trunc d_hidden"], Float[Tensor, "d_hidden d_hidden"]]
AblationAccuracies = dict[str, dict[int, float]]
EdgeMasks = dict[str, dict[int, Bool[Tensor, "out in"]]]


class ScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schedule_type: Literal["exponential", "linear"]
    early_stopping_threshold: Optional[float] = Field(
        None,
        description="The threshold to use for stopping the ablation calculations early. The"
        "experiment will stop when the ablated score is more than `early_stopping_threshold` away "
        "from the unablated score. If None, we don't stop early.",
    )
    specific_points: Optional[list[int]] = Field(
        None,
        description="A list of number of vecs remaining to add to the schedule. If None, we use"
        "the default schedule.",
    )

    def get_ablation_schedule(self, n_vecs: int) -> list[int]:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _add_specific_ablation_points(self, ablation_schedule: list[int], n_vecs: int) -> list[int]:
        """Add each number of vecs remaining in self.specific_points to the ablation schedule."""
        if self.specific_points is not None:
            # Ignore the specific points that are greater than the number of vecs
            specific_ablated_vecs = [n_vecs - x for x in self.specific_points if x <= n_vecs]
            # Add our specific points for the number of vecs remaining to the ablation schedule
            ablation_schedule = sorted(
                list(set(ablation_schedule + specific_ablated_vecs)), reverse=True
            )
        return ablation_schedule


class ExponentialScheduleConfig(ScheduleConfig):
    schedule_type: Literal["exponential"]
    ablate_every_vec_cutoff: Optional[int] = Field(
        None,
        description="The point in the exponential schedule at which we start ablating every"
        "individual vector. If None, always ablate every vector.",
    )
    exp_base: Optional[float] = Field(2.0, description="The base of the exponential schedule.")

    def get_ablation_schedule(self, n_vecs: int) -> list[int]:
        """Create an exponential schedule for the number of vectors to ablate.

        The schedule is exponential with a base of 2, with the exception that from
        `self.ablate_every_vec_cutoff` to `n_vecs` we ablate every vector. The schedule also
        includes a run with no ablations as well as with the number of vecs remaining given in
        `self.specific_points`.

        Args:
            n_vecs: Total number of vectors.

        Returns:
            The schedule for the number of vectors to ablate.

        Examples:
            >>> ExponentialScheduleConfig("exponential", None).get_ablation_schedule(12)
            [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
            >>> ExponentialScheduleConfig("exponential", 0).get_ablation_schedule(12)
            [12, 11, 9, 5, 0]  # Exponential schedule (2^x) from the beginning.
            >>> ExponentialScheduleConfig("exponential", 1).get_ablation_schedule(12)
            [12, 11, 10, 8, 4, 0]  # Exponential schedule (2^x) after the first 1 value
            >>> ExponentialScheduleConfig("exponential", 3).get_ablation_schedule(12)
            [12, 11, 10, 9, 8, 6, 2, 0]
            >>> ExponentialScheduleConfig("exponential", 3).get_ablation_schedule(24)
            [24, 23, 22, 21, 20, 18, 14, 6, 0]
        """
        cutoff = self.ablate_every_vec_cutoff
        exp_base = self.exp_base if self.exp_base is not None else 2.0

        if cutoff is None:
            return list(range(n_vecs, -1, -1))

        assert cutoff < n_vecs, "ablate_every_vec_cutoff must be smaller than n_vecs"
        assert cutoff >= 0, "ablate_every_vec_cutoff must be positive"
        # The section in which we ablate every vector.
        ablate_every_vecs: list[int] = list(range(n_vecs, n_vecs - cutoff - 1, -1))
        # The section in which we ablate according to 2^x.
        ablate_exponential: list[int] = []
        prev_val = ablate_every_vecs[-1]
        for x in range(n_vecs):
            exp_val = int(prev_val - exp_base**x)
            if exp_val > 0:
                ablate_exponential.append(exp_val)
                prev_val = exp_val
            else:
                # No more values to append, just add the case for no ablation and exit
                ablate_exponential.append(0)
                break

        # combine the two sections
        ablation_schedule = ablate_every_vecs + ablate_exponential
        assert ablation_schedule[0] == n_vecs, "The first element of the schedule must be n_vecs."
        assert ablation_schedule[-1] == 0, "The last element of the schedule must be 0."

        ablation_schedule = self._add_specific_ablation_points(ablation_schedule, n_vecs)
        return ablation_schedule


class LinearScheduleConfig(ScheduleConfig):
    schedule_type: Literal["linear"]
    n_points: int = Field(
        ...,
        description=(
            "The number of points to use in the linear ablation schedule. Must be specified if "
            "schedule_type is linear and cannot be specified if schedule_type is exponential."
        ),
    )

    def get_ablation_schedule(self, n_vecs: int) -> list[int]:
        """Create a linear schedule for the number of vectors to ablate.

        The points are evenly spaced between `n_vecs` and 0, including the endpoints and any points
        in `self.specific_points` are also added.

        Args:
            n_vecs: Total number of vectors.

        Returns:
            The schedule for the number of vectors to ablate.

        Examples:
            >>> LinearScheduleConfig("linear", 3).get_ablation_schedule(12)
            [12, 6, 0]
        """
        assert self.n_points >= 2, f"{self.n_points} must be at least 2."
        assert self.n_points <= n_vecs, f"{self.n_points} must be <= {n_vecs}."

        ablation_schedule = [int(a) for a in np.linspace(n_vecs, 0, self.n_points, dtype=int)]

        ablation_schedule = self._add_specific_ablation_points(ablation_schedule, n_vecs)
        return ablation_schedule


class AblationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    exp_name: str
    out_dir: Optional[RootPath] = Field(
        Path(__file__).parent / "out",
        description="Directory for the output files. Defaults to `./out/`. If None, no output "
        "is written. If a relative path, it is relative to the root of the rib repo.",
    )
    ablation_type: Literal["rib", "orthogonal"]
    edge_ablation: bool = Field(
        False,
        description="Whether to perform edge ablation experiments. If False, we perform node "
        "ablation experiments.",
    )
    rib_results_path: RootPath
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


@torch.inference_mode()
def ablate_node_layers_and_eval(
    basis_matrices: list[tuple[BasisVecs, BasisVecsPinv]],
    ablation_node_layers: list[str],
    hooked_model: HookedModel,
    data_loader: DataLoader,
    eval_fn: Callable,
    module_names: list[str],
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
        module_names: The names of the modules to apply the ablations. Can be any valid pytorch
            module in hooked_model.model. These typically correspond to section_names (e.g.
            "sections.section_0") when the model is a SequentialTransformer or raw layers (e.g.
            "layers.2") when the model is an MLP.
        schedule_config: The config for the ablation schedule.
        device: The device to run the model on.
        dtype: The data type to cast the inputs to. Ignored if int32 or int64.

    Returns:
        A dictionary mapping node layers to ablation accuracies/losses.
    """
    results: AblationAccuracies = {}
    for ablation_node_layer, module_name, (basis_vecs, basis_vecs_pinv) in zip(
        ablation_node_layers, module_names, basis_matrices, strict=True
    ):
        ablation_schedule = schedule_config.get_ablation_schedule(n_vecs=basis_vecs.shape[0])

        base_score: Optional[float] = None

        # Track the results for the case when there is no ablation. There may be many of these, so we
        # store them to avoid recomputing.
        n_truncated_vecs = basis_vecs.shape[0] - basis_vecs.shape[1]

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

            if n_ablated_vecs_trunc == 0 and base_score is not None:
                # We may have already calculated the score for the case when there is no ablation
                results[ablation_node_layer][n_vecs_remaining] = base_score
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

            score = eval_fn(
                hooked_model, data_loader, hooks=[rotation_hook], dtype=dtype, device=device
            )
            results[ablation_node_layer][n_vecs_remaining] = score

            if schedule_config.early_stopping_threshold is not None:
                if i == 0:
                    base_score = score
                else:
                    # If the score is more than `early_stopping_threshold` away from the base result,
                    # then we stop ablating vectors.
                    if abs(score - base_score) > schedule_config.early_stopping_threshold:
                        break

    return results


@torch.inference_mode()
def ablate_edges_and_eval(
    basis_matrices: list[tuple[BasisVecs, BasisVecsPinv]],
    ablation_node_layers: list[str],
    edges: list[Edges],
    hooked_model: HookedModel,
    data_loader: DataLoader,
    eval_fn: Callable,
    module_names: list[str],
    schedule_config: Union[ExponentialScheduleConfig, LinearScheduleConfig],
    device: str,
    dtype: Optional[torch.dtype] = None,
) -> tuple[AblationAccuracies, EdgeMasks]:
    """Perform a series of edge ablation experiments across layers and multiple # of edges to keep.

    Note that we want our ablation schedules for different bases to match up, even though different
    bases may have different number of basis vectors due to truncation. We therefore create our
    ablation schedule assuming a non-truncated basis (i.e. using the full possible # of edges,
    out_dim * in_dim. This can be very large).

    Args:
        basis_matrices: List of basis vector matrices (Cs) and their pseudoinverses (C_pinv).
        ablation_node_layers: The names of the node layers whose (rotated) inputs we want to ablate.
        edges: The edge weights computed by RIB. We use these to determine edge ablation order.
        hooked_model: The hooked model.
        data_loader: The data loader to use for testing.
        eval_fn: The function to use to evaluate the model.
        module_names: The names of the modules to apply the ablations. Can be any valid pytorch
            module in hooked_model.model. These typically correspond to section_names (e.g.
            "sections.section_0") when the model is a SequentialTransformer or raw layers (e.g.
            "layers.2") when the model is an MLP.
        schedule_config: The config for the ablation schedule.
        device: The device to run the model on.
        dtype: The data type to cast the inputs to. Ignored if int32 or int64.

    Returns:
        A dictionary mapping node layers to ablation accuracies/losses.
    """
    base_score = eval_fn(hooked_model, data_loader, hooks=[], dtype=dtype, device=device)

    results: AblationAccuracies = {}
    edge_masks: EdgeMasks = {}
    basis_pairs = zip(basis_matrices[:-1], basis_matrices[1:])
    for ablation_node_layer, module_name, basis_pair, layer_edges in zip(
        ablation_node_layers[:-1], module_names[:-1], basis_pairs, edges, strict=True
    ):
        (in_C, in_C_inv), (out_C, out_C_inv) = basis_pair
        total_possible_edges = in_C.shape[0] * out_C.shape[1]
        ablation_schedule = schedule_config.get_ablation_schedule(n_vecs=total_possible_edges)
        results[ablation_node_layer] = {}
        edge_masks[ablation_node_layer] = {}
        # Iterate through possible number of ablated vectors, starting from no ablated vectors
        for num_edges_ablated in tqdm(
            ablation_schedule[::-1], total=len(ablation_schedule), desc=f"Ablating {module_name}"
        ):
            num_edges_kept = total_possible_edges - num_edges_ablated
            if num_edges_kept > layer_edges.E_hat.numel():
                results[ablation_node_layer][num_edges_kept] = base_score
                continue
            if num_edges_kept > 0:
                threshold = torch.topk(layer_edges.E_hat.flatten(), k=num_edges_kept).values[-1]
                edge_mask = layer_edges.E_hat >= threshold
            else:
                edge_mask = torch.zeros_like(layer_edges.E_hat, dtype=torch.bool)
            edge_masks[ablation_node_layer][num_edges_kept] = edge_mask

            hook = Hook(
                name=module_name,
                data_key="edge_ablation",
                fn=edge_ablation_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={
                    "edge_mask": edge_mask.to(device),
                    "in_C": in_C.to(device),
                    "in_C_inv": in_C_inv.to(device),
                    "out_C": out_C.to(device),
                    "out_C_inv": out_C_inv.to(device),
                },
            )

            score = eval_fn(hooked_model, data_loader, hooks=[hook], dtype=dtype, device=device)
            results[ablation_node_layer][num_edges_kept] = score

            if schedule_config.early_stopping_threshold is not None:
                # If the score is more than `early_stopping_threshold` away from the base result,
                # then we stop ablating vectors.
                if abs(score - base_score) > schedule_config.early_stopping_threshold:
                    break

    return results, edge_masks


def load_basis_matrices(
    rib_results: RibBuildResults,
    ablation_node_layers: list[str],
    ablation_type: Literal["rib", "orthogonal"],
    dtype: torch.dtype,
    device: str,
    none_ok: bool = False,
) -> list[tuple[BasisVecs, BasisVecsPinv]]:
    """Load the basis matrices and their pseudoinverses.

    Supports both rib and orthogonal basis matrices. Converts each matrix to the specified dtype
    and device.

    By default asserts that all basis matrices are non-None. If none_ok is True then an identity
    matrix is returned in place of None matricies.
    """
    if ablation_type == "rib":
        basis_matrix_key = "interaction_rotations"
    elif ablation_type == "orthogonal":
        basis_matrix_key = "eigenvectors"
    else:
        raise ValueError(f"ablation_type must be one of ['rib', 'orthogonal']")

    # Get the basis vecs and their pseudoinverses using the module_names as keys
    basis_matrices: list[tuple[BasisVecs, BasisVecsPinv]] = []
    basis_infos_list = getattr(rib_results, basis_matrix_key)
    basis_infos_dict = {info.node_layer_name: info for info in basis_infos_list}
    for module_name in ablation_node_layers:
        basis_info = basis_infos_dict[module_name]
        if ablation_type == "rib":
            assert isinstance(basis_info, InteractionRotation)
            if none_ok and basis_info.C is None:
                assert basis_info.C_pinv is None, f"{module_name} has a C_pinv matrix."
                basis_vecs = torch.eye(basis_info.out_dim, dtype=dtype, device=device)
                basis_vecs_pinv = torch.eye(basis_info.out_dim, dtype=dtype, device=device)
            else:
                assert basis_info.C is not None, f"{module_name} has no C matrix."
                assert basis_info.C_pinv is not None, f"{module_name} has no C_pinv matrix."
                basis_vecs = basis_info.C.to(dtype=dtype, device=device)
                basis_vecs_pinv = basis_info.C_pinv.to(dtype=dtype, device=device)
        elif ablation_type == "orthogonal":
            assert isinstance(basis_info, Eigenvectors)
            if none_ok and basis_info.U is None:
                basis_vecs = torch.eye(basis_info.out_dim, dtype=dtype, device=device)
                basis_vecs_pinv = torch.eye(basis_info.out_dim, dtype=dtype, device=device)
            else:
                assert basis_info.U is not None, f"{module_name} has no U matrix."
                basis_vecs = basis_info.U.to(dtype=dtype, device=device)
                # Pseudoinverse of an orthonormal matrix is its transpose
                basis_vecs_pinv = basis_vecs.T.detach().clone()
        basis_matrices.append((basis_vecs, basis_vecs_pinv))
    return basis_matrices


def load_bases_and_ablate(
    config_path_or_obj: Union[str, AblationConfig], force: bool = False
) -> AblationAccuracies:
    """Load basis matrices and run ablation experiments.

    The process is as follows:
        1. Load pre-saved basis matrices (typcially RIB bases (Cs) or orthogonal bases (Us)).
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
        force: Whether to overwrite existing output files.`

    Returns:
        A dictionary mapping node layers to accuracies/losses. If the config has an out_dir, the
        results are also written to a file in that directory.
    """
    start_time = time.time()
    config = load_config(config_path_or_obj, config_model=AblationConfig)

    if config.out_dir is not None:
        config.out_dir.mkdir(parents=True, exist_ok=True)
        edge_node = "edge" if config.edge_ablation else "node"
        out_file = config.out_dir / f"{config.exp_name}_{edge_node}_ablation_results.json"
        if not check_outfile_overwrite(out_file, force):
            raise FileExistsError("Not overwriting output file")

    set_seed(config.seed)
    rib_results = RibBuildResults(**torch.load(config.rib_results_path))

    assert set(config.ablation_node_layers) <= set(
        rib_results.config.node_layers
    ), "The node layers in the config must be a subset of the node layers in the RIB graph."
    if config.edge_ablation:
        # config.node_layers must be a subsequence of loaded_config.node_layers
        assert "|".join(config.ablation_node_layers) in "|".join(
            rib_results.config.node_layers
        ), "node_layers in the config must be a subsequence of the node layers in the RIB graph."
        assert (
            config.ablation_type == "rib"
        ), "Can't do edge ablation with Us, as we don't have edges for U basis"
        assert len(rib_results.edges) > 0, "No edges found in the RIB results."
        assert rib_results.contains_all_edges
    else:
        assert "output" not in config.ablation_node_layers, "Cannot ablate the output node layer."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = TORCH_DTYPES[config.dtype]

    basis_matrices = load_basis_matrices(
        rib_results=rib_results,
        ablation_node_layers=config.ablation_node_layers,
        ablation_type=config.ablation_type,
        dtype=dtype,
        device=device,
        none_ok=config.edge_ablation,
    )

    model, dataset = load_model_and_dataset_from_rib_config(
        rib_results.config,
        dataset_config=config.dataset,
        device=device,
        dtype=dtype,
        node_layers=config.ablation_node_layers,
    )
    model.to(device=torch.device(device), dtype=dtype)
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    hooked_model = HookedModel(model)

    # Test model accuracy/loss before running ablations, ta be sure
    eval_fn: Callable = (
        eval_model_accuracy if config.eval_type == "accuracy" else eval_cross_entropy_loss
    )
    no_ablation_result = eval_fn(hooked_model, data_loader, dtype=dtype, device=device)
    logger.info("Model %s on dataset: %.4f", config.eval_type, no_ablation_result)

    if isinstance(model, MLP):
        module_names = config.ablation_node_layers
    else:
        assert isinstance(model, SequentialTransformer)
        module_names = [f"sections.{sec}" for sec in model.sections if sec != "pre"]

    if config.edge_ablation:
        edges_dict = {info.in_node_layer_name: info for info in rib_results.edges}
        edges = [edges_dict[layer] for layer in config.ablation_node_layers[:-1]]
        ablation_results, edge_masks = ablate_edges_and_eval(
            basis_matrices=basis_matrices,
            ablation_node_layers=config.ablation_node_layers,
            edges=edges,
            hooked_model=hooked_model,
            data_loader=data_loader,
            eval_fn=eval_fn,
            module_names=module_names,
            schedule_config=config.schedule,
            device=device,
            dtype=dtype,
        )
    else:
        ablation_results = ablate_node_layers_and_eval(
            basis_matrices=basis_matrices,
            ablation_node_layers=config.ablation_node_layers,
            hooked_model=hooked_model,
            data_loader=data_loader,
            eval_fn=eval_fn,
            module_names=module_names,
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
        "no_ablation_result": no_ablation_result,
    }
    if config.edge_ablation:
        results["edge_masks"] = edge_masks

    if config.out_dir is not None:
        with open(out_file, "w") as f:
            json.dump(results, f, default=lambda x: x.tolist())  # serialize edge_mask tensors
        logger.info("Wrote results to %s", out_file)

    return ablation_results
