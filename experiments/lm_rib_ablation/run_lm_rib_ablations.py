"""Ablate vectors from the interaction basis and calculate the difference in loss.

The process is as follows:
    1. Load a SequentialTransformer model and dataset.
    2. For each number of ablated nodes `n`, create a rotation matrix that has the effect of
    rotating to and from the interaction basis with `n` fewer basis vectors.
    3. Run the test set through the model, applying the above rotations at each node layer, and
    calculate the resulting accuracy.
    5. Repeat steps 3 and 4 for a range of values of n, plotting the resulting accuracies.

A node layer is positioned at the input of each module_name specified in the config. In this script,
we don't create a node layer at the output of the final module, as ablating nodes in this layer
is not useful.

Usage:
    python run_lm_rib_ablations.py <path/to/yaml_config_file>

"""

import json
from pathlib import Path
from typing import Optional

import fire
import torch
from jaxtyping import Float
from pydantic import BaseModel, Field, field_validator
from torch import Tensor
from torch.utils.data import DataLoader

from rib.ablations import ablate_and_test
from rib.hook_manager import HookedModel
from rib.loader import create_data_loader, load_sequential_transformer
from rib.log import logger
from rib.types import TORCH_DTYPES
from rib.utils import (
    calc_ablation_schedule,
    eval_model_accuracy,
    load_config,
    overwrite_output,
    set_seed,
)


class Config(BaseModel):
    exp_name: Optional[str]
    interaction_graph_path: Path
    ablate_every_vec_cutoff: Optional[int] = Field(
        None,
        description="The point at which we start ablating every individual vector. If None, always ablate every vector.",
    )
    node_layers: list[str]
    batch_size: int
    dtype: str
    seed: int

    @field_validator("dtype")
    def dtype_validator(cls, v):
        assert v in TORCH_DTYPES, f"dtype must be one of {TORCH_DTYPES}"
        return v


def run_ablations(
    interaction_matrices: list[
        tuple[Float[Tensor, "d_hidden d_hidden_trunc"], Float[Tensor, "d_hidden_trunc d_hidden"]]
    ],
    node_layers: list[str],
    hooked_model: HookedModel,
    data_loader: DataLoader,
    graph_module_names: list[str],
    ablate_every_vec_cutoff: Optional[int],
    device: str,
) -> dict[str, dict[int, float]]:
    """Rotate to and from the interaction basis and compare accuracies with and without ablation.

    Args:
        interaction_matrices: The interaction rotation matrix and its pseudoinverse.
        node_layers: The names of the node layers to build the graph with.
        hooked_model: The hooked model.
        data_loader: The data loader to use for testing.
        graph_module_names: The names of the modules we want to build the graph around.
        ablate_every_vec_cutoff: The point in the ablation schedule to start ablating every vector.
        device: The device to run the model on.

    Returns:
        A dictionary mapping node layers to accuracy results.
    """
    results: dict[str, dict[int, float]] = {}
    for hook_name, module_name, (C, C_pinv) in zip(
        node_layers, graph_module_names, interaction_matrices
    ):
        ablation_schedule = calc_ablation_schedule(
            ablate_every_vec_cutoff=ablate_every_vec_cutoff,
            n_vecs=C.shape[1],
        )
        module_accuracies: dict[int, float] = ablate_and_test(
            hooked_model=hooked_model,
            module_name=module_name,
            interaction_rotation=C,
            interaction_rotation_pinv=C_pinv,
            test_loader=data_loader,
            device=device,
            ablation_schedule=ablation_schedule,
            hook_name=hook_name,
        )
        results[hook_name] = module_accuracies

    return results


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)

    set_seed(config.seed)
    interaction_graph_info = torch.load(config.interaction_graph_path)

    # Get the interaction rotations and their pseudoinverses (C and C_pinv)
    interaction_matrices: list[
        tuple[Float[Tensor, "d_hidden d_hidden_trunc"], Float[Tensor, "d_hidden_trunc d_hidden"]]
    ] = []
    for module_name in config.node_layers:
        for rotation_info in interaction_graph_info["interaction_rotations"]:
            if rotation_info["node_layer_name"] == module_name:
                interaction_matrices.append((rotation_info["C"], rotation_info["C_pinv"]))
                break
    assert len(interaction_matrices) == len(
        config.node_layers
    ), f"Could not find all modules in the interaction graph config. "

    out_file = Path(__file__).parent / "out" / f"{config.exp_name}_ablation_results.json"
    if out_file.exists() and not overwrite_output(out_file):
        print("Exiting.")
        return

    out_file.parent.mkdir(parents=True, exist_ok=True)

    assert (
        interaction_graph_info["config"]["tlens_pretrained"] is None
        and interaction_graph_info["config"]["tlens_model_path"] is not None
    ), "Currently can't build graphs for pretrained models due to memory limits."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = TORCH_DTYPES[config.dtype]
    tlens_model_path = Path(interaction_graph_info["config"]["tlens_model_path"])

    seq_model, _ = load_sequential_transformer(
        node_layers=config.node_layers,
        last_pos_only=interaction_graph_info["config"]["last_pos_only"],
        tlens_pretrained=interaction_graph_info["config"]["tlens_pretrained"],
        tlens_model_path=tlens_model_path,
        dtype=dtype,
        device=device,
    )

    seq_model.eval()
    seq_model.to(device=torch.device(device), dtype=dtype)
    seq_model.fold_bias()
    hooked_model = HookedModel(seq_model)

    data_loader = create_data_loader(
        dataset_name=interaction_graph_info["config"]["dataset"],
        tlens_model_path=tlens_model_path,
        seed=config.seed,
        batch_size=config.batch_size,
    )

    # Test model accuracy before ablation
    accuracy = eval_model_accuracy(hooked_model, data_loader, dtype=dtype, device=device)
    logger.info("Accuracy before ablation: %.2f%%", accuracy * 100)

    graph_module_names = [f"sections.{sec}" for sec in seq_model.sections if sec != "pre"]

    accuracies: dict[str, dict[int, float]] = run_ablations(
        interaction_matrices=interaction_matrices,
        node_layers=config.node_layers,
        hooked_model=hooked_model,
        data_loader=data_loader,
        graph_module_names=graph_module_names,
        ablate_every_vec_cutoff=config.ablate_every_vec_cutoff,
        device=device,
    )

    results = {
        "exp_name": config.exp_name,
        "config": json.loads(config.model_dump_json()),
        "accuracies": accuracies,
    }
    if config.exp_name is not None:
        with open(out_file, "w") as f:
            json.dump(results, f)
        logger.info("Wrote results to %s", out_file)


if __name__ == "__main__":
    fire.Fire(main)
