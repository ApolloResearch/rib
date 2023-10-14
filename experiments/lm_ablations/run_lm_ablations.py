"""Run an LM on a dataset while rotating to and from a (truncated) rib or orthogonal basis.

The process is as follows:
    1. Load a SequentialTransformer model and dataset.
    2. For each number of ablated nodes `n`, create a rotation matrix that has the effect of
    rotating to and from the new basis with `n` fewer basis vectors.
    3. Run the test set through the model, applying the above rotations at each node layer, and
    calculate the resulting accuracy.
    5. Repeat steps 3 and 4 for a range of values of n, plotting the resulting accuracies.

A node layer is positioned at the input of each module specified in `node_layers` in the config.
In this script, we don't create a node layer at the output of the final module, as ablating nodes in
this layer is not useful.

Usage:
    python run_lm_ablations.py <path/to/yaml_config_file>

"""

import json
from pathlib import Path
from typing import Literal, Optional

import fire
import torch
from pydantic import BaseModel, Field, field_validator

from rib.ablations import load_basis_matrices, run_ablations
from rib.hook_manager import HookedModel
from rib.loader import create_data_loader, load_dataset, load_sequential_transformer
from rib.log import logger
from rib.types import TORCH_DTYPES
from rib.utils import eval_model_accuracy, load_config, overwrite_output, set_seed


class Config(BaseModel):
    exp_name: Optional[str]
    ablation_type: Literal["rib", "orthogonal"]
    interaction_graph_path: Path
    ablate_every_vec_cutoff: Optional[int] = Field(
        None,
        description="The point at which we start ablating every individual vector. If None, always ablate every vector.",
    )
    dataset: Literal["modular_arithmetic", "wikitext"]
    node_layers: list[str]
    batch_size: int
    dtype: str
    eps: Optional[float] = 1e-5
    seed: int

    @field_validator("dtype")
    def dtype_validator(cls, v):
        assert v in TORCH_DTYPES, f"dtype must be one of {TORCH_DTYPES}"
        return v


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)

    out_file = Path(__file__).parent / "out" / f"{config.exp_name}_ablation_results.json"
    if out_file.exists() and not overwrite_output(out_file):
        print("Exiting.")
        return

    out_file.parent.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    interaction_graph_info = torch.load(config.interaction_graph_path)

    assert (
        interaction_graph_info["config"]["tlens_pretrained"] is None
        and interaction_graph_info["config"]["tlens_model_path"] is not None
    ), "Currently can't build graphs for pretrained models due to memory limits."

    assert set(config.node_layers) <= set(
        interaction_graph_info["config"]["node_layers"]
    ), "The node layers in the config must be a subset of the node layers in the interaction graph."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = TORCH_DTYPES[config.dtype]

    basis_matrices = load_basis_matrices(
        interaction_graph_info=interaction_graph_info,
        node_layers=config.node_layers,
        ablation_type=config.ablation_type,
        dtype=dtype,
        device=device,
    )

    tlens_model_path = Path(interaction_graph_info["config"]["tlens_model_path"])

    seq_model, _ = load_sequential_transformer(
        node_layers=config.node_layers,
        last_pos_module_type=interaction_graph_info["config"]["last_pos_module_type"],
        tlens_pretrained=interaction_graph_info["config"]["tlens_pretrained"],
        tlens_model_path=tlens_model_path,
        eps=config.eps,
        dtype=dtype,
        device=device,
    )

    seq_model.eval()
    seq_model.to(device=torch.device(device), dtype=dtype)
    seq_model.fold_bias()
    hooked_model = HookedModel(seq_model)

    assert (
        interaction_graph_info["config"]["dataset"] == "modular_arithmetic"
    ), "Currently only supports modular arithmetic."

    dataset = load_dataset(
        dataset_type=config.dataset,
        return_set="test",
        tlens_model_path=tlens_model_path,
        seed=config.seed,
        frac_train=0.5,  # Take a random 50% split of the dataset
    )
    data_loader = create_data_loader(dataset, shuffle=True, batch_size=config.batch_size)

    # Test model accuracy before ablation
    accuracy = eval_model_accuracy(hooked_model, data_loader, dtype=dtype, device=device)
    logger.info("Accuracy before ablation: %.2f%%", accuracy * 100)

    graph_module_names = [f"sections.{sec}" for sec in seq_model.sections if sec != "pre"]

    accuracies: dict[str, dict[int, float]] = run_ablations(
        basis_matrices=basis_matrices,
        node_layers=config.node_layers,
        hooked_model=hooked_model,
        data_loader=data_loader,
        graph_module_names=graph_module_names,
        ablate_every_vec_cutoff=config.ablate_every_vec_cutoff,
        device=device,
    )

    if config.exp_name is not None:
        results = {
            "config": json.loads(config.model_dump_json()),
            "accuracies": accuracies,
        }
        with open(out_file, "w") as f:
            json.dump(results, f)
        logger.info("Wrote results to %s", out_file)


if __name__ == "__main__":
    fire.Fire(main)
