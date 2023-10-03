"""Run an mlp on MNIST while rotating to and from a (truncated) rib or orthogonal basis.

The process is as follows:
    1. Load an MLP trained on MNIST and a test set of MNIST images.
    2. For each number of ablated nodes `n`, create a rotation matrix that has the effect of
    rotating to and from the new basis with `n` fewer basis vectors.
    3. Run the test set through the MLP, applying the above rotations at each node layer, and
    calculate the resulting accuracy.
    5. Repeat steps 3 and 4 for a range of values of n, plotting the resulting accuracies.

A node layer is positioned at the input of each module specified in `node_layers` in the config.
In this script, we don't create a node layer at the output of the final module, as ablating nodes in
this layer is not useful.

Usage:
    python run_mnist_ablations.py <path/to/yaml_config_file>

"""

import json
from pathlib import Path
from typing import Literal, Optional

import fire
import torch
from pydantic import BaseModel, Field, field_validator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rib.ablations import load_basis_matrices, run_ablations
from rib.hook_manager import HookedModel
from rib.log import logger
from rib.models import MLP
from rib.types import TORCH_DTYPES
from rib.utils import (
    REPO_ROOT,
    eval_model_accuracy,
    load_config,
    overwrite_output,
    set_seed,
)


class Config(BaseModel):
    exp_name: Optional[str]
    ablation_type: Literal["rib", "orthogonal"]
    interaction_graph_path: Path
    ablate_every_vec_cutoff: Optional[int] = Field(
        None,
        description="The point at which we start ablating every individual vector. If None, always ablate every vector.",
    )
    dtype: str
    node_layers: list[str]
    batch_size: int
    seed: int

    @field_validator("dtype")
    def dtype_validator(cls, v):
        assert v in TORCH_DTYPES, f"dtype must be one of {TORCH_DTYPES}"
        return v


def load_mlp(config_dict: dict, mlp_path: Path) -> MLP:
    mlp = MLP(
        hidden_sizes=config_dict["model"]["hidden_sizes"],
        input_size=784,
        output_size=10,
        activation_fn=config_dict["model"]["activation_fn"],
        bias=config_dict["model"]["bias"],
        fold_bias=config_dict["model"]["fold_bias"],
    )
    mlp.load_state_dict(torch.load(mlp_path))
    return mlp


def load_mnist_dataloader(train: bool = False, batch_size: int = 64) -> DataLoader:
    transform = transforms.ToTensor()
    test_data = datasets.MNIST(
        root=REPO_ROOT / ".data", train=train, download=True, transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return test_loader


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)

    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = TORCH_DTYPES[config.dtype]
    interaction_graph_info = torch.load(config.interaction_graph_path)

    basis_matrices = load_basis_matrices(
        interaction_graph_info=interaction_graph_info,
        node_layers=config.node_layers,
        ablation_type=config.ablation_type,
        dtype=dtype,
        device=device,
    )

    out_file = Path(__file__).parent / "out" / f"{config.exp_name}_ablation_results.json"
    if out_file.exists() and not overwrite_output(out_file):
        print("Exiting.")
        return

    out_file.parent.mkdir(parents=True, exist_ok=True)

    mlp = load_mlp(
        config_dict=interaction_graph_info["model_config_dict"],
        mlp_path=interaction_graph_info["config"]["mlp_path"],
    )
    mlp.eval()
    mlp.to(device)
    mlp.to(dtype)
    hooked_mlp = HookedModel(mlp)

    test_loader = load_mnist_dataloader(train=False, batch_size=config.batch_size)

    # Test model accuracy before ablation
    accuracy = eval_model_accuracy(hooked_mlp, test_loader, dtype=dtype, device=device)
    logger.info("Accuracy before ablation: %.2f%%", accuracy * 100)

    accuracies: dict[str, dict[int, float]] = run_ablations(
        basis_matrices=basis_matrices,
        node_layers=config.node_layers,
        hooked_model=hooked_mlp,
        data_loader=test_loader,
        graph_module_names=config.node_layers,
        ablate_every_vec_cutoff=config.ablate_every_vec_cutoff,
        device=device,
        dtype=dtype,
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