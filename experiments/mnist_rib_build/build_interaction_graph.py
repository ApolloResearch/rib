"""Calculate the interaction graph of an MLP trained on MNIST.

The full algorithm is Algorithm 1 of https://www.overleaf.com/project/6437d0bde0eaf2e8c8ac3649
The process is as follows:
    1. Load an MLP trained on MNIST and a test set of MNIST images.
    2. Collect gram matrices at each node layer.
    3. Calculate the interaction rotation matrices (labelled C in the paper) for each node layer. A
        node layer is positioned at the input of each module_name specified in the config, as well
        as at the output of the final module.
    4. Calculate the edges of the interaction graph between each node layer.

Usage:
    python build_interaction_graph.py <path/to/yaml_config_file>

"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import fire
import torch
import yaml
from pydantic import BaseModel, field_validator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rib.data_accumulator import collect_gram_matrices, collect_interaction_edges
from rib.hook_manager import HookedModel
from rib.interaction_algos import calculate_interaction_rotations
from rib.log import logger
from rib.models import MLP
from rib.types import TORCH_DTYPES
from rib.utils import REPO_ROOT, load_config, overwrite_output


class Config(BaseModel):
    exp_name: str
    mlp_path: Path
    batch_size: int
    seed: int
    truncation_threshold: float  # Remove eigenvectors with eigenvalues below this threshold.
    rotate_output: bool  # Whether to rotate the output layer to its eigenbasis.
    dtype: str  # Data type of all tensors (except those overriden in certain functions).
    module_names: list[str]

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
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return test_loader


def main(config_path_str: str) -> Optional[dict[str, Any]]:
    """Implement the main algorithm and store the graph to disk."""
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)
    torch.manual_seed(config.seed)

    with open(config.mlp_path.parent / "config.yaml", "r") as f:
        model_config_dict = yaml.safe_load(f)

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_interaction_graph_file = out_dir / f"{config.exp_name}_interaction_graph.pt"
    if out_interaction_graph_file.exists() and not overwrite_output(out_interaction_graph_file):
        logger.info("Exiting.")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = load_mlp(model_config_dict, config.mlp_path)
    mlp.eval()
    mlp.to(device=torch.device(device), dtype=TORCH_DTYPES[config.dtype])
    hooked_mlp = HookedModel(mlp)

    test_loader = load_mnist_dataloader(train=False, batch_size=config.batch_size)

    gram_matrices = collect_gram_matrices(
        hooked_model=hooked_mlp,
        module_names=config.module_names,
        data_loader=test_loader,
        device=device,
        dtype=config.dtype,
        collect_output_gram=True,
    )

    Cs = calculate_interaction_rotations(
        gram_matrices=gram_matrices,
        module_names=config.module_names,
        hooked_model=hooked_mlp,
        data_loader=test_loader,
        device=device,
        dtype=config.dtype,
        truncation_threshold=config.truncation_threshold,
        rotate_output=config.rotate_output,
    )

    E_hats = collect_interaction_edges(
        Cs=Cs,
        hooked_model=hooked_mlp,
        data_loader=test_loader,
        device=device,
        dtype=config.dtype,
    )

    results = {
        "exp_name": config.exp_name,
        "gram_matrices": gram_matrices,
        "interaction_rotations": [asdict(C_info) for C_info in Cs],
        "edges": [(module, E_hats[module]) for module in config.module_names],
        "config": json.loads(config.model_dump_json()),
        "model_config_dict": model_config_dict,
    }

    # Save the results (which include torch tensors) to file
    torch.save(results, out_interaction_graph_file)
    logger.info("Saved results to %s", out_interaction_graph_file)
    return results


if __name__ == "__main__":
    fire.Fire(main)
