import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Union

import fire
import torch
import yaml
from pydantic import BaseModel, field_validator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rib.data_accumulator import collect_relu_interactions
from rib.hook_manager import HookedModel
from rib.models import MLP
from rib.types import TORCH_DTYPES
from rib.utils import REPO_ROOT, load_config, set_seed


class Config(BaseModel):
    exp_name: str
    mlp_path: Path
    batch_size: int
    seed: int
    # Remove eigenvectors with eigenvalues below this threshold.
    truncation_threshold: float
    # Whether to rotate the output layer to its eigenbasis.
    rotate_output: bool
    # Data type of all tensors (except those overriden in certain functions).
    dtype: str
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


def print_all_modules(mlp):
    for name, module in mlp.named_modules():
        print(name, ":", module)


def main(config_path_str: str) -> None:
    """Test for ReLU interactions (separate to main RIB algorithm)."""
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)
    set_seed(config.seed)

    with open(config.mlp_path.parent / "config.yaml", "r") as f:
        model_config_dict = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = load_mlp(model_config_dict, config.mlp_path)
    mlp.eval()  # Run in inference only
    mlp.to(device=torch.device(device), dtype=TORCH_DTYPES[config.dtype])
    hooked_mlp = HookedModel(mlp)

    test_loader = load_mnist_dataloader(
        train=False, batch_size=config.batch_size)

    relu_matrices = collect_relu_interactions(
        hooked_model=hooked_mlp,
        module_names=config.module_names,
        data_loader=test_loader,
        dtype=TORCH_DTYPES[config.dtype],
        device=device,
    )

    print(relu_matrices)


def load_local_config(config_path_str):
    """Load config (specifically, including MLP config) from local config file.

    Rest of RIB loads MLP config from FluidStack servers.
    """
    config_path = Path(config_path_str)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    # Module printing code
    #     CONFIG_PATH_STR = "experiments/mnist_relu_interactions/relu_interactions.yaml"
    #     config = load_local_config(CONFIG_PATH_STR)
    #     mlp = MLP(**config["mlp_config"])
    #     print_all_modules(mlp)

    pass
