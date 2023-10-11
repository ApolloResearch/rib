import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Union

import fire
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml
from pydantic import BaseModel, field_validator
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
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


def load_mlp(config_dict: dict, mlp_path: Path, device: str) -> MLP:
    mlp = MLP(
        hidden_sizes=config_dict["model"]["hidden_sizes"],
        input_size=784,
        output_size=10,
        activation_fn=config_dict["model"]["activation_fn"],
        bias=config_dict["model"]["bias"],
        fold_bias=config_dict["model"]["fold_bias"],
    )
    mlp.load_state_dict(torch.load(
        mlp_path, map_location=torch.device(device)))
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


def main(config_path_str: str, relu_metric_type: int) -> None:
    """Test for ReLU interactions (separate to main RIB algorithm).

    TODO: test on high bias - should be entire giant block fit.
    Histogram
    """
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)
    set_seed(config.seed)

    with open(config.mlp_path.parent / "config.yaml", "r") as f:
        model_config_dict = yaml.safe_load(f)

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    file_name = "relu_similarity_matrices"
    relu_matrices_save_file = Path(__file__).parent / file_name

    if relu_matrices_save_file.exists():
        with relu_matrices_save_file.open("rb") as f:
            relu_matrices = pickle.load(f)

    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mlp = load_mlp(model_config_dict, config.mlp_path, device=device)
        # print_all_modules(mlp) # Check module names were correctly defined
        mlp.eval()  # Run in inference only
        mlp.to(device=torch.device(device), dtype=TORCH_DTYPES[config.dtype])
        hooked_mlp = HookedModel(mlp)

        test_loader = load_mnist_dataloader(
            train=True, batch_size=config.batch_size)

        relu_matrices = collect_relu_interactions(
            hooked_model=hooked_mlp,
            module_names=config.module_names,
            data_loader=test_loader,
            dtype=TORCH_DTYPES[config.dtype],
            device=device,
            relu_metric_type=relu_metric_type
        )

        with open(relu_matrices_save_file, "wb") as f:
            pickle.dump(relu_matrices, f)

    for i, similarity_matrix in enumerate(list(relu_matrices.values())):

        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, annot=False,
                    cmap="YlGnBu", cbar=True, square=True)
        plt.savefig(
            out_dir / f"{config.exp_name}_similarity_matrix_{i}_type_{relu_metric_type}.png")

        plt.figure(figsize=(10,8))
        flat_similarity = np.array(similarity_matrix).flatten()
        print(torch.max(similarity_matrix))
        sns.histplot(flat_similarity, bins=100, kde=False)
        plt.savefig(
            out_dir / f"{config.exp_name}_similarity_histogram_{i}_type_{relu_metric_type}.png")

        match relu_metric_type:
            case 0:
                # Threshold
                threshold = 0.95
                similarity_matrix[similarity_matrix < threshold] = 0
                # Transform into distance matrix
                distance_matrix = 1 - similarity_matrix
            case 1:
                similarity_matrix = torch.min(
                    similarity_matrix, similarity_matrix.T) # Make symmetric
                # similarity_matrix = rescale(similarity_matrix)
                distance_matrix = similarity_matrix
                # Threshold
                threshold = 2
                distance_matrix[distance_matrix > threshold] = torch.max(distance_matrix)

        # Deal with zeros on the diagonal
        distance_matrix = distance_matrix.fill_diagonal_(0)

        # Create linkage matrix using clustering algorithm
        linkage_matrix = linkage(squareform(
            distance_matrix), method='complete')
        order = leaves_list(linkage_matrix)
        rearranged_similarity_matrix = similarity_matrix[order, :][:, order]

        plt.figure(figsize=(10, 8))
        sns.heatmap(rearranged_similarity_matrix, annot=False,
                    cmap="YlGnBu", cbar=True, square=True)
        plt.title("Reordered Similarity Matrix")
        plt.savefig(
            out_dir / f"{config.exp_name}_rearranged_similarity_matrix_{i}_type_{relu_metric_type}.png")


def load_local_config(config_path_str):
    """Load config (specifically, including MLP config) from local config file.

    Rest of RIB loads MLP config from FluidStack servers.
    """
    config_path = Path(config_path_str)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def rescale(tensor: torch.Tensor) -> torch.Tensor:
    """Rescale tensor to [0, 1] range."""
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


if __name__ == "__main__":
    # Module printing code
    #     CONFIG_PATH_STR = "experiments/mnist_relu_interactions/relu_interactions.yaml"
    #     config = load_local_config(CONFIG_PATH_STR)
    #     mlp = MLP(**config["mlp_config"])
    #     print_all_modules(mlp)
    fire.Fire(main)
