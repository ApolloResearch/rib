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
from jaxtyping import Float
from torch import Tensor
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
    """Use for choosing which modules go in config file."""
    for name, module in mlp.named_modules():
        print(name, ":", module)


# Weights ========================================================

def edit_weights_fn(model: HookedModel, layer_name: str) -> None:
    """Weight matrix dimensions are rows=output, cols=input."""
    layer = get_nested_attribute(model, layer_name)

    weights = layer.weight.data.clone()
    print(f"Weights shape{weights.shape}")
    output_neurons, input_neurons = weights.shape
    weights[:output_neurons//2, -1] = 1e8
    layer.weight.data.copy_(weights)


def get_nested_attribute(obj, attr_name):
    attrs = attr_name.split('.')
    current_attr = obj
    for attr in attrs:
        current_attr = getattr(current_attr, attr)
    return current_attr


# Helper functions for main ========================================================

def relu_plotting(similarity_matrices: list[Float[Tensor, "d_hidden d_hidden"]], out_dir: Path, relu_metric_type: int, edit_weights: bool) -> None:
    for i, similarity_matrix in enumerate(list(similarity_matrices.values())):
        # Plot raw matrix values before clustering
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, annot=False,
                    cmap="YlGnBu", cbar=True, square=True)
        plt.savefig(
            out_dir / f"mat_{i}_type_{relu_metric_type}_editweights_{edit_weights}.png")

        # Plot histogram
        plt.figure(figsize=(10,8))
        flat_similarity = np.array(similarity_matrix).flatten()
        sns.histplot(flat_similarity, bins=100, kde=False)
        plt.savefig(
            out_dir / f"hist_{i}_type_{relu_metric_type}_editweights_{edit_weights}.png")

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
                # Threshold distance matrix
                threshold = 1
                distance_matrix[distance_matrix > threshold] = threshold

        # Deal with zeros on the diagonal
        distance_matrix = distance_matrix.fill_diagonal_(0)

        # Create linkage matrix using clustering algorithm
        linkage_matrix = linkage(squareform(
            distance_matrix), method='complete')
        order = leaves_list(linkage_matrix)
        rearranged_similarity_matrix = similarity_matrix[order, :][:, order]

        # Plot sorted similarity matrix via indices obtained from distance matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(rearranged_similarity_matrix, annot=False,
                    cmap="YlGnBu", cbar=True, square=True)
        plt.title("Reordered Similarity Matrix")
        plt.savefig(
            out_dir / f"rearr_mat_{i}_type_{relu_metric_type}_editweights_{edit_weights}.png")


def load_local_config(config_path_str: str) -> dict:
    """Load config (specifically, including MLP config) from local config file.

    Rest of RIB loads MLP config from FluidStack servers.
    """
    config_path = Path(__file__).parent / config_path_str
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def check_and_open_file(file_path: str, get_var_fn: callable, config_path_str: str, **kwargs) -> Any:
    """Load information from pickle file into a variable and return it."""
    if file_path.exists():
        with file_path.open("rb") as f:
            var = pickle.load(f)
    else:
        var = get_var_fn(config_path_str, file_path, **kwargs)

    return var


def get_relu_similarities(config_path_str: str, file_path: Path, relu_metric_type: int, edit_weights: bool) -> dict[str, Float[Tensor, "d_hidden d_hidden"]]:
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)
    set_seed(config.seed)

    with open(config.mlp_path.parent / "config.yaml", "r") as f:
        model_config_dict = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = load_mlp(model_config_dict, config.mlp_path, device=device)
    # print_all_modules(mlp) # Check module names were correctly defined
    layer_list = ["layers.0.linear", "layers.1.linear"]
    if edit_weights:
        for layer in layer_list:
            edit_weights_fn(mlp, layer)
    mlp.eval()  # Run in inference only
    mlp.to(device=torch.device(device), dtype=TORCH_DTYPES[config.dtype])
    hooked_mlp = HookedModel(mlp)

    train_loader = load_mnist_dataloader(
        train=True, batch_size=config.batch_size)

    relu_similarity_matrix: dict[str, Float[Tensor, "d_hidden d_hidden"]] = collect_relu_interactions(
        hooked_model=hooked_mlp,
        module_names=config.module_names,
        data_loader=train_loader,
        dtype=TORCH_DTYPES[config.dtype],
        device=device,
        relu_metric_type=relu_metric_type
    )

    with open(file_path, "wb") as f:
        pickle.dump(relu_similarity_matrix, f)

    return relu_similarity_matrix


# Main ========================================================

def relu_similarity_main(config_path_str: str, relu_metric_type: int, edit_weights: bool) -> None:
    """MAIN FUNCTION 1. Test for ReLU interactions (separate to main RIB algorithm)."""
    out_dir = Path(__file__).parent / "out_relu_interactions"
    out_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"relu_similarity_matrices_editweights_{edit_weights}"
    relu_matrices_save_file = Path(__file__).parent / file_name

    relu_matrices = check_and_open_file(
        file_path=relu_matrices_save_file,
        get_var_fn=get_relu_similarities,
        config_path_str=config_path_str,
        relu_metric_type=relu_metric_type,
        edit_weights=edit_weights,
    )

    relu_plotting(relu_matrices, out_dir, relu_metric_type, edit_weights)


if __name__ == "__main__":
    fire.Fire(relu_similarity_main)
    """Run above: python run_relu_interactions.py relu_interactions.yaml 1 False"""