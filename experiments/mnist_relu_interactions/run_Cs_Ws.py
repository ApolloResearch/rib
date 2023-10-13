import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Union

import fire
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from jaxtyping import Float
from pydantic import BaseModel, field_validator
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rib.data_accumulator import (
    collect_activations_and_rotate,
    collect_gram_matrices,
    collect_test_edges,
)
from rib.hook_fns import acts_forward_hook_fn
from rib.hook_manager import HookedModel
from rib.interaction_algos import (
    Eigenvectors,
    InteractionRotation,
    calculate_interaction_rotations,
)
from rib.models import MLP
from rib.types import TORCH_DTYPES
from rib.utils import REPO_ROOT, load_config, set_seed


class Config(BaseModel):
    exp_name: str
    mlp_path: Path
    batch_size: int
    seed: int
    truncation_threshold: float  # Remove eigenvectors with eigenvalues below this threshold.
    rotate_output: bool  # Whether to rotate the output layer to its eigenbasis.
    n_intervals: int  # The number of intervals to use for integrated gradients.
    dtype: str  # Data type of all tensors (except those overriden in certain functions).
    node_layers: list[str]

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
    output_neurons, input_neurons = weights.shape
    weights[:output_neurons//2, -1] = 1e8
    layer.weight.data.copy_(weights)


def get_nested_attribute(obj, attr_name):
    attrs = attr_name.split('.')
    current_attr = obj
    for attr in attrs:
        current_attr = getattr(current_attr, attr)
    return current_attr


def extract_weights(model: torch.nn.Module) -> list[torch.Tensor]:
    """Recursively extract weights from each layer of the model and return them as a list."""
    weights_list = []

    for name, layer in model.named_children():
        if hasattr(layer, 'weight'):
            weights = get_nested_attribute(model, name + '.weight').data.clone()
            weights_list.append(weights)
        # Recursively extract weights from nested children
        weights_list.extend(extract_weights(layer))

    return weights_list


# Helper functions for main ========================================================

def get_Cs(
    config_path_str: str,
    file_path: str,
) -> dict[str,
    Union[list[InteractionRotation],
    list[Eigenvectors],
    list[Float[Tensor, "d_hidden_trunc d_hidden_extra_trunc"]],
    list[Float[Tensor, "d_hidden_extra_trunc d_hidden_trunc"]]],
]:
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)
    set_seed(config.seed)

    with open(config.mlp_path.parent / "config.yaml", "r") as f:
        model_config_dict = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = load_mlp(model_config_dict, config.mlp_path, device=device)
    # print_all_modules(mlp) # Check module names were correctly defined
    mlp.eval()  # Run in inference only
    mlp.to(device=torch.device(device), dtype=TORCH_DTYPES[config.dtype])
    hooked_mlp = HookedModel(mlp)

    train_loader = load_mnist_dataloader(
        train=True, batch_size=config.batch_size)

    gram_matrices = collect_gram_matrices(
        hooked_model=hooked_mlp,
        module_names=config.node_layers,
        data_loader=train_loader,
        dtype=TORCH_DTYPES[config.dtype],
        device=device,
        collect_output_gram=True,
    )

    # Calls on collect_M_dash_and_Lambda_dash
    # Builds sqrt sorted Lambda matrix and its inverse
    Cs, Us, Lambda_abs_sqrts, Lambda_abs_sqrt_pinvs, U_D_sqrt_pinv_Vs, U_D_sqrt_Vs = calculate_interaction_rotations(
        gram_matrices=gram_matrices,
        module_names=config.node_layers,
        hooked_model=hooked_mlp,
        data_loader=train_loader,
        dtype=TORCH_DTYPES[config.dtype],
        device=device,
        n_intervals=config.n_intervals,
        truncation_threshold=config.truncation_threshold,
        rotate_output=config.rotate_output,
    )

    C_list = [C_info.C for C_info in Cs]
    C_pinv_list = [C_info.C_pinv for C_info in Cs]

    with open(file_path, "wb") as f:
        pickle.dump({"C": C_list, "C_pinv": C_pinv_list, "Lambda_abs_sqrts": Lambda_abs_sqrts, "Lambda_abs_sqrt_pinvs": Lambda_abs_sqrt_pinvs, "U_D_sqrt_pinv_Vs": U_D_sqrt_pinv_Vs, "U_D_sqrt_Vs": U_D_sqrt_Vs}, f)

    return {"C": C_list, "C_pinv": C_pinv_list, "Lambda_abs_sqrts": Lambda_abs_sqrts, "Lambda_abs_sqrt_pinvs": Lambda_abs_sqrt_pinvs, "U_D_sqrt_pinv_Vs": U_D_sqrt_pinv_Vs, "U_D_sqrt_Vs": U_D_sqrt_Vs}


def plot(matrix_list: list[Float[Tensor, "d_hidden d_hidden"]], var_name: str, out_dir: Path) -> None:
    for i, matrix in enumerate(list(matrix_list)):
        # Calculate figsize based on matrix dimensions and the given figsize_per_unit
        nrows, ncols = matrix.shape
        figsize = (int(ncols * 0.1), int(nrows * 0.1))
        plt.figure(figsize=figsize)
        sns.heatmap(matrix, annot=False,
                    cmap="YlGnBu", cbar=True, square=True)
        plt.savefig(
            out_dir / f"{var_name}_{i}.png")


def get_rotated_Ws(
    config_path_str: str,
    file_path: str,
    C_pinv_list: list[Float[Tensor, "d_hidden1 d_hidden2"]],
) -> list[Float[Tensor, "layer_count d_hidden d_hidden"]]:
    """Extract W^l, perform paper-equivalent right multiplication of psuedoinverse
    of C^l."""
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)
    set_seed(config.seed)

    with open(config.mlp_path.parent / "config.yaml", "r") as f:
        model_config_dict = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = load_mlp(model_config_dict, config.mlp_path, device=device)
    mlp.eval()

    weights_list = extract_weights(mlp)
    rotated_weights_list = []
    for weight_matrix, C_pinv in zip(weights_list, C_pinv_list):
        rotated_weights_list.append(weight_matrix @ C_pinv)

    with open(file_path, "wb") as f:
        pickle.dump(rotated_weights_list, f)

    return rotated_weights_list


def check_and_open_file(file_path: Path, get_var_fn: callable, config_path_str: str, **kwargs) -> Union[Any, tuple[Any, ...]]:
    """Load information from pickle file into a variable and return it.

    Note the return type is overloaded to allow for tuples.
    """
    if file_path.exists():
        with file_path.open("rb") as f:
            var = pickle.load(f)
    else:
        var = get_var_fn(config_path_str, file_path, **kwargs)

    return var


def get_f_hats(
    config_path_str: str,
    file_path: str,
    Cs_list: list[Float[Tensor, "d_hidden d_hidden"]],
) -> list[Float[Tensor, "batch d_hidden"]]:
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)
    set_seed(config.seed)

    with open(config.mlp_path.parent / "config.yaml", "r") as f:
        model_config_dict = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = load_mlp(model_config_dict, config.mlp_path, device=device)
    # print_all_modules(mlp) # Check module names were correctly defined
    mlp.eval()  # Run in inference only
    mlp.to(device=torch.device(device), dtype=TORCH_DTYPES[config.dtype])
    hooked_mlp = HookedModel(mlp)

    train_loader = load_mnist_dataloader(
        train=True, batch_size=config.batch_size)

    # Get functions rotated by non rescaled Cs
    f_hats = collect_activations_and_rotate(
        Cs=Cs_list,
        hooked_model=hooked_mlp,
        module_names=config.node_layers,
        data_loader=train_loader,
        dtype=TORCH_DTYPES[config.dtype],
        device=device,
    )

    with open(file_path, "wb") as f:
        pickle.dump(f_hats, f)

    return f_hats


def get_edges(
    config_path_str: str,
    file_path: str,
    Cs_list: list[Float[Tensor, "d_hidden d_hidden"]],
    Cs_unscaled_list: list[Float[Tensor, "d_hidden d_hidden"]],
    W_hat_list: list[Float[Tensor, "d_hidden d_hidden"]],
) -> list[Float[Tensor, ""]]:
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)
    set_seed(config.seed)

    with open(config.mlp_path.parent / "config.yaml", "r") as f:
        model_config_dict = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = load_mlp(model_config_dict, config.mlp_path, device=device)
    # print_all_modules(mlp) # Check module names were correctly defined
    mlp.eval()  # Run in inference only
    mlp.to(device=torch.device(device), dtype=TORCH_DTYPES[config.dtype])
    hooked_mlp = HookedModel(mlp)

    train_loader = load_mnist_dataloader(
        train=True, batch_size=config.batch_size)

    edges = collect_test_edges(
        Cs_unscaled=Cs_unscaled_list,
        Cs=Cs_list,
        W_hats=W_hat_list,
        hooked_model=hooked_mlp,
        module_names=config.node_layers,
        data_loader=train_loader,
        dtype=TORCH_DTYPES[config.dtype],
        device=device,
    )

    with open(file_path, "wb") as f:
        pickle.dump(edges, f)

    return edges


def Cs_Ws_main(config_path_str: str) -> None:
    """MAIN FUNCTION 2. Check how sparse Cs and rotated Ws are in equation:
    C^{l+1} O(x) W C+^l C^l f(x)
    """
    out_dir = Path(__file__).parent / "out_Cs_Ws"
    out_dir.mkdir(parents=True, exist_ok=True)

    Cs_save_file = Path(__file__).parent / "Cs"
    Ws_save_file = Path(__file__).parent / "Ws"
    fhats_save_file = Path(__file__).parent / "fhats"
    edges_save_file = Path(__file__).parent / "edges"

    # List of InteractionRotation objects
    Cs_and_Lambdas: dict[str, list[InteractionRotation]] = check_and_open_file(
        file_path=Cs_save_file,
        get_var_fn=get_Cs,
        config_path_str=config_path_str
    )

    C_list, C_pinv_list, Lambda_abs_sqrts_list, Lambda_abs_sqrt_pinvs_list, U_D_sqrt_pinv_Vs_list, U_D_sqrt_Vs_list = Cs_and_Lambdas["C"], Cs_and_Lambdas["C_pinv"], Cs_and_Lambdas["Lambda_abs_sqrts"], Cs_and_Lambdas["Lambda_abs_sqrt_pinvs"], Cs_and_Lambdas["U_D_sqrt_pinv_Vs"], Cs_and_Lambdas["U_D_sqrt_Vs"]

    rescaled_C_list = []
    rescaled_C_pinv_list = []
    for C, C_pinv, Lambda_abs_sqrt, Lambda_abs_sqrt_pinv in zip(C_list, C_pinv_list, Lambda_abs_sqrts_list, Lambda_abs_sqrt_pinvs_list):
        rescaled_C_list.append(C @ Lambda_abs_sqrt_pinv)
        rescaled_C_pinv_list.append(Lambda_abs_sqrt @ C_pinv)

    # Can either use original or rescaled C list in fn below
    W_list = check_and_open_file(
        file_path=Ws_save_file,
        get_var_fn=get_rotated_Ws,
        config_path_str=config_path_str,
        C_pinv_list=U_D_sqrt_pinv_Vs_list
    )

    edges = check_and_open_file(
        file_path=edges_save_file,
        get_var_fn=get_edges,
        config_path_str=config_path_str,
        Cs_list=C_list,
        Cs_unscaled_list=U_D_sqrt_pinv_Vs_list,
        W_hat_list=W_list,
    )

    plot(rescaled_C_list, "C", out_dir)
    plot(W_list, "W", out_dir)
    plot(U_D_sqrt_Vs_list, "U_D_sqrt_V", out_dir)
    plot(U_D_sqrt_pinv_Vs_list, "U_D_sqrt_pinv_V", out_dir)
    plot(edges, "edges", out_dir)


if __name__ == "__main__":
    fire.Fire(Cs_Ws_main)

