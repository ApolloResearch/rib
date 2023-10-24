import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Union

import fire
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from jaxtyping import Float, Int
from pydantic import BaseModel, field_validator
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rib.data_accumulator import (
    calculate_all_swapped_iterative_relu_loss,
    calculate_swapped_relu_loss,
    collect_gram_matrices,
    collect_relu_interactions,
)
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
    logits_node_layer: bool  # Whether to build an extra output node layer for the logits.
    rotate_final_node_layer: bool  # Whether to rotate the output layer to its eigenbasis.
    n_intervals: int  # The number of intervals to use for integrated gradients.
    dtype: str  # Data type of all tensors (except those overriden in certain functions).
    module_names: list[str]
    node_layers: list[str]
    relu_metric_type: int
    edit_weights: bool

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

def relu_plotting(similarity_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]], out_dir: Path, config: Config) -> None:
    for i, similarity_matrix in enumerate(list(similarity_matrices.values())):
        # Plot raw matrix values before clustering
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, annot=False,
                    cmap="YlGnBu", cbar=True, square=True)
        plt.savefig(
            out_dir / f"mat_{i}_type_{config.relu_metric_type}_editweights_{config.edit_weights}.png")

        # Plot histogram
        plt.figure(figsize=(10,8))
        flat_similarity = np.array(similarity_matrix).flatten()
        sns.histplot(flat_similarity, bins=100, kde=False)
        plt.savefig(
            out_dir / f"hist_{i}_type_{config.relu_metric_type}_editweights_{config.edit_weights}.png")

        match config.relu_metric_type:
            case 0:
                # Threshold
                threshold = 0.95
                similarity_matrix[similarity_matrix < threshold] = 0
                distance_matrix = 1 - similarity_matrix
            case 1 | 2:
                distance_matrix = torch.max(similarity_matrix, similarity_matrix.T) # Make symmetric
                threshold = 15
                distance_matrix[distance_matrix > threshold] = threshold
            case 3:
                distance_matrix = torch.max(torch.abs(similarity_matrix), torch.abs(similarity_matrix).T)

        # Deal with zeros on diagonal
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
            out_dir / f"rearr_mat_{i}_type_{config.relu_metric_type}_editweights_{config.edit_weights}.png")


def load_local_config(config_path_str: str) -> dict:
    """Load config (specifically, including MLP config) from local config file.

    Rest of RIB loads MLP config from FluidStack servers.
    """
    config_path = Path(__file__).parent / config_path_str
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_Cs(
    model: nn.Module,
    config: Config,
    file_path: str,
) -> dict[str,
    Union[list[InteractionRotation],
    list[Eigenvectors],
    list[Float[Tensor, "d_hidden_trunc d_hidden_extra_trunc"]],
    list[Float[Tensor, "d_hidden_extra_trunc d_hidden_trunc"]]],
]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device=torch.device(device), dtype=TORCH_DTYPES[config.dtype])
    hooked_model = HookedModel(model)

    train_loader = load_mnist_dataloader(train=True, batch_size=config.batch_size)

    # Only need gram matrix for logits if we're rotating the final node layer
    collect_output_gram = config.logits_node_layer and config.rotate_final_node_layer

    gram_matrices = collect_gram_matrices(
        hooked_model=hooked_model,
        module_names=config.node_layers,
        data_loader=train_loader,
        dtype=TORCH_DTYPES[config.dtype],
        device=device,
        collect_output_gram=collect_output_gram,
    )

    # Calls on collect_M_dash_and_Lambda_dash
    # Builds sqrt sorted Lambda matrix and its inverse
    Cs, Us, Lambda_abs_sqrts, Lambda_abs_sqrt_pinvs, U_D_sqrt_pinv_Vs, U_D_sqrt_Vs, Lambda_dashes = calculate_interaction_rotations(
        gram_matrices=gram_matrices,
        module_names=config.node_layers,
        hooked_model=hooked_model,
        data_loader=train_loader,
        dtype=TORCH_DTYPES[config.dtype],
        device=device,
        n_intervals=config.n_intervals,
        truncation_threshold=config.truncation_threshold,
        rotate_final_node_layer=config.rotate_final_node_layer,
    )

    C_list = [C_info.C for C_info in Cs]
    C_pinv_list = [C_info.C_pinv for C_info in Cs]

    with open(file_path, "wb") as f:
        torch.save({"C": C_list, "C_pinv": C_pinv_list, "Lambda_abs_sqrts": Lambda_abs_sqrts, "Lambda_abs_sqrt_pinvs": Lambda_abs_sqrt_pinvs, "U_D_sqrt_pinv_Vs": U_D_sqrt_pinv_Vs, "U_D_sqrt_Vs": U_D_sqrt_Vs, "Lambda_dashes": Lambda_dashes, "Cs raw": Cs, "Us raw": Us, "gram matrices": gram_matrices,}, f)

    return {"C": C_list, "C_pinv": C_pinv_list, "Lambda_abs_sqrts": Lambda_abs_sqrts, "Lambda_abs_sqrt_pinvs": Lambda_abs_sqrt_pinvs, "U_D_sqrt_pinv_Vs": U_D_sqrt_pinv_Vs, "U_D_sqrt_Vs": U_D_sqrt_Vs, "Lambda_dashes": Lambda_dashes, "Cs raw": Cs, "Us raw": Us, "gram matrices": gram_matrices}


def get_relu_similarities(
    model: nn.Module,
    config: Config,
    file_path: Path,
    Cs: list[Float[Tensor, "d_hidden1 d_hidden2"]],
    Lambda_dashes: list[Float[Tensor, "d_hidden d_hidden"]],
) -> dict[str, Float[Tensor, "d_hidden d_hidden"]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer_list = ["layers.0.linear", "layers.1.linear"]
    if config.edit_weights:
        for layer in layer_list:
            edit_weights_fn(model, layer)
    model.to(device=torch.device(device), dtype=TORCH_DTYPES[config.dtype])
    hooked_mlp = HookedModel(model)

    train_loader = load_mnist_dataloader(
        train=True, batch_size=config.batch_size)

    relu_similarity_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]] = collect_relu_interactions(
        hooked_model=hooked_mlp,
        module_names=config.module_names,
        data_loader=train_loader,
        dtype=TORCH_DTYPES[config.dtype],
        device=device,
        relu_metric_type=config.relu_metric_type,
        Cs=Cs,
        Lambda_dashes=Lambda_dashes,
        layer_module_names=config.node_layers,
        n_intervals=config.n_intervals,
        unhooked_model=model,
    )

    with open(file_path, "wb") as f:
        torch.save(relu_similarity_matrices, f)

    return relu_similarity_matrices


def check_and_open_file(
    file_path: Path,
    get_var_fn: callable,
    model: nn.Module,
    config: Config,
    device: str,
    **kwargs
) -> Union[Any, tuple[Any, ...]]:
    """Load information from pickle file into a variable and return it.

    Note the return type is overloaded to allow for tuples.
    """
    if file_path.exists():
        with file_path.open("rb") as f:
            var = torch.load(f, map_location=device)
    else:
        var = get_var_fn(model, config, file_path, **kwargs)

    return var


def swap_single_layer(
    hooked_model: HookedModel,
    relu_matrices: list[Float[Tensor, "d_hidden d_hidden"]],
    config: Config,
    layer_num: int
) -> None:
    """Now that we have gotten our similarity matrices, swap in forward pass.
    Check loss and accuracy drop, and plot as a function of similarity tolerance for pruning."""
    num_valid_swaps_list: list[int] = []
    loss_change_list: list[float] = []
    acc_change_list: list[float] = []
    random_loss_change_list: list[float] = []
    random_acc_change_list: list[float] = []
    tols = [[1e-3, 7e-4, 4e-4, 1e-4, 7e-5, 4e-5], [1e-4, 7e-5, 4e-5, 1e-5, 7e-6, 4e-6]]
    matrix = list(relu_matrices.values())[layer_num]
    module_name = list(config.module_names)[layer_num]

    for i in range(len(tols[0])):
        matrix = torch.abs(matrix)
        tol = tols[layer_num][i]
        diag_mask = torch.eye(matrix.size(0), matrix.size(1)).bool().to(matrix.device)
        matrix = matrix.masked_fill_(diag_mask, float('inf'))

        row_min_vals, row_min_idxs = torch.min(matrix, dim=-1)
        # Keep when minimum value in row is above some threshold
        keep_indices = torch.nonzero(row_min_vals > tol)
        num_valid_swaps = matrix.shape[0] - len(keep_indices)
        num_valid_swaps_list.append(num_valid_swaps)

        # Where no replacement wanted, set the element to equal index itself
        row_min_idxs[keep_indices] = keep_indices

        print(f"num_valid_swaps: {num_valid_swaps}")

        unhooked_loss, hooked_loss, random_hooked_loss, unhooked_accuracy, hooked_accuracy, random_hooked_accuracy = calculate_swapped_relu_loss(
            hooked_model=hooked_model,
            module_name=module_name, # Used for hooks
            data_loader=load_mnist_dataloader(train=False, batch_size=config.batch_size),
            dtype=TORCH_DTYPES[config.dtype],
            device=device,
            replacement_idx_list=row_min_idxs,
            num_replaced=num_valid_swaps,
        )

        print(
            f"unhooked loss {unhooked_loss}",
            f"hooked loss {hooked_loss}",
            f"random hooked loss {random_hooked_loss}"
            f"unhooked accuracy {unhooked_accuracy}",
            f"hooked accuracy {hooked_accuracy}",
            f"random hooked accuracy {random_hooked_accuracy}"
        )

        loss_change_list.append((hooked_loss - unhooked_loss) / unhooked_loss)
        acc_change_list.append(( hooked_accuracy - unhooked_accuracy) / unhooked_accuracy)
        random_loss_change_list.append((random_hooked_loss - unhooked_loss) / unhooked_loss)
        random_acc_change_list.append((random_hooked_accuracy - unhooked_accuracy) / unhooked_accuracy)

    plot_changes(num_valid_swaps_list, loss_change_list, random_loss_change_list, acc_change_list, random_acc_change_list, out_dir, layer_num)


def plot_changes(
    num_valid_swaps_list: list[int],
    loss_change_list: list[float],
    random_loss_change_list: list[float],
    acc_change_list: list[float],
    random_acc_change_list: list[float],
    out_dir: Path,
    layer_num: int = -1,
) -> None:
    plt.figure(figsize=(10, 5))
    # Loss change
    plt.subplot(1, 2, 1)
    plt.plot(num_valid_swaps_list, loss_change_list, label="not random")
    plt.plot(num_valid_swaps_list, random_loss_change_list, label="random")
    plt.xlabel("num_valid_swaps")
    plt.ylabel("Loss change")
    plt.legend()

    # Accuracy change
    plt.subplot(1, 2, 2)
    plt.plot(num_valid_swaps_list, acc_change_list, label="not random")
    plt.plot(num_valid_swaps_list, random_acc_change_list, label="random")
    plt.xlabel("num_valid_swaps")
    plt.ylabel("Accuracy change")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_dir / f"post_swap_all_layer_{layer_num}.png")


def swap_all_layers(
    hooked_model: HookedModel,
    relu_matrices: list[Float[Tensor, "d_hidden d_hidden"]],
    config: Config,
    device: str,
    out_dir: str,
    loss_threshold: float = 1.0,
    acc_threshold: float = 1.0,
) -> None:
    """Assuming universal synchronisation matrices (no need to recalculate in forward pass due to
    changes in previous layer), swap out subsets of all layers.

    If it is true that the sychronisation swaps made in the previous layer(s) are accurate, the
    fundamental computation in future layers should not break, because only activations (0s or 1s)
    are swapped, not pre-activations or post-activations.
    """
    # Outer loop: tolerance value; inner loop: layer
    num_valid_swaps_list_list: list[list[int]] = []
    loss_change_list: list[float] = []
    acc_change_list: list[float] = []
    random_loss_change_list: list[float] = []
    random_acc_change_list: list[float] = []
    # Need to manually fix as many lists as there are functions
    # tols = [[1e-3, 7e-4, 4e-4, 1e-4, 7e-5, 4e-5], [1e-4, 7e-5, 4e-5, 1e-5, 7e-6, 4e-6]]
    tols = [[1e-3, 7e-4, 4e-4, 1e-4, 7e-5, 4e-5], [5e-4, 5e-4, 5e-4, 5e-4, 5e-4, 5e-4]]
    matrix_list = list(relu_matrices.values())
    module_list = list(config.module_names)

    for i in range(len(tols[0])):
        num_valid_swaps_list: list[int] = []
        # Outer loop: modules; inner loop: indices of hidden layer
        replacement_idx_list: list[Int[Tensor, "d_hidden"]] = []
        for layer_num, (module, matrix) in enumerate(zip(module_list, matrix_list)):
            matrix = torch.abs(matrix)
            tol = tols[layer_num][i]
            diag_mask = torch.eye(matrix.size(0), matrix.size(1)).bool().to(matrix.device)
            matrix = matrix.masked_fill_(diag_mask, float('inf'))

            row_min_vals, row_min_idxs = torch.min(matrix, dim=-1)
            # Keep when minimum value in row is above some threshold
            keep_indices = torch.nonzero(row_min_vals > tol)
            num_valid_swaps = matrix.shape[0] - len(keep_indices)
            num_valid_swaps_list.append(num_valid_swaps)

            # Where no replacement wanted, set the element to equal index itself
            row_min_idxs[keep_indices] = keep_indices
            replacement_idx_list.append(row_min_idxs)

        print(f"num_valid_swaps: {num_valid_swaps_list}")
        num_valid_swaps_list_list.append(num_valid_swaps_list)

        unhooked_loss, hooked_loss, random_hooked_loss, unhooked_accuracy, hooked_accuracy, random_hooked_accuracy = calculate_all_swapped_iterative_relu_loss(
            hooked_model=hooked_model,
            module_names=module_list, # Used for hooks
            data_loader=load_mnist_dataloader(train=False, batch_size=config.batch_size),
            dtype=TORCH_DTYPES[config.dtype],
            device=device,
            replacement_idx_list=replacement_idx_list,
            num_replaced_list=num_valid_swaps_list,
        )

        print(
            f"unhooked loss {unhooked_loss}",
            f"hooked loss {hooked_loss}",
            f"random hooked loss {random_hooked_loss}\n"
            f"unhooked accuracy {unhooked_accuracy}",
            f"hooked accuracy {hooked_accuracy}",
            f"random hooked accuracy {random_hooked_accuracy}"
        )

        loss_change_list.append((hooked_loss - unhooked_loss) / unhooked_loss)
        acc_change_list.append(( hooked_accuracy - unhooked_accuracy) / unhooked_accuracy)
        random_loss_change_list.append((random_hooked_loss - unhooked_loss) / unhooked_loss)
        random_acc_change_list.append((random_hooked_accuracy - unhooked_accuracy) / unhooked_accuracy)

    mean_swaps_list = [sum(swaps) / len(swaps) for swaps in num_valid_swaps_list_list]
    plot_changes(mean_swaps_list, loss_change_list, random_loss_change_list, acc_change_list, random_acc_change_list, out_dir)


# def swap_all_layers_iterative(
#     hooked_model: HookedModel,
#     relu_matrices: list[Float[Tensor, "d_hidden d_hidden"]],
#     config: Config,
#     device: str,
#     out_dir: str,
#     loss_threshold: float = 10,
#     acc_threshold: float = -0.05,
# ) -> None:
#     """Assuming universal synchronisation matrices (no need to recalculate in forward pass due to
#     changes in previous layer),
#     **iteratively** swap out subsets of all layers, until loss or accuracy change meets pre-defined threshold.
#     """
#     # Outer loop: tolerance value; inner loop: layer
#     num_valid_swaps_list_list: list[list[int]] = []
#     loss_change_list: list[float] = []
#     acc_change_list: list[float] = []
#     random_loss_change_list: list[float] = []
#     random_acc_change_list: list[float] = []
#     # Need to manually fix as many lists as there are functions
#     # tols need to be on small end so they lead to changes within loss/acc threshold
#     tols = [[7e-5, 6e-5, 5e-5, 4e-5, 3e-5], [1e-4, 1e-4, 1e-4, 1e-4, 1e-4]]
#     matrix_list = list(relu_matrices.values())
#     module_list = list(config.module_names)
#     mat_dim = matrix_list[0].shape[0] # Used to initialise vectors below

#     for i in range(len(tols[0])):
#         num_valid_swaps_list: list[int] = []
#         # Outer loop: modules; inner loop: indices of hidden layer
#         its = 0

#         # Keep going until hit threshold, then save final loss and acc values
#         while loss_change < loss_threshold or accuracy_change > accuracy_threshold:

#             if its == 0: # Stores intermediate values between iterations of while loop
#                 combined_replacement_idx_list: List[Int[Tensor, "d_hidden"]] = [torch.zeros(mat_dim, dtype=torch.int)] * len(module_list)
#                 # At start, ignore none of the rows - all valid for considering swaps
#                 combined_idx_replaced_mask: List[Bool[Tensor, "d_hidden"]] = [torch.zeros(mat_dim, dtype=torch.bool)] * len(module_list)

#             for layer_num, (module, matrix) in enumerate(zip(module_list, matrix_list)):
#                 matrix = torch.abs(matrix)
#                 tol = tols[layer_num][i]
#                 diag_mask = torch.eye(mat_dim, matrix.size(1)).bool().to(matrix.device)
#                 matrix = matrix.masked_fill_(diag_mask, float('inf'))
#                 # Ignore all rows/m indices (repeat mask along cols) that were already replaced
#                 matrix = matrix.masked_fill_(repeat(combined_idx_replaced_mask, 'd1 -> d1 d2', d2=mat_dim), float('inf'))

#                 row_min_vals, row_min_idxs = torch.min(matrix, dim=-1)
#                 # Keep when minimum value in row is above some threshold
#                 # These indices only will be non-inf in the next iteration (IMPORTANT)
#                 # At each iteration, keep_indices is subset of rows without swaps in all iterations before
#                 keep_indices = torch.nonzero(row_min_vals > tol)
#                 num_valid_swaps = matrix.shape[0] - len(keep_indices)
#                 num_valid_swaps_list.append(num_valid_swaps)

#                 # Where no replacement wanted, set the element to equal index itself
#                 row_min_idxs[keep_indices] = keep_indices

#                 # Generate new mask of replaced indices for next iteration
#                 combined_idx_replaced_mask[layer_num][~keep_indices] = True

#                 # Replace index tensor with newly calculated replacement idxs where mask where not already replaced
#                 combined_replacement_idx_list[layer_num][~combined_idx_replaced_mask] = row_min_idxs[~combined_idx_replaced_mask[layer_num]]

#         print(f"num_valid_swaps: {num_valid_swaps_list}")
#         num_valid_swaps_list_list.append(num_valid_swaps_list)

#         unhooked_loss, hooked_loss, random_hooked_loss, unhooked_accuracy, hooked_accuracy, random_hooked_accuracy = calculate_all_swapped_iterative_relu_loss(
#             hooked_model=hooked_model,
#             module_names=module_list, # Used for hooks
#             data_loader=load_mnist_dataloader(train=True, batch_size=config.batch_size),
#             dtype=TORCH_DTYPES[config.dtype],
#             device=device,
#             replacement_idx_list=combined_replacement_idx_list,
#             num_replaced_list=num_valid_swaps_list,
#         )

#         print(
#             f"unhooked loss {unhooked_loss}",
#             f"hooked loss {hooked_loss}",
#             f"random hooked loss {random_hooked_loss}\n"
#             f"unhooked accuracy {unhooked_accuracy}",
#             f"hooked accuracy {hooked_accuracy}",
#             f"random hooked accuracy {random_hooked_accuracy}"
#         )

#         loss_change_list.append((hooked_loss - unhooked_loss) / unhooked_loss)
#         acc_change_list.append(( hooked_accuracy - unhooked_accuracy) / unhooked_accuracy)
#         random_loss_change_list.append((random_hooked_loss - unhooked_loss) / unhooked_loss)
#         random_acc_change_list.append((random_hooked_accuracy - unhooked_accuracy) / unhooked_accuracy)

#     mean_swaps_list = [sum(swaps) / len(swaps) for swaps in num_valid_swaps_list_list]
#     plot_changes(mean_swaps_list, loss_change_list, random_loss_change_list, acc_change_list, random_acc_change_list, out_dir)


# Main ========================================================

def mlp_relu_main(config_path_str: str) -> None:
    """MAIN FUNCTION 1. Test for ReLU interactions (separate to main RIB algorithm)."""
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)
    set_seed(config.seed)

    relu_matrices_save_file = Path(__file__).parent / f"type_{config.relu_metric_type}_edit_weights_{config.edit_weights}"
    Cs_save_file = Path(__file__).parent / "Cs"

    out_dir = Path(__file__).parent / f"out_mlp_relu/type_{config.relu_metric_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config.mlp_path.parent / "config.yaml", "r") as f:
        model_config_dict = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_mlp(model_config_dict, config.mlp_path, device=device)
    model.to(device=torch.device(device), dtype=TORCH_DTYPES[config.dtype])
    hooked_model = HookedModel(model)
    # print_all_modules(model) # Check module names were correctly defined
    model.eval()  # Run in inference only

    # dict of InteractionRotation objects or Tensors
    Cs_and_Lambdas: dict[str, list[Union[InteractionRotation, Float[Tensor, ...]]]] = check_and_open_file(
        file_path=Cs_save_file,
        get_var_fn=get_Cs,
        config=config,
        model=model,
        device=device,
    )

    relu_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]] = check_and_open_file(
        file_path=relu_matrices_save_file,
        get_var_fn=get_relu_similarities,
        config=config,
        model=model,
        device=device,
        Cs=Cs_and_Lambdas["C"],
        Lambda_dashes=Cs_and_Lambdas["Lambda_dashes"],
    )

    # relu_plotting(relu_matrices, out_dir, config)

    swap_all_layers(hooked_model, relu_matrices, config, device, out_dir)


if __name__ == "__main__":
    fire.Fire(mlp_relu_main)
    """Run above: python run_relu_interactions.py relu_interactions.yaml
    Check ReLU metric used and whether weights are edited in yaml file
    """