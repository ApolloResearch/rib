from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import torch
from einops import rearrange
from jaxtyping import Float, Int
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from torch import Tensor
from torch.utils.data import DataLoader

from rib.data_accumulator import (
    calculate_all_swapped_relu_loss,
    calculate_swapped_relu_loss,
)
from rib.hook_manager import HookedModel
from rib.types import TORCH_DTYPES

if TYPE_CHECKING:   # Prevent circular import to import type annotations
    from experiments.relu_interactions.transformer_relu import Config


def relu_plot_and_cluster(
    similarity_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]],
    out_dir: Path,
    config: "Config",
) -> tuple[list[Int[Tensor, "d_hidden"]], list[list[int]], list[list[Int[Tensor, "cluster_size"]]]]:
    """Form clustering. Plot original and clustered matrices.

    Returns:
        return_index_list: Outer idx layers. Tensor idx hidden neurons.
        all_num_valid_swaps:
        all_cluser_idxs: Outer idx layers. Inner idx cluster number. Tensor idx hidden neurons in cluster.
    """
    return_index_list: list[Int[Tensor, "d_hidden"]] = []
    all_num_valid_swaps: list[list[int]] = []
    all_cluster_idxs: list[list[Int[Tensor, "cluster_size"]]] = []

    for i, similarity_matrix in enumerate(list(similarity_matrices.values())):
        similarity_matrix = similarity_matrix.detach().cpu()
        layer_cluster_idxs = []
        layer_num_valid_swaps = []
        # Plot raw matrix values before clustering
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, annot=False,
                    cmap="YlGnBu", cbar=True, square=True)
        plt.savefig(
            out_dir / f"mat_{i}_type_{config.relu_metric_type}.png")

        # Plot raw histogram
        plt.figure(figsize=(10, 8))
        flat_similarity = np.array(similarity_matrix).flatten()
        sns.histplot(flat_similarity, bins=100, kde=False)
        plt.savefig(
            out_dir / f"hist_{i}_type_{config.relu_metric_type}.png")

        match config.relu_metric_type:
            case 0:
                threshold = 1
                similarity_matrix[similarity_matrix < threshold] = 0
                distance_matrix = 1 - similarity_matrix
            case 1 | 2:
                distance_matrix = torch.max(
                    similarity_matrix, similarity_matrix.T)  # Make symmetric
                threshold = 1
                distance_matrix[distance_matrix > threshold] = threshold
            case 3:
                distance_matrix = torch.max(
                    torch.abs(similarity_matrix), torch.abs(similarity_matrix).T)

        distance_matrix = distance_matrix.fill_diagonal_(
            0)  # Deal with zeros on diagonal
        mean_distance = torch.mean(distance_matrix)
        std_distance = torch.sqrt(torch.var(distance_matrix))
        print(f"mean distance {mean_distance} std distance {std_distance}")

        # Create linkage matrix using clustering algorithm
        X = squareform(distance_matrix)
        Z = linkage(X, method="complete")
        order = leaves_list(Z)

        # Plot dendrogram to determine cut height
        plt.figure(figsize=(40, 7))
        dendrogram(Z)
        plt.savefig(out_dir / f"dendrogram.png", dpi=300)

        # Cut dendrogram
        # ith `clusters` element is flat cluster number to which original observation i belonged
        # Idxs of `clusters` is original element idxs
        clusters = fcluster(Z, t=config.threshold, criterion="distance")
        unique_clusters = np.unique(clusters)
        print(f"unique clusters {len(unique_clusters)}")

        # Important: this index vector will be what's passed into hook function
        # To replace elements of operator vector within clusters with `centroid member` of each cluster
        indices_of_original_O = np.arange(distance_matrix.shape[0])
        for cluster in unique_clusters:
            # 1D array of indices of original matrix
            cluster_idx = np.where(clusters == cluster)[0]
            layer_cluster_idxs.append(torch.tensor(cluster_idx))

            layer_num_valid_swaps.append(cluster_idx.shape[0] - 1)
            cluster_distances = distance_matrix[np.ix_(cluster_idx, cluster_idx)]

            # Use symmetry of matrix, sum only over one dimension to compute total distance to all
            # other members
            # And find minimum index in this cluster subarray
            centroid_idx_in_cluster = np.argmin(cluster_distances.sum(axis=1))

            # Map this back to indices of original array
            centroid_idx_original = cluster_idx[centroid_idx_in_cluster]

            # Set all index elements to cluster centroid index (note we do not return or manipulate
            # *distance matrix values*)
            # Want instead indices with which to permute the O(x) vector in forward hook
            indices_of_original_O[cluster_idx] = centroid_idx_original

        all_cluster_idxs.append(layer_cluster_idxs)
        all_num_valid_swaps.append(layer_num_valid_swaps)
        print(f"swaps {layer_num_valid_swaps}")

        # Cast indices to tensor and add to list of returns - one item per layer
        return_index_list.append(torch.tensor(indices_of_original_O))
        layer_num_swaps_dict = {i: swap_num for i, swap_num in enumerate(layer_num_valid_swaps)}
        degeneracy_total = sum([(n+1)**2-n-1 for n in layer_num_valid_swaps])
        print(f"swaps dict {layer_num_swaps_dict}")
        print(f"degeneracy total {degeneracy_total}")

        with open('relu_clusters.txt', 'w') as file:
            for item in all_cluster_idxs:
                file.write("%s\n" % item)

        rearranged_similarity_matrix = distance_matrix[order, :][:, order]

        # Plot sorted similarity matrix via indices obtained from distance matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(rearranged_similarity_matrix, annot=False,
                    cmap="YlGnBu", cbar=True, square=True)
        plt.title("Reordered Similarity Matrix")
        plt.savefig(
            out_dir / f"rearr_mat_{i}_type_{config.relu_metric_type}.png")

    return return_index_list, all_num_valid_swaps, all_cluster_idxs


def detect_edges(matrix, sigma=1):
    smoothed = scipy.ndimage.gaussian_filter(matrix, sigma=sigma)
    laplacian = scipy.ndimage.laplace(smoothed)

    # Find zero-crossings
    zero_crossings = np.logical_xor(
        laplacian > 0, scipy.ndimage.binary_erosion(laplacian < 0))
    edges = zero_crossings.astype(int)

    np.save('log_edges', edges)
    # Visualize
    plt.figure(figsize=(10, 8))
    sns.heatmap(edges, annot=False, cmap="YlGnBu", cbar=True, square=True)
    plt.title("Edge detection results")
    plt.savefig("edges_test.png")

    return edges


# Swapping ReLUs ==========================================================

# def swap_single_layer(
#     hooked_model: HookedModel,
#     relu_matrices: list[Float[Tensor, "d_hidden d_hidden"]],
#     config: "Config",
#     data_loader: DataLoader,
#     device: str,
#     layer_num: int,
#     out_dir: str,
# ) -> None:
#     """Now that we have gotten our similarity matrices, swap in forward pass.
#     Check loss and accuracy drop, and plot as a function of similarity tolerance for pruning."""
#     num_valid_swaps_list: list[int] = []
#     loss_change_list: list[float] = []
#     acc_change_list: list[float] = []
#     random_loss_change_list: list[float] = []
#     random_acc_change_list: list[float] = []
#     tols = [[1e-3, 7e-4, 4e-4, 1e-4, 7e-5, 4e-5],
#             [1e-4, 7e-5, 4e-5, 1e-5, 7e-6, 4e-6]]
#     matrix = list(relu_matrices.values())[layer_num]
#     module_name = list(config.activation_layers)[layer_num]

#     for i in range(len(tols[0])):
#         tol = tols[layer_num][i]
#         row_min_idxs, num_valid_swaps = _find_indices_to_replace(matrix, tol)
#         num_valid_swaps_list.apeend(num_valid_swaps)

#         print(f"num_valid_swaps: {num_valid_swaps}")

#         unhooked_loss, hooked_loss, random_hooked_loss, unhooked_accuracy, hooked_accuracy, random_hooked_accuracy = calculate_swapped_relu_loss(
#             hooked_model=hooked_model,
#             module_name=module_name,  # Used for hooks
#             data_loader=data_loader,
#             dtype=TORCH_DTYPES[config.dtype],
#             device=device,
#             replacement_idxs=row_min_idxs,
#             num_replaced=num_valid_swaps,
#         )

#         print(
#             f"unhooked loss {unhooked_loss}",
#             f"hooked loss {hooked_loss}",
#             f"random hooked loss {random_hooked_loss}"
#             f"unhooked accuracy {unhooked_accuracy}",
#             f"hooked accuracy {hooked_accuracy}",
#             f"random hooked accuracy {random_hooked_accuracy}"
#         )

#         loss_change_list.append((hooked_loss - unhooked_loss) / unhooked_loss)
#         acc_change_list.append(
#             (hooked_accuracy - unhooked_accuracy) / unhooked_accuracy)
#         random_loss_change_list.append(
#             (random_hooked_loss - unhooked_loss) / unhooked_loss)
#         random_acc_change_list.append(
#             (random_hooked_accuracy - unhooked_accuracy) / unhooked_accuracy)

#     plot_changes(num_valid_swaps_list, loss_change_list, random_loss_change_list,
#                  acc_change_list, random_acc_change_list, out_dir, layer_num)


# def swap_all_layers(
#     hooked_model: HookedModel,
#     relu_matrices: list[Float[Tensor, "d_hidden d_hidden"]],
#     tols: list[list[float]],
#     config: "Config",
#     data_loader: DataLoader,
#     device: str,
#     out_dir: str,
# ) -> None:
#     """Assuming universal synchronisation matrices (no need to recalculate in forward pass due to
#     changes in previous layer), swap out subsets of all layers. Plot resulting loss and accuracy
#     loss.

#     Currently only tested on MLP.

#     If it is true that the sychronisation swaps made in the previous layer(s) are accurate, the
#     fundamental computation in future layers should not break, because only activations (0s or 1s)
#     are swapped, not pre-activations or post-activations.
#     """
#     # list[float]
#     loss_change_list, acc_change_list, random_loss_change_list, random_acc_change_list = [], [], [], []
#     # Outer loop: tolerance value; inner loop: layer
#     num_valid_swaps_list_list: list[list[int]] = []
#     matrix_list = list(relu_matrices.values())
#     module_list = list(config.activation_layers)

#     for i in range(len(tols[0])):
#         # Count number of swaps in each layer - for plotting
#         num_valid_swaps_list: list[int] = []
#         # Outer loop: modules; inner loop: indices of hidden layer
#         # Passed into hook function
#         replacement_idx_list: list[Int[Tensor, "d_hidden"]] = []
#         for layer_num, (module, matrix) in enumerate(zip(module_list, matrix_list)):
#             tol = tols[layer_num][i]
#             row_min_idxs, num_valid_swaps = _find_indices_to_replace(
#                 matrix, tol)
#             replacement_idx_list.append(row_min_idxs)
#             num_valid_swaps_list.append(num_valid_swaps)

#         print(f"Num valid swaps: {num_valid_swaps_list}")
#         num_valid_swaps_list_list.append(num_valid_swaps_list)

#         unhooked_loss, hooked_loss, random_hooked_loss, unhooked_accuracy, hooked_accuracy, random_hooked_accuracy = calculate_all_swapped_relu_loss(
#             hooked_model=hooked_model,
#             module_names=module_list,  # Used for hooks
#             data_loader=data_loader,
#             dtype=TORCH_DTYPES[config.dtype],
#             device=device,
#             replacement_idx_list=replacement_idx_list,
#             num_replaced_list=num_valid_swaps_list,
#             use_residual_stream=config.use_residual_stream,
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
#         acc_change_list.append(
#             (hooked_accuracy - unhooked_accuracy) / unhooked_accuracy)
#         random_loss_change_list.append(
#             (random_hooked_loss - unhooked_loss) / unhooked_loss)
#         random_acc_change_list.append(
#             (random_hooked_accuracy - unhooked_accuracy) / unhooked_accuracy)

#     mean_swaps_list = [sum(swaps) / len(swaps)
#                        for swaps in num_valid_swaps_list_list]
#     plot_changes(mean_swaps_list, loss_change_list, random_loss_change_list,
#                  acc_change_list, random_acc_change_list, out_dir)


def swap_all_layers_using_clusters(
    replacement_idxs_from_cluster: list[Int[Tensor, "d_hidden"]],
    num_valid_swaps_from_cluster: list[int],
    hooked_model: HookedModel,
    config: "Config",
    data_loader: DataLoader,
    device: str,
) -> None:
    """Swap based on indices of clusters. Currently only tested on transformer.

    Hook activation layers only.

    Args:
        replacement_idxs_from_cluster: Swap indices from clustering algorithm run beforehand.
        num_valid_swaps_from_cluster: List of total swaps made, calculated while running clustering.
    """
    module_list = list(config.activation_layers)

    unhooked_loss, hooked_loss, random_hooked_loss, unhooked_accuracy, hooked_accuracy, random_hooked_accuracy = calculate_all_swapped_relu_loss(
        hooked_model=hooked_model,
        module_names=module_list,  # Used for hooks
        data_loader=data_loader,
        dtype=TORCH_DTYPES[config.dtype],
        device=device,
        replacement_idx_list=replacement_idxs_from_cluster,
        num_replaced_list=num_valid_swaps_from_cluster,
        use_residual_stream=config.use_residual_stream,
    )

    print(
        f"unhooked loss {unhooked_loss}",
        f"hooked loss {hooked_loss}",
        f"random hooked loss {random_hooked_loss}\n"
        f"unhooked accuracy {unhooked_accuracy}",
        f"hooked accuracy {hooked_accuracy}",
        f"random hooked accuracy {random_hooked_accuracy}"
    )


def _find_indices_to_replace(matrix: Float[Tensor, "d1 d1"], tol: float) -> tuple[Float[Tensor, "d1"], int]:
    """Use for swapping function to, for each row (m, index to swap), find:
    - The minimum distance to another neuron
    - Whether this minimum distance is below a given threshold -> if true, then swap mth row with nth column
    """
    matrix = torch.abs(matrix)
    diag_mask = torch.eye(matrix.size(0), matrix.size(1)
                          ).bool().to(matrix.device)
    matrix = matrix.masked_fill_(diag_mask, float('inf'))

    row_min_vals, row_min_idxs = torch.min(matrix, dim=-1)
    # Keep when minimum value in row is above some threshold
    keep_indices = torch.nonzero(row_min_vals > tol)
    num_valid_swaps = matrix.shape[0] - len(keep_indices)

    # Where no replacement wanted, set the element to equal index itself
    row_min_idxs[keep_indices] = keep_indices

    return row_min_idxs, num_valid_swaps


# Weights/ Models ========================================================

def edit_weights_fn(model: HookedModel, layer_name: str) -> None:
    """Make the bias really big. Used for toy test case in MLP.

    Weight matrix dimensions are (output, input).
    """
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


def extract_weights_mlp(model: torch.nn.Module) -> list[torch.Tensor]:
    """Recursively extract weights from each layer of the model and return them as a list.
    This works for an MLP but not SequentialTransformer.
    """
    weights_list = []

    for name, layer in model.named_children():
        if hasattr(layer, 'weight'):
            weights = get_nested_attribute(
                model, name + '.weight').data.clone().cpu()

            batch_size, d_hidden = weights.shape
            to_append_weights = torch.zeros(d_hidden)
            # Create vector with last element 1, append as row of W (dims output, input)
            to_append_weights[-1] = 1
            # Tranpose weights later for multiplication since canonical code form (input, output)
            weights = torch.cat(
                [weights, rearrange(to_append_weights, 'd -> 1 d')], dim=0)
            assert weights[-1, -1] == 1

            weights_list.append(weights)

        # Recursively extract weights from nested children
        weights_list.extend(extract_weights_mlp(layer))

    return weights_list


def print_all_modules(model):
    """Use for choosing which modules go in config file."""
    for name, module in model.named_modules():
        print(name, ":", module)


# Plotting ===============================================================

def plot_changes(
    num_valid_swaps_list: list[int],
    loss_change_list: list[float],
    random_loss_change_list: list[float],
    acc_change_list: list[float],
    random_acc_change_list: list[float],
    out_dir: Path,
    layer_num: int = -1,
) -> None:
    """Visualise effect om loss and accuracy of swapping ReLUs."""
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


def plot_temp(matrix: Float[Tensor, "d1 d2"], title: str) -> None:
    out_dir = Path(__file__).parent / "relu_temp_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    if matrix.dim() == 1:
        aspect_ratio = 20 / matrix.shape[0]
        matrix = matrix.unsqueeze(-1)
    else:
        aspect_ratio = matrix.shape[1] / matrix.shape[0]
    # Set the vertical size, and let the horizontal size adjust based on the aspect ratio
    vertical_size = 7
    horizontal_size = vertical_size * aspect_ratio
    plt.figure(figsize=(horizontal_size, vertical_size))
    sns.heatmap(matrix.detach().cpu(), annot=False,
                cmap="YlGnBu", cbar=True, square=True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{title}.png")
    plt.close()


def plot_matrix_list(matrix_list: list[Float[Tensor, "d_hidden d_hidden"]], var_name: str, out_dir: Path) -> None:
    for i, matrix in enumerate(list(matrix_list)):
        # Calculate figsize based on matrix dimensions and the given figsize_per_unit
        nrows, ncols = matrix.shape
        figsize = (int(ncols * 0.1), int(nrows * 0.1))
        plt.figure(figsize=figsize)
        sns.heatmap(matrix.detach().cpu(), annot=False,
                    cmap="YlGnBu", cbar=True, square=True)
        plt.savefig(
            out_dir / f"{var_name}_{i}.png")


def plot_eigenvalues(eigenvalues: Float[Tensor, "d_hidden"], out_dir: Path, title: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(eigenvalues)), eigenvalues, color='blue')
    plt.title('Eigenvalues in Descending Order')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    plt.savefig(out_dir / f"eigenvalues_{title}.png")
    plt.close()


def plot_eigenvectors(eigenvectors: np.ndarray, out_dir: Path, title: str) -> None:
    num_vectors = eigenvectors.shape[1]
    num_components = eigenvectors.shape[0]
    x = np.arange(num_components)  # the label locations

    plt.figure(figsize=(10, 6))

    # Width of a bar
    width = 0.8 / num_vectors

    for i in range(num_vectors):
        plt.bar(x - width/2 + i * width, eigenvectors[:, i], width, label=f'Eigenvector {i+1}', alpha=0.7)

    plt.title(f'All Eigenvectors - {title}')
    plt.xlabel('Component')
    plt.ylabel('Value')
    plt.xticks(x)
    plt.legend()
    plt.grid(True)

    plt.savefig(out_dir / f"eigenvectors_{title}.png")
    plt.close()