"""This script builds a RIB graph for a language model.
We build the graph using a SequentialTransformer model, with weights ported over from a
transformer-lens model.

Steps to build the graph:
1. Load a model from transformerlens (either from_pretrained or via ModelConfig).
2. Fold in the biases into the weights.
3. Convert the model to a SequentialTransformer model, which has nn.Modules corresponding to each
    node layer.
5. Collect the gram matrices at each node layer.
6. Calculate the interaction basis matrices (labelled C in the paper) for each node layer, starting
    from the final node layer and working backwards.
7. Calculate the edges of the interaction graph between each node layer.

Usage:
    python run_lm_rib_build.py <path/to/config.yaml>

The config.yaml should contain the `node_layers` field. This describes the sections of the
graph that will be built: A graph layer will be built on the inputs to each specified node layer,
as well as the output of the final node layer. For example, if `node_layers` is ["attn.0",
"mlp_act.0"], this script will build the following graph layers:
- One on the inputs to the "attn.0" node layer. This will include the residual stream concatenated
    with the output of ln1.0.
- One on the input to "mlp_act.0". This will include the residual stream concatenated with the
    output of "mlp_in.0".
- One on the output of "mlp_act.0". This will include the residual stream concatenated with the
    output of "mlp_act.0".
"""
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import torch
from jaxtyping import Float, Int
from torch import Tensor

from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage, cophenet
from scipy.ndimage import label
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs

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
from rib.loader import create_data_loader, load_dataset, load_sequential_transformer
from rib.log import logger
from rib.types import TORCH_DTYPES
from rib.utils import (
    eval_cross_entropy_loss,
    eval_model_accuracy,
    load_config,
    overwrite_output,
    set_seed,
)

if TYPE_CHECKING:   # Prevent circular import to import type annotations
    from experiments.relu_interactions.mlp_relu import Config


def relu_plot_and_cluster(
        similarity_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]],
        out_dir: Path,
        config: "Config",
) -> list[Int[Tensor, "d_hidden"]]:
    """Form clustering. Plot original and clustered matrices.

    Returns:
        all_layer_clusters: List of lists. Outer list: layers. Inner list: indices of the
    """
    return_index_list = []

    for i, similarity_matrix in enumerate(list(similarity_matrices.values())):

        # Plot raw matrix values before clustering
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, annot=False,
                    cmap="YlGnBu", cbar=True, square=True)
        plt.savefig(
            out_dir / f"mat_{i}_type_{config.relu_metric_type}.png")

        # Plot raw histogram
        plt.figure(figsize=(10,8))
        flat_similarity = np.array(similarity_matrix).flatten()
        sns.histplot(flat_similarity, bins=100, kde=False)
        plt.savefig(
            out_dir / f"hist_{i}_type_{config.relu_metric_type}.png")

        match config.relu_metric_type:
            case 0:
                threshold = 0.95
                similarity_matrix[similarity_matrix < threshold] = 0
                distance_matrix = 1 - similarity_matrix
            case 1 | 2:
                distance_matrix = torch.max(similarity_matrix, similarity_matrix.T) # Make symmetric
                threshold = 15
                distance_matrix[distance_matrix > threshold] = threshold
            case 3:
                distance_matrix = torch.max(torch.abs(similarity_matrix), torch.abs(similarity_matrix).T)

        distance_matrix = distance_matrix.fill_diagonal_(0) # Deal with zeros on diagonal
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
        threshold = 0.001
        clusters = fcluster(Z, t=threshold, criterion="distance")
        unique, unique_reverse_indices = np.unique(clusters, return_inverse=True)
        print(f"unique cluster vals {unique}")

        # Important: this index vector will be what's passed into hook function
        # To replace elements of operator vector within clusters with `centroid member` of each cluster
        indices_of_original_O = np.arange(distance_matrix.shape[0])
        for cluster in unique:
            cluster_idx = np.where(clusters == cluster)[0] # 1D array of indices of original matrix
            cluster_distances = distance_matrix[np.ix_(cluster_idx, cluster_idx)]
            # Use symmetry of matrix, sum only over one dimension to compute total distance to all
            # other members
            # And find minimum index in this cluster subarray
            centroid_idx_in_cluster = np.argmin(cluster_distances.sum(axis=0))
            # Map this back to indices of original array
            centroid_idx_original = cluster_idx[centroid_index]
            # Set all index elements to cluster centroid index (note we do not return or manipulate
            # *distance matrix values*)
            # Want instead indices with which to permute the O(x) vector in forward hook
            indices_of_original_O[cluster_idx] = centroid_idx_original

        # Cast indices to tensor and add to list of returns - one item per layer
        return_index_list.append(torch.tensor(indices_of_original_O))

        # # Okay, the above works - now automatic distance selection in fcluster
        # """Method 1: silhouette point."""
        # t_min = (mean_distance - 1.6 * std_distance)
        # t_max = (mean_distance + 1 * std_distance)
        # max_silhouette_score = -1 # Initialise at lowest value possible
        # optimal_t = None
        # for t in np.linspace(t_min, t_max, 50):
        #     clusters = fcluster(Z, t=t, criterion='distance')
        #     if len(set(clusters)) > 1:  # Silhouette score requires at least 2 clusters
        #         silhouette_avg = silhouette_score(distance_matrix, clusters, metric='precomputed')
        #         if silhouette_avg > max_silhouette_score:
        #             max_silhouette_score = silhouette_avg
        #             optimal_t = t
        # silhouette_cluster_num = len(np.unique(clusters))
        # print(f"Silhouette point optimised cluster count {silhouette_cluster_num}")

        # """Method 2 - Davies-Bouldin index."""
        # min_db_index = float('inf')
        # optimal_t = None
        # for t in np.linspace(t_min, t_max, 50):
        #     clusters = fcluster(Z, t=t, criterion='distance')
        #     if len(set(clusters)) > 1:  # Davies-Bouldin index requires at least 2 clusters
        #         db_index = davies_bouldin_score(distance_matrix, clusters)
        #         if db_index < min_db_index:
        #             min_db_index = db_index
        #             optimal_t = t
        # db_cluster_num = len(np.unique(clusters))
        # print(f"Davies-Bouldin optimised cluster count {db_cluster_num}")

        rearranged_similarity_matrix = distance_matrix[order, :][:, order]

        # Plot sorted similarity matrix via indices obtained from distance matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(rearranged_similarity_matrix, annot=False,
                    cmap="YlGnBu", cbar=True, square=True)
        plt.title("Reordered Similarity Matrix")
        plt.savefig(
            out_dir / f"rearr_mat_{i}_type_{config.relu_metric_type}.png")

    return return_index_list


def detect_edges(matrix, sigma=1):
    smoothed = scipy.ndimage.gaussian_filter(matrix, sigma=sigma)
    laplacian = scipy.ndimage.laplace(smoothed)

    # Find zero-crossings
    zero_crossings = np.logical_xor(laplacian > 0, scipy.ndimage.binary_erosion(laplacian < 0))
    edges = zero_crossings.astype(int)

    np.save('log_edges', edges)
    # Visualize
    plt.figure(figsize=(10, 8))
    sns.heatmap(edges, annot=False, cmap="YlGnBu", cbar=True, square=True)
    plt.title("Edge detection results")
    plt.savefig("edges_test.png")

    return edges


# Weights/ Models ========================================================

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


def print_all_modules(model):
    """Use for choosing which modules go in config file."""
    for name, module in model.named_modules():
        print(name, ":", module)


# Plotting ===============================================================


def plot_temp(matrix: Float[Tensor, "d1 d2"], title: str) -> None:
    from pathlib import Path

    import matplotlib.pyplot as plt
    import seaborn as sns
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
    sns.heatmap(matrix.detach().cpu(), annot=False, cmap="YlGnBu", cbar=True, square=True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{title}.png")
    plt.close()