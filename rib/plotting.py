"""Plotting functions

plot_rib_graph:
    - Plot an interaction graph given a results file contain the graph edges.

plot_ablation_accuracies:
    - Plot accuracy vs number of remaining basis vectors.

"""
from pathlib import Path
from typing import Literal, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


def create_node_layers(edges: list[tuple[str, torch.Tensor]]) -> list[np.ndarray]:
    """Create a list of node layers from the given edges."""
    layers = []
    for i, (_, weight_matrix) in enumerate(edges):
        # Only add nodes to the graph for the current layer if it's the first layer
        if i == 0:
            current_layer = np.arange(weight_matrix.shape[1])
            layers.append(current_layer)
        else:
            # Ensure that the current layer's nodes are the same as the previous layer's nodes
            assert len(layers[-1]) == weight_matrix.shape[1], (
                f"The number of nodes implied by edge matrix {i} ({weight_matrix.shape[1]}) "
                f"does not match the number implied by the previous edge matrix ({len(layers[-1])})."
            )
            current_layer = layers[-1]
        next_layer = np.arange(weight_matrix.shape[0]) + max(current_layer) + 1
        layers.append(next_layer)
    return layers


def add_edges_to_graph(
    graph: nx.Graph,
    edges: list[tuple[str, torch.Tensor]],
    layers: list[np.ndarray],
    pos_color: str = "blue",
    neg_color: str = "red",
) -> None:
    """Add edges to the graph object (note that there is one more layer than there are edges).

    Args:
        graph: The graph object to add edges to.
        edges: A list of tuples of (module_name, edge_weights), each with shape
            (n_nodes_in_l+1, n_nodes_in_l).
        layers: A list of node layers, where each layer is an array of node indices.
        pos_color: The color to use for positive edges. Defaults to "blue".
        neg_color: The color to use for negative edges. Defaults to "red".
    """
    for module_idx, (_, edge_matrix) in enumerate(edges):
        for j in range(edge_matrix.shape[1]):
            for i in range(edge_matrix.shape[0]):
                color = pos_color if edge_matrix[i, j] > 0 else neg_color
                # Draw the edge from the node in the current layer to the node in the next layer
                graph.add_edge(
                    layers[module_idx][j],
                    layers[module_idx + 1][i],
                    weight=edge_matrix[i, j],
                    color=color,
                )


def plot_interaction_graph(
    edges: list[tuple[str, torch.Tensor]],
    plot_file: Path,
    exp_name: str,
    n_nodes_ratio: float = 1.0,
) -> None:
    """Plot the interaction graph for the given edges.

    Args:
        edges: A list of tuples of (module_name, edge_weights), each with shape
            (n_nodes_in_l+1, n_nodes_in_l).
        plot_file: The file to save the plot to.
        exp_name: The name of the experiment
        n_nodes_ratio: Ratio of the number of nodes in the first layer to the number of nodes in
            the other layers. Defaults to 1.0.

    """

    # Convert the edges to float32 (bfloat16 will cause errors and we don't need higher precision)
    edges = [(module_name, edge_matrix.float()) for module_name, edge_matrix in edges]

    # Create the undirected graph
    graph = nx.Graph()

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    layers = create_node_layers(edges)
    # Add nodes to the graph object
    for layer in layers:
        graph.add_nodes_from(layer)

    add_edges_to_graph(graph, edges, layers)

    # Create positions for each node
    pos: dict[int, tuple[int, Union[int, float]]] = {}
    for i, layer in enumerate(layers):
        # Add extra spacing for all nodes after the first layer if there are fewer of them
        spacing = 1 if i == 0 else n_nodes_ratio
        for j, node in enumerate(layer):
            pos[node] = (i, j * spacing)

    # Draw nodes
    colors = ["black", "green", "orange", "purple"]  # Add more colors if you have more layers
    options = {"edgecolors": "tab:gray", "node_size": 100, "alpha": 0.3}
    for i, layer in enumerate(layers):
        nx.draw_networkx_nodes(
            graph, pos, nodelist=layer, node_color=colors[i % len(colors)], **options
        )
    # Draw edges
    width_factor = 15
    # for edge in graph.edges(data=True):
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=[(edge[0], edge[1]) for edge in graph.edges(data=True)],
        width=[width_factor * edge[2]["weight"] for edge in graph.edges(data=True)],
        alpha=1,
        edge_color=[edge[2]["color"] for edge in graph.edges(data=True)],
    )

    plt.tight_layout()
    ax.axis("off")
    plt.savefig(plot_file)


def plot_ablation_accuracies(
    accuracies: dict[str, dict[str, float]],
    plot_file: Path,
    exp_name: str,
    experiment_type: Literal["orthog", "rib"],
) -> None:
    """Plot accuracy vs number of remaining basis vectors.

    Args:
        accuracies: A dictionary mapping module names to an inner dictionary mapping the number of
            ablated vectors to model accuracy.
        plot_file: The file to save the plot to.
        exp_name: The name of the experiment
        experiment_type: Either 'orthog' or 'rib'.
    """
    module_names = list(accuracies.keys())
    n_plots = len(module_names)
    _, axs = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots), dpi=140)

    if n_plots == 1:
        axs = [axs]

    for i, module_name in enumerate(module_names):
        # module_accuracies = {int(k): v for k, v in accuracies[module_name].items()}
        # We want to show number of vecs remaining, not number of vecs ablated.
        # We thus reverse the list of ablated vecs
        n_ablated_vecs = sorted(list(int(k) for k in accuracies[module_name]), reverse=True)
        n_vecs = max(n_ablated_vecs)
        n_vecs_remaining = [n_vecs - n for n in n_ablated_vecs]
        y_values = [
            accuracies[module_name][str(n_ablated_vecs)] for n_ablated_vecs in n_ablated_vecs
        ]
        axs[i].plot(n_vecs_remaining, y_values, "-o", label="MNIST test")

        title_extra = (
            "n_remaining_eigenvalues" if experiment_type == "orthog" else "n_remaining_basis_vecs"
        )
        xlabel_extra = (
            "Number of remaining eigenvalues"
            if experiment_type == "orthog"
            else "Number of remaining interaction basis vectors"
        )

        axs[i].set_title(f"{exp_name}-MLP MNIST acc vs {title_extra} for input to {module_name}")
        axs[i].set_xlabel(xlabel_extra)
        axs[i].set_ylabel("Accuracy")
        axs[i].set_ylim(0, 1)
        axs[i].grid(True)
        axs[i].legend()

    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.4)

    plt.savefig(plot_file)
