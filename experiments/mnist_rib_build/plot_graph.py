"""Plot an interaction graph given a results file contain the graph edges.

Usage:
    python plot_graph.py <path/to/results_pt_file>

    The results_pt_file should be the output of the build_interaction_graph.py script.
"""
from pathlib import Path
from typing import Union

import fire
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from rib.utils import overwrite_output


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


def main(results_file: str) -> None:
    results = torch.load(results_file)

    nodes_input_layer = 40
    nodes_per_layer = 10
    # results["edges"] contains a list of edges which are tuples of
    # (module, edge_weights), each with shape (n_nodes_in_l+1, n_nodes_in_l)
    edges: list[tuple[str, torch.Tensor]] = []
    for i, (module_name, weight_matrix) in enumerate(results["edges"]):
        n_nodes_in = nodes_input_layer if i == 0 else nodes_per_layer
        # Normalize the edge weights by the sum of the absolute values of the weights
        weight_matrix /= torch.sum(torch.abs(weight_matrix))
        # Only keep the first nodes_per_layer nodes in each layer
        edges.append((module_name, weight_matrix[:nodes_per_layer, :n_nodes_in]))

    out_dir = Path(__file__).parent / "out"
    plot_file = out_dir / f"{results['exp_name']}_interaction_graph.png"

    if plot_file.exists() and not overwrite_output(plot_file):
        print("Exiting.")
        return

    plot_interaction_graph(
        edges=edges,
        plot_file=plot_file,
        exp_name=results["exp_name"],
        n_nodes_ratio=nodes_input_layer / nodes_per_layer,
    )


if __name__ == "__main__":
    fire.Fire(main)
