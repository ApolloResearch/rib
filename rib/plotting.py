"""Plotting functions

plot_graph:
    - Plot a rib graph given a results file contain the graph edges.

plot_ablation_results:
    - Plot accuracy/loss vs number of remaining basis vectors.

"""
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


def _create_node_layers(edges: list[torch.Tensor]) -> list[np.ndarray]:
    """Create a list of node layers from the given edges."""
    layers = []
    for i, weight_matrix in enumerate(edges):
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


def _add_edges_to_graph(
    graph: nx.Graph,
    edges: list[torch.Tensor],
    layers: list[np.ndarray],
    pos_color: Optional[str] = "blue",
    neg_color: Optional[str] = "red",
) -> None:
    """Add edges to the graph object (note that there is one more layer than there are edges).

    Args:
        graph (nx.Graph): The graph object to add edges to.
        edges (list[torch.Tensor]): A list of edges, each with shape (n_nodes_in_l+1, n_nodes_in_l).
        layers (list[np.ndarray]): A list of node layers with each layer an array of node indices.
        pos_color (Optional[str]): The color to use for positive edges. Defaults to "blue".
        neg_color (Optional[str]): The color to use for negative edges. Defaults to "red".
    """
    for module_idx, edge_matrix in enumerate(edges):
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


def _prepare_edges_for_plotting(
    raw_edges: list[torch.Tensor], nodes_per_layer: list[int], hide_const_edges: bool = False
) -> list[torch.Tensor]:
    """Convert edges to float, normalize, and truncate to desired number of nodes in each layer.

    Args:
        raw_edges (list[torch.Tensor]): List of edges tensors, each with shape
            (n_nodes_in_l+1, n_nodes_in_l).
        nodes_per_layer (list[int]): The number of nodes in each layer.

    Returns:
        list[torch.Tensor]: A list of edges, each with shape (n_nodes_in_l+1, n_nodes_in_l).
    """
    edges: list[torch.Tensor] = []
    for i, weight_matrix in enumerate(raw_edges):
        # Convert edges to float32 (bfloat16 will cause errors and we don't need higher precision)
        weight_matrix = weight_matrix.float()
        if hide_const_edges:
            const_node_index = 0
            # Set edges outgoing from this node to zero (edges.shape ~ l+1, l). The incoming edges
            # should be zero except for a non-rotated last layer where they are important.
            weight_matrix[:, const_node_index] = 0
        # Normalize the edge weights by the sum of the absolute values of the weights
        weight_matrix /= torch.sum(torch.abs(weight_matrix))
        # Only keep the desired number of nodes in each layer
        in_nodes = nodes_per_layer[i]
        out_nodes = nodes_per_layer[i + 1]
        edges.append(weight_matrix[:out_nodes, :in_nodes])
    return edges


def plot_ablation_results(
    results: list[dict[str, dict[str, float]]],
    out_file: Optional[Path],
    exp_names: list[str],
    eval_type: Literal["accuracy", "ce_loss"],
    ablation_types: list[Literal["orthogonal", "rib"]],
    log_scale: bool = False,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
) -> None:
    """Plot accuracy/loss vs number of remaining basis vectors.

    Args:
        results: A list of dictionares mapping node layers to an inner dictionary that maps the
            number of basis vectors remaining to the accuracy/loss.
        out_file: The file to save the plot to.
        exp_names: The names of the rib_scripts.
        ablation_types: The type of ablation performed for each experiment ("orthogonal" or "rib").
        log_scale: Whether to use a log scale for the x-axis. Defaults to False.
        xlim: The limits for the x-axis. Defaults to None.
        ylim: The limits for the y-axis. Defaults to None.
    """
    # Verify that all results have the same node layers
    node_layers_per_exp = [set(result.keys()) for result in results]
    assert all(
        node_layers == node_layers_per_exp[0] for node_layers in node_layers_per_exp[1:]
    ), "All results must have the same node layers."

    node_layers = results[0].keys()
    n_plots = len(node_layers)
    _, axs = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots), dpi=140)

    if n_plots == 1:
        axs = [axs]

    for i, node_layer in enumerate(node_layers):
        for exp_name, ablation_type, exp_results in zip(exp_names, ablation_types, results):
            n_vecs_remaining = sorted(list(int(k) for k in exp_results[node_layer]))
            y_values = [exp_results[node_layer][str(i)] for i in n_vecs_remaining]
            axs[i].plot(n_vecs_remaining, y_values, "-o", label=exp_name)

            axs[i].set_title(f"{eval_type} vs n_remaining_basis_vecs for input to {node_layer}")
            axs[i].set_xlabel("Number of remaining basis vecs")
            axs[i].set_ylabel(eval_type)
            if xlim is not None:
                axs[i].set_xlim(*xlim)
            if ylim is not None:
                axs[i].set_ylim(*ylim)

            if log_scale:
                axs[i].set_xscale("log")

            axs[i].grid(True)
            axs[i].legend()

    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.text(
        1,
        0,
        date,
        fontsize=12,
        color="gray",
        ha="right",
        va="bottom",
        alpha=0.5,
        transform=axs[0].transAxes,
    )

    # Make a title for the entire figure
    plt.suptitle("_".join(exp_names))

    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.4)
    if out_file is not None:
        plt.savefig(out_file)


def plot_rib_graph(
    raw_edges: list[torch.Tensor],
    layer_names: list[str],
    exp_name: str,
    nodes_per_layer: Union[int, list[int]],
    out_file: Optional[Path] = None,
    node_labels: Optional[list[list[str]]] = None,
    hide_const_edges: bool = False,
) -> None:
    """Plot the RIB graph for the given edges.

    Args:
        raw_edges (list[torch.Tensor]): List of edges with shape (n_nodes_in_l+1, n_nodes_in_l)
        layer_names (list[str]): The names of the layers. These correspond to the first dimension
            of each tensor in raw_edges, and also includes a name for the final node_layer.
        exp_name (str): The name of the experiment.
        nodes_per_layer (Union[int, list[int]]): The number of nodes in each layer. If int, then
            all layers have the same number of nodes. If list, then the number of nodes in each
            layer is given by the list.
        out_file (Path): The file to save the plot to.
    """
    if isinstance(nodes_per_layer, int):
        # Note that there is one more layer than there edge matrices
        nodes_per_layer = [nodes_per_layer] * (len(raw_edges) + 1)

    max_layer_height = max(nodes_per_layer)

    edges = _prepare_edges_for_plotting(
        raw_edges, nodes_per_layer, hide_const_edges=hide_const_edges
    )

    # Create the undirected graph
    graph = nx.Graph()

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    layers = _create_node_layers(edges)
    # Add nodes to the graph object
    for layer in layers:
        graph.add_nodes_from(layer)

    _add_edges_to_graph(graph, edges, layers)

    # Create positions for each node
    pos: dict[int, tuple[int, Union[int, float]]] = {}
    for i, layer in enumerate(layers):
        # Add extra spacing for nodes that have fewer nodes than the biggest layer
        spacing = 1 if i == 0 else max_layer_height / len(layer)
        for j, node in enumerate(layer):
            pos[node] = (i, j * spacing)

    # Draw nodes
    colors = ["black", "green", "orange", "purple"]  # Add more colors if you have more layers
    options = {"edgecolors": "tab:gray", "node_size": 100, "alpha": 0.3}
    for i, (layer_name, layer) in enumerate(zip(layer_names, layers)):
        nx.draw_networkx_nodes(
            graph, pos, nodelist=layer, node_color=colors[i % len(colors)], **options
        )
        # Add layer label above the nodes
        plt.text(i, max_layer_height, layer_name, ha="center", va="center", fontsize=12)

    # Label nodes if node_labels is provided
    if node_labels is not None:
        node_label_dict = {}
        for i, layer in enumerate(layers):
            for j, node in enumerate(layer):
                node_label_dict[node] = node_labels[i][j].replace("|", "\n")
        nx.draw_networkx_labels(graph, pos, node_label_dict, font_size=8)

    # Draw edges
    width_factor = 15
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=[(edge[0], edge[1]) for edge in graph.edges(data=True)],
        width=[width_factor * edge[2]["weight"] for edge in graph.edges(data=True)],
        alpha=1,
        edge_color=[edge[2]["color"] for edge in graph.edges(data=True)],
    )

    plt.suptitle(exp_name)
    plt.tight_layout()
    ax.axis("off")
    if out_file is not None:
        plt.savefig(out_file)
