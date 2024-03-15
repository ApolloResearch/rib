"""Plotting functions

plot_graph:
    - Plot a rib graph given a results file contain the graph edges.

plot_ablation_results:
    - Plot accuracy/loss vs number of remaining basis vectors.

"""
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import colorcet
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from rib.data_accumulator import Edges
from rib.log import logger
from rib.modularity import IdentityEdgeNorm


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
    coords_to_index: dict[tuple[int, int], int],
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
    for l, edge_matrix in enumerate(edges):
        for j in range(edge_matrix.shape[1]):  # L
            for i in range(edge_matrix.shape[0]):  # L+1
                color = pos_color if edge_matrix[i, j] > 0 else neg_color
                # Draw the edge from the node in the current layer to the node in the next layer
                graph.add_edge(
                    coords_to_index[(l, j)],
                    coords_to_index[(l + 1, i)],
                    weight=edge_matrix[i, j],
                    color=color,
                )


def _prepare_edges_for_plotting(
    raw_edges: list[torch.Tensor],
    nodes_per_layer: list[int],
    hide_const_edges: bool = False,
) -> list[torch.Tensor]:
    """Convert edges to float, normalize, and truncate to desired number of nodes in each layer.

    Args:
        raw_edges (list[torch.Tensor]): List of edges tensors, each with shape
            (n_nodes_in_l+1, n_nodes_in_l).
        nodes_per_layer (list[int]): The number of nodes in each layer.
        line_width_factor: Scale factor to convert edge weights into line widths.

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
        # Only keep the desired number of nodes in each layer
        in_nodes = nodes_per_layer[i]
        out_nodes = nodes_per_layer[i + 1]
        edges.append(weight_matrix[:out_nodes, :in_nodes])
    return edges


def plot_ablation_results(
    results: list[dict[str, dict[str, float]]],
    no_ablation_results_list: list[float],
    out_file: Optional[Path],
    exp_names: list[str],
    eval_type: Literal["accuracy", "ce_loss"],
    ablation_types: list[Literal["orthogonal", "rib"]],
    log_scale_x: bool = False,
    log_scale_y: bool = False,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    baseline_is_zero: bool = False,
) -> None:
    """Plot accuracy/loss vs number of remaining basis vectors.

    Args:
        results: A list of dictionares mapping node layers to an inner dictionary that maps the
            number of basis vectors remaining to the accuracy/loss.
        no_ablation_results_list: A list of the accuracy/loss for the no ablation case for each
            experiment.
        out_file: The file to save the plot to.
        exp_names: The names of the rib_scripts.
        ablation_types: The type of ablation performed for each experiment ("orthogonal" or "rib").
        log_scale: Whether to use a log scale for the x-axis. Defaults to False.
        xlim: The limits for the x-axis. Defaults to None.
        ylim: The limits for the y-axis. Defaults to None.
        ylim_relative: The limits for the y-axis, relative to no_ablation_result. Is overwritten by
            ylim if both are provided. Defaults to None.
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

    # Check that all ablation curves use the same baseline. If not that is okay but should be
    # warned about. Multiple baselines will produce multiple grey baseline lines.
    if not all(
        no_ablation_result == no_ablation_results_list[0]
        for no_ablation_result in no_ablation_results_list[1:]
    ):
        logger.warning(
            "Different baselines detected! Are you sure you want to compare these results?"
        )

    for i, node_layer in enumerate(node_layers):
        for j, [exp_name, ablation_type, exp_results, no_ablation_result] in enumerate(
            zip(exp_names, ablation_types, results, no_ablation_results_list)
        ):
            n_vecs_remaining = sorted(list(int(k) for k in exp_results[node_layer]))
            y_values = np.array([exp_results[node_layer][str(i)] for i in n_vecs_remaining])
            color = plt.cm.get_cmap("tab10")(j)
            if baseline_is_zero:
                y_values -= no_ablation_result
                if not log_scale_y:
                    axs[i].axhline(0, color="grey", linestyle="--")
            else:
                axs[i].axhline(no_ablation_result, color="grey", linestyle="--")
            axs[i].plot(n_vecs_remaining, y_values, "-o", color=color, label=exp_name)

            axs[i].set_title(f"{eval_type} vs n_remaining_basis_vecs for input to {node_layer}")
            axs[i].set_xlabel("Number of remaining basis vecs")
            axs[i].set_ylabel(eval_type)
            if xlim is not None:
                axs[i].set_xlim(*xlim)
            if ylim is not None:
                axs[i].set_ylim(*ylim)
            if log_scale_x:
                axs[i].set_xscale("log")
            if log_scale_y:
                axs[i].set_yscale("log")

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
    edges: list[Edges],
    cluster_list: Optional[list[list[int]]] = None,
    sort_by_cluster: bool = False,
    edge_norm: Optional[Callable[[torch.Tensor, str], torch.Tensor]] = None,
    line_width_factor: Optional[float] = None,
    out_file: Optional[Path] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    nodes_per_layer: Union[int, list[int]] = 100,
    hide_const_edges: bool = False,
    colors: Optional[list[str]] = None,
    show_node_labels: bool = True,
    node_labels: Optional[list[list[str]]] = None,
) -> None:
    """Plot the a graph for the given edges (not necessarily a RIB graph).

    Args:
        edges (list[Edges]): List of Edges. Internally this is a list of tensors (E_hat) with
            shape (n_nodes_in_l+1, n_nodes_in_l)
        clusters: TODO
        edge_norm: A function to normalize the edges (by layer) before plotting.
        line_width_factor: Scale factor to convert edge weights into line widths. If None, will
            choose a facctor such that, among all layers, the thickest line is 20.
        out_file (Path): The file to save the plot to. If None, no plot is saved
        ax: The axis to plot the graph on. If None, then a new figure is created.
        title (str): The plot suptitle, typically the name of the experiment.
        nodes_per_layer (Union[int, list[int]]): The max number of nodes in each layer. If int, then
            all layers have the same max number of nodes. If list, then the max number of nodes in
            each layer is given by the list.
        hide_const_edges (bool): Whether to hide the outgoing edges from constant nodes. Note that
            this does _not_n check results.center, it is recommended to set hide_const_edges to
            results.center.
        colors (Optional[list[str]]): The colors to use for the nodes in each layer. If None, then
            the tab10 colormap is used. Is overwritten by clusters if both are provided.
        show_node_labels (bool): Whether to show the node labels. Defaults to True.
        node_labels: The labels for each node in the graph. If None, then use RIB dim indices.
    """
    # Process args
    layer_names = [edge.in_node_layer for edge in edges] + [edges[-1].out_node_layer]
    n_layers = len(layer_names)
    assert n_layers == len(edges) + 1
    if isinstance(nodes_per_layer, int):
        nodes_per_layer = [nodes_per_layer] * n_layers
    max_layer_height = max(nodes_per_layer)

    # Normalize the edges
    edge_norm = edge_norm or IdentityEdgeNorm()
    processed_edges = _prepare_edges_for_plotting(
        [edge_norm(edge.E_hat, edge.in_node_layer) for edge in edges],
        nodes_per_layer,
        hide_const_edges=True,
    )
    del edges

    # Create figure and normalize the line width
    width = n_layers * 2
    height = max(nodes_per_layer) / 3
    ax = ax or plt.subplots(1, 1, figsize=(width, height), constrained_layout=True)[1]
    bbox = ax.get_position()  # Get the bounding box of the axes in figure coordinates
    _, fig_height = ax.get_figure().get_size_inches()  #  type: ignore
    axes_height_inches = fig_height * bbox.height
    max_edge_weight = max([edge.max().item() for edge in processed_edges])
    line_width_factor = line_width_factor or axes_height_inches / max_edge_weight
    logger.info(f"Using line width factor {line_width_factor}")

    # Create the graph & add nodes and edges
    # layers = _create_node_layers(processed_edges)

    # def _create_node_layers(edges: list[torch.Tensor]) -> list[np.ndarray]:
    #     """Create a list of node layers from the given edges."""
    #     layers = []
    #     for i, weight_matrix in enumerate(edges):
    #         # Only add nodes to the graph for the current layer if it's the first layer
    #         if i == 0:
    #             current_layer = np.arange(weight_matrix.shape[1])
    #             layers.append(current_layer)
    #         else:
    #             # Ensure that the current layer's nodes are the same as the previous layer's nodes
    #             assert len(layers[-1]) == weight_matrix.shape[1], (
    #                 f"The number of nodes implied by edge matrix {i} ({weight_matrix.shape[1]}) "
    #                 f"does not match the number implied by the previous edge matrix ({len(layers[-1])})."
    #             )
    #             current_layer = layers[-1]
    #         next_layer = np.arange(weight_matrix.shape[0]) + max(current_layer) + 1
    #         layers.append(next_layer)
    #     return layers

    # Property of every node:
    # internal index
    # color
    # label
    # cluster
    # position

    coords_to_index = {}
    graph = nx.Graph()
    running_index = 0
    for i in range(n_layers):
        n_nodes = processed_edges[i - 1].shape[0] if i != 0 else processed_edges[0].shape[1]
        if i < n_layers - 1:
            assert processed_edges[i].shape[1] == n_nodes, "Consistency check failed"
        # Derive positions based on clusters
        positions = list(range(n_nodes))
        if cluster_list is not None:
            clusters = cluster_list[i]
            ordering = sorted(positions, key=lambda x: clusters[x] if clusters[x] != 0 else 999)
            for position, j in enumerate(ordering):
                positions[j] = position
        else:
            clusters = [0] * n_nodes
            # ordering = list(range(n_nodes))
        # Derive colors based on clusters or layers
        if cluster_list is not None:
            colormap = ["#bbbbbb"] + colorcet.glasbey
            color = lambda j: colormap[clusters[j] % 256]
        else:
            colors = colors or [
                f"#{''.join([f'{int(i * 255):02x}' for i in x[:3]])}"
                for x in plt.get_cmap("tab10").colors
            ]
            color = lambda j: colors[i % len(colors)]
        # Add nodes to the graph
        for j, pos in enumerate(positions):
            # for j in ordering[:max_nodes_per_layer]:
            pos = positions[j]
            graph.add_node(
                running_index,
                layer_idx=i,
                rib_idx=j,
                cluster=clusters[j],
                position=pos,
                color=color(j),
            )
            coords_to_index[(i, j)] = running_index
            running_index += 1

    # Property of every edge
    # weight
    # color

    _add_edges_to_graph(graph, processed_edges, coords_to_index)

    nodes = graph.nodes(data=True)

    pos_dict = {node: (data["layer_idx"], data["position"]) for node, data in nodes}

    # Draw nodes:
    options = {"edgecolors": "tab:gray", "node_size": 50, "alpha": 0.6, "ax": ax}
    for node, data in graph.nodes(data=True):
        if data["position"] <= nodes_per_layer[data["layer_idx"]]:
            nx.draw_networkx_nodes(
                graph, pos_dict, nodelist=[node], node_color=data["color"], **options
            )
    # Draw layer labels
    for i, layer_name in enumerate(layer_names):
        ax.text(i, max_layer_height, layer_name, ha="center", va="center", fontsize=12)
        # nodelist = [node for node in graph.nodes if nodes[node]["layer_idx"] == i]
        # node_colors = [
        #     nodes[node]["color"] for node in nodelist if nodes[node]["layer_idx"] == i
        # ]
        # nx.draw_networkx_nodes(
        #     graph, pos_dict, nodelist=nodelist, node_color=node_colors, ax=ax, **options
        # )
        # # Add layer label above the nodes
        # ax.text(i, max_layer_height, layer_name, ha="center", va="center", fontsize=12)

    # Draw edges
    nx.draw_networkx_edges(
        graph,
        pos_dict,
        edgelist=[(edge[0], edge[1]) for edge in graph.edges(data=True)],
        width=[line_width_factor * edge[2]["weight"] for edge in graph.edges(data=True)],
        alpha=1,
        edge_color=[edge[2]["color"] for edge in graph.edges(data=True)],
        ax=ax,
    )

    # Draw labels
    if show_node_labels:
        for node, data in graph.nodes(data=True):
            label = (
                str(data["rib_idx"])
                if node_labels is None
                else node_labels[data["layer_idx"]][data["rib_idx"]].replace("|", "\n")
            )
            label += f"\nRIB dir {data['rib_idx']}"
            label += f"\ncluster {data['cluster']}"
            nx.draw_networkx_labels(
                graph,
                pos_dict,
                {node: label},
                font_size=4,
                ax=ax,
                bbox={"ec": "k", "fc": "white", "alpha": 0.30, "boxstyle": "round,pad=0.2"},
                font_color=data["color"],
            )

        # node_label_dict = {}
        # node_color_dict = {}
        # for i, layer in enumerate(layers):
        #     node_colors = [colormap[cluster_idx % 256] for cluster_idx in clusters[i]]
        #     for j, node in enumerate(layer):
        #         if node_labels is not None:
        #             node_label_dict[node] = node_labels[i][j].replace("|", "\n")
        #             node_color_dict[node] = node_colors[j]
        #         else:
        #             node_label_dict[node] = str(j)
        #             node_color_dict[node] = node_colors[j]
        # label_options = {"ec": "k", "fc": "white", "alpha": 0.30, "boxstyle": "round,pad=0.2"}
        # for idx, (node, label) in enumerate(node_label_dict.items()):
        #     nx.draw_networkx_labels(
        #         graph,
        #         pos_dict,
        #         {node: label},
        #         font_size=4,
        #         ax=ax,
        #         bbox=label_options,
        #         font_color=node_color_dict[node],
        #     )

    # # Create positions for each node
    # pos: dict[int, tuple[int, Union[int, float]]] = {}
    # for i, layer in enumerate(layers):
    #     # Add extra spacing for nodes that have fewer nodes than the biggest layer
    #     spacing = max_layer_height / len(layer)
    #     if clusters is None:
    #         ordering = list(range(len(layer)))
    #     else:
    #         # truncate clusters to only include the nodes we keep in the graph
    #         clusters[i] = clusters[i][: nodes_per_layer[i]]
    #         ordering = sorted(range(len(layer)), key=lambda x: clusters[i][x])

    #     if sort_by_cluster:
    #         for j, node in enumerate(layer[ordering]):
    #             pos[node] = (i, j * spacing)
    #     else:
    #         for j, node in enumerate(layer):
    #             pos[node] = (i, j * spacing)

    # for i in range(n_layers):
    #     for j in range(max_nodes_per_layer[i]):
    #         nodes[i * max_nodes_per_layer[i] + j]["pos"] = (i, j)

    # # Draw nodes
    # if colors is None:
    #     # tab10 colormap, convert from rgb to hex to avoid matplotlib warning
    #     to_hex = lambda x: f"{int(x * 255):02x}"
    #     colors = [f"#{to_hex(r)}{to_hex(g)}{to_hex(b)}" for r, g, b in plt.get_cmap("tab10").colors]  # type: ignore
    #
    # for i, (layer_name, layer) in enumerate(zip(layer_names, layers)):
    #     node_colors: Union[str, list[str]]
    #     if clusters is not None:
    #         colormap = ["#bbbbbb"] + colorcet.glasbey
    #         node_colors = [colormap[cluster_idx % 256] for cluster_idx in clusters[i]]
    #     else:
    #         # pos: dict[int, tuple[int, float]]
    #         # layer: array[int]
    #         # node_colors: list[str]
    #         #
    #         node_colors = colors[i % len(colors)]

    # Draw edges
    # nx.draw_networkx_edges(
    #     graph,
    #     pos,
    #     edgelist=[(edge[0], edge[1]) for edge in graph.edges(data="full")],
    #     width=[line_width_factor * edge[2]["weight"] for edge in graph.edges(data="full")],
    #     alpha=1,
    #     edge_color=[edge[2]["color"] for edge in graph.edges(data="full")],
    #     ax=ax,
    # )

    if cluster_list is not None:
        n_unique_clusters = len(set([c for layer in cluster_list for c in layer]))
        title = title + " | " if title is not None else ""
        title += f"{n_unique_clusters} clusters"

    if title is not None:
        plt.suptitle(title)

    ax.axis("off")
    if out_file is not None:
        plt.savefig(out_file, dpi=600)


def plot_graph_by_layer(
    edges: list[Edges],
    clusters: Optional[list[list[int]]] = None,
    edge_norm: Optional[Callable[[torch.Tensor, str], torch.Tensor]] = None,
    line_width_factor: Optional[float] = None,
    out_file: Optional[Path] = None,
    title: Optional[str] = None,
    nodes_per_layer=100,
    hide_const_edges: bool = True,
):
    """
    Plots a RIB graph with every transformer block on it's own row. Note: We skip all node layers
    without a block, i.e. ln_final and ln_final_out.

    Can be called from the command line interface with `--by_layer` flag, you'll need to call the
    function from python if you want to use edge_norm.

    Arguments like plot_rib_graph except for the node_labers argument, which is not supported here.

    Args:
        edges (list[Edges]): List of Edges. Internally this is a list of tensors (E_hat) with
            shape (n_nodes_in_l+1, n_nodes_in_l)
        clusters: TODO
        edge_norm: A function to normalize the edges (by layer) before plotting.
        line_width_factor: Scale factor to convert edge weights into line widths. If None, will
            choose a facctor such that, among all layers, the thickest line is 20.
        out_file (Path): The file to save the plot to. If None, no plot is saved
        title (str): The plot suptitle, typically the name of the experiment.
        nodes_per_layer (Union[int, list[int]]): The max number of nodes in each layer. If int, then
            all layers have the same max number of nodes. If list, then the max number of nodes in
            each layer is given by the list.
        hide_const_edges (bool): Whether to hide the outgoing edges from constant nodes. Note that
            this does _not_n check results.center, it is recommended to set hide_const_edges to
            results.center.
    """
    node_layers = [edge.in_node_layer for edge in edges] + [edges[-1].out_node_layer]

    # How many blocks do the results span:
    get_block = lambda name: int(name.split(".")[1]) if "." in name else None
    blocks_in_results = [get_block(nl) for nl in node_layers if get_block(nl) is not None]
    assert blocks_in_results, "No blocks found in the results"
    blocks = range(min(blocks_in_results), max(blocks_in_results) + 1)  # type: ignore
    # Make figure for all blocks
    fig, axs = plt.subplots(len(blocks), 1, figsize=(8, len(blocks) * 6))
    axs = axs if len(blocks) > 1 else [axs]
    # Make individual plots for each block
    for ax, block in tqdm(zip(axs, blocks, strict=True), total=len(blocks), desc="Plotting Blocks"):
        # Get the edges for each block
        block_edges = [edge for edge in edges if get_block(edge.in_node_layer) == block]
        # Get the clusters for each block
        if clusters is not None:
            assert len(clusters) == len(node_layers), "Clusters must be provided for each layer"
            block_layers = [edge.in_node_layer for edge in block_edges] + [
                block_edges[-1].out_node_layer
            ]
            block_clusters = [
                clusters[node_layers.index(nl)] for nl in node_layers if nl in block_layers
            ]
        else:
            block_clusters = None
        # Call main plotting function without out_file
        plot_rib_graph(
            edges=block_edges,
            title=title,
            nodes_per_layer=nodes_per_layer,
            edge_norm=edge_norm,
            out_file=None,
            node_labels=None,
            hide_const_edges=hide_const_edges,
            ax=ax,
            line_width_factor=line_width_factor,
            clusters=block_clusters,
        )
    # Save the figure
    if out_file is not None:
        plt.savefig(out_file, dpi=400)
        logger.info(f"Saved plot to {Path(out_file).absolute()}")
