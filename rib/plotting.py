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
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from rib.data_accumulator import Edges
from rib.log import logger
from rib.modularity import IdentityEdgeNorm, sort_clusters


def _add_edges_to_graph(
    graph: nx.Graph,
    edges: list[torch.Tensor],
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
                graph.add_edge((l, j), (l + 1, i), weight=edge_matrix[i, j], color=color)


def _prepare_edges_for_plotting(
    edges: list[Edges],
    edge_norm: Callable[[torch.Tensor, str], torch.Tensor],
    max_nodes_per_layer: list[int],
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
    raw_edges = [edge.E_hat for edge in edges]
    out_edges: list[torch.Tensor] = []
    for i, weight_matrix in enumerate(raw_edges):
        # Convert edges to float32 (bfloat16 will cause errors and we don't need higher precision)
        weight_matrix = weight_matrix.float()
        if hide_const_edges:
            const_node_index = 0
            # Set edges outgoing from this node to zero (edges.shape ~ l+1, l). The incoming edges
            # should be zero except for a non-rotated last layer where they are important.
            weight_matrix[:, const_node_index] = 0
        # Normalize the edges
        weight_matrix = edge_norm(weight_matrix, edges[i].in_node_layer)
        # Only keep the desired number of nodes in each layer
        in_nodes = max_nodes_per_layer[i]
        out_nodes = max_nodes_per_layer[i + 1]
        out_edges.append(weight_matrix[:out_nodes, :in_nodes])
    return out_edges


def adjust_plot(ax, n_layers, max_layer_height, buffer_size=0.2):
    """Adjust the plot to remove axes etc."""
    # Adjust space between ends of plot and nodes
    ax.set_ylim(-buffer_size, max_layer_height - 1 + buffer_size)
    ax.set_xlim(-buffer_size, n_layers - 1 + buffer_size)
    # Remove unwanted plot enelemts
    ax.grid(False)
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", top=False, labeltop=True, bottom=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


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
    sorting: Literal["rib", "cluster", "clustered_rib"] = "rib",
    edge_norm: Optional[Callable[[torch.Tensor, str], torch.Tensor]] = None,
    line_width_factor: Optional[float] = None,
    out_file: Optional[Path] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    max_nodes_per_layer: Union[int, list[int]] = 100,
    hide_const_edges: bool = False,
    colors: Optional[list[str]] = None,
    show_node_labels: bool = True,
    node_labels: Optional[list[list[str]]] = None,
) -> None:
    """Plot the a graph for the given edges (not necessarily a RIB graph).

    Args:
        edges (list[Edges]): List of Edges. Internally this is a list of tensors (E_hat) with
            shape (n_nodes_in_l+1, n_nodes_in_l)
        clusters: List of cluster indices for every node to color and optionally sort the nodes.
        sorting: The sorting method to use for the nodes. Can be "rib", "cluster", or
            "clustered_rib". Ignored if no clusters provided. "rib" sorts by the RIB index,
            "cluster" sorts by the cluster index, and "clustered_rib" sorts by RIB index but keeps
            nodes in the same cluster together.
        edge_norm: A function to normalize the edges (by layer) before plotting.
        line_width_factor: Scale factor to convert edge weights into line widths. If None, will
            choose a facctor such that, among all layers, the thickest line is 20.
        out_file (Path): The file to save the plot to. If None, no plot is saved
        ax: The axis to plot the graph on. If None, then a new figure is created.
        title (str): The plot suptitle, typically the name of the experiment.
        max_nodes_per_layer (Union[int, list[int]]): The max number of nodes in each layer. If int, then
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
    if isinstance(max_nodes_per_layer, int):
        max_nodes_per_layer = [max_nodes_per_layer] * n_layers

    # Normalize the edges
    edge_norm = edge_norm or IdentityEdgeNorm()
    processed_edges = _prepare_edges_for_plotting(
        edges=edges,
        edge_norm=edge_norm,
        max_nodes_per_layer=max_nodes_per_layer,
        hide_const_edges=hide_const_edges,
    )
    del edges

    # Get actual number of nodes per layer
    nodes_per_layer = [0] * n_layers
    for i in range(n_layers):
        n_nodes = processed_edges[i - 1].shape[0] if i != 0 else processed_edges[0].shape[1]
        nodes_per_layer[i] = min(n_nodes, max_nodes_per_layer[i])
    max_layer_height = max(nodes_per_layer)

    # Create figure and normalize the line width
    width = n_layers * 1.5
    height = 1 + max_layer_height / 2
    ax = ax or plt.subplots(1, 1, figsize=(width, height), constrained_layout=True)[1]
    fig = ax.get_figure()
    bbox = ax.get_position()  # Get the bounding box of the axes in figure coordinates
    _, fig_height = fig.get_size_inches()  # type: ignore
    axes_height_inches = fig_height * bbox.height
    max_edge_weight = max([edge.max().item() for edge in processed_edges])
    line_width_factor = line_width_factor or axes_height_inches / max_edge_weight
    logger.info(f"Using line width factor {line_width_factor}")

    # Create the graph & add nodes and edges

    graph = nx.Graph()
    for i in range(n_layers):
        n_nodes = processed_edges[i - 1].shape[0] if i != 0 else processed_edges[0].shape[1]
        if i < n_layers - 1:
            assert processed_edges[i].shape[1] == n_nodes, "Consistency check failed"
        # Clusters can contain the -1 entries from arr we get len(clusters) > n_nodes. Thus we
        # slice the clusters to the correct length. here.
        clusters = cluster_list[i][:n_nodes] if cluster_list is not None else None
        # Derive positions based on clusters:
        #   Clusters: cluster of each node (clusters in nodes-order)
        #   Positions: position of each node (positions in nodes-order)
        #   Ordering [used in sort_clusters only]: node at each position (nodes in position-order)
        if sorting == "rib":
            positions = list(range(n_nodes))
        else:
            assert clusters is not None, "Clusters must be provided for sorting other than 'rib'"
            positions = sort_clusters(clusters, sorting)
        # Derive colors based on clusters or layers
        if clusters is not None:
            colormap = ["#bbbbbb"] + colorcet.glasbey
            color = lambda j: colormap[clusters[j] % 256]
        else:
            colors = [mpl.colors.rgb2hex(mpl.cm.tab10(i)) for i in range(10)]  # type: ignore
            color = lambda _: colors[i % len(colors)]
        # Add nodes to the graph
        for j in range(nodes_per_layer[i]):
            graph.add_node(
                (i, j),
                layer_idx=i,
                rib_idx=j,
                cluster=clusters[j] if clusters is not None else None,
                position=positions[j],
                color=color(j),
            )

    # Processed edges already have edges beyond nodes_per_layer limit removed.
    _add_edges_to_graph(graph, processed_edges)

    pos_dict = {
        node: (data["layer_idx"], data["position"]) for node, data in graph.nodes(data=True)
    }

    # Draw nodes:
    options = {"edgecolors": "tab:gray", "node_size": 50, "alpha": 0.6, "ax": ax}
    nx.draw_networkx_nodes(
        graph,
        pos_dict,
        nodelist=graph.nodes,
        node_color=[d[1] for d in graph.nodes.data("color")],
        **options,
    )

    # Draw layer labels
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layer_names)

    # Draw edges
    nx.draw_networkx_edges(
        graph,
        pos_dict,
        edgelist=[(u, v) for u, v in graph.edges],
        width=[line_width_factor * weight for _, _, weight in graph.edges(data="weight")],
        alpha=1,
        edge_color=[color for _, _, color in graph.edges(data="color")],
        ax=ax,
    )

    # Draw labels
    if show_node_labels:
        for node, data in graph.nodes(data=True):
            label = f"D{data['rib_idx']}"
            if data["cluster"] is not None:
                label += f" C{data['cluster']}"
            if node_labels is not None:
                label += "\n" + node_labels[data["layer_idx"]][data["rib_idx"]].replace("|", "\n")
            nx.draw_networkx_labels(
                graph,
                pos_dict,
                {node: label},
                font_size=6,
                ax=ax,
                bbox={"ec": "k", "fc": "white", "alpha": 0.30, "boxstyle": "round,pad=0.2"},
                font_color=data["color"],
            )

    if cluster_list is not None:
        n_unique_clusters = len(set([c for layer in cluster_list for c in layer]))
        title = title + " | " if title is not None else ""
        title += f"{n_unique_clusters} clusters"

    if title is not None:
        fig.suptitle(title)  # type: ignore

    adjust_plot(ax, n_layers, max_layer_height)
    if out_file is not None:
        plt.savefig(out_file, dpi=600)


def plot_graph_by_layer(
    edges: list[Edges],
    clusters: Optional[list[list[int]]] = None,
    out_file: Optional[Path] = None,
    **kwargs,
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
        clusters: List of cluster indices for every node to color and optionally sort the nodes.
        out_file (Path): The file to save the plot to. If None, no plot is saved
        kwargs: Additional arguments to pass to plot_rib_graph
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
            cluster_list=block_clusters,
            ax=ax,
            out_file=None,
            node_labels=None,
            **kwargs,
        )
    # Save the figure
    if out_file is not None:
        plt.savefig(out_file, dpi=400)
        logger.info(f"Saved plot to {Path(out_file).absolute()}")
