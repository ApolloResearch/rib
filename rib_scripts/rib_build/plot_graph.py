"""Plot a graph given a results file contain the graph edges.

Usage:
    python plot_graph.py <path/to/results.pt> [--nodes_per_layer <int>]
        [--labels_file <path/to/labels.csv>] [--out_file <path/to/out.png>]
        [--force] [--by_layer] [--edge_norm_factor <float>]

    The results.pt should be the output of the run_rib_build.py script.
"""

import csv
from pathlib import Path
from typing import Callable, Optional, Union

import fire
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from rib.data_accumulator import Edges
from rib.log import logger
from rib.modularity import SqrtNorm
from rib.plotting import plot_rib_graph
from rib.rib_builder import ResultsLike, to_results
from rib.utils import check_out_file_overwrite, handle_overwrite_fail


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
    Plots a RIB graph with every transformer block on it's own row.

    Can be called from the command line interface with `--by_layer` flag.

    You'll need to call the function from python if you want to use edge_norm.

    Note: We skip all node layers without a block, i.e. ln_final and ln_final_out.

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


def main(
    results: ResultsLike,
    nodes_per_layer: Union[int, list[int]] = 64,
    labels_file: Optional[str] = None,
    out_file: Optional[Union[str, Path]] = None,
    force: bool = False,
    hide_const_edges: bool = True,
    by_layer: Optional[bool] = False,
    line_width_factor: Optional[float] = None,
) -> None:
    """Plot an RIB graph given a results file contain the graph edges.

    Args:
        results: The results file containing the graph edges.
        nodes_per_layer: The number of nodes per layer to plot.
        labels_file: A csv file containing the labels for the nodes.
        out_file: The path to save the plot.
        force: Whether to overwrite the out_file if it already exists.
        hide_const_edges: Whether to hide the nodes corresponding to constant RIB dirs. This is
            ignored if the RIB build is non-centered
        by_layer: Whether to plot the graph by layer.
        line_width_factor: Scale factor to convert edge weights into line widths. If None, will
            choose a facctor such that, among all layers, the thickest line is 20.
    """
    results = to_results(results)
    if out_file is None:
        out_dir = Path(__file__).parent / "out"
        out_file = out_dir / f"{results.exp_name}_rib_graph.png"
    else:
        out_file = Path(out_file)

    if not check_out_file_overwrite(out_file, force):
        handle_overwrite_fail()

    assert results.edges, "The results file does not contain any edges."
    hide_const_edges = hide_const_edges or results.config.center

    # Add labels if provided
    if labels_file is not None:
        with open(labels_file, "r", newline="") as file:
            reader = csv.reader(file)
            node_labels = list(reader)
    else:
        node_labels = None

    if by_layer:
        if node_labels is not None:
            raise NotImplementedError("Would just need to find the right subset of labels")
        results = to_results(results)
        edges = results.edges
        plot_graph_by_layer(
            edges,
            title=results.exp_name,
            nodes_per_layer=nodes_per_layer,
            out_file=out_file,
            edge_norm=SqrtNorm(),
            hide_const_edges=results.config.center and hide_const_edges,
            line_width_factor=line_width_factor,
        )
        return None

    edge_layers = [edges.in_node_layer for edges in results.edges] + [
        results.edges[-1].out_node_layer
    ]
    assert edge_layers == results.config.node_layers, (
        "The layers of the edges do not match the layers of the nodes. "
        f"Edge layers: {edge_layers}, node layers: {results.config.node_layers}"
        "Something must have gone wrong when building the graph."
    )
    plot_rib_graph(
        edges=results.edges,
        title=results.exp_name,
        nodes_per_layer=nodes_per_layer,
        out_file=out_file,
        node_labels=node_labels,
        hide_const_edges=results.config.center and hide_const_edges,
        line_width_factor=line_width_factor,
    )

    logger.info(f"Saved plot to {out_file}")


if __name__ == "__main__":
    fire.Fire(main)
