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
import tqdm

from rib.data_accumulator import Edges
from rib.log import logger
from rib.modularity import SqrtNorm
from rib.plotting import plot_rib_graph
from rib.rib_builder import ResultsLike, to_results
from rib.utils import check_out_file_overwrite, handle_overwrite_fail


def plot_by_layer(
    edges: list[Edges],
    title: Optional[str] = None,
    nodes_per_layer=100,
    out_file=None,
    edge_norm: Optional[Callable[[torch.Tensor, str], torch.Tensor]] = None,
    hide_const_edges: bool = True,
    manual_edge_norm_factor: Optional[float] = None,
    clusters: Optional[list[list[int]]] = None,
):
    """
    Plots a RIB graph with every transformer block on it's own row.

    Can be called from the command line interface with `--by_layer` flag.

    You'll need to call the function from python if you want to use edge_norm.

    Note: doesn't plot edges in the final layernorm.

    Args:
        results: The results file containing the graph edges.
        nodes_per_layer: The number of nodes per layer to plot.
        out_file: The path to save the plot.
        edge_norm: A function to normalize the edge weights pre-plotting.
        hide_const_edges: Whether to hide the nodes corresponding to constant RIB dirs. This is
            ignored if the RIB build is non-centered
        manual_edge_norm_factor: If None (default), scales each set of edges by the sum of all edge
            weights. If non-none, will instead scale by `(1/manual_edge_norm_factor)`, keeping
            edge widths consistent across layers for the same E_hat value.
    """
    node_layers = [edge.in_node_layer for edge in edges] + [edges[-1].out_node_layer]

    def get_block(name: str) -> Optional[int]:
        split = name.split(".")
        if len(split) == 2:
            return int(split[1])
        else:
            return None

    blocks_in_results = [get_block(nl) for nl in node_layers if get_block(nl) is not None]
    assert blocks_in_results
    blocks = range(min(blocks_in_results), max(blocks_in_results) + 1)  # type: ignore

    fig, axs = plt.subplots(len(blocks), 1, figsize=(8, len(blocks) * 6))
    axs = axs if len(blocks) > 1 else [axs]

    for ax, block in tqdm.tqdm(
        zip(axs, blocks, strict=True), total=len(blocks), desc="Plotting Blocks"
    ):
        block_edges = [edge for edge in edges if get_block(edge.in_node_layer) == block]
        block_layers = [edge.in_node_layer for edge in block_edges] + [
            block_edges[-1].out_node_layer
        ]

        if clusters is not None:
            block_clusters = [
                nl_clusters
                for nl, nl_clusters in zip(node_layers, clusters, strict=True)
                if nl in block_layers
            ]
        else:
            block_clusters = None

        plot_rib_graph(
            edges=block_edges,
            title=title,
            nodes_per_layer=nodes_per_layer,
            edge_norm=edge_norm,
            out_file=None,
            node_labels=None,
            hide_const_edges=hide_const_edges,
            ax=ax,
            manual_edge_norm_factor=manual_edge_norm_factor,
            clusters=block_clusters,
        )

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
    manual_edge_norm_factor: Optional[float] = None,
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
        manual_edge_norm_factor: If None (default), scales each set of edges by the sum of all edge
            weights. If non-none, will instead scale by `(1/manual_edge_norm_factor)`, keeping
            edge widths consistent across layers for the same E_hat value.
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
        plot_by_layer(
            edges,
            title=results.exp_name,
            nodes_per_layer=nodes_per_layer,
            out_file=out_file,
            edge_norm=SqrtNorm(),
            hide_const_edges=results.config.center and hide_const_edges,
            manual_edge_norm_factor=manual_edge_norm_factor,
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
        manual_edge_norm_factor=manual_edge_norm_factor,
    )

    logger.info(f"Saved plot to {out_file}")


if __name__ == "__main__":
    fire.Fire(main)
