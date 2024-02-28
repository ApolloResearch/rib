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

from rib.log import logger
from rib.plotting import plot_rib_graph
from rib.rib_builder import RibBuildResults
from rib.utils import check_out_file_overwrite, handle_overwrite_fail

ResultsLike = Union[RibBuildResults, Path, str]


def _to_results(results: ResultsLike) -> RibBuildResults:
    if isinstance(results, RibBuildResults):
        return results
    elif isinstance(results, (str, Path)):
        return RibBuildResults(**torch.load(results))
    else:
        raise ValueError(f"Invalid results type: {type(results)}")


def plot_by_layer(
    results: ResultsLike,
    nodes_per_layer=100,
    out_file=None,
    edge_norm: Optional[Callable[[torch.Tensor, str], torch.Tensor]] = None,
    hide_const_edges: bool = True,
    const_edge_norm: Optional[float] = None,
):
    """
    Plots a RIB graph with every transformer block on it's own row.

    Can be called from the command line interface with `--by_layer` flag.

    You'll need to call the function from python if you want to use edge_norm.

    Args:
        results: The results file containing the graph edges.
        nodes_per_layer: The number of nodes per layer to plot.
        out_file: The path to save the plot.
        edge_norm: A function to normalize the edge weights pre-plotting.
        hide_const_edges: Whether to hide the nodes corresponding to constant RIB dirs. This is
            ignored if the RIB build is non-centered
        const_edge_norm: If non-none, will use a fixed normalization value for all layers instead
            of normalizing edges layer by layer.
    """
    results = _to_results(results)

    def get_block(name: str) -> Optional[int]:
        split = name.split(".")
        if len(split) == 2:
            return int(split[1])
        else:
            return None

    blocks_in_results = [
        get_block(nl) for nl in results.config.node_layers if get_block(nl) is not None
    ]
    assert blocks_in_results
    blocks = range(min(blocks_in_results), max(blocks_in_results) + 1)  # type: ignore

    fig, axs = plt.subplots(len(blocks), 1, figsize=(8, len(blocks) * 6))
    axs = axs if len(blocks) > 1 else [axs]

    for ax, block in zip(axs, blocks, strict=True):
        edges = [edge for edge in results.edges if get_block(edge.in_node_layer) == block]
        layers = [edge.in_node_layer for edge in edges] + [edges[-1].out_node_layer]
        if edge_norm is not None:
            raw_edges = [edge_norm(edge.E_hat, edge.in_node_layer) for edge in edges]
        else:
            raw_edges = [edge.E_hat for edge in edges]

        plot_rib_graph(
            raw_edges=raw_edges,
            layer_names=layers,
            exp_name=results.exp_name,
            nodes_per_layer=nodes_per_layer,
            out_file=None,
            node_labels=None,
            hide_const_edges=results.config.center and hide_const_edges,
            ax=ax,
            const_edge_norm=const_edge_norm,
        )

    if out_file is not None:
        plt.savefig(out_file, dpi=200)

        logger.info(f"Saved plot to {Path(out_file).absolute()}")


def main(
    results: ResultsLike,
    nodes_per_layer: Union[int, list[int]] = 64,
    labels_file: Optional[str] = None,
    out_file: Optional[Union[str, Path]] = None,
    force: bool = False,
    hide_const_edges: bool = True,
    by_layer: Optional[bool] = False,
    const_edge_norm: Optional[float] = None,
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
        const_edge_norm: If non-none, will use a fixed normalization value for all layers instead
            of normalizing edges layer by layer.
    """
    results = _to_results(results)
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
        plot_by_layer(
            results,
            nodes_per_layer=nodes_per_layer,
            out_file=out_file,
            hide_const_edges=results.config.center and hide_const_edges,
            const_edge_norm=const_edge_norm,
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
        raw_edges=[edges.E_hat for edges in results.edges],
        layer_names=edge_layers,
        exp_name=results.exp_name,
        nodes_per_layer=nodes_per_layer,
        out_file=out_file,
        node_labels=node_labels,
        hide_const_edges=results.config.center and hide_const_edges,
        const_edge_norm=const_edge_norm,
    )

    logger.info(f"Saved plot to {out_file}")


if __name__ == "__main__":
    fire.Fire(main)
