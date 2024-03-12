"""Plot a graph given a results file contain the graph edges.

Usage:
    python plot_graph.py <path/to/results.pt> [--nodes_per_layer <int>]
        [--labels_file <path/to/labels.csv>] [--out_file <path/to/out.png>]
        [--force] [--by_layer] [--edge_norm_factor <float>]

    The results.pt should be the output of the run_rib_build.py script.
"""

import csv
from pathlib import Path
from typing import Optional, Union

import fire

from rib.log import logger
from rib.modularity import EdgeNorm, IdentityEdgeNorm, SqrtNorm
from rib.plotting import plot_graph_by_layer, plot_rib_graph
from rib.rib_builder import ResultsLike, to_results
from rib.utils import check_out_file_overwrite, handle_overwrite_fail


def main(
    results: ResultsLike,
    nodes_per_layer: Union[int, list[int]] = 64,
    labels_file: Optional[str] = None,
    out_file: Optional[Union[str, Path]] = None,
    force: bool = False,
    hide_const_edges: bool = True,
    by_layer: Optional[bool] = False,
    line_width_factor: Optional[float] = None,
    norm: str = "sqrt",
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

    edge_norm: EdgeNorm
    if norm.lower() == "sqrt" or norm.lower() == "sqrtnorm":
        edge_norm = SqrtNorm()
    elif norm.lower() == "none" or norm.lower() == "identity":
        edge_norm = IdentityEdgeNorm()
    else:
        raise ValueError(f"Unknown norm: {norm}")

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
            edge_norm=edge_norm,
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
        edge_norm=edge_norm,
        hide_const_edges=results.config.center and hide_const_edges,
        line_width_factor=line_width_factor,
    )

    logger.info(f"Saved plot to {out_file}")


if __name__ == "__main__":
    fire.Fire(main)
