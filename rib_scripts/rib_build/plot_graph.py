"""Plot a graph given a results file contain the graph edges.

Usage:
    python plot_graph.py <path/to/results.pt> [--nodes_per_layer <int>]
        [--labels_file <path/to/labels.csv>] [--out_file <path/to/out.png>]
        [--force]

    The results.pt should be the output of the run_rib_build.py script.
"""

import csv
from pathlib import Path
from typing import Optional, Union

import fire
import torch

from rib.log import logger
from rib.plotting import plot_rib_graph
from rib.rib_builder import RibBuildResults
from rib.utils import check_out_file_overwrite


def main(
    results_file: str,
    nodes_per_layer: Union[int, list[int]] = 40,
    labels_file: Optional[str] = None,
    out_file: Optional[Union[str, Path]] = None,
    force: bool = False,
    hide_const_edges: Optional[bool] = None,
) -> None:
    """Plot an RIB graph given a results file contain the graph edges."""
    results = RibBuildResults(**torch.load(results_file))
    out_dir = Path(__file__).parent / "out"
    if out_file is None:
        out_file = out_dir / f"{results.exp_name}_rib_graph.png"
    else:
        out_file = Path(out_file)

    if not check_out_file_overwrite(out_file, force):
        return

    assert results.edges, "The results file does not contain any edges."
    # Add labels if provided
    if labels_file is not None:
        with open(labels_file, "r", newline="") as file:
            reader = csv.reader(file)
            node_labels = list(reader)
    else:
        node_labels = None

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
        hide_const_edges=hide_const_edges or results.config.center,
    )

    logger.info(f"Saved plot to {out_file}")


if __name__ == "__main__":
    fire.Fire(main)
