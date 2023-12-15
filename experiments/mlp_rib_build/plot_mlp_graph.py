"""Plot an interaction graph given a results file contain the graph edges.

# TODO: Merge with experiments/lm_rib_build/plot_lm_graph.py
Usage:
    python plot_mlp_graph.py <path/to/results_pt_file>

    The results_pt_file should be the output of the run_mlp_rib_build.py script.
"""
import csv
from pathlib import Path
from typing import Optional, Union

import fire
import torch

from rib.log import logger
from rib.plotting import plot_interaction_graph
from rib.utils import check_outfile_overwrite


def main(
    results_file: str,
    nodes_per_layer: Optional[Union[int, list[int]]] = None,
    labels_file: Optional[str] = None,
    out_file: Optional[Union[str, Path]] = None,
    force: bool = False,
) -> None:
    """Plot an interaction graph given a results file contain the graph edges."""
    results = torch.load(results_file)
    out_dir = Path(__file__).parent / "out"
    if out_file is None:
        out_file = out_dir / f"{results['exp_name']}_rib_graph.png"
    else:
        out_file = Path(out_file)

    # Input layer is much larger for mnist so include more nodes in it
    nodes_per_layer = [40, 10, 10, 10] if nodes_per_layer is None else nodes_per_layer
    # nodes_per_layer = (
    #     [max(e[1].shape) for e in results["edges"]] + [max(results["edges"][-1][1].shape)]
    #     if nodes_per_layer is None
    #     else nodes_per_layer
    # )

    if not check_outfile_overwrite(out_file, force):
        return

    # Ensure that we have edges
    assert results["edges"], "The results file does not contain any edges."

    # Add labels if provided
    if labels_file is not None:
        with open(labels_file, "r", newline="") as file:
            reader = csv.reader(file)
            node_labels = list(reader)
    else:
        node_labels = None

    plot_interaction_graph(
        raw_edges=results["edges"],
        layer_names=results["config"]["node_layers"],
        exp_name=results["exp_name"],
        nodes_per_layer=nodes_per_layer,
        out_file=out_file,
        node_labels=node_labels,
    )

    logger.info("Saved plot to %s", out_file)


if __name__ == "__main__":
    fire.Fire(main)
