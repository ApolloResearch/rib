"""Plot an interaction graph given a results file contain the graph edges.

Usage:
    python plot_lm_graph.py <path/to/results_pt_file>

    The results_pt_file should be the output of the run_lm_rib_build.py script.
"""
import csv
from pathlib import Path

import fire
import torch

from rib.plotting import plot_interaction_graph
from rib.utils import check_outfile_overwrite


def main(
    results_file: str,
    nodes_per_layer: int = 40,
    labels_file: str = None,
    out_file: str = None,
    force: bool = False,
) -> None:
    """Plot an interaction graph given a results file contain the graph edges."""
    results = torch.load(results_file)
    out_dir = Path(__file__).parent / "out"
    out_file = out_file or out_dir / f"{results['exp_name']}_rib_graph.png"
    out_file = Path(out_file)

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

    print(f"Saved plot to {out_file}")


if __name__ == "__main__":
    fire.Fire(main)
