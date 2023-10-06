"""Plot an interaction graph given a results file contain the graph edges.

Usage:
    python plot_lm_graph.py <path/to/results_pt_file>

    The results_pt_file should be the output of the build_interaction_graph.py script.
"""
from pathlib import Path

import fire
import torch

from rib.plotting import plot_interaction_graph
from rib.utils import overwrite_output


def main(results_file: str) -> None:
    """Plot an interaction graph given a results file contain the graph edges.

    TODO: Handle showing a different number of nodes per layer as opposed to the same n_nodes in
    all layers after the first one.
    """
    results = torch.load(results_file)
    out_dir = Path(__file__).parent / "out"
    out_file = out_dir / f"{results['exp_name']}_interaction_graph.png"

    if out_file.exists() and not overwrite_output(out_file):
        print("Exiting.")
        return

    plot_interaction_graph(
        raw_edges=results["edges"],
        exp_name=results["exp_name"],
        nodes_input_layer=40,
        nodes_per_layer=40,
        out_file=out_file,
    )


if __name__ == "__main__":
    fire.Fire(main)
