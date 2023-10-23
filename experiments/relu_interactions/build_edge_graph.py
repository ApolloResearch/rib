"""Plot an interaction graph given a results file contain the graph edges.

Usage:
    python plot_mnist_graph.py <path/to/results_pt_file>

    The results_pt_file should be the output of the run_mnist_rib_build.py script.
"""
from pathlib import Path

import fire
import torch

from rib.plotting import plot_interaction_graph
from rib.utils import overwrite_output


def main(results_file: str) -> None:
    """Plot an interaction graph given a results file contain the graph edges."""
    results = torch.load(results_file)
    out_dir = Path(__file__).parent / "out_Cs_Ws"
    out_file = out_dir / f"{results['exp_name']}_interaction_graph.png"

    if out_file.exists() and not overwrite_output(out_file):
        print("Exiting.")
        return

    # Input layer is much larger so include more nodes in it
    nodes_per_layer = [40, 10, 10, 10]

    plot_interaction_graph(
        raw_edges=results["edges"],
        exp_name=results["exp_name"],
        nodes_per_layer=nodes_per_layer,
        out_file=out_file,
        layer_names=['layers.0', 'layers.1', 'layers.2']
    )


if __name__ == "__main__":
    fire.Fire(main)
