"""Plot an interaction graph given a results file contain the graph edges.

Usage:
    python plot_lm_graph.py <path/to/results_pt_file>

    The results_pt_file should be the output of the run_mnist_rib_build.py script.
"""
from pathlib import Path

import fire
import torch

from rib.plotting import plot_interaction_graph
from rib.utils import overwrite_output


def main(results_file: str, force: bool = False) -> None:
    """Plot an interaction graph given a results file contain the graph edges."""
    results = torch.load(results_file)
    out_dir = Path(__file__).parent / "out"
    out_file = out_dir / f"{results['exp_name']}_rib_graph.png"

    if out_file.exists():
        if force:
            print(f"Overwriting {out_file} (based on cmdline argument)")
        elif not overwrite_output(out_file):
            print("Exiting.")
            return
        else:
            print(f"Overwriting {out_file} (based on user prompt)")

    # Ensure that we have edges
    assert results["edges"], "The results file does not contain any edges."

    # Set all layers to have the same number of nodes
    nodes_per_layer = 40

    plot_interaction_graph(
        raw_edges=results["edges"],
        layer_names=results["config"]["node_layers"],
        exp_name=results["exp_name"],
        nodes_per_layer=nodes_per_layer,
        out_file=out_file,
    )


if __name__ == "__main__":
    fire.Fire(main)
