"""Plot an interaction graph given a results file contain the graph edges.

Usage:
    python plot_mlp_graph.py <path/to/results_pt_file>

    The results_pt_file should be the output of the run_mlp_rib_build.py script.
"""
from pathlib import Path

import fire
import torch

from rib.plotting import plot_interaction_graph
from rib.utils import check_outfile_overwrite


def main(results_file: str, force: bool = False) -> None:
    """Plot an interaction graph given a results file contain the graph edges."""
    results = torch.load(results_file)
    out_dir = Path(__file__).parent / "out"
    out_file = out_dir / f"{results['exp_name']}_rib_graph.png"

    if not check_outfile_overwrite(out_file, force):
        return

    nodes_per_layer = [10]

    layer_names = results["config"]["node_layers"] + ["output"]

    plot_interaction_graph(
        raw_edges=results["edges"],
        layer_names=layer_names,
        exp_name=results["exp_name"],
        nodes_per_layer=nodes_per_layer,
        out_file=out_file,
    )


if __name__ == "__main__":
    fire.Fire(main)
