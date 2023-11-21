"""Plot an interaction graph given a results file contain the graph edges.

Usage:
    python plot_mnist_graph.py <path/to/results_pt_file>

    The results_pt_file should be the output of the run_mnist_rib_build.py script.
"""
from pathlib import Path

import fire
import torch

from rib.plotting import plot_interaction_graph
from rib.utils import check_outfile_overwrite
from modular_dnn import BlockDiagonalDNN


def main(results_file: str, force: bool = True) -> None:
    """Plot an interaction graph given a results file contain the graph edges."""
    results = torch.load(results_file)
    out_dir = Path(results_file).parent
    out_file = out_dir / "rib_graph.png"
    out_file_mlp = out_dir / "mlp_graph.png"
    # print(results.keys())
    mlp = results["mlp"]
    mlp_edges = []
    for name in results["model_config_dict"]["node_layers"][:-1]:
        num = int(name.split(".")[1])
        mlp_edges.append((name, mlp.layers[num].W.data))

    if not check_outfile_overwrite(out_file, force):
        return

    # Input layer is much larger so include more nodes in it
    nodes_per_layer = 10
    layer_names = results["model_config_dict"]["node_layers"] + ["output"]

    plot_interaction_graph(
        raw_edges=results["edges"],
        layer_names=layer_names,
        exp_name=results["exp_name"],
        nodes_per_layer=nodes_per_layer,
        out_file=out_file,
    )

    plot_interaction_graph(
        raw_edges=mlp_edges,
        layer_names=layer_names,
        exp_name=results["exp_name"],
        nodes_per_layer=nodes_per_layer,
        out_file=out_file_mlp,
    )


if __name__ == "__main__":
    fire.Fire(main)
