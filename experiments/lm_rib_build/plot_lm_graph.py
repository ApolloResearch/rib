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

    # How many nodes get displayed. If None, display all nodes.
    nodes_input_layer = 40
    nodes_per_layer = 40
    # results["edges"] contains a list of edges which are tuples of
    # (module, edge_weights), each with shape (n_nodes_in_l+1, n_nodes_in_l)
    edges: list[tuple[str, torch.Tensor]] = []
    for i, (module_name, weight_matrix) in enumerate(results["edges"]):
        n_nodes_in = nodes_input_layer if i == 0 else nodes_per_layer
        # Normalize the edge weights by the sum of the absolute values of the weights
        weight_matrix /= torch.sum(torch.abs(weight_matrix))
        # Only keep the first nodes_per_layer nodes in each layer
        edges.append((module_name, weight_matrix[:nodes_per_layer, :n_nodes_in]))

    out_dir = Path(__file__).parent / "out"
    out_file = out_dir / f"{results['exp_name']}_interaction_graph.png"

    if out_file.exists() and not overwrite_output(out_file):
        print("Exiting.")
        return

    plot_interaction_graph(
        edges=edges,
        out_file=out_file,
        exp_name=results["exp_name"],
        n_nodes_ratio=nodes_input_layer / nodes_per_layer,
    )


if __name__ == "__main__":
    fire.Fire(main)
