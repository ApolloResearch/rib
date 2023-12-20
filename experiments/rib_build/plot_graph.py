"""Plot a graph given a results file contain the graph edges.

Usage:
    python plot_graph.py <path/to/results.pt> [--nodes_per_layer <int>]
        [--labels_file <path/to/labels.csv>] [--out_file <path/to/out.png>]
        [--force]

    The results.pt should be the output of the run_rib_build.py script.
"""
import fire

from rib.plotting import plot_graph

if __name__ == "__main__":
    fire.Fire(plot_graph)
