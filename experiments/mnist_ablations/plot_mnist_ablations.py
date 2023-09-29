"""Plot the results of the mnist_ablations experiment.

Usage:
    python plot_mnist_ablations.py <path/to/results_json_file>

    The results_json_file should be the output of the run_mnist_ablations.py script.
"""
import json
from pathlib import Path

import fire

from rib.plotting import plot_ablation_accuracies
from rib.utils import overwrite_output


def main(results_file: str) -> None:
    with open(results_file, "r") as f:
        results = json.load(f)

    exp_name = results["config"]["exp_name"]
    plot_file = Path(__file__).parent / "out" / f"{exp_name}_accuracy_vs_ablated_vecs.png"

    if plot_file.exists() and not overwrite_output(plot_file):
        print("Exiting.")
        return

    plot_ablation_accuracies(
        accuracies=results["accuracies"],
        plot_file=plot_file,
        exp_name=exp_name,
        model_name="MLP MNIST",
        ablation_type=results["config"]["ablation_type"],
    )


if __name__ == "__main__":
    fire.Fire(main)
