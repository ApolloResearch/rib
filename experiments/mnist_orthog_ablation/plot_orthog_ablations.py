"""Plot the results of the mnist_orthog_ablation experiment.

Usage:
    python plot_orthog_ablations.py <path/to/results_json_file>

    The results_json_file should be the output of the run_orthog_ablations.py script.
"""
import json
from pathlib import Path

import fire

from rib.plotting import plot_ablation_accuracies
from rib.utils import overwrite_output


def main(results_file: str) -> None:
    with open(results_file, "r") as f:
        results = json.load(f)

    plot_file = (
        Path(__file__).parent
        / "out"
        / f"{results['config']['exp_name']}_accuracy_vs_orthogonal_ablation.png"
    )

    if plot_file.exists() and not overwrite_output(plot_file):
        print("Exiting.")
        return

    plot_ablation_accuracies(
        accuracies=results["accuracies"],
        plot_file=plot_file,
        exp_name=results["config"]["exp_name"],
        model_name="MLP MNIST",
        experiment_type="orthog",
    )


if __name__ == "__main__":
    fire.Fire(main)
