"""Plot the results of the mnist_orthog_ablation experiment.

Usage:
    python plot_orthog_ablations.py <path/to/results_json_file>

    The results_json_file should be the output of the run_orthog_ablations.py script.
"""
import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt


def plot_accuracies(
    accuracies: dict[str, list[float]],
    plot_file: Path,
    exp_name: str,
) -> None:
    """Plot accuracies for all hook points.

    Args:
        accuracies: A dictionary mapping hook names to accuracy results.
        plot_file: The file to save the plot to.
        exp_name: The name of the experiment
    """
    module_names = list(accuracies.keys())
    n_plots = len(module_names)
    _, axs = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots), dpi=140)

    if n_plots == 1:
        axs = [axs]

    for i, module_name in enumerate(module_names):
        x_values = [len(accuracies[module_name]) - i for i in range(len(accuracies[module_name]))]
        axs[i].plot(x_values, accuracies[module_name], label="MNIST test")

        axs[i].set_title(
            f"{exp_name}-MLP MNIST acc vs n_remaining_eigenvalues for input to {module_name}"
        )
        axs[i].set_xlabel("Number of remaining eigenvalues")
        axs[i].set_ylabel("Accuracy")
        axs[i].set_ylim(0, 1)
        axs[i].grid(True)
        axs[i].legend()

    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.4)

    plt.savefig(plot_file)

    plt.clf()


def main(results_file: str) -> None:
    with open(results_file, "r") as f:
        results = json.load(f)

    out_dir = Path(__file__).parent / "out"
    plot_file = out_dir / f"{results['exp_name']}_accuracy_vs_orthogonal_ablation.png"

    # If the file exists, warn the user and kill the program
    if plot_file.exists():
        print(f"Plot file {plot_file} already exists. Exiting.")
        return

    plot_accuracies(
        accuracies=results["accuracies"],
        plot_file=plot_file,
        exp_name=results["exp_name"],
    )


if __name__ == "__main__":
    fire.Fire(main)
