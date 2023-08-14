"""Plotting functions"""
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt


def plot_ablation_accuracies(
    accuracies: dict[str, dict[int, float]],
    plot_file: Path,
    exp_name: str,
    experiment_type: Literal["orthog", "rib"],
) -> None:
    """Plot accuracy vs number of remaining basis vectors.

    Args:
        accuracies: A dictionary mapping module names to an inner dictionary mapping the number of
            ablated vectors to model accuracy.
        plot_file: The file to save the plot to.
        exp_name: The name of the experiment
        experiment_type: Either 'orthog' or 'rib'.
    """
    module_names = list(accuracies.keys())
    n_plots = len(module_names)
    _, axs = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots), dpi=140)

    if n_plots == 1:
        axs = [axs]

    for i, module_name in enumerate(module_names):
        # module_accuracies = {int(k): v for k, v in accuracies[module_name].items()}
        # We want to show number of vecs remaining, not number of vecs ablated.
        # We thus reverse the list of ablated vecs
        n_ablated_vecs = sorted(list(int(k) for k in accuracies[module_name]), reverse=True)
        n_vecs = max(n_ablated_vecs)
        n_vecs_remaining = [n_vecs - n for n in n_ablated_vecs]
        y_values = [
            accuracies[module_name][str(n_ablated_vecs)] for n_ablated_vecs in n_ablated_vecs
        ]
        axs[i].plot(n_vecs_remaining, y_values, "-o", label="MNIST test")

        title_extra = (
            "n_remaining_eigenvalues" if experiment_type == "orthog" else "n_remaining_basis_vecs"
        )
        xlabel_extra = (
            "Number of remaining eigenvalues"
            if experiment_type == "orthog"
            else "Number of remaining interaction basis vectors"
        )

        axs[i].set_title(f"{exp_name}-MLP MNIST acc vs {title_extra} for input to {module_name}")
        axs[i].set_xlabel(xlabel_extra)
        axs[i].set_ylabel("Accuracy")
        axs[i].set_ylim(0, 1)
        axs[i].grid(True)
        axs[i].legend()

    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.4)

    plt.savefig(plot_file)
