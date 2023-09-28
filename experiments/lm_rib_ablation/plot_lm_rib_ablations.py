"""Plot the results of the lm_rib_ablation experiment.

Usage:
    python plot_lm_rib_ablations.py <path/to/results_json_file>

    The results_json_file should be the output of the run_lm_rib_ablations.py script.
"""
import json
from pathlib import Path

import fire

from rib.plotting import plot_ablation_accuracies
from rib.utils import overwrite_output


def main(results_file: str) -> None:
    with open(results_file, "r") as f:
        results = json.load(f)

    out_dir = Path(__file__).parent / "out"
    plot_file = out_dir / f"{results['exp_name']}_accuracy_vs_interaction_ablation.png"

    if plot_file.exists() and not overwrite_output(plot_file):
        print("Exiting.")
        return

    plot_ablation_accuracies(
        accuracies=results["accuracies"],
        plot_file=plot_file,
        exp_name=results["exp_name"],
        model_name="LM",
        experiment_type="rib",
        log_scale=False,
        xmax=20,
    )


if __name__ == "__main__":
    fire.Fire(main)
