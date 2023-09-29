"""Plot the results of lm_ablations experiments.

Supports plotting multiple experiments on the same plot. To do so, pass in multiple results files as
command line arguments.

All experiments must have the same node layers.

Usage:
    python plot_lm_ablations.py <path/to/results_json_file1> <path/to/results_json_file2> ...

    The results_json_files should be outputs of the run_lm_ablations.py script.
"""
import json
from pathlib import Path

import fire

from rib.plotting import plot_ablation_accuracies
from rib.utils import overwrite_output


def main(*results_files: str) -> None:
    accuracies_list = []
    exp_names = []
    ablation_types = []

    for results_file in results_files:
        with open(results_file, "r") as f:
            results = json.load(f)

        accuracies_list.append(results["accuracies"])
        exp_names.append(results["config"]["exp_name"])
        ablation_types.append(results["config"]["ablation_type"])

    out_filename = "_".join(exp_names)
    out_file = Path(__file__).parent / "out" / f"{out_filename}_accuracy_vs_ablated_vecs.png"

    if out_file.exists() and not overwrite_output(out_file):
        print("Exiting.")
        return

    plot_ablation_accuracies(
        accuracies=accuracies_list,
        out_file=out_file,
        exp_names=[f"{exp_name} LM" for exp_name in exp_names],
        ablation_types=ablation_types,
        log_scale=False,
        xmax=20,
    )


if __name__ == "__main__":
    fire.Fire(main)
