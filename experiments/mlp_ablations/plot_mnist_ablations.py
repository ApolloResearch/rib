"""Plot the results of mlp_ablations experiments.

Supports plotting multiple experiments on the same plot. To do so, pass in multiple results files as
command line arguments.

All experiments must have the same node layers.

Usage:
    python plot_mlp_ablations.py <path/to/results_json_file1> <path/to/results_json_file2> ...

    The results_json_files should be outputs of the run_mlp_ablations.py script.
"""
import json
from pathlib import Path

import fire

from rib.log import logger
from rib.plotting import plot_ablation_results
from rib.utils import check_outfile_overwrite


def main(*results_files: str, force: bool = False) -> None:
    accuracies_list = []
    exp_names = []
    ablation_types = []

    for results_file in results_files:
        if not results_file.endswith(".json"):
            # Raise a warning to the user, but don't exit.
            logger.warning(f"Skipping {results_file} because it is not a JSON file.")
            continue
        with open(results_file, "r") as f:
            results = json.load(f)

        accuracies_list.append(results["accuracies"])
        exp_names.append(results["config"]["exp_name"])
        ablation_types.append(results["config"]["ablation_type"])

    out_filename = "_".join(exp_names)
    out_file = Path(__file__).parent / "out" / f"{out_filename}_accuracy_vs_ablated_vecs.png"

    if not check_outfile_overwrite(out_file, force):
        return

    plot_ablation_results(
        results=accuracies_list,
        out_file=out_file,
        exp_names=[f"{exp_name} MLP" for exp_name in exp_names],
        eval_type="accuracy",
        ablation_types=ablation_types,
    )


if __name__ == "__main__":
    fire.Fire(main)
