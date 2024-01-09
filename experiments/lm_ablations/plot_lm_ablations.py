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
from typing import Optional, Union

import fire

from rib.log import logger
from rib.plotting import plot_ablation_results
from rib.utils import check_outfile_overwrite


def main(
    *results_files: str,
    force: bool = False,
    out_file: Optional[Union[str, Path]] = None,
    logscale: bool = False,
    exp_names: Optional[list[str]] = None,
    x_shift: Optional[dict[str, float]] = None,
    line_style: Optional[dict[str, str]] = None,
    colors: Optional[dict[str, str]] = None,
) -> None:
    results_list = []
    exp_names = exp_names or []
    ablation_types = []
    eval_type = None

    for results_file in results_files:
        if not results_file.endswith(".json"):
            # Raise a warning to the user, but don't exit.
            logger.warning(f"Skipping {results_file} because it is not a JSON file.")
            continue
        with open(results_file, "r") as f:
            results = json.load(f)

        results_list.append(results["results"])
        exp_names.append(results["config"]["exp_name"])
        ablation_types.append(results["config"]["ablation_type"])
        if eval_type is None:
            eval_type = results["config"]["eval_type"]
        else:
            assert eval_type == results["config"]["eval_type"], (
                "All results must have the same eval_type. "
                f"Expected {eval_type}, got {results['config']['eval_type']}."
            )

    assert eval_type is not None, "Eval type should have been set by now."

    if out_file is None:
        out_filename = "_".join(exp_names)
        out_file = Path(__file__).parent / "out" / f"{out_filename}_{eval_type}_vs_ablated_vecs.png"
    else:
        out_file = Path(out_file)

    if not check_outfile_overwrite(out_file, force):
        return

    plot_ablation_results(
        results=results_list,
        out_file=out_file,
        exp_names=[f"{exp_name}" for exp_name in exp_names],
        eval_type=eval_type,
        ablation_types=ablation_types,
        log_scale=logscale,
        xlim=(0.0, 20.0) if eval_type == "accuracy" else None,
        ylim=(0.0, 1.0) if eval_type == "accuracy" else (3.2, 5),
        x_shift=x_shift,
        line_style=line_style,
        colors=colors,
    )

    logger.info(f"Saved plot to {out_file}")


if __name__ == "__main__":
    fire.Fire(main)
