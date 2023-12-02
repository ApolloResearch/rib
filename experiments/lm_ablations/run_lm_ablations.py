"""Run an LM on a dataset while rotating to and from a (truncated) rib or orthogonal basis.

The process is as follows:
    1. Load a SequentialTransformer model and dataset.
    2. For each number of ablated nodes `n`, create a rotation matrix that has the effect of
    rotating to and from the new basis with `n` fewer basis vectors.
    3. Run the test set through the model, applying the above rotations at each node layer, and
    calculate the resulting accuracy/loss.
    5. Repeat steps 3 and 4 for a range of values of n, plotting the resulting accuracies/losses.

A node layer is positioned at the input of each module specified in `node_layers` in the config.
In this script, we don't create a node layer at the output of the final module, as ablating nodes in
this layer is not useful.

Usage:
    python run_lm_ablations.py <path/to/yaml_config_file>

"""

import json
import time
from pathlib import Path
from typing import Callable, Literal, Optional, Union, cast

import fire
import torch
from pydantic import BaseModel, ConfigDict, Field

from rib.ablations import (
    AblationAccuracies,
    ExponentialScheduleConfig,
    LinearScheduleConfig,
    load_basis_matrices,
    run_ablations,
)
from rib.data import HFDatasetConfig, ModularArithmeticDatasetConfig
from rib.hook_manager import HookedModel
from rib.loader import create_data_loader, load_dataset, load_sequential_transformer
from rib.log import logger
from rib.types import TORCH_DTYPES, RootPath, StrDtype
from rib.utils import (
    check_outfile_overwrite,
    eval_cross_entropy_loss,
    eval_model_accuracy,
    load_config,
    set_seed,
)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exp_name: str
    out_dir: Optional[RootPath] = Field(
        Path(__file__).parent / "out",
        description="Directory for the output files. Defaults to `./out/`. If None, no output "
        "is written. If a relative path, it is relative to the root of the rib repo.",
    )
    ablation_type: Literal["rib", "orthogonal"]
    interaction_graph_path: RootPath
    schedule: Union[ExponentialScheduleConfig, LinearScheduleConfig] = Field(
        ...,
        discriminator="schedule_type",
        description="The schedule to use for ablations.",
    )
    dataset: Union[ModularArithmeticDatasetConfig, HFDatasetConfig] = Field(
        ...,
        discriminator="source",
        description="The dataset to use to build the graph.",
    )
    ablation_node_layers: list[str]
    batch_size: int
    dtype: StrDtype
    seed: int
    eval_type: Literal["accuracy", "ce_loss"] = Field(
        ...,
        description="The type of evaluation to perform on the model before building the graph.",
    )


def main(config_path_or_obj: Union[str, Config], force: bool = False) -> AblationAccuracies:
    start_time = time.time()
    config = load_config(config_path_or_obj, config_model=Config)

    if config.out_dir is not None:
        config.out_dir.mkdir(parents=True, exist_ok=True)
        out_file = config.out_dir / f"{config.exp_name}_ablation_results.json"
        if not check_outfile_overwrite(out_file, force):
            raise FileExistsError("Not overwriting output file")

    set_seed(config.seed)
    interaction_graph_info = torch.load(config.interaction_graph_path)

    assert set(config.ablation_node_layers) <= set(
        interaction_graph_info["config"]["node_layers"]
    ), "The node layers in the config must be a subset of the node layers in the interaction graph."

    assert "output" not in config.ablation_node_layers, "Cannot ablate the output node layer."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = TORCH_DTYPES[config.dtype]

    basis_matrices = load_basis_matrices(
        interaction_graph_info=interaction_graph_info,
        ablation_node_layers=config.ablation_node_layers,
        ablation_type=config.ablation_type,
        dtype=dtype,
        device=device,
    )

    tlens_model_path = (
        Path(interaction_graph_info["config"]["tlens_model_path"])
        if interaction_graph_info["config"]["tlens_model_path"] is not None
        else None
    )

    # Note that we specify node_layers as config.ablation_node_layers, even though the original
    # graph was built with interaction_graph_info["config"]["node_layers"]. This is because
    # changing the sections in the model has no effect on the computation, and we would like the
    # sections to match ablation_node_layers so we can hook them easily.
    seq_model, _ = load_sequential_transformer(
        node_layers=config.ablation_node_layers,
        last_pos_module_type=interaction_graph_info["config"]["last_pos_module_type"],
        tlens_pretrained=interaction_graph_info["config"]["tlens_pretrained"],
        tlens_model_path=tlens_model_path,
        fold_bias=True,
        dtype=dtype,
        device=device,
    )

    seq_model.eval()
    hooked_model = HookedModel(seq_model)

    # This script doesn't need train and test sets (i.e. the "both" argument)
    return_set = cast(Literal["train", "test", "all"], config.dataset.return_set)
    dataset = load_dataset(
        dataset_config=config.dataset,
        return_set=return_set,
        model_n_ctx=seq_model.cfg.n_ctx,
        tlens_model_path=tlens_model_path,
    )
    data_loader = create_data_loader(
        dataset, shuffle=False, batch_size=config.batch_size, seed=config.seed
    )

    # Test model accuracy/loss before graph building, ta be sure
    eval_fn: Callable = (
        eval_model_accuracy if config.eval_type == "accuracy" else eval_cross_entropy_loss
    )
    eval_results = eval_fn(hooked_model, data_loader, dtype=dtype, device=device)
    logger.info("Model %s on dataset: %.4f", config.eval_type, eval_results)

    graph_module_names = [f"sections.{sec}" for sec in seq_model.sections if sec != "pre"]

    ablation_results: AblationAccuracies = run_ablations(
        basis_matrices=basis_matrices,
        ablation_node_layers=config.ablation_node_layers,
        hooked_model=hooked_model,
        data_loader=data_loader,
        eval_fn=eval_fn,
        graph_module_names=graph_module_names,
        schedule_config=config.schedule,
        device=device,
    )

    time_taken = f"{(time.time() - start_time) / 60:.1f} minutes"
    logger.info("Finished in %s.", time_taken)

    results = {
        "config": json.loads(config.model_dump_json()),
        "results": ablation_results,
        "time_taken": time_taken,
    }
    if config.out_dir is not None:
        with open(out_file, "w") as f:
            json.dump(results, f)
        logger.info("Wrote results to %s", out_file)

    return ablation_results


if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: "")
