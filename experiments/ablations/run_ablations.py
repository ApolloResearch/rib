"""Run a model on a dataset while rotating to and from a (truncated) rib or orthogonal basis.

The process is as follows:
    1. Load a model and dataset.
    2. For each number of ablated nodes `n`, create a rotation matrix that has the effect of
    rotating to and from the new basis with `n` fewer basis vectors.
    3. Run the test set through the model, applying the above rotations at each node layer, and
    calculate the resulting accuracy/loss.
    5. Repeat steps 3 and 4 for a range of values of n, plotting the resulting accuracies/losses.

A node layer is positioned at the input of each module specified in `node_layers` in the config.
In this script, we don't create a node layer at the output of the final module, as ablating nodes in
this layer is not useful.

Usage:
    python run_ablations.py <path/to/yaml_config_file>

"""
import json
import time
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import fire
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import DataLoader

from rib.ablations import (
    AblationAccuracies,
    ExponentialScheduleConfig,
    LinearScheduleConfig,
    load_basis_matrices,
    run_ablations,
)
from rib.data import (
    HFDatasetConfig,
    ModularArithmeticDatasetConfig,
    VisionDatasetConfig,
)
from rib.hook_manager import HookedModel
from rib.loader import load_model_and_dataset_from_rib_results
from rib.log import logger
from rib.models import MLP, SequentialTransformer
from rib.types import TORCH_DTYPES, RootPath, StrDtype
from rib.utils import (
    check_outfile_overwrite,
    eval_cross_entropy_loss,
    eval_model_accuracy,
    load_config,
    set_seed,
)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
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
    dataset: Union[ModularArithmeticDatasetConfig, HFDatasetConfig, VisionDatasetConfig] = Field(
        ...,
        discriminator="dataset_type",
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

    model, dataset = load_model_and_dataset_from_rib_results(
        interaction_graph_info,
        device=device,
        dtype=dtype,
        node_layers=config.ablation_node_layers,
        dataset_config=config.dataset,
        return_set=config.dataset.return_set,
    )
    model.to(device=torch.device(device), dtype=dtype)
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    hooked_model = HookedModel(model)

    # Test model accuracy/loss before running ablations, ta be sure
    eval_fn: Callable = (
        eval_model_accuracy if config.eval_type == "accuracy" else eval_cross_entropy_loss
    )
    eval_results = eval_fn(hooked_model, data_loader, dtype=dtype, device=device)
    logger.info("Model %s on dataset: %.4f", config.eval_type, eval_results)

    if isinstance(model, MLP):
        graph_module_names = config.ablation_node_layers
    else:
        assert isinstance(model, SequentialTransformer)
        graph_module_names = [f"sections.{sec}" for sec in model.sections if sec != "pre"]

    ablation_results: AblationAccuracies = run_ablations(
        basis_matrices=basis_matrices,
        ablation_node_layers=config.ablation_node_layers,
        hooked_model=hooked_model,
        data_loader=data_loader,
        eval_fn=eval_fn,
        graph_module_names=graph_module_names,
        schedule_config=config.schedule,
        device=device,
        dtype=dtype,
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
