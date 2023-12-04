"""Run an mlp on MNIST while rotating to and from a (truncated) rib or orthogonal basis.

The process is as follows:
    1. Load an MLP trained on MNIST and a test set of MNIST images.
    2. For each number of ablated nodes `n`, create a rotation matrix that has the effect of
    rotating to and from the new basis with `n` fewer basis vectors.
    3. Run the test set through the MLP, applying the above rotations at each node layer, and
    calculate the resulting accuracy.
    5. Repeat steps 3 and 4 for a range of values of n, plotting the resulting accuracies.

A node layer is positioned at the input of each module specified in `node_layers` in the config.
In this script, we don't create a node layer at the output of the final module, as ablating nodes in
this layer is not useful.

Usage:
    python run_mnist_ablations.py <path/to/yaml_config_file>

"""

import json
from pathlib import Path
from typing import Literal, Optional, Union

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
from rib.data import VisionDatasetConfig
from rib.hook_manager import HookedModel
from rib.loader import create_data_loader, load_dataset, load_mlp
from rib.log import logger
from rib.models.mlp import MLPConfig
from rib.types import TORCH_DTYPES, RootPath, StrDtype
from rib.utils import (
    check_outfile_overwrite,
    eval_model_accuracy,
    load_config,
    set_seed,
)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exp_name: str
    ablation_type: Literal["rib", "orthogonal"]
    interaction_graph_path: RootPath
    schedule: Union[ExponentialScheduleConfig, LinearScheduleConfig] = Field(
        ...,
        discriminator="schedule_type",
        description="The schedule to use for ablations.",
    )
    dtype: StrDtype
    ablation_node_layers: list[str]
    batch_size: int
    seed: int
    out_dir: Optional[RootPath] = Field(
        Path(__file__).parent / "out",
        description="Directory for the output files. Defaults to `./out/`. If None, no output "
        "is written. If a relative path, it is relative to the root of the rib repo.",
    )
    dataset: VisionDatasetConfig = VisionDatasetConfig()


def main(config_path_or_obj: Union[str, Config], force: bool = False) -> AblationAccuracies:
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

    mlp_config = MLPConfig(**interaction_graph_info["model_config_dict"]["model"])
    mlp = load_mlp(
        config=mlp_config,
        mlp_path=interaction_graph_info["config"]["mlp_path"],
        fold_bias=True,
        device=device,
    )
    mlp.eval()
    mlp.to(device)
    mlp.to(dtype)
    hooked_mlp = HookedModel(mlp)

    dataset = load_dataset(config.dataset, "train")
    data_loader = create_data_loader(
        dataset, shuffle=True, batch_size=config.batch_size, seed=config.seed
    )

    # Test model accuracy before ablation
    accuracy = eval_model_accuracy(hooked_mlp, data_loader, dtype=dtype, device=device)
    logger.info("Accuracy before ablation: %.2f%%", accuracy * 100)

    accuracies: AblationAccuracies = run_ablations(
        basis_matrices=basis_matrices,
        ablation_node_layers=config.ablation_node_layers,
        hooked_model=hooked_mlp,
        data_loader=data_loader,
        eval_fn=eval_model_accuracy,
        graph_module_names=config.ablation_node_layers,
        schedule_config=config.schedule,
        device=device,
        dtype=dtype,
    )

    results = {
        "config": json.loads(config.model_dump_json()),
        "accuracies": accuracies,
    }

    if config.out_dir is not None:
        with open(out_file, "w") as f:
            json.dump(results, f)
        logger.info("Wrote results to %s", out_file)

    return accuracies


if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: "")
