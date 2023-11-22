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
from pydantic import BaseModel, Field, field_validator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rib.ablations import (
    ExponentialScheduleConfig,
    LinearScheduleConfig,
    load_basis_matrices,
    run_ablations,
)
from rib.hook_manager import HookedModel
from rib.loader import load_mlp
from rib.log import logger
from rib.models import MLP
from rib.types import TORCH_DTYPES
from rib.utils import (
    REPO_ROOT,
    eval_model_accuracy,
    load_config,
    overwrite_output,
    set_seed,
)


class Config(BaseModel):
    exp_name: Optional[str]
    ablation_type: Literal["rib", "orthogonal"]
    interaction_graph_path: Path
    schedule: Union[ExponentialScheduleConfig, LinearScheduleConfig] = Field(
        ...,
        discriminator="schedule_type",
        description="The schedule to use for ablations.",
    )
    dtype: str
    ablation_node_layers: list[str]
    batch_size: int
    seed: int

    @field_validator("dtype")
    @classmethod
    def dtype_validator(cls, v: str):
        assert v in TORCH_DTYPES, f"dtype must be one of {TORCH_DTYPES}"
        return v


def load_mnist_dataloader(train: bool = False, batch_size: int = 64) -> DataLoader:
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(
        root=REPO_ROOT / ".data", train=train, download=True, transform=transform
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


def main(config_path_or_obj: Union[str, Config]) -> None:
    config = load_config(config_path_or_obj, config_model=Config)

    out_file = Path(__file__).parent / "out" / f"{config.exp_name}_ablation_results.json"
    if out_file.exists() and not overwrite_output(out_file):
        print("Exiting.")
        return

    out_file.parent.mkdir(parents=True, exist_ok=True)

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

    mlp = load_mlp(
        config_dict=interaction_graph_info["model_config_dict"],
        mlp_path=interaction_graph_info["config"]["mlp_path"],
        device=device,
    )
    mlp.eval()
    mlp.to(device)
    mlp.to(dtype)
    hooked_mlp = HookedModel(mlp)

    test_loader = load_mnist_dataloader(train=False, batch_size=config.batch_size)

    # Test model accuracy before ablation
    accuracy = eval_model_accuracy(hooked_mlp, test_loader, dtype=dtype, device=device)
    logger.info("Accuracy before ablation: %.2f%%", accuracy * 100)

    accuracies: dict[str, dict[int, float]] = run_ablations(
        basis_matrices=basis_matrices,
        ablation_node_layers=config.ablation_node_layers,
        hooked_model=hooked_mlp,
        data_loader=test_loader,
        eval_fn=eval_model_accuracy,
        graph_module_names=config.ablation_node_layers,
        schedule_config=config.schedule,
        device=device,
        dtype=dtype,
    )

    if config.exp_name is not None:
        results = {
            "config": json.loads(config.model_dump_json()),
            "accuracies": accuracies,
        }
        with open(out_file, "w") as f:
            json.dump(results, f)
        logger.info("Wrote results to %s", out_file)


if __name__ == "__main__":
    fire.Fire(main)
