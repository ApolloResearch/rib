"""Run an mlp on MNIST while rotating to and from a (truncated) interaction basis.

The process is as follows:
    1. Load an MLP trained on MNIST and a test set of MNIST images.
    2. For each number of ablated nodes `n`, create a rotation matrix that has the effect of
    rotating to and from the interaction basis with `n` fewer basis vectors.
    3. Run the test set through the MLP, applying the above rotations at each node layer, and
    calculate the resulting accuracy.
    5. Repeat steps 3 and 4 for a range of values of n, plotting the resulting accuracies.

A node layer is positioned at the input of each module_name specified in the config. In this script,
we don't create a node layer at the output of the final module, as ablating nodes in this layer
is not useful.

Usage:
    python run_rib_ablations.py <path/to/yaml_config_file>

This script will take 4 minutes to run on cpu or gpu for 2-hidden-layer 100-hidden-unit MLPs with
two modules.
"""

import json
from pathlib import Path
from typing import Optional

import fire
import torch
from jaxtyping import Float
from pydantic import BaseModel, Field, field_validator
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rib.ablations import ablate_and_test
from rib.hook_manager import HookedModel
from rib.log import logger
from rib.models import MLP
from rib.types import TORCH_DTYPES
from rib.utils import REPO_ROOT, calc_ablation_schedule, load_config, overwrite_output


class Config(BaseModel):
    exp_name: Optional[str]
    interaction_graph_path: Path
    ablate_every_vec_cutoff: Optional[int] = Field(
        None,
        description="The point at which we start ablating every individual vector. If None, always ablate every vector.",
    )
    dtype: str
    module_names: list[str]
    batch_size: int

    @field_validator("dtype")
    def dtype_validator(cls, v):
        assert v in TORCH_DTYPES, f"dtype must be one of {TORCH_DTYPES}"
        return v


def load_mlp(config_dict: dict, mlp_path: Path) -> MLP:
    mlp = MLP(
        hidden_sizes=config_dict["model"]["hidden_sizes"],
        input_size=784,
        output_size=10,
        activation_fn=config_dict["model"]["activation_fn"],
        bias=config_dict["model"]["bias"],
        fold_bias=config_dict["model"]["fold_bias"],
    )
    mlp.load_state_dict(torch.load(mlp_path))
    return mlp


def load_mnist_dataloader(train: bool = False, batch_size: int = 64) -> DataLoader:
    transform = transforms.ToTensor()
    test_data = datasets.MNIST(
        root=REPO_ROOT / ".data", train=train, download=True, transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return test_loader


def run_ablations(
    model_config_dict: dict,
    mlp_path: Path,
    module_names: list[str],
    interaction_matrices: list[
        tuple[Float[Tensor, "d_hidden d_hidden_trunc"], Float[Tensor, "d_hidden_trunc d_hidden"]]
    ],
    ablate_every_vec_cutoff: Optional[int],
    dtype: torch.dtype,
    device: str,
    batch_size: int = 512,
) -> dict[str, dict[int, float]]:
    """Rotate to and from orthogonal basis and compare accuracies with and without ablation.

    Args:
        model_config_dict: The config dictionary used for training the mlp.
        mlp_path: The path to the saved mlp.
        module_names: The names of the modules whos inputs we want to ablate.
        interaction_matrices: The interaction rotation matrix and its pseudoinverse.
        ablate_every_vec_cutoff: The point in the ablation schedule to start ablating every vector.
        dtype: The data type to use for model computations.
        device: The device to run the model on.
        batch_size: The batch size to pass through the model.


    Returns:
        A dictionary mapping node lyae to accuracy results.
    """
    mlp = load_mlp(model_config_dict, mlp_path)
    mlp.eval()
    mlp.to(device)
    mlp.to(dtype)
    hooked_mlp = HookedModel(mlp)

    test_loader = load_mnist_dataloader(train=False, batch_size=batch_size)
    results: dict[str, dict[int, float]] = {}
    for module_name, (C, C_pinv) in zip(module_names, interaction_matrices):
        ablation_schedule = calc_ablation_schedule(
            ablate_every_vec_cutoff=ablate_every_vec_cutoff,
            n_vecs=C.shape[1],
        )

        module_accuracies: dict[int, float] = ablate_and_test(
            hooked_model=hooked_mlp,
            module_name=module_name,
            interaction_rotation=C,
            interaction_rotation_pinv=C_pinv,
            test_loader=test_loader,
            dtype=dtype,
            device=device,
            ablation_schedule=ablation_schedule,
            hook_name=module_name,
        )
        results[module_name] = module_accuracies

    return results


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = TORCH_DTYPES[config.dtype]
    interaction_graph_info = torch.load(config.interaction_graph_path)

    # Get the interaction rotations and their pseudoinverses (C and C_pinv) using the module_names
    # as keys
    interaction_matrices: list[
        tuple[Float[Tensor, "d_hidden d_hidden_trunc"], Float[Tensor, "d_hidden_trunc d_hidden"]]
    ] = []
    for module_name in config.module_names:
        for rotation_info in interaction_graph_info["interaction_rotations"]:
            if rotation_info["node_layer_name"] == module_name:
                C = rotation_info["C"].to(dtype=dtype, device=device)
                C_pinv = rotation_info["C_pinv"].to(dtype=dtype, device=device)
                interaction_matrices.append((C, C_pinv))
                break
    assert len(interaction_matrices) == len(
        config.module_names
    ), f"Could not find all modules in the interaction graph config. "

    out_file = Path(__file__).parent / "out" / f"{config.exp_name}_ablation_results.json"
    if out_file.exists() and not overwrite_output(out_file):
        print("Exiting.")
        return

    out_file.parent.mkdir(parents=True, exist_ok=True)

    accuracies: dict[str, dict[int, float]] = run_ablations(
        model_config_dict=interaction_graph_info["model_config_dict"],
        mlp_path=interaction_graph_info["config"]["mlp_path"],
        module_names=config.module_names,
        interaction_matrices=interaction_matrices,
        ablate_every_vec_cutoff=config.ablate_every_vec_cutoff,
        dtype=dtype,
        device=device,
        batch_size=config.batch_size,
    )
    results = {
        "exp_name": config.exp_name,
        "accuracies": accuracies,
    }
    if config.exp_name is not None:
        with open(out_file, "w") as f:
            json.dump(results, f)
        logger.info("Wrote results to %s", out_file)


if __name__ == "__main__":
    fire.Fire(main)
