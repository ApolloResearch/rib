"""Run an mlp on MNIST while rotating to and from a (truncated) orthogonal basis.

The process is as follows:
    1. Load an MLP trained on MNIST and a test set of MNIST images.
    2. Calculate gram matrices at each node layer.
    3. Calculate a rotation matrix at each node layer representing the operation of rotating to and
       from the partial eigenbasis of the gram matrix. The partial eigenbasis is equal to the entire
       eigenbasis with the zeroed out eigenvectors corresponding to the n smallest eigenvalues. The
       values of n follow the ablation schedule given below.
    4. Run the test set through the MLP, applying the rotations at each node layer, and calculate
       the resulting accuracy.
    5. Repeat steps 3 and 4 for a range of values of n, plotting the resulting accuracies.

Ablation Schedule:
    Our ablation schedule is exponential with a base of 2, with the exception that we start
    ablating every vector for the final vectors specified by `ablate_every_vec_cutoff` in the
    config. The schedule also includes a run with no ablations. See
    `rib.utils.calc_ablation_schedule` for more details.

A node layer is positioned at the input of each module_name specified in the config. In this script,
we don't create a node layer at the output of the final module, as ablating nodes in this layer
is not useful.


Usage:
    python run_orthog_ablations.py <path/to/yaml_config_file>

This script will take 4 minutes to run on cpu or gpu for 2-layer 100-hidden-unit MLPs with two hook
points.
"""

import json
from pathlib import Path
from typing import Optional

import fire
import torch
import yaml
from pydantic import BaseModel, Field, field_validator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rib.ablations import ablate_and_test
from rib.data_accumulator import collect_gram_matrices
from rib.hook_manager import HookedModel
from rib.linalg import eigendecompose
from rib.log import logger
from rib.models import MLP
from rib.types import TORCH_DTYPES
from rib.utils import (
    REPO_ROOT,
    calc_ablation_schedule,
    load_config,
    overwrite_output,
    set_seed,
)


class Config(BaseModel):
    exp_name: Optional[Path]  # If None, don't write out results
    mlp_path: Path
    ablate_every_vec_cutoff: Optional[int] = Field(
        None,
        description="The point at which we start ablating every individual vector. If None, always ablate every vector.",
    )
    dtype: str
    module_names: list[str]
    seed: int

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
    ablate_every_vec_cutoff: Optional[int],
    dtype: torch.dtype,
) -> dict[str, dict[int, float]]:
    """Rotate to and from orthogonal basis and compare accuracies with and without ablation.

    Args:
        model_config_dict: The config dictionary used for training the mlp.
        mlp_path: The path to the saved mlp.
        module_names: The names of the modules we want to build the graph around.
        ablate_every_vec_cutoff: The point in the ablation schedule to start ablating every vector.
        dtype: The data type to use for model computations.


    Returns:
        A dictionary mapping node lyae to accuracy results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = load_mlp(model_config_dict, mlp_path)
    mlp.eval()
    mlp.to(device)
    mlp.to(dtype)
    hooked_mlp = HookedModel(mlp)

    test_loader = load_mnist_dataloader(train=False, batch_size=512)

    gram_matrices = collect_gram_matrices(
        hooked_mlp, module_names, test_loader, dtype=dtype, device=device, collect_output_gram=False
    )

    results: dict[str, dict[int, float]] = {}
    for module_name in module_names:
        _, eigenvecs = eigendecompose(gram_matrices[module_name])

        ablation_schedule = calc_ablation_schedule(
            ablate_every_vec_cutoff=ablate_every_vec_cutoff,
            n_vecs=len(eigenvecs),
        )
        module_accuracies: dict[int, float] = ablate_and_test(
            hooked_model=hooked_mlp,
            module_name=module_name,
            interaction_rotation=eigenvecs,
            interaction_rotation_pinv=eigenvecs.T,
            test_loader=test_loader,
            device=device,
            ablation_schedule=ablation_schedule,
            hook_name=module_name,
        )

        results[module_name] = module_accuracies

    return results


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)

    set_seed(config.seed)

    with open(config.mlp_path.parent / "config.yaml", "r") as f:
        model_config_dict = yaml.safe_load(f)

    if config.exp_name is not None:
        out_file = Path(__file__).parent / "out" / f"{config.exp_name}_ablation_results.json"
        if out_file.exists() and not overwrite_output(out_file):
            print("Exiting.")
            return
        out_file.parent.mkdir(parents=True, exist_ok=True)

    accuracies: dict[str, dict[int, float]] = run_ablations(
        model_config_dict=model_config_dict,
        mlp_path=config.mlp_path,
        module_names=config.module_names,
        ablate_every_vec_cutoff=config.ablate_every_vec_cutoff,
        dtype=TORCH_DTYPES[config.dtype],
    )
    results = {
        "config": json.loads(config.model_dump_json()),
        "accuracies": accuracies,
    }
    if config.exp_name is not None:
        with open(out_file, "w") as f:
            json.dump(results, f)
        logger.info("Wrote results to %s", out_file)


if __name__ == "__main__":
    fire.Fire(main)
