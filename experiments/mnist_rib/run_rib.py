"""Calculate the interaction basis for each layer of an MLP trained on MNIST.

The process is as follows:
    1. Load an MLP trained on MNIST and a test set of MNIST images.
    2. Calculate gram matrices at the given hook points.
    3. Eigendecompose the gram matrices. For the final layer, the interaction basis is equal to the
        eigenbasis of the gram matrix.
    4. (for other layers) TODO: complete
    3. Calculate a rotation matrix at each hook point representing the operation of rotating to and
       from the partial interaction basis of the gram matrix. The partial interaction basis is equal
       to the entire interaction basis with the zeroed out columns corresponding to the n smallest
       interactions, where we let n range from 0 to the total number of nodes (i.e. the dimension of
       the gram matrix).
    4. Run the test set through the MLP, applying the rotations at each hook point, and calculate
       the resulting accuracy.
    5. Repeat steps 3 and 4 for a range of values of n, plotting the resulting accuracies.

Usage:
    python run_rib.py <path/to/yaml_config_file>

"""

import json
from dataclasses import dataclass
from pathlib import Path

import fire
import torch
import yaml
from jaxtyping import Float
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from typing_extensions import Literal

from rib.hook_manager import Hook, HookedModel
from rib.linalg import calc_rotation_matrix, eigendecompose
from rib.log import logger
from rib.models import MLP
from rib.utils import (
    REPO_ROOT,
    eval_model_accuracy,
    load_config,
    run_dataset_through_model,
)


class HookConfig(BaseModel):
    hook_name: str
    module_name: str  # The module to hook into
    hook_type: Literal["forward", "pre_forward"]
    layer_size: int  # The size of the data at the hook point


class Config(BaseModel):
    mlp_name: str
    mlp_path: Path
    hook_configs: list[HookConfig]


@dataclass
class EigenInfo:
    """Information about the eigendecomposition of a gram matrix."""

    hook_name: str
    eigenvals: Float[Tensor, "d_hidden"]
    eigenvecs: Float[Tensor, "d_hidden d_hidden"]

@dataclass
class InteractionInfo:
    """Information about the interaction basis of a gram matrix."""

    hook_name: str
    interaction_basis: Float[Tensor, "d_hidden d_hidden"]


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
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return test_loader


def calc_interaction_matrices(
    hooked_mlp: HookedModel,
    hook_configs: list[HookConfig],
    eigen_infos: list[EigenInfo],
    test_loader: DataLoader,
    eigenvecs: Float[Tensor, "d_hidden d_hidden"],
    device: str,
) -> list[InteractionInfo]:
    """Calculate the interaction matrix for each layer of the MLP.

    We assume that the hook_configs (and eigen_infos) are in the same order as the modules of the
    network. Importantly, we also assume that no computation happens between these modules. I.e.
    the input to the 2nd layer is the output of the 1st layer, and so on.

    Args:
        hooked_mlp: The hooked model.
        hook_configs: The configs for the hook points.
        eigen_infos: The eigen information for each layer.
        test_loader: The DataLoader for the test data.
        device: The device to run the model on.

    Returns:
        A list of objects containing the interaction matrix for each layer.
    """


    interaction_info: list[InteractionInfo] = []
    # Iterate through the configs in backwards order
    for i in range(len(hook_configs) - 1, -1, -1):
        if i == len(hook_configs) - 1:
            # The interaction matrix for the final layer is equal to the eigenbasis.
            interaction_matrix = eigen_infos[i].eigenvecs
        else:
            # Get the jacobian of the next layer w.r.t the outputs of the current layer.
            jac = get_jacobian(hook_configs[i + 1])
    # Iterate through possible number of ablated vectors.
    for n_ablated_vecs in tqdm(
        range(hook_config.layer_size + 1),
        total=len(range(hook_config.layer_size + 1)),
        desc=f"Ablating {hook_config.module_name}",
    ):
        rotation_matrix = calc_rotation_matrix(eigenvecs, n_ablated_vecs=n_ablated_vecs)
        rotation_hook = Hook(
            name=hook_config.hook_name,
            data_key="rotation",
            fn_name=rotate_hook_fn_name,
            module_name=hook_config.module_name,
            fn_kwargs={"rotation_matrix": rotation_matrix},
        )

        accuracy_ablated = eval_model_accuracy(
            hooked_mlp, test_loader, hooks=[rotation_hook], device=device
        )
        accuracies.append(accuracy_ablated)

    return accuracies


def run_interaction_algorithm(
    model_config_dict: dict, mlp_path: Path, hook_configs: list[HookConfig]
) -> dict[str, list[float]]:
    """Calculate the interaction basis for each layer of an MLP trained on MNIST.

    Args:
        model_config_dict: The config dictionary used for training the mlp.
        mlp_path: The path to the saved mlp.
        hook_configs: Information about the hook points.

    Returns:
        A dictionary mapping hook points to accuracy results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = load_mlp(model_config_dict, mlp_path)
    mlp.eval()
    mlp.to(device)
    hooked_mlp = HookedModel(mlp)

    gram_hooks: list[Hook] = []
    for hook_config in hook_configs:
        assert hook_config.hook_type in ["forward", "pre_forward"]
        gram_hook_fn_name = f"gram_{hook_config.hook_type}_hook_fn"
        gram_hooks.append(
            Hook(
                name=hook_config.hook_name,
                data_key="gram",
                fn_name=gram_hook_fn_name,
                module_name=hook_config.module_name,
            )listsList[])

    test_loader = load_mnist_dataloader(
        train=False, batch_size=model_config_dict["train"]["batch_size"]
    )
    run_dataset_through_model(hooked_mlp, test_loader, gram_hooks, device=device)
    len_dataset = len(test_loader.dataset)  # type: ignore

    eigen_info: list[EigenInfo] = []
    results: dict[str, list[float]] = {}
    for hook_config in hook_configs:
        # Scale the gram matrix by the number of samples in the dataset.
        hooked_mlp.hooked_data[hook_config.hook_name]["gram"] = (
            hooked_mlp.hooked_data[hook_config.hook_name]["gram"] / len_dataset
        )
        eigenvals, eigenvecs = eigendecompose(hooked_mlp.hooked_data[hook_config.hook_name]["gram"])
        eigen_info.append(EigenInfo(hook_name=hook_config.hook_name, eigenvals=eigenvals, eigenvecs=eigenvecs))

    interaction_bases = calc_interaction_matrices(
        hook_mlp=hooked_mlp,
        hook_configs=hook_configs,
        eigen_info=eigen_info,
        test_loader=test_loader,
        device=device,
    )
    return interaction_bases


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)

    with open(config.mlp_path.parent / "config.yaml", "r") as f:
        model_config_dict = yaml.safe_load(f)

    out_file = Path(__file__).parent / "out" / f"{config.mlp_name}_rib_results.json"
    if out_file.exists():
        logger.error(f"Output file {out_file} already exists. Exiting.")
        return

    out_file.parent.mkdir(parents=True, exist_ok=True)

    accuracies: dict[str, list[float]] = run_interaction_algorithm(
        model_config_dict=model_config_dict,
        mlp_path=config.mlp_path,
        hook_configs=config.hook_configs,
    )
    results = {
        "mlp_name": config.mlp_name,
        "accuracies": accuracies,
    }
    with open(out_file, "w") as f:
        json.dump(results, f)
    logger.info(f"Saved results to {out_file}")


if __name__ == "__main__":
    fire.Fire(main)
