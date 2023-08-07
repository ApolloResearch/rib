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

import fire
import torch
from jaxtyping import Float
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from rib.hook_manager import Hook, HookedModel
from rib.linalg import calc_rotation_matrix
from rib.log import logger
from rib.models import MLP
from rib.utils import REPO_ROOT, eval_model_accuracy, load_config


class Config(BaseModel):
    exp_name: str
    interaction_graph_path: Path
    module_names: list[str]
    batch_size: int


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


def ablate_and_test(
    hooked_mlp: HookedModel,
    module_name: str,
    interaction_rotation: Float[Tensor, "d_hidden d_hidden_trunc"],
    interaction_rotation_pinv: Float[Tensor, "d_hidden_trunc d_hidden"],
    test_loader: DataLoader,
    device: str,
) -> list[float]:
    """Ablate eigenvectors and test the model accuracy.

    Args:
        hooked_mlp: The hooked model.
        module_name: The name of the module whose inputs we want to rotate and ablate.
        interaction_rotation: The matrix that rotates activations to the interaction basis. (C)
        interaction_rotation_pinv: The pseudo-inverse of the interaction rotation matrix. (C^+)
        hook_config: The config for the hook point.
        test_loader: The DataLoader for the test data.
        device: The device to run the model on.

    Returns:
        A list of accuracies for each number of ablated vectors.
    """

    accuracies: list[float] = []

    n_interaction_nodes = interaction_rotation.shape[1]
    # Iterate through possible number of ablated vectors.
    for n_ablated_vecs in tqdm(
        range(n_interaction_nodes + 1),
        total=n_interaction_nodes,
        desc=f"Ablating {module_name}",
    ):
        rotation_matrix = calc_rotation_matrix(
            vecs=interaction_rotation,
            vecs_pinv=interaction_rotation_pinv,
            n_ablated_vecs=n_ablated_vecs,
        )
        rotation_hook = Hook(
            name=module_name,
            data_key="rotation",
            fn_name="rotate_orthog_pre_forward_hook_fn",
            module_name=module_name,
            fn_kwargs={"rotation_matrix": rotation_matrix},
        )

        accuracy_ablated = eval_model_accuracy(
            hooked_mlp, test_loader, hooks=[rotation_hook], device=device
        )
        accuracies.append(accuracy_ablated)

    return accuracies


def run_ablations(
    model_config_dict: dict,
    mlp_path: Path,
    module_names: list[str],
    interaction_matrices: list[
        tuple[Float[Tensor, "d_hidden d_hidden_trunc"], Float[Tensor, "d_hidden_trunc d_hidden"]]
    ],
    batch_size: int = 512,
) -> dict[str, list[float]]:
    """Rotate to and from orthogonal basis and compare accuracies with and without ablation.

    Args:
        model_config_dict: The config dictionary used for training the mlp.
        mlp_path: The path to the saved mlp.
        module_names: The names of the modules whos inputs we want to ablate.
        interaction_matrices: The interaction rotation matrix and its pseudoinverse.
        batch_size: The batch size to pass through the model.


    Returns:
        A dictionary mapping node lyae to accuracy results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = load_mlp(model_config_dict, mlp_path)
    mlp.eval()
    mlp.to(device)
    hooked_mlp = HookedModel(mlp)

    test_loader = load_mnist_dataloader(train=False, batch_size=batch_size)

    results: dict[str, list[float]] = {}
    for module_name, (C, C_pinv) in zip(module_names, interaction_matrices):
        accuracies: list[float] = ablate_and_test(
            hooked_mlp=hooked_mlp,
            module_name=module_name,
            interaction_rotation=C,
            interaction_rotation_pinv=C_pinv,
            test_loader=test_loader,
            device=device,
        )
        results[module_name] = accuracies

    return results


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)

    interaction_graph_info = torch.load(config.interaction_graph_path)

    # Get the interaction rotations and their pseudoinverses (C and C_pinv) using the module_names
    # as keys
    interaction_matrices: list[
        tuple[Float[Tensor, "d_hidden d_hidden_trunc"], Float[Tensor, "d_hidden_trunc d_hidden"]]
    ] = []
    for module_name in config.module_names:
        for rotation_info in interaction_graph_info["interaction_rotations"]:
            if rotation_info["node_layer_name"] == module_name:
                interaction_matrices.append((rotation_info["C"], rotation_info["C_pinv"]))
                break
    assert len(interaction_matrices) == len(
        config.module_names
    ), f"Could not find all modules in the interaction graph config. "

    out_file = Path(__file__).parent / "out" / f"{config.exp_name}_ablation_results.json"
    if out_file.exists():
        # Ask user if they want to overwrite the output.
        response = input(f"Output file {out_file} already exists. Overwrite? (y/n) ")
        if response.lower() != "y":
            print("Exiting.")
            return
        else:
            print("Will overwrite output file %s", out_file)

    out_file.parent.mkdir(parents=True, exist_ok=True)

    accuracies: dict[str, list[float]] = run_ablations(
        model_config_dict=interaction_graph_info["model_config_dict"],
        mlp_path=interaction_graph_info["config"]["mlp_path"],
        module_names=config.module_names,
        interaction_matrices=interaction_matrices,
        batch_size=config.batch_size,
    )
    results = {
        "exp_name": config.exp_name,
        "accuracies": accuracies,
    }
    with open(out_file, "w") as f:
        json.dump(results, f)
    logger.info("Wrote results to %s", out_file)


if __name__ == "__main__":
    fire.Fire(main)