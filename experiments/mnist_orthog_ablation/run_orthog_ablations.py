"""Run an mlp on MNIST while rotating to and from a (truncated) orthogonal basis.

The process is as follows:
    1. Load an MLP trained on MNIST and a test set of MNIST images.
    2. Calculate gram matrices at each node layer.
    3. Calculate a rotation matrix at each node layer representing the operation of rotating to and
       from the partial eigenbasis of the gram matrix. The partial eigenbasis is equal to the entire
       eigenbasis with the zeroed out eigenvectors corresponding to the n smallest eigenvalues,
       where we let n range from 0 to the total number of eigenvalues (i.e. the dimension of the
       gram matrix).
    4. Run the test set through the MLP, applying the rotations at each node layer, and calculate
       the resulting accuracy.
    5. Repeat steps 3 and 4 for a range of values of n, plotting the resulting accuracies.

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

import fire
import torch
import yaml
from jaxtyping import Float
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from rib.data_accumulator import collect_gram_matrices
from rib.hook_manager import Hook, HookedModel
from rib.linalg import calc_rotation_matrix, eigendecompose
from rib.log import logger
from rib.models import MLP
from rib.utils import REPO_ROOT, eval_model_accuracy, load_config, overwrite_output


class Config(BaseModel):
    exp_name: str
    mlp_path: Path
    module_names: list[str]


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
    test_loader: DataLoader,
    eigenvecs: Float[Tensor, "d_hidden d_hidden"],
    device: str,
) -> list[float]:
    """Ablate eigenvectors and test the model accuracy.

    Args:
        hooked_mlp: The hooked model.
        module_name: The name of the module whose inputs we want to rotate and ablate.
        hook_config: The config for the hook point.
        test_loader: The DataLoader for the test data.
        eigenvecs: A matrix whose columns are the eigenvectors of the gram matrix.
        device: The device to run the model on.

    Returns:
        A list of accuracies for each number of ablated vectors.
    """

    accuracies: list[float] = []

    # Iterate through possible number of ablated vectors.
    for n_ablated_vecs in tqdm(
        range(len(eigenvecs) + 1),
        total=len(range(len(eigenvecs) + 1)),
        desc=f"Ablating {module_name}",
    ):
        rotation_matrix = calc_rotation_matrix(
            vecs=eigenvecs, vecs_pinv=eigenvecs.T, n_ablated_vecs=n_ablated_vecs
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
    model_config_dict: dict, mlp_path: Path, module_names: list[str]
) -> dict[str, list[float]]:
    """Rotate to and from orthogonal basis and compare accuracies with and without ablation.

    Args:
        model_config_dict: The config dictionary used for training the mlp.
        mlp_path: The path to the saved mlp.
        module_names: The names of the modules we want to build the graph around.


    Returns:
        A dictionary mapping node lyae to accuracy results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = load_mlp(model_config_dict, mlp_path)
    mlp.eval()
    mlp.to(device)
    hooked_mlp = HookedModel(mlp)

    test_loader = load_mnist_dataloader(train=False, batch_size=512)

    gram_matrices = collect_gram_matrices(
        hooked_mlp, module_names, test_loader, device=device, collect_output_gram=False
    )

    results: dict[str, list[float]] = {}
    for module_name in module_names:
        _, eigenvecs = eigendecompose(gram_matrices[module_name])
        accuracies: list[float] = ablate_and_test(
            hooked_mlp=hooked_mlp,
            module_name=module_name,
            test_loader=test_loader,
            eigenvecs=eigenvecs,
            device=device,
        )
        results[module_name] = accuracies

    return results


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)

    with open(config.mlp_path.parent / "config.yaml", "r") as f:
        model_config_dict = yaml.safe_load(f)

    out_file = Path(__file__).parent / "out" / f"{config.exp_name}_ablation_results.json"
    if out_file.exists() and not overwrite_output(out_file):
        print("Exiting.")
        return

    out_file.parent.mkdir(parents=True, exist_ok=True)

    accuracies: dict[str, list[float]] = run_ablations(
        model_config_dict=model_config_dict,
        mlp_path=config.mlp_path,
        module_names=config.module_names,
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
