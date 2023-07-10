"""Run an mlp on MNIST while rotating to and from a (truncated) orthogonal basis.

The process is as follows:
    1. Load an MLP trained on MNIST and a test set of MNIST images.
    2. Calculate gram matrices at the given hook points.
    3. Calculate a rotation matrix at each hook point representing the operation of rotating to and
       from the partial eigenbasis of the gram matrix. The partial eigenbasis is equal to the entire
       eigenbasis with the zeroed out eigenvectors corresponding to the n smallest eigenvalues,
       where we let n range from 0 to the total number of eigenvalues (i.e. the dimension of the
       gram matrix).
    4. Run the test set through the MLP, applying the rotations at each hook point, and calculate
       the resulting accuracy.
    5. Repeat steps 3 and 4 for a range of values of n, plotting the resulting accuracies.

Usage:
    python mnist_orthogonal_ablation.py <path_to_config>

This script will take 4 minutes to run on cpu or gpu for 2-layer 100-hidden-unit MLPs with two hook
points.
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
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
from rib.models import MLP
from rib.utils import eval_model_accuracy, run_dataset_through_model


class HookConfig(BaseModel):
    hook_name: str
    module_name: str  # The module to hook into
    hook_type: Literal["forward", "pre_forward"]
    layer_size: int  # The size of the data at the hook point


class Config(BaseModel):
    mlp_name: str
    mlp_path: Path
    hook_configs: list[HookConfig]


def load_config(config_path: Path) -> Config:
    """Load the config from a YAML file into a Pydantic model."""
    assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)
    return config


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
    mlp.eval()
    return mlp


def load_mnist_dataloader(train: bool = False, batch_size: int = 64) -> DataLoader:
    transform = transforms.ToTensor()
    test_data = datasets.MNIST(
        root=Path(__file__).parent.parent / ".data", train=train, download=True, transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return test_loader


def scale_gram_matrices(hooked_mlp: HookedModel, module_names: list[str], num_samples: int) -> None:
    """Divide the gram matrices by the number of samples."""
    for module_name in module_names:
        hooked_mlp.hooked_data[module_name]["gram"] = (
            hooked_mlp.hooked_data[module_name]["gram"] / num_samples
        )


def plot_accuracies(
    hook_names: list[str],
    results: dict[str, list[float]],
    plot_dir: Path,
    mlp_name: str,
) -> None:
    """Plot accuracies for all hook points.

    Args:
        hook_names: The names of the hook points.
        results: A dictionary mapping hook points to accuracy results.
        plot_dir: The directory where the plots should be saved.
        mlp_name: The name of the mlp
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    n_plots = len(hook_names)
    _, axs = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots), dpi=140)

    if n_plots == 1:
        axs = [axs]

    for i, hook_name in enumerate(hook_names):
        x_values = [len(results[hook_name]) - i for i in range(len(results[hook_name]))]
        axs[i].plot(x_values, results[hook_name], label="MNIST test")

        axs[i].set_title(f"{mlp_name}-MLP MNIST acc vs n_remaining_eigenvalues for: {hook_name}")
        axs[i].set_xlabel("Number of remaining eigenvalues")
        axs[i].set_ylabel("Accuracy")
        axs[i].set_ylim(0, 1)
        axs[i].grid(True)
        axs[i].legend()

    filename = f"{mlp_name}_accuracy_vs_orthogonal_ablation.png"
    plt.savefig(plot_dir / filename)

    plt.clf()


def ablate_and_test(
    hooked_mlp: HookedModel,
    hook_config: HookConfig,
    test_loader: DataLoader,
    eigenvecs: Float[Tensor, "d_hidden d_hidden"],
    device: str,
) -> list[float]:
    """Ablate eigenvectors and test the model accuracy.

    Args:
        hooked_mlp: The hooked model.
        hook_config: The config for the hook point.
        test_loader: The DataLoader for the test data.
        eigenvecs: A matrix whose columns are the eigenvectors of the gram matrix.
        device: The device to run the model on.

    Returns:
        A list of accuracies for each number of ablated vectors.
    """
    assert hook_config.hook_type in ["forward", "pre_forward"]
    rotate_hook_fn_name = "rotate_" + hook_config.hook_type + "_hook_fn"

    accuracies: list[float] = []

    # Iterate through possible number of ablated vectors.
    for n_ablated_vecs in tqdm(
        range(hook_config.layer_size + 1),
        total=len(range(hook_config.layer_size + 1)),
        desc=f"Ablating {hook_config.module_name}",
    ):
        rotation_matrix = calc_rotation_matrix(eigenvecs, n_ablated_vecs=n_ablated_vecs)
        rotation_hook = Hook(
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


def run_ablations(
    model_config_dict: dict, mlp_path: Path, hook_configs: list[HookConfig]
) -> dict[str, list[float]]:
    """Rotate to and from orthogonal basis and compare accuracies with and without ablation.

    Args:
        model_config_dict: The config dictionary used for training the mlp.
        mlp_path: The path to the saved mlp.
        hook_configs: Information about the hook points.

    Returns:
        A dictionary mapping hook points to accuracy results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = load_mlp(model_config_dict, mlp_path)
    mlp.to(device)
    hooked_mlp = HookedModel(mlp)

    gram_hooks: list[Hook] = []
    for hook_config in hook_configs:
        assert hook_config.hook_type in ["forward", "pre_forward"]
        gram_hook_fn_name = f"gram_{hook_config.hook_type}_hook_fn"
        gram_hooks.append(
            Hook(data_key="gram", fn_name=gram_hook_fn_name, module_name=hook_config.module_name)
        )

    test_loader = load_mnist_dataloader(
        train=False, batch_size=model_config_dict["train"]["batch_size"]
    )

    run_dataset_through_model(hooked_mlp, test_loader, gram_hooks, device=device)
    len_dataset = len(test_loader.dataset)  # type: ignore

    results: dict[str, list[float]] = {}
    for hook_config in hook_configs:
        hooked_mlp.hooked_data[hook_config.module_name]["gram"] = (
            hooked_mlp.hooked_data[hook_config.module_name]["gram"] / len_dataset
        )
        _, eigenvecs = eigendecompose(hooked_mlp.hooked_data[hook_config.module_name]["gram"])
        accuracies: list[float] = ablate_and_test(
            hooked_mlp=hooked_mlp,
            hook_config=hook_config,
            test_loader=test_loader,
            eigenvecs=eigenvecs,
            device=device,
        )
        results[hook_config.hook_name] = accuracies

    return results


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path)

    with open(config.mlp_path.parent / "config.yaml", "r") as f:
        model_config_dict = yaml.safe_load(f)

    plot_dir = Path(__file__).parent.parent / "plots" / "mnist"

    results = run_ablations(
        model_config_dict=model_config_dict,
        mlp_path=config.mlp_path,
        hook_configs=config.hook_configs,
    )
    plot_accuracies(
        hook_names=[cfg.hook_name for cfg in config.hook_configs],
        results=results,
        plot_dir=plot_dir,
        mlp_name=config.mlp_name,
    )


if __name__ == "__main__":
    fire.Fire(main)
