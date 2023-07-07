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

"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import torch
import yaml
from pydantic import BaseModel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from rib.hooks import Hook, HookedModel, gram_matrix_hook_fn, rotate_and_ablate_hook_fn
from rib.linalg import calc_eigen_info
from rib.models import MLP
from rib.models.utils import get_model_attr
from rib.utils import calc_model_accuracy, run_dataset_through_model


class Config(BaseModel):
    mlp_name: str
    mlp_path: Path
    hook_points: list[str]


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


def scale_gram_matrices(hooked_mlp: HookedModel, hook_points: list[str], num_samples: int) -> None:
    """Divide the gram matrices by the number of samples."""
    for hook_point in hook_points:
        hooked_mlp.hooked_data[hook_point]["gram"] = (
            hooked_mlp.hooked_data[hook_point]["gram"] / num_samples
        )


def plot_accuracies(
    hook_points: list[str], results: dict[str, list[float]], plot_dir: Path, mlp_name: str
) -> None:
    """Plot accuracies for all hook points.

    Args:
        hook_points: The hook points.
        results: A dictionary mapping hook points to accuracy results.
        plot_dir: The directory where the plots should be saved.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)

    for hook_point in hook_points:
        # hook_point is typically "layers.linear_0"
        layer_name = hook_point.split(".", 1)[1]
        plt.plot(range(len(results[hook_point])), results[hook_point])
        plt.title(f"{mlp_name}-MLP MNIST acc vs n_ablated_vecs for layer: {layer_name}")
        plt.xlabel("Number of ablated vectors")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.grid(True)

        filename = f"{mlp_name}_accuracy_vs_orthogonal_ablation_{layer_name.replace('.', '-')}.png"
        plt.savefig(plot_dir / filename)

        plt.clf()


def ablate_and_test(
    hooked_mlp: HookedModel, hook_point: str, test_loader: DataLoader, layer_size: int
) -> list[float]:
    """Ablate eigenvectors and test the model accuracy.

    Args:
        hooked_mlp: The hooked model.
        hook_point: The hook point in which to apply the rotations.
        test_loader: The DataLoader for the test data.
        layer_size: The size of the layer at the hook point.

    Returns:
        A list of accuracies for each number of ablated vectors.
    """
    accuracies: list[float] = []
    # Iterate through possible number of ablated vectors.
    # We use every second value as this is a slow process.
    for n_ablated_vecs in tqdm(
        range(0, layer_size + 1, 2), total=layer_size // 2 + 1, desc=f"Ablating {hook_point}"
    ):
        eigens = calc_eigen_info(
            hooked_mlp=hooked_mlp,
            hook_points=[hook_point],
            matrix_key="gram",
            zero_threshold=None,  # e.g. [None, 1e-13]
            n_ablated_vecs=n_ablated_vecs,
        )

        rotation_hook = Hook(
            name="rotation",
            fn=rotate_and_ablate_hook_fn,
            hook_point=hook_point,
            hook_kwargs={"rotation_matrix": eigens[hook_point].rotation_matrix},
        )

        accuracy_ablated = calc_model_accuracy(hooked_mlp, test_loader, [rotation_hook])
        accuracies.append(accuracy_ablated)

    return accuracies


def run_ablations(
    model_config_dict: dict, mlp_path: Path, hook_points: list[str]
) -> dict[str, list[float]]:
    """Rotate to and from orthogonal basis and compare accuracies with and without ablation.

    Args:
        model_config_dict: The config dictionary used for training the mlp.
        mlp_path: The path to the saved mlp.
        hook_points: The hook points in which to apply the rotations.
        plot_dir: The directory to save plots to.
        mlp_name: The name of the mlp (for plotting).

    Returns:
        A dictionary mapping hook points to accuracy results.
    """
    mlp = load_mlp(model_config_dict, mlp_path)
    hooked_mlp = HookedModel(mlp)

    gram_hooks = [
        Hook(name="gram", fn=gram_matrix_hook_fn, hook_point=hook_point)
        for hook_point in hook_points
    ]
    test_loader = load_mnist_dataloader(
        train=False, batch_size=model_config_dict["train"]["batch_size"]
    )

    run_dataset_through_model(hooked_mlp, test_loader, gram_hooks)
    len_dataset = len(test_loader.dataset)  # type: ignore

    results: dict[str, list[float]] = {}
    for hook_point in hook_points:
        hooked_mlp.hooked_data[hook_point]["gram"] = (
            hooked_mlp.hooked_data[hook_point]["gram"] / len_dataset
        )
        # TODO: Find more principled way to get layer size.
        hook_layer: str = hook_point.split("_")[-1]
        layer_size = get_model_attr(hooked_mlp.model, f"layers.linear_{hook_layer}").weight.size(0)  # type: ignore

        accuracies: list[float] = ablate_and_test(hooked_mlp, hook_point, test_loader, layer_size)
        results[hook_point] = accuracies

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
        hook_points=config.hook_points,
    )
    plot_accuracies(
        hook_points=config.hook_points, results=results, plot_dir=plot_dir, mlp_name=config.mlp_name
    )


if __name__ == "__main__":
    fire.Fire(main)
