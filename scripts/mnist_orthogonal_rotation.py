"""
Run an mlp with and without rotating to a (truncated) orthogonal basis.

The process is as follows:
1. Load an MLP trained on MNIST and a test set of MNIST images.
2. Calculate gram matrices at the given hook points.
3. Calculate a rotation matrix at each hook point representing the operation of rotating to and from the partial eigenbasis of the gram matrix. The partial eigenbasis is equal to the entire eigenbasis with the n_ablated_vecs eigenvectors corresponding to the n_ablated_vecs smallest eigenvalues zeroed out.
4. Run the test set through the MLP, applying the rotations at each hook point, and calculate the resulting accuracy.

"""

from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rib.hooks import Hook, HookedModel, gram_matrix_hook_fn, rotate_and_ablate_hook_fn
from rib.linalg import calc_eigen_info
from rib.models import MLP
from rib.utils import calc_model_accuracy, run_dataset_through_model


def load_mlp(config_dict: dict, model_path: Path) -> MLP:
    mlp = MLP(
        hidden_sizes=config_dict["model"]["hidden_sizes"],
        input_size=784,
        output_size=10,
        activation_fn=config_dict["model"]["activation_fn"],
        bias=config_dict["model"]["bias"],
        fold_bias=config_dict["model"]["fold_bias"],
    )
    mlp.load_state_dict(torch.load(model_path))
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


def main(config_dict: dict, model_path: Path, hook_points: list[str], n_ablated_vecs: int) -> None:
    """Rotate to and from orthogonal basis and compare accuracies with and without ablation.

    Args:
        config_dict: The config dictionary used for training the mlp.
        model_path: The path to the saved mlp.
        hook_points: The hook points in which to apply the rotations.
        n_ablated_vecs: The number of eigenvectors (corresponding to the smallest eigenvalues) to set to zero.
    """
    mlp = load_mlp(config_dict, model_path)

    hooked_mlp = HookedModel(mlp)
    gram_hooks = [
        Hook(name="gram", fn=gram_matrix_hook_fn, hook_point=hook_point)
        for hook_point in hook_points
    ]

    test_loader = load_mnist_dataloader(train=False, batch_size=config_dict["train"]["batch_size"])

    run_dataset_through_model(hooked_mlp, test_loader, gram_hooks)

    len_dataset = len(test_loader.dataset)  # type: ignore
    scale_gram_matrices(hooked_mlp, hook_points, len_dataset)

    eigens = calc_eigen_info(
        hooked_mlp=hooked_mlp,
        hook_points=hook_points,
        matrix_key="gram",
        zero_threshold=None,  # e.g. [None, 1e-13]
        n_ablated_vecs=n_ablated_vecs,
    )

    rotation_hooks = [
        Hook(
            name="rotation",
            fn=rotate_and_ablate_hook_fn,
            hook_point=hook_point,
            hook_kwargs={"rotation_matrix": eigens[hook_point].rotation_matrix},
        )
        for hook_point in hook_points
    ]
    accuracy = calc_model_accuracy(hooked_mlp, test_loader, [])
    accuracy_ablated = calc_model_accuracy(hooked_mlp, test_loader, rotation_hooks)

    print("Accuracy: ", accuracy)
    print("Accuracy ablated: ", accuracy_ablated)


if __name__ == "__main__":
    model_dir = Path("/mnt/ssd-apollo/checkpoints/MNIST/lr-0.001_bs-64_2023-07-03_15-26-43")
    model_path = model_dir / "model_epoch_12.pt"
    with open(model_dir / "config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)

    hook_points = ["layers.act_0"]
    n_ablated_vecs = 20

    main(
        config_dict=config_dict,
        model_path=model_path,
        hook_points=hook_points,
        n_ablated_vecs=n_ablated_vecs,
    )
