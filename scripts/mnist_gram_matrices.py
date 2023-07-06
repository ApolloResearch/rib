from pathlib import Path

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rib.hooks import Hook, HookedModel, gram_matrix_hook_fn
from rib.linalg import eigendecompose
from rib.models import MLP


@torch.inference_mode()
def run_dataset_through_model(
    hooked_model: HookedModel, dataloader: DataLoader, hooks: list[Hook]
) -> None:
    for batch in dataloader:
        data, _ = batch
        hooked_model(data, hooks=hooks)


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


def load_dataloader(train: bool = False) -> DataLoader:
    transform = transforms.ToTensor()
    test_data = datasets.MNIST(
        root=Path(__file__).parent.parent / ".data", train=train, download=True, transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    return test_loader


def calc_eigen_matrices(
    hooked_mlp: HookedModel, hook_points: list[str]
) -> list[str, list[str, Tensor]]:
    """Calculate eigenvalues and eigenvectors of the gram matrices for each hook point.

    Args:
        hooked_mlp: The hooked model.
        hook_points: The hook points for which to calculate the eigenvalues and eigenvectors.

    Returns:
        A dictionary of dictionaries. The outer dictionary is keyed by hook point. The inner
        dictionary is keyed by `vals` and `vecs`, corresponding to the eigenvalues and eigenvectors
    """
    eigen_matrices = {}
    for hook_point in hook_points:
        gram_matrix = hooked_mlp.hooked_data[hook_point]["gram"]
        eigenvalues, eigenvectors = eigendecompose(gram_matrix, descending=True)
        eigen_matrices[hook_point] = {"vals": eigenvalues, "vecs": eigenvectors}
    return eigen_matrices


def main(config_dict: dict, model_path: Path, hook_points: list[str]) -> None:
    mlp = load_mlp(config_dict, model_path)

    hooked_mlp = HookedModel(mlp)
    hooks = [
        Hook(name="gram", fn=gram_matrix_hook_fn, hook_point=hook_point)
        for hook_point in hook_points
    ]

    test_loader = load_dataloader(train=False)

    run_dataset_through_model(hooked_mlp, test_loader, hooks)

    eigen_matrices = calc_eigen_matrices(hooked_mlp, hook_points)
    for hook_point in hook_points:
        gram_matrix = hooked_mlp.hooked_data[hook_point]["gram"]
        print(f"Gram matrix for {hook_point}:")
        print(f"Shape: {gram_matrix.shape}")
        print(f"Eigenvalues: {eigen_matrices[hook_point]['vals'][:3]}")
        print(f"Eigenvectors: {eigen_matrices[hook_point]['vecs'][:3, :3]}")


if __name__ == "__main__":
    model_dir = Path("/mnt/ssd-apollo/checkpoints/MNIST/lr-0.001_bs-64_2023-07-03_15-26-43")
    model_path = model_dir / "model_epoch_12.pt"
    with open(model_dir / "config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)

    hook_points = ["layers.act_0", "layers.linear_1"]

    main(config_dict, model_path, hook_points=hook_points)
