from pathlib import Path
from typing import List

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rib.hooks import Hook, HookedModel, gram_matrix_hook_fn
from rib.models import MLP


@torch.inference_mode()
def run_dataset_through_model(
    hooked_model: HookedModel, dataloader: DataLoader, hooks: List[Hook]
) -> None:
    for batch in dataloader:
        data, _ = batch
        hooked_model(data, hooks=hooks)


def main(config_dict: dict, model_path: Path, hook_points: List[str]) -> None:
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

    hooked_mlp = HookedModel(mlp)
    hooks = [
        Hook(name="gram", fn=gram_matrix_hook_fn, hook_point=hook_point)
        for hook_point in hook_points
    ]

    # Load the MNIST dataset
    transform = transforms.ToTensor()
    test_data = datasets.MNIST(
        root=Path(__file__).parent.parent / ".data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    run_dataset_through_model(hooked_mlp, test_loader, hooks)

    for hook_point in hook_points:
        gram_matrix = hooked_mlp.hooked_data[hook_point]["gram"]
        print(f"Gram matrix for {hook_point}:")
        print(f"Shape: {gram_matrix.shape}")


if __name__ == "__main__":
    model_dir = Path("/mnt/ssd-apollo/checkpoints/MNIST/lr-0.001_bs-64_2023-07-03_15-26-43")
    model_path = model_dir / "model_epoch_12.pt"
    with open(model_dir / "config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)

    hook_points = ["layers.act_0", "layers.linear_1"]

    main(config_dict, model_path, hook_points=hook_points)
