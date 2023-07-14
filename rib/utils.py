from pathlib import Path
from typing import Type, TypeVar

import torch
import yaml
from jaxtyping import Float
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import DataLoader

from rib.hook_manager import Hook, HookedModel

T = TypeVar("T", bound=BaseModel)

REPO_ROOT = Path(__file__).parent.parent


def run_dataset_through_model(
    hooked_model: HookedModel, dataloader: DataLoader, hooks: list[Hook], device: str = "cuda"
) -> None:
    """Simply pass all batches through a hooked model."""
    assert len(hooks) > 0, "Hooks have not been applied to this model."
    for batch in dataloader:
        data, _ = batch
        data = data.to(device)
        hooked_model(data, hooks=hooks)


@torch.inference_mode()
def eval_model_accuracy(
    hooked_model: HookedModel, dataloader: DataLoader, hooks: list[Hook], device: str = "cuda"
) -> float:
    """Run the model on the dataset and return the accuracy.

    Args:
        hooked_model: The model to evaluate.
        dataloader: The dataloader for the dataset.
        hooks: The hooks to use.
        device: The device to run the model on.

    Returns:
        The accuracy of the model on the dataset.
    """

    correct_predictions: int = 0
    total_predictions: int = 0

    for batch in dataloader:
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        output: Float[Tensor, "batch d_vocab"] = hooked_model(data, hooks=hooks)

        # Assuming output is raw logits and labels are class indices.
        predicted_labels: Float[Tensor, "batch"] = output.argmax(dim=1)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_predictions += labels.shape[0]

    accuracy: float = correct_predictions / total_predictions
    return accuracy


def load_config(config_path: Path, config_model: Type[T]) -> T:
    """Load the config from a YAML file into a Pydantic model."""
    assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = config_model(**config_dict)
    return config
