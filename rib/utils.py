import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader

from rib.hooks import Hook, HookedModel


@torch.inference_mode()
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
def calc_model_accuracy(
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
