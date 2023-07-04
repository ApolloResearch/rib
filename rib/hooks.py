from functools import partial
from typing import Any, Callable, Dict, List

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader

from rib.utils import get_model_attr


class HookedModel:
    """A wrapper around a PyTorch model that allows hooks to be added and removed."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.hook_data: Dict[str, Any] = {}

    def __call__(self, *args, **kwargs) -> Any:
        self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Any:
        """Run the forward pass of the model and remove all hooks."""
        output = self.model(*args, **kwargs)
        self.remove_hooks()
        return output

    def add_forward_hooks(self, hook_points: List[str], hook_fn: Callable) -> None:
        """Add a hook to the model at each of the specified hook points."""
        for hook_point in hook_points:
            hook_module = get_model_attr(self.model, hook_point)
            hook_fn_partial = partial(hook_fn, hook_data=self.hook_data, hook_point=hook_point)
            handle = hook_module.register_forward_hook(hook_fn_partial)
            self.hooks.append(handle)

    def remove_hooks(self) -> None:
        """Remove all hooks from the model."""
        for handle in self.hooks:
            handle.remove()


def gram_matrix_hook_fn(
    module: torch.nn.Module,
    inputs: Float[Tensor, "batch d_hidden"],
    outputs: Float[Tensor, "batch d_hidden"],
    hook_data: Dict[str, Any],
    hook_point: str,
) -> None:
    """Hook function for calculating gram matrix.

    We add the gram matrix to previously stored data for this hook point.
    This is equivalent to taking the gram matrix of activations concatenated over batches.
    """
    gram_matrix = outputs.T @ outputs

    if hook_point in hook_data:
        hook_data[hook_point] += gram_matrix
    else:
        hook_data[hook_point] = gram_matrix


@torch.inference_mode()
def collect_hook_data(hooked_model: HookedModel, dataloader: DataLoader) -> None:
    """Run a dataset through a model.

    Assumes that hooks have been added to the model.
    """
    assert hooked_model.hooks, "No hooks have been added to the model."
    for batch in dataloader:
        data, _ = batch
        with torch.no_grad():
            _ = hooked_model(data)
