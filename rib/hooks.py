from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Optional

import torch
from jaxtyping import Float
from torch import Tensor

from rib.models.utils import get_model_attr


@dataclass
class Hook:
    """Defines a hook object that can be added to a model.

    Attributes:
        name: Name of the hook. This is used as the key in the hooked_data dict in HookedModel.
        fn: Function to run at the hook point.
        hook_point: String representing the attribute of the model to add the hook to.
            Nested attributes are split by periods (e.g. "layers.linear_0").
        kwargs: Additional keyword arguments to pass to the hook function.
    """

    name: str
    fn: Callable
    hook_point: str
    hook_kwargs: dict[str, Any] = field(default_factory=dict)


class HookedModel(torch.nn.Module):
    """A wrapper around a PyTorch model that allows hooks to be added and removed.

    TODO: Handle backward hooks.

    Example:
        >>> model = torch.nn.Sequential()
        >>> model.add_module("linear_0", torch.nn.Linear(3, 2))
        >>> hooked_model = HookedModel(model)
        >>> hook = Hook(name="gram", fn=gram_matrix_hook_fn, hook_point="linear_0")
        >>> hooked_model(torch.randn(6, 3), hooks=[hook])
        >>> hooked_model.hooked_data["linear_0"]["gram"]
        tensor([[ 1.2023, -0.0311],
                [-0.0311,  0.9988]])
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self.hooked_data: dict[str, Any] = {}

    def __call__(self, *args, hooks: Optional[list[Hook]] = None, **kwargs) -> Any:
        return self.forward(*args, hooks=hooks, **kwargs)

    def forward(self, *args, hooks: Optional[list[Hook]] = None, **kwargs) -> Any:
        """Run the forward pass of the model and remove all hooks."""
        if hooks is not None:
            self.add_forward_hooks(hooks)
        try:
            output = self.model(*args, **kwargs)
        finally:
            self.remove_hooks()
        return output

    def add_forward_hooks(self, hooks: list[Hook]) -> None:
        """Add a hook to the model at each of the specified hook points."""
        for hook in hooks:
            hook_module = get_model_attr(self.model, hook.hook_point)
            hook_fn_partial = partial(
                hook.fn,
                hooked_data=self.hooked_data,
                hook_point=hook.hook_point,
                hook_name=hook.name,
                **hook.hook_kwargs,
            )
            handle = hook_module.register_forward_hook(hook_fn_partial)
            self.hook_handles.append(handle)

    def remove_hooks(self) -> None:
        """Remove all hooks from the model."""
        for handle in self.hook_handles:
            handle.remove()


@torch.inference_mode()
def gram_matrix_hook_fn(
    module: torch.nn.Module,
    inputs: Float[Tensor, "batch d_hidden"],
    outputs: Float[Tensor, "batch d_hidden"],
    hooked_data: dict[str, Any],
    hook_point: str,
    hook_name: str,
    **_: Any,
) -> None:
    """Hook function for calculating gram matrix.

    We add the gram matrix to previously stored data for this hook point.
    This is equivalent to taking the gram matrix of activations concatenated over batches.

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module (not used).
        outputs: Outputs of the module.
        hooked_data: Dictionary of hook data.
        hook_point: Model attribute that the hook is attached to. Used as a first-level key in
            `hooked_data`.
        hook_name: Name of the hook, used as a second-level key in `hooked_data`.
        **_: Additional keyword arguments (not used).
    """
    gram_matrix = outputs.T @ outputs

    # If no data exists, initialize with zeros
    hooked_data.setdefault(hook_point, {}).setdefault(hook_name, torch.zeros_like(gram_matrix))
    # Add gram matrix to data
    hooked_data[hook_point][hook_name] += gram_matrix


def rotate_and_ablate_hook_fn(
    module: torch.nn.Module,
    inputs: Float[Tensor, "batch d_hidden"],
    outputs: Float[Tensor, "batch d_hidden"],
    rotation_matrix: Float[Tensor, "d_hidden d_hidden"],
    **_: Any,
) -> Float[Tensor, "batch d_hidden"]:
    """Hook function for rotating activations.

    The output activations are rotated by the specified rotation matrix.

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module (not used).
        outputs: Outputs of the module.
        rotation_matrix: Rotation matrix to apply to the activations.
        **_: Additional keyword arguments (not used).

    Returns:
        Rotated activations.
    """
    return outputs @ rotation_matrix.T.float()
