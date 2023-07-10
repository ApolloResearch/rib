from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor


@torch.inference_mode()
def gram_forward_hook_fn(
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

    Note that the gram matrix will be stored on the same device as the outputs.

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


def rotate_forward_hook_fn(
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
    return outputs @ rotation_matrix.float()


HOOK_REGISTRY = {
    "gram_forward_hook_fn": (gram_forward_hook_fn, "forward"),
    "rotate_forward_hook_fn": (rotate_forward_hook_fn, "forward"),
}
