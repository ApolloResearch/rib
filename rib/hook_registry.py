from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor


def update_gram_data(
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: str,
    gram_matrix: Float[Tensor, "d_hidden d_hidden"],
) -> None:
    """Update the hooked data with the gram matrix.

    We add the gram matrix to previously stored data for this hook point.
    This is equivalent to taking the gram matrix of activations concatenated over batches.

    Note that the gram matrix will be stored on the same device as the output.

    Args:
        hooked_data: Dictionary of hook data that will be updated.
        hook_name: Name of hook. Used as a first-level key in `hooked_data`.
        data_key: Name of the hook, used as a second-level key in `hooked_data`.
        gram_matrix: Gram matrix to add to the hooked data.
    """
    # If no data exists, initialize with zeros
    hooked_data.setdefault(hook_name, {}).setdefault(data_key, torch.zeros_like(gram_matrix))
    hooked_data[hook_name][data_key] += gram_matrix


@torch.inference_mode()
def gram_forward_hook_fn(
    module: torch.nn.Module,
    inputs: tuple[Float[Tensor, "batch d_hidden"]],
    output: Float[Tensor, "batch d_hidden"],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: str,
    **_: Any,
) -> None:
    """Hook function for calculating and updating the gram matrix.

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module (not used).
        output: output of the module.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook.
        data_key: Name of the hook.
        **_: Additional keyword arguments (not used).
    """
    gram_matrix = output.T @ output

    update_gram_data(hooked_data, hook_name, data_key, gram_matrix)


def gram_pre_forward_hook_fn(
    module: torch.nn.Module,
    inputs: tuple[Float[Tensor, "batch d_hidden"]],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: str,
    **_: Any,
) -> None:
    """Hook function for calculating gram matrix and updating the gram matrix.

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook.
        data_key: Name of the hook.
        **_: Additional keyword arguments (not used).
    """
    gram_matrix = inputs[0].T @ inputs[0]

    update_gram_data(hooked_data, hook_name, data_key, gram_matrix)


def rotate_forward_hook_fn(
    module: torch.nn.Module,
    inputs: tuple[Float[Tensor, "batch d_hidden"]],
    output: Float[Tensor, "batch d_hidden"],
    rotation_matrix: Float[Tensor, "d_hidden d_hidden"],
    **_: Any,
) -> Float[Tensor, "batch d_hidden"]:
    """Hook function for rotating activations.

    The output activations are rotated by the specified rotation matrix.

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module (not used).
        output: output of the module.
        rotation_matrix: Rotation matrix to apply to the activations.
        **_: Additional keyword arguments (not used).

    Returns:
        Rotated activations.
    """
    return output @ rotation_matrix.float()


def rotate_pre_forward_hook_fn(
    module: torch.nn.Module,
    inputs: tuple[Float[Tensor, "batch d_hidden"]],
    rotation_matrix: Float[Tensor, "d_hidden d_hidden"],
    **_: Any,
) -> tuple[Float[Tensor, "batch d_hidden"]]:
    """Hook function for rotating the input tensor to a module.

    The input is rotated by the specified rotation matrix.

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module (not used).
        rotation_matrix: Rotation matrix to apply to the activations.
        **_: Additional keyword arguments (not used).

    Returns:
        Rotated activations.
    """
    return (inputs[0] @ rotation_matrix.float(),)


HOOK_REGISTRY = {
    "gram_forward_hook_fn": (gram_forward_hook_fn, "forward"),
    "gram_pre_forward_hook_fn": (gram_pre_forward_hook_fn, "pre_forward"),
    "rotate_forward_hook_fn": (rotate_forward_hook_fn, "forward"),
    "rotate_pre_forward_hook_fn": (rotate_pre_forward_hook_fn, "pre_forward"),
}
