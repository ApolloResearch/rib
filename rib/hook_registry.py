from typing import Any, Callable, Union

import torch
from jaxtyping import Float
from torch import Tensor

from rib.linalg import batched_jacobian, calc_interaction_matrix


def add_to_hooked_matrix(
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: str,
    hooked_matrix: Float[Tensor, "d_hidden d_hidden"],
) -> None:
    """Update the hooked data matrix with the given matrix.

    We add the hooked matrix to previously stored data matrix for this hook point.

    Note that the data matrix will be stored on the same device as the output.

    Args:
        hooked_data: Dictionary of hook data that will be updated.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
        hooked_matrix: Matrix to add to the hooked data.
    """
    # If no data exists, initialize with zeros
    hooked_data.setdefault(hook_name, {}).setdefault(data_key, torch.zeros_like(hooked_matrix))
    hooked_data[hook_name][data_key] += hooked_matrix


def gram_forward_hook_fn(
    module: torch.nn.Module,
    inputs: tuple[Float[Tensor, "batch d_hidden"]],
    output: Float[Tensor, "batch d_hidden"],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    **_: Any,
) -> None:
    """Hook function for calculating and updating the gram matrix.

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module (not used).
        output: output of the module.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
        **_: Additional keyword arguments (not used).
    """
    assert isinstance(data_key, str), "data_key must be a string."
    out_copy = output.detach().clone()
    gram_matrix = out_copy.T @ out_copy
    add_to_hooked_matrix(hooked_data, hook_name, data_key, gram_matrix)


def gram_pre_forward_hook_fn(
    module: torch.nn.Module,
    inputs: tuple[Float[Tensor, "batch d_hidden"]],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    **_: Any,
) -> None:
    """Calculates the gram matrix for the batch and adds it to the global.

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
        **_: Additional keyword arguments (not used).
    """
    assert isinstance(data_key, str), "data_key must be a string."
    in_copy = inputs[0].detach().clone()
    gram_matrix = in_copy.T @ in_copy
    add_to_hooked_matrix(hooked_data, hook_name, data_key, gram_matrix)


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
    out_copy = output.detach().clone()
    return out_copy @ rotation_matrix


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
    in_copy = inputs[0].detach().clone()
    return (in_copy @ rotation_matrix,)


def M_dash_and_Lambda_dash_forward_hook_fn(
    module: torch.nn.Module,
    inputs: tuple[Float[Tensor, "batch in_hidden"]],
    output: Float[Tensor, "batch out_hidden"],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    next_layer_C: Float[Tensor, "out_hidden out_hidden"],
    **_: Any,
) -> None:
    """Hook function for accumulating the M' and Lambda' matrices.

    For calculating the Jacobian, we need to run the inputs through the module. Unfortunately,
    this causes an infinite recursion because the module has a hook which runs this function. To
    avoid this, we (hackily) remove the hook before running the module and then add it back after.

    Args:
        module: Module that the hook is attached to.
        inputs: Inputs to the module.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of 2nd-level keys to store in `hooked_data`.
        next_layer_C: The C matrix for the next layer (C^{l+1} in the paper).
        **_: Additional keyword arguments (not used).
    """
    assert isinstance(data_key, list), "data_key must be a list of strings."
    assert len(data_key) == 2, "data_key must be a list of length 2 to store M' and Lambda'."
    # Remove the foward hook to avoid recursion when calculating the jacobian
    module._forward_hooks.popitem()
    assert not module._forward_hooks, "Module has multiple forward hooks"

    in_acts = inputs[0].detach().clone()
    out_acts = output.detach().clone()
    O: Float[Tensor, "batch out_hidden in_hidden"] = batched_jacobian(module, in_acts)

    with torch.inference_mode():
        # This corresponds to the left half of the inner products in the M' and Lambda' equations
        # In latex: $\sum_i \hat{f}^{l+1}(X) {C^{l+1}}^T O^l$
        f_hat_next_layer_C_O: Float[Tensor, "batch in_hidden"] = torch.einsum(
            "bi,Ii,bIj->bj", out_acts, next_layer_C, O
        )
        M_dash: Float[Tensor, "in_hidden in_hidden"] = f_hat_next_layer_C_O.T @ f_hat_next_layer_C_O
        Lambda_dash: Float[Tensor, "in_hidden in_hidden"] = f_hat_next_layer_C_O.T @ in_acts

        add_to_hooked_matrix(hooked_data, hook_name, data_key[0], M_dash)
        add_to_hooked_matrix(hooked_data, hook_name, data_key[1], Lambda_dash)


HookRegistryType = dict[str, tuple[Callable[..., Any], str]]

HOOK_REGISTRY: HookRegistryType = {
    "gram_forward_hook_fn": (gram_forward_hook_fn, "forward"),
    "gram_pre_forward_hook_fn": (gram_pre_forward_hook_fn, "pre_forward"),
    "rotate_forward_hook_fn": (rotate_forward_hook_fn, "forward"),
    "rotate_pre_forward_hook_fn": (rotate_pre_forward_hook_fn, "pre_forward"),
    "M_dash_and_Lambda_dash_forward_hook_fn": (M_dash_and_Lambda_dash_forward_hook_fn, "forward"),
}
