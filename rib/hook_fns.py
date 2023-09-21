from functools import partial
from typing import Any, Union

import torch
from jaxtyping import Float
from torch import Tensor
from torch.func import jacrev, vmap

from rib.linalg import edge_norm


def _add_to_hooked_matrix(
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
    inputs: Union[
        tuple[Float[Tensor, "batch d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden1"], Float[Tensor, "batch pos d_hidden2"]],
    ],
    output: Union[
        Float[Tensor, "batch d_hidden"],
        Float[Tensor, "batch pos d_hidden"],
        tuple[Float[Tensor, "batch pos d_hidden1"], Float[Tensor, "batch pos d_hidden2"]],
    ],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    **_: Any,
) -> None:
    """Hook function for calculating and updating the gram matrix.

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module (not used).
        output: Output of the module. Handles modules with one or two outputs of varying d_hiddens
            and positional indices. If no positional indices, assumes one output.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
        **_: Additional keyword arguments (not used).
    """
    assert isinstance(data_key, str), "data_key must be a string."
    # Output may be tuple of tensors if there are two outputs
    outputs = output if isinstance(output, tuple) else (output,)

    # Concat over the hidden dimension
    out_acts = torch.cat([x.detach().clone() for x in outputs], dim=-1)

    if out_acts.dim() == 3:  # tensor with pos dimension
        einsum_pattern = "bpi, bpj -> ij"
    elif out_acts.dim() == 2:  # tensor without pos dimension
        einsum_pattern = "bi, bj -> ij"
    else:
        raise ValueError("Unexpected tensor rank")

    gram_matrix = torch.einsum(einsum_pattern, out_acts, out_acts)
    _add_to_hooked_matrix(hooked_data, hook_name, data_key, gram_matrix)


def gram_pre_forward_hook_fn(
    module: torch.nn.Module,
    inputs: Union[
        tuple[Float[Tensor, "batch d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden1"], Float[Tensor, "batch pos d_hidden2"]],
    ],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    **_: Any,
) -> None:
    """Calculate the gram matrix for inputs with positional indices and add it to the global.

    First, we concatenate all inputs along the d_hidden dimension. Our gram matrix is then
    calculated by summing over the batch and position dimension (if there is a pos dimension).

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module. Handles modules with one or two inputs of varying d_hiddens
            and positional indices. If no positional indices, assumes one input.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
        **_: Additional keyword arguments (not used).
    """
    assert isinstance(data_key, str), "data_key must be a string."

    # Concat over the hidden dimension
    in_acts = torch.cat([x.detach().clone() for x in inputs], dim=-1)

    if in_acts.dim() == 3:  # tensor with pos dimension
        einsum_pattern = "bpi, bpj -> ij"
    elif in_acts.dim() == 2:  # tensor without pos dimension
        einsum_pattern = "bi, bj -> ij"
    else:
        raise ValueError("Unexpected tensor rank")

    gram_matrix = torch.einsum(einsum_pattern, in_acts, in_acts)

    _add_to_hooked_matrix(hooked_data, hook_name, data_key, gram_matrix)


def rotate_orthog_pre_forward_hook_fn(
    module: torch.nn.Module,
    inputs: tuple[Float[Tensor, "batch d_hidden"]],
    rotation_matrix: Float[Tensor, "d_hidden d_hidden"],
    **_: Any,
) -> tuple[Float[Tensor, "batch d_hidden"]]:
    """Hook function for rotating the input tensor to a module.

    The input is rotated by the specified rotation matrix.

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module that we rotate.
        rotation_matrix: Rotation matrix to apply to the activations.
        **_: Additional keyword arguments (not used).

    Returns:
        Rotated activations.
    """
    in_acts = inputs[0].detach().clone()
    return (in_acts @ rotation_matrix,)


def M_dash_and_Lambda_dash_pre_forward_hook_fn(
    module: torch.nn.Module,
    inputs: Union[
        tuple[Float[Tensor, "batch in_hidden"]],
        tuple[Float[Tensor, "batch pos in_hidden"], ...],
    ],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    C_out: Float[Tensor, "out_hidden_combined out_hidden_combined_trunc"],
    **_: Any,
) -> None:
    """Hook function for accumulating the M' and Lambda' matrices.

    Args:
        module: Module that the hook is attached to.
        inputs: Inputs to the module. Handles modules with one or two inputs of varying d_hiddens
            and positional indices. If no positional indices, assumes one input.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of 2nd-level keys to store in `hooked_data`.
        C_out: The C matrix for the next layer (C^{l+1} in the paper).
        **_: Additional keyword arguments (not used).
    """
    from rib.models.sequential_transformer.components import MLPIn

    assert isinstance(data_key, list), "data_key must be a list of strings."
    assert len(data_key) == 2, "data_key must be a list of length 2 to store M' and Lambda'."
    # Remove the pre foward hook to avoid recursion when calculating the jacobian
    module._forward_pre_hooks.popitem()
    assert not module._forward_hooks, "Module has multiple forward hooks"

    # Ensure that the inputs have requires_grad=True
    for x in inputs:
        x.requires_grad_(True)

    integral_step = 1  # TODO: Make this configurable or use numerical integration package
    for alpha in torch.arange(integral_step, 1 + integral_step, integral_step):
        alpha_inputs = tuple(alpha * x for x in inputs)
        output = module(*alpha_inputs)
        outputs = (output,) if isinstance(output, torch.Tensor) else output

        # Concatenate the outputs over the hidden dimension
        out_acts = torch.cat(outputs, dim=-1)

        f_hat: Union[
            Float[Tensor, "batch out_hidden_combined_trunc"],
            Float[Tensor, "batch pos out_hidden_combined_trunc"],
        ] = (
            out_acts @ C_out
        )
        f_hat_norm: Float[Tensor, ""] = (f_hat**2).sum()
        if isinstance(module[0], MLPIn):  # type: ignore
            print("INSIDE seciton 2")
        # Accumulate the grad of f_hat_norm w.r.t the input tensors (ignoring all other gradients)
        f_hat_norm.backward(inputs=inputs, retain_graph=True)

    has_pos = inputs[0].dim() == 3
    if has_pos:
        einsum_pattern = "bpj,bpJ->jJ"
    else:
        einsum_pattern = "bj,bJ->jJ"

    with torch.inference_mode():
        in_grads_list: list[Tensor] = []
        for x in inputs:
            assert x.grad is not None, "Input tensor does not have a gradient."
            in_grads_list.append(x.grad)
        in_grads: Union[
            Float[Tensor, "batch in_hidden_combined"],
            Float[Tensor, "batch pos in_hidden_combined"],
        ] = torch.cat(in_grads_list, dim=-1)

        M_dash = torch.einsum(einsum_pattern, in_grads, in_grads)
        # Concatenate the inputs over the hidden dimension
        in_acts = torch.cat(inputs, dim=-1)
        Lambda_dash = torch.einsum(einsum_pattern, in_grads, in_acts)

        _add_to_hooked_matrix(hooked_data, hook_name, data_key[0], M_dash)
        _add_to_hooked_matrix(hooked_data, hook_name, data_key[1], Lambda_dash)


def interaction_edge_pre_forward_hook_fn(
    module: torch.nn.Module,
    inputs: Union[
        tuple[Float[Tensor, "batch in_hidden"]],
        tuple[Float[Tensor, "batch pos _"], ...],
    ],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    C_in: Float[Tensor, "in_hidden in_hidden_trunc"],
    C_in_pinv: Float[Tensor, "in_hidden_trunc in_hidden"],
    C_out: Float[Tensor, "out_hidden out_hidden_trunc"],
    **_: Any,
) -> None:
    """Hook function for accumulating the edges (denoted E_hat) of the interaction graph.

    For calculating the Jacobian, we need to run the inputs through the module. Unfortunately,
    this causes an infinite recursion because the module has a hook which runs this function. To
    avoid this, we (hackily) remove the hook before running the module and then add it back after.

    Args:
        module: Module that the hook is attached to.
        inputs: Inputs to the module. Handles modules with one or two inputs of varying d_hiddens
            and positional indices. If no positional indices, assumes one input.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of 2nd-level keys to store in `hooked_data`.
        C_in: The C matrix for the current layer (C^l in the paper).
        C_in_pinv: The pseudoinverse of the C matrix for the current layer ((C^l)^+ in the paper).
        C_out: The C matrix for the next layer (C^{l+1} in the paper).
        **_: Additional keyword arguments (not used).
    """
    assert isinstance(data_key, str), "data_key must be a string."

    # Remove the pre-foward hook to avoid recursion when calculating the jacobian
    module._forward_pre_hooks.popitem()
    assert not module._forward_hooks, "Module has multiple forward hooks"

    has_pos = inputs[0].dim() == 3
    if has_pos:
        # jac_size is (batch, out_hidden_trunc, pos, in_hidden_trunc)
        jac_size = [inputs[0].shape[0], C_out.shape[1], inputs[0].shape[1], C_in.shape[1]]
    else:
        # jac_size is (batch, out_hidden_trunc, in_hidden_trunc)
        jac_size = [inputs[0].shape[0], C_out.shape[1], C_in.shape[1]]

    jac_out: Union[
        Float[Tensor, "batch out_hidden_trunc in_hidden_trunc"],
        Float[Tensor, "batch out_hidden_trunc pos in_hidden_trunc"],
    ] = torch.zeros(size=jac_size, device=inputs[0].device, dtype=inputs[0].dtype)
    # We first concatenate the inputs over the hidden dimension
    in_acts = torch.cat(inputs, dim=-1)
    # For each integral step, we calculate derivatives w.r.t alpha * in_acts @ C_in
    f_hat = in_acts @ C_in

    integral_step = 1  # TODO: Make this configurable or use numerical integration package
    for alpha in torch.arange(integral_step, 1 + integral_step, integral_step):
        alpha_input = alpha * f_hat
        edge_norm_partial = partial(
            edge_norm,
            module=module,
            C_in_pinv=C_in_pinv,
            C_out=C_out,
            in_hidden_dims=[x.shape[-1] for x in inputs],
            has_pos=has_pos,
        )
        alpha_jac_out = vmap(jacrev(edge_norm_partial))(alpha_input)

        jac_out += alpha_jac_out

    has_pos = inputs[0].dim() == 3
    if has_pos:
        einsum_pattern = "bipj,bpj->ij"
    else:
        einsum_pattern = "bij,bj->ij"

    with torch.inference_mode():
        E = torch.einsum(einsum_pattern, jac_out, f_hat)

        _add_to_hooked_matrix(hooked_data, hook_name, data_key, E)


def acts_forward_hook_fn(
    module: torch.nn.Module,
    inputs: Union[
        tuple[Float[Tensor, "batch d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden1"], Float[Tensor, "batch pos d_hidden2"]],
    ],
    output: Union[
        Float[Tensor, "batch d_hidden"],
        Float[Tensor, "batch pos d_hidden"],
        tuple[Float[Tensor, "batch pos d_hidden1"], Float[Tensor, "batch pos d_hidden2"]],
    ],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    **_: Any,
) -> None:
    """Hook function for storing the output activations.

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module (not used).
        output: Output of the module. Handles modules with one or two outputs of varying d_hiddens
            and positional indices. If no positional indices, assumes one output.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
        **_: Additional keyword arguments (not used).
    """
    assert isinstance(data_key, str), "data_key must be a string."
    outputs = output if isinstance(output, tuple) else (output,)
    detached_outputs = [x.detach().cpu() for x in outputs]
    # Store the output activations
    hooked_data[hook_name] = {data_key: detached_outputs}
