from typing import Any, Union

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from rib.linalg import batched_jacobian


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


def _concatenate_with_embedding_reshape(
    inputs: Union[
        tuple[Float[Tensor, "batch d_hidden"]],
        tuple[Float[Tensor, "batch pos _"], ...],
    ],
) -> Float[Tensor, "batch d_hidden_combined"]:
    """
    Fold in position dim to hidden dim and concatenate the inputs over the hidden dimension.

    For tensors with a rank of 3 (assumed to have positional embeddings), the positional and hidden
    dimensions are combined. For tensors with a rank of 2, they remain unchanged.

    Args:
        inputs: A tuple containing one or two tensors to be concatenated. If the tensors contain a
            position dimensions (i.e. rank of 3), the positional and hidden dimensions are combined.

    Returns:
        The concatenated tensors with the positional and hidden dimensions combined.

    Raises:
        ValueError: If a tensor rank is neither 2 nor 3.
    """
    combined_tensors = []

    for x in inputs:
        if x.dim() == 3:  # tensor with pos embedding
            pattern = "batch pos d_hidden -> batch (pos d_hidden)"
        elif x.dim() == 2:  # tensor without pos embedding
            pattern = "batch d_hidden -> batch d_hidden"
        else:
            raise ValueError("Unexpected tensor rank")

        combined_tensors.append(rearrange(x.detach().clone(), pattern))

    return torch.cat(combined_tensors, dim=-1)


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

    out_acts = _concatenate_with_embedding_reshape(outputs)  # type: ignore

    gram_matrix = out_acts.T @ out_acts
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

    First, we combine the pos and hidden dimensions into a single dimension. Then, if there are two
    inputs, we concatenate them along this combined dimension. We then calculate the gram matrix.

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

    in_acts = _concatenate_with_embedding_reshape(inputs)

    gram_matrix = in_acts.T @ in_acts
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


def M_dash_and_Lambda_dash_forward_hook_fn(
    module: torch.nn.Module,
    inputs: Union[
        tuple[Float[Tensor, "batch in_hidden"]],
        tuple[Float[Tensor, "batch pos _"], ...],
    ],
    output: Union[
        Float[Tensor, "batch out_hidden"],
        tuple[Float[Tensor, "batch pos _"], ...],
    ],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    C_out: Float[Tensor, "out_hidden_combined out_hidden_combined_trunc"],
    **_: Any,
) -> None:
    """Hook function for accumulating the M' and Lambda' matrices.

    For calculating the Jacobian, we need to run the inputs through the module. Unfortunately,
    this causes an infinite recursion because the module has a hook which runs this function. To
    avoid this, we (hackily) remove the hook before running the module and then add it back after.

    Args:
        module: Module that the hook is attached to.
        inputs: Inputs to the module. Handles modules with one or two inputs of varying d_hiddens
            and positional indices. If no positional indices, assumes one input.
        output: Output of the module. Handles modules with one or two outputs of varying d_hiddens
            and positional indices. If no positional indices, assumes one output.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of 2nd-level keys to store in `hooked_data`.
        C_out: The C matrix for the next layer (C^{l+1} in the paper).
        **_: Additional keyword arguments (not used).
    """
    assert isinstance(data_key, list), "data_key must be a list of strings."
    assert len(data_key) == 2, "data_key must be a list of length 2 to store M' and Lambda'."
    # Remove the foward hook to avoid recursion when calculating the jacobian
    module._forward_hooks.popitem()
    assert not module._forward_hooks, "Module has multiple forward hooks"

    out_tuple_len = len(output) if isinstance(output, tuple) else 0

    O: Float[Tensor, "batch out_hidden_combined in_hidden_combined"] = batched_jacobian(
        module, inputs, out_tuple_len=out_tuple_len
    )

    in_acts = _concatenate_with_embedding_reshape(inputs)

    # outputs = output if isinstance(output, tuple) else (output,)
    outputs = (output,) if isinstance(output, torch.Tensor) else output
    out_acts = _concatenate_with_embedding_reshape(outputs)

    with torch.inference_mode():
        # This corresponds to the left half of the inner products in the M' and Lambda' equations
        # In latex: $\sum_i \hat{f}^{l+1}(X) {C^{l+1}}^T O^l$
        f_hat_C_out_O: Float[Tensor, "batch in_hidden_combined"] = torch.einsum(
            "bi,iI,Ik,bkj->bj", out_acts, C_out, C_out.T, O
        )
        M_dash: Float[Tensor, "in_hidden_combined in_hidden_combined"] = (
            f_hat_C_out_O.T @ f_hat_C_out_O
        )
        Lambda_dash: Float[Tensor, "in_hidden_combined in_hidden_combined"] = (
            f_hat_C_out_O.T @ in_acts
        )

        _add_to_hooked_matrix(hooked_data, hook_name, data_key[0], M_dash)
        _add_to_hooked_matrix(hooked_data, hook_name, data_key[1], Lambda_dash)


def interaction_edge_forward_hook_fn(
    module: torch.nn.Module,
    inputs: tuple[Float[Tensor, "batch in_hidden"]],
    output: Float[Tensor, "batch out_hidden"],
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
        inputs: Inputs to the module.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of 2nd-level keys to store in `hooked_data`.
        C_in: The C matrix for the current layer (C^l in the paper).
        C_in_pinv: The pseudoinverse of the C matrix for the current layer ((C^l)^+ in the paper).
        C_out: The C matrix for the next layer (C^{l+1} in the paper).
        **_: Additional keyword arguments (not used).
    """
    assert isinstance(data_key, str), "data_key must be a string."

    # Remove the foward hook to avoid recursion when calculating the jacobian
    module._forward_hooks.popitem()
    assert not module._forward_hooks, "Module has multiple forward hooks"

    O: Float[Tensor, "batch out_hidden in_hidden"] = batched_jacobian(module, inputs)

    in_acts: Float[Tensor, "batch in_hidden"] = inputs[0].detach().clone()
    out_acts: Float[Tensor, "batch out_hidden"] = output.detach().clone()

    with torch.inference_mode():
        # LHS of Hadamard product
        f_hat_out_T_f_hat_in: Float[Tensor, "out_hidden_trunc in_hidden_trunc"] = torch.einsum(
            "bi,ik,bj,jm->bkm", out_acts, C_out, in_acts, C_in
        )
        # RHS of Hadamard product
        C_out_O_C_in_pinv_T: Float[Tensor, "out_hidden_trunc in_hidden_trunc"] = torch.einsum(
            "ik,bij,jm->bkm", C_out, O, C_in_pinv.T
        )
        E = (f_hat_out_T_f_hat_in * C_out_O_C_in_pinv_T).sum(dim=0)

        _add_to_hooked_matrix(hooked_data, hook_name, data_key, E)
