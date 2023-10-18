"""Defines hook functions that are used in a HookedModel.

All forward hook functions must contain "forward" in their function names, and all pre-forward
hook functions must contain "pre_forward" in their function names. This is done to ensure that
the correct type of hook is registered to the module.

By default, a HookedModel passes in the arguments `hooked_data`, `hook_name`, and `data_key` to
each hook function. Therefore, these arguments must be included in the signature of each hook.

Otherwise, the hook function operates like a regular pytorch hook function.
"""
from functools import partial
from typing import Any, Optional, Union

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from rib.linalg import (
    edge_norm,
    integrated_gradient_trapezoidal_jacobian,
    integrated_gradient_trapezoidal_norm,
)


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
    hooked_data.setdefault(hook_name, {}).setdefault(
        data_key, torch.zeros_like(hooked_matrix))
    hooked_data[hook_name][data_key] += hooked_matrix


def gram_forward_hook_fn(
    module: torch.nn.Module,
    inputs: Union[
        tuple[Float[Tensor, "batch d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden1"],
              Float[Tensor, "batch pos d_hidden2"]],
    ],
    output: Union[
        Float[Tensor, "batch d_hidden"],
        Float[Tensor, "batch pos d_hidden"],
        tuple[Float[Tensor, "batch pos d_hidden1"],
              Float[Tensor, "batch pos d_hidden2"]],
    ],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
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
        tuple[Float[Tensor, "batch pos d_hidden1"],
              Float[Tensor, "batch pos d_hidden2"]],
    ],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
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


def rotate_pre_forward_hook_fn(
    module: torch.nn.Module,
    inputs: tuple[Float[Tensor, "batch d_hidden"]],
    rotation_matrix: Float[Tensor, "d_hidden d_hidden"],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
) -> tuple[Float[Tensor, "batch d_hidden"], ...]:
    """Hook function for rotating the input tensor to a module. Edits the forward pass but does not save.

    The input is rotated by the specified rotation matrix.

    Handles multiple inputs by concatenating over the hidden dimension and then splitting the
    rotated tensor back into the original input sizes.

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module that we rotate.
        rotation_matrix: Rotation matrix to apply to the activations.
        hooked_data: Dictionary of hook data (not used).
        hook_name: Name of hook (not used).
        data_key: Name of data (not used).

    Returns:
        Rotated activations.
    """
    # Concatenate over the hidden dimension
    in_hidden_dims = [x.shape[-1] for x in inputs]
    in_acts = torch.cat(inputs, dim=-1)
    rotated = in_acts @ rotation_matrix
    adjusted_inputs = tuple(torch.split(rotated, in_hidden_dims, dim=-1))
    return adjusted_inputs


def M_dash_and_Lambda_dash_pre_forward_hook_fn(
    module: torch.nn.Module,
    inputs: Union[
        tuple[Float[Tensor, "batch in_hidden"]],
        tuple[Float[Tensor, "batch pos in_hidden"], ...],
    ],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    C_out: Optional[Float[Tensor, "out_hidden_combined out_hidden_combined_trunc"]],
    n_intervals: int,
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
        n_intervals: Number of intervals to use for the trapezoidal rule. If 0, this is equivalent
            to taking a point estimate at alpha == 1.
    """
    assert isinstance(data_key, list), "data_key must be a list of strings."
    assert len(
        data_key) == 2, "data_key must be a list of length 2 to store M' and Lambda'."
    # Remove the pre foward hook to avoid recursion when calculating the jacobian
    module._forward_pre_hooks.popitem()
    assert not module._forward_hooks, "Module has multiple forward hooks"

    in_grads = integrated_gradient_trapezoidal_norm(
        module=module,
        inputs=inputs,
        C_out=C_out,
        n_intervals=n_intervals,
    )

    has_pos = inputs[0].dim() == 3

    einsum_pattern = "bpj,bpJ->jJ" if has_pos else "bj,bJ->jJ"

    with torch.inference_mode():
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
    C_out: Optional[Float[Tensor, "out_hidden out_hidden_trunc"]],
    n_intervals: int,
) -> None:
    """Hook function for accumulating the edges (denoted E_hat) of the interaction graph.

    For calculating the Jacobian, we need to run the inputs through the module. Unfortunately,
    this causes an infinite recursion because the module has a hook which runs this function. To
    avoid this, we (hackily) remove the hook before running the module and then add it back after.

    The trapezoidal rule is used to approximate the integrated gradient. If n_intervals == 0, the
    integrated gradient effectively takes a point estimate for the integral at alpha == 1.

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
        n_intervals: Number of intervals to use for the trapezoidal rule. If 0, this is equivalent
            to taking a point estimate at alpha == 1.
    """
    assert isinstance(data_key, str), "data_key must be a string."

    # Remove the pre-foward hook to avoid recursion when calculating the jacobian
    module._forward_pre_hooks.popitem()
    assert not module._forward_hooks, "Module has multiple forward hooks"

    has_pos = inputs[0].dim() == 3

    # We first concatenate the inputs over the hidden dimension
    in_acts = torch.cat(inputs, dim=-1)
    # For each integral step, we calculate derivatives w.r.t alpha * in_acts @ C_in
    f_hat = in_acts @ C_in

    # Setup function for calculating the edge norm
    edge_norm_partial = partial(
        edge_norm,
        module=module,
        C_in_pinv=C_in_pinv,
        C_out=C_out,
        in_hidden_dims=[x.shape[-1] for x in inputs],
        has_pos=has_pos,
    )

    jac_out = integrated_gradient_trapezoidal_jacobian(
        fn=edge_norm_partial,
        in_tensor=f_hat,
        n_intervals=n_intervals,
    )
    einsum_pattern = "bipj,bpj->ij" if has_pos else "bij,bj->ij"

    with torch.inference_mode():
        E = torch.einsum(einsum_pattern, jac_out, f_hat)

        _add_to_hooked_matrix(hooked_data, hook_name, data_key, E)


def acts_forward_hook_fn(
    module: torch.nn.Module,
    inputs: Union[
        tuple[Float[Tensor, "batch d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden1"],
              Float[Tensor, "batch pos d_hidden2"]],
    ],
    output: Union[
        Float[Tensor, "batch d_hidden"],
        Float[Tensor, "batch pos d_hidden"],
        tuple[Float[Tensor, "batch pos d_hidden1"],
              Float[Tensor, "batch pos d_hidden2"]],
    ],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
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
    """
    assert isinstance(data_key, str), "data_key must be a string."
    outputs = output if isinstance(output, tuple) else (output,)
    outputs = [x.detach().cpu() for x in outputs]
    # Store the output activations
    hooked_data[hook_name] = {data_key: outputs}


def acts_pre_forward_hook_fn(
    module: torch.nn.Module,
    inputs: Union[
        tuple[Float[Tensor, "batch d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden1"],
              Float[Tensor, "batch pos d_hidden2"]],
    ],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
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
    """
    assert isinstance(data_key, str), "data_key must be a string."
    inputs = torch.cat([x.detach().clone() for x in inputs], dim=-1)
    # Store the output activations
    hooked_data[hook_name] = {data_key: inputs}


@torch.no_grad()
def relu_interaction_forward_hook_fn(
    module: torch.nn.Module,
    inputs: Union[
        tuple[Float[Tensor, "batch d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden1"],
              Float[Tensor, "batch pos d_hidden2"]],
    ],
    output: Union[
        Float[Tensor, "batch d_hidden"],
        Float[Tensor, "batch pos d_hidden"],
        tuple[Float[Tensor, "batch pos d_hidden1"],
              Float[Tensor, "batch pos d_hidden2"]],
    ],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    relu_metric_type: int,
    C_next_layer: Float[Tensor, "d_hidden_out d_hidden_trunc_next_layer_output"],
) -> None:
    """Hook function to store pre and post output activations for ReLU operators.

    TODO (nonurgent): fix 0/0 cases in type 0 (type 1 handles infs and nans with duct tape).

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module.
        output: Output of the module. Handles modules with one or two outputs of varying d_hiddens
            and positional indices. If no positional indices, assumes one output.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
        relu_metric_type: Temporary variable that exists for multiple versions of this function while I play
        around with what works.
            0: ReLU pointwise ratio. Calculate sum_x (O_i(x) == O_j(x)) = B_ij.
            1: Commutator without gradients. Scales the method in 0 by the value of the
            pre-activation itself (applies an importance scaling based on the incoming information).
            sum_x ( [A^(i,j),O(x)] p(x) )^2 = sum_x ( (O_i(x) - O_j(x)) p(x) )^2 =
            sum_x (O_i(x) p_j(x) - f_j(x))^2. This numerator is then scaled by sum_x f_j(x)^2 (the
            size of the function itself). We separately accumulate the numerator and denominator
            over the dataset, then divide by dataset size outside of this function.
            2: Now we rotate into the basis of f^{l+1} using C^{l+1}, swapping elements of the
            operator around and compare this value to the value of the rotated functions in the next layer.
        C_next_layer: Basis rotation matrix used only for metric type 2. Dimensions
    """
    assert isinstance(data_key, list) or isinstance(data_key, str), "data_key must be a str or list of strings."
    # Code below would be `in_acts = torch.cat(inputs,d dim=-1)` if not detaching
    # Inputs always tuple
    inputs = torch.cat([x for x in inputs], dim=-1)

    outputs = output if isinstance(output, tuple) else (output,)
    outputs = torch.cat([x for x in outputs], dim=-1) # Concat over hidden dimension
    operator: Float[Tensor, "batch d_hidden_out"] = torch.div(outputs, inputs)

    match relu_metric_type:
        case 0:
            # Use reshaping and broadcasting to compare every element to every other element
            if operator.dim() == 2:
                batch_size, d_hidden = operator.shape
                operator_expanded_i: Float[Tensor, "batch 1 d_hidden"] = rearrange(operator, 'b d -> b 1 d')
                operator_expanded_j: Float[Tensor, "batch d_hidden 1"] = rearrange(operator, 'b d -> b d 1')
                relu_interaction_matrix = torch.einsum('bik -> ik', operator_expanded_i == operator_expanded_j) / batch_size

            elif operator.dim() == 3:
                batch_size, token_len, d_hidden_concat = operator.shape
                operator_expanded_j = rearrange(operator, 'b p d -> b p 1 d')
                operator_expanded_k = rearrange(operator, 'b p d -> b p d 1')
                # operator_matrix[batch,i,j,k,] is 1 if operator[i,j] == operator[i,k] and 0 otherwise
                relu_interaction_matrix = torch.einsum('bpik -> ik', operator_expanded_j == operator_expanded_k) / batch_size

            # Store operator (ReLU pointwise ratio)
            _add_to_hooked_matrix(hooked_data, hook_name, data_key, relu_interaction_matrix)

        case 1:
            batch_size, d_hidden = operator.shape # [256, 101]
            # Element-wise product O_j * p_j
            diag_values: Float[Tensor, "batch d_hidden"] = operator.mul(inputs)
            # Repeat diag_values along rows - row_matrix_ij = O_j * p_j
            row_matrix = repeat(diag_values, 'b d -> b d_hidden d', d_hidden=d_hidden) # [256, 101, 101]
            assert (row_matrix[0, 0, :].squeeze() == diag_values[0]).all()
            assert (row_matrix[:, 5, 7] == operator[:, 7] * inputs[:, 7]).all()
            denominator = torch.einsum('bik -> ik', row_matrix.pow(2))

            # Outer product O_i * p_j
            outer_product: Float[Tensor, "batch d_hidden d_hidden"] = torch.bmm(rearrange(operator, 'b n -> b n 1'), rearrange(inputs, 'b n -> b 1 n'))
            assert (outer_product[:, 3, 4] == operator[:, 3] * inputs[:, 4]).all()
            # 1/|X| sum_x [ (O_i * p_j- O_j * p_j)^2 / 1/|X| sum_x (O_j * p_j)^2 ] is final expression
            # Accumulate numerator and denominator separately
            numerator = torch.einsum('bik -> ik', (outer_product - row_matrix).pow(2) )

            # Store commutator matrix for numerator and denominator to accumulate separately
            _add_to_hooked_matrix(hooked_data, hook_name, data_key[0], numerator)
            _add_to_hooked_matrix(hooked_data, hook_name, data_key[1], denominator)

        case 2:
            batch_size, d_hidden = operator.shape
            # Denominator is l2 norm of functions - i.e. output of this ReLU layer
            denominator = torch.einsum('bi -> ', outputs)

            f_next_layer_hats: Float[Tensor, "batch d_hidden_trunc_next"] = outputs @ C_next_layer # E.g. [256, 89]
            # Expand twice for m, n dimensions of final matrix, keeping last dim d1 as vector dimension
            operator_expanded = repeat(operator, 'b d1 -> b d2 d3 d1', d2=d_hidden, d3=d_hidden).clone()

            for m in range(d_hidden):
                operator_expanded[:, m, :, m] = operator
            assert operator_expanded[0, 3, 7, 3] == operator[0, 7]

            C_next_layer_repeated = repeat(C_next_layer, 'd1 d2 -> b d3 d4 d1 d2' , b=batch_size, d3=d_hidden, d4=d_hidden)
            p_repeated = repeat(inputs, 'b d1 -> b d2 d3 d1', d2=d_hidden, d3=d_hidden)
            O_p = p_repeated * operator_expanded
            # Need to unsqueeze and then squeeze to avoid issues with PyTorch auto-broadcasting
            C_O_p = ( O_p.unsqueeze(-2) @ C_next_layer_repeated).squeeze(-2)
            f_next_layer_hats_unsqueezed = repeat(f_next_layer_hats, "b d1 -> b d2 d3 d1", d2=d_hidden, d3=d_hidden)
            squared_diff = (f_next_layer_hats_unsqueezed - C_O_p).pow(2)
            # Sum over batch dimension
            numerator = torch.einsum('bijk -> ij', squared_diff)

            _add_to_hooked_matrix(hooked_data, hook_name, data_key[0], numerator)
            _add_to_hooked_matrix(hooked_data, hook_name, data_key[1], denominator)
            _add_to_hooked_matrix(hooked_data, hook_name, "preactivations", inputs.sum(dim=0))


def test_edges_forward_hook_fn(
    module: torch.nn.Module,
    inputs: Union[
        tuple[Float[Tensor, "batch d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden1"],
              Float[Tensor, "batch pos d_hidden2"]],
    ],
    output: Union[
        Float[Tensor, "batch d_hidden"],
        Float[Tensor, "batch pos d_hidden"],
        tuple[Float[Tensor, "batch pos d_hidden1"],
              Float[Tensor, "batch pos d_hidden2"]],
    ],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    C_unscaled: Float[Tensor, "d_hidden1 d_hidden2"],
    C_next_layer: Float[Tensor, "d_hidden_out d_hidden_truncated"],
    W_hat: Float[Tensor, "d_hidden_out d_hiddden_in"],
) -> None:
    """Calculates C^l+1_scaled O^l W_hat^l f_hat^l.

    W_hat^l and f_hat^l are calculated using unscaled C matrices (we leave the dimensions
    degenerate).

    To use this function, make sure to hook the whole layer (e.g. layers.0) and not just the
    activation layers:
        - Input: f^l
        - Output: f^l+1

    Note all matrices obey convention that outputs=columns (opposite to notation in Overleaf main
    text) except for weight matrix.

    Args:
        C_unscaled: The unscaled C matrix for the current layer (C^l in the paper).
        C_scaled_next_layer: The scaled C matrix for the next layer (C^{l+1} in the paper).
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module (not used).
        output: Output of the module. Handles modules with one or two outputs of varying d_hiddens
            and positional indices. If no positional indices, assumes one output.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
    """
    assert isinstance(data_key, str), "data_key must be a string."

    # Code below would be `in_acts = torch.cat(inputs,d dim=-1)` if not detaching
    # Inputs are always tuple
    inputs = torch.cat([x.detach().clone() for x in inputs], dim=-1)
    batch_size = inputs.shape[0]

    # Make outputs standard tuple and concat over hidden dimension
    outputs = output if isinstance(output, tuple) else (output,)
    outputs = torch.cat([x.detach().clone() for x in outputs], dim=-1)

    next_layer_preactivations = module.linear(inputs)
    operator: Union[Float[Tensor, "batch d_hidden_out"], Float[Tensor, "batch pos d_hidden_out_concat"]] = torch.div(outputs, next_layer_preactivations)
    if operator.dim() == 2:
        diag_operator_matrix: Float[Tensor, "batch d_hidden_out d_hidden_out"] = torch.diag_embed(operator)
    elif operator.dim() == 3:
        print("oopsie woopsie, case not handled yet")

    f_hats: Float[Tensor, "batch d_hidden_trunc_curr"] = inputs @ C_unscaled # E.g. [256, 488]
    f_next_layer_hats: Float[Tensor, "batch d_hidden_trunc_next"] = outputs @ C_next_layer # E.g. [256, 89]

    # Get whole load of matrices to left of f in equation
    C_next_layer_unsqueezed = repeat(C_next_layer, 'd_hidden_out d_hidden_trunc_next -> batch_size d_hidden_out d_hidden_trunc_next', batch_size=batch_size)
    W_hat_transpose_unsqueezed = repeat(W_hat.T, 'd_hidden_out d_hidden_trunc_in -> batch_size d_hidden_out d_hidden_trunc_in', batch_size=batch_size)
    # print(f"W_hat_transpose {W_hat_transpose_unsqueezed.shape}")
    # print(f"C next layer {C_next_layer_unsqueezed.shape}")
    C_O_W_hat: Float[Tensor, "batch d_hidden_trunc_curr d_hidden_trunc_next"] = W_hat_transpose_unsqueezed @ diag_operator_matrix @ C_next_layer_unsqueezed

    batch_size, d_hidden_trunc_curr, d_hidden_trunc_next = C_O_W_hat.shape
    rows_f_next_layer_hats = repeat(f_next_layer_hats, 'b d_hidden_trunc_next -> b d d_hidden_trunc_next', d=d_hidden_trunc_curr)
    cols_f_hats = repeat(f_hats, 'b d_hidden_trunc_curr -> b d_hidden_trunc_curr d', d=d_hidden_trunc_next)
    # E_ij = hat{f^{l+1}_i} * C_O_W_hat_ij * hat{f^l_j} where * denotes scalar product
    # Equivalent to Hadamard product of hat{f^{l+1}_i} repeated along rows with C_O_W_hat
    # Then Hadamard product with hat{f^l_j} repeated along columns
    # Finally, sum over batch dimension (don't forget divison by dataset size in accumulator function)
    edge_matrix = torch.einsum('bij -> ij', rows_f_next_layer_hats * C_O_W_hat * cols_f_hats)

    _add_to_hooked_matrix(hooked_data, hook_name, data_key, edge_matrix.detach())
