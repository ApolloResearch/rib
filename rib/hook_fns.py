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
import torch.nn as nn
from einops import rearrange, repeat
from jaxtyping import Float, Int
from torch import Tensor

from rib.linalg import (
    edge_norm,
    integrated_gradient_trapezoidal_jacobian,
    integrated_gradient_trapezoidal_norm,
)
from rib.models.utils import get_model_attr


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

    So far, only metrics 1 and 4 are edited to use transformers.

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
        inputs: Inputs to the module.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
    """
    assert isinstance(data_key, str), "data_key must be a string."
    inputs = torch.cat([x.detach().clone() for x in inputs], dim=-1)
    # Store the input activations
    hooked_data[hook_name] = {data_key: inputs}


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
    C_next_layer: Float[Tensor, "d_hidden1 d_hidden2"],
    unhooked_model: nn.Module,
    module_name_next_layer: str,
    C_next_next_layer: Float[Tensor, "d_hidden1 d_hidden2"],
    n_intervals: int,
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
            3: Now we multiply the difference in function size (f_m^{l+1}-O_m*p_m) by the integrated
            gradient expression g_m^{l+1}. Note that g_m^{l+1} f_m^{l+1} is not the same as the
            entries of Lambda, because Lambda is constructed with the g_m containing derivative of
        C_next_layer: Basis rotation matrix used only for metric type 2. Dimensions
        model: Pass whole model in to literally grab next layer for in_grads.
        module_name_next_layer: Also for getting next layer for in_grads.
        C_next_next_layer: Also for in_grads.
        n_intervals: Also for in_grads.
    """
    assert isinstance(data_key, list) or isinstance(data_key, str), "data_key must be a str or list of strings."

    output_is_tuple: bool = True if isinstance(output, tuple) else False
    outputs = output if isinstance(output, tuple) else (output,)
    raw_output = outputs # For passing into g_j finding function for metric 3
    out_hidden_dims = [x.shape[-1] for x in outputs]

    # Don't want residual stream for operator syncing
    inputs = inputs[1]
    outputs = outputs[1]

    # inputs = torch.cat([x.detach().clone() for x in inputs], dim=-1) # Inputs always tuple
    # batch_size = inputs.shape[0]
    # outputs = torch.cat([x.detach().clone() for x in outputs], dim=-1)

    operator: Float[Tensor, "batch d_hidden_out"] = torch.div(outputs, inputs)

    has_pos = True if operator.dim() == 3 else False
    if not has_pos:
        batch_size, d_hidden = operator.shape # [256, 101]
    else:
        batch_size, token_len, d_hidden = operator.shape

    match relu_metric_type:
        case 0:
            """Compare every operator element to every other element - resulting matrix entries either 0 or 1."""
            if not has_pos:
                operator_expanded_i: Float[Tensor, "batch 1 d_hidden"] = rearrange(operator, 'b d -> b 1 d')
                operator_expanded_j: Float[Tensor, "batch d_hidden 1"] = rearrange(operator, 'b d -> b d 1')
                relu_interaction_matrix = torch.einsum('bik -> ik', operator_expanded_i == operator_expanded_j)

            if has_pos:
                operator_expanded_j = rearrange(operator, 'b p d -> b p 1 d')
                operator_expanded_k = rearrange(operator, 'b p d -> b p d 1')
                # operator_matrix[batch,i,j,k,] is 1 if operator[i,j] == operator[i,k] and 0 otherwise
                relu_interaction_matrix = torch.einsum('bpik -> ik', operator_expanded_j == operator_expanded_k)

            _add_to_hooked_matrix(hooked_data, hook_name, "relu_num", relu_interaction_matrix)

        case 1:
            # Element-wise product O_j * p_j
            diag_values: Float[Tensor, "batch d_hidden"] = operator.mul(inputs)
            # Repeat diag_values along rows - row_matrix_ij = O_j * p_j
            row_matrix = repeat(diag_values, 'b d -> b d_hidden d', d_hidden=d_hidden) # [256, 101, 101]
            assert (row_matrix[0, 0, :].squeeze() == diag_values[0]).all()
            assert (row_matrix[:, 5, 7] == operator[:, 7] * inputs[:, 7]).all()

            # Outer product O_i * p_j
            outer_product: Float[Tensor, "batch d_hidden d_hidden"] = torch.bmm(rearrange(operator, 'b n -> b n 1'), rearrange(inputs, 'b n -> b 1 n'))
            assert (outer_product[:, 3, 4] == operator[:, 3] * inputs[:, 4]).all()
            # 1/|X| sum_x [ (O_i * p_j- O_j * p_j)^2 / 1/|X| sum_x (O_j * p_j)^2 ] is final expression
            # Accumulate numerator and denominator separately
            numerator = torch.einsum('bik -> ik', (outer_product - row_matrix).pow(2) )

            # Store commutator matrix for numerator and denominator to accumulate separately
            _add_to_hooked_matrix(hooked_data, hook_name, "relu_num", numerator)

        case 2:
            with torch.no_grad():
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

                f_next_layer_hats: Float[Tensor, "batch d_hidden_trunc_next"] = outputs @ C_next_layer # E.g. [256, 89]
                f_next_layer_hats_unsqueezed = repeat(f_next_layer_hats, "b d1 -> b d2 d3 d1", d2=d_hidden, d3=d_hidden)

                squared_diff = (f_next_layer_hats_unsqueezed - C_O_p).pow(2)
                numerator = torch.einsum('bijk -> ij', squared_diff) # Sum over batch dimension and indices of vector (for each final matrix element)

                _add_to_hooked_matrix(hooked_data, hook_name, "relu_num", numerator.detach())

        case 3:
            """Note when running for any given metric, we are in layer l-1 for the lth layer metric!
            This is so we can hook the correct O^l(x) and p^l(x).
            Also makes calculating g_j^l a pain, as g_j^l depends on C^{l+1} and should be
            calculated for module l, not l-1. Pass in separate next module name and next_next_layer
            C^{l+1}."""
            ## For first term of numerator (... indicates either "b" or "b p")
            g_j_next_layer: Float[Tensor, "... d_hidden_combined"] = integrated_gradient_trapezoidal_norm(
                module=get_model_attr(unhooked_model, module_name_next_layer),
                inputs=raw_output, # Inputs (ALWAYS TUPLE) to next layer are outputs of activation module -> may need to fix concat for transformers
                C_out=C_next_next_layer,
                n_intervals=n_intervals,
            )

            if output_is_tuple: g_j_next_layer = torch.split(g_j_next_layer, out_hidden_dims, dim=-1)[1]

            if has_pos: # Fold position into batch dimension
                g_j_next_layer = rearrange(g_j_next_layer, "b p d_hidden_combined -> (b p) d_hidden_combined")
                inputs = rearrange(inputs, "b p d_hidden_combined -> (b p) d_hidden_combined")
                outputs = rearrange(outputs, "b p d_hidden_combined -> (b p) d_hidden_combined")
                operator = rearrange(operator, "b p d_hidden_combined -> (b p) d_hidden_combined")

            with torch.no_grad():
                ## Second term of numerator
                g_j_hadamard_p: Float[Tensor, "... d_hidden"] = g_j_next_layer * inputs
                cols_g_j_hadamard_p = repeat(g_j_hadamard_p, 'b d1 -> b d1 d2', d2=d_hidden)
                rows_operators = repeat(operator, 'b d1 -> b d2 d1', d2=d_hidden)
                assert (rows_operators[:, 0, :] == operator).all()
                # Hadamard product and sum over batch dimension
                numerator_term_2: Float[Tensor, "batch d_hidden d_hidden"] = cols_g_j_hadamard_p * rows_operators
                assert numerator_term_2[0, 4, 8] == (g_j_next_layer[0, 4] * operator[0, 8] * inputs[0, 4])

                ## First term of numerator
                numerator_term_1: Float[Tensor, "batch d_hidden d_hidden"] = repeat(g_j_next_layer * outputs, 'b d1 -> b d1 d2', d2=d_hidden)
                Lambda_dash = torch.einsum('bi,bj -> bij', g_j_next_layer, outputs)
                whole_numerator: Float[Tensor, "batch d_hidden d_hidden"] = numerator_term_1 - numerator_term_2

                [numerator_term_2, whole_numerator, numerator_term_1] = apply_batch_einsum(numerator_term_2, whole_numerator, numerator_term_1)
                _add_to_hooked_matrix(hooked_data, hook_name, "relu_num", numerator_term_2)
                _add_to_hooked_matrix(hooked_data, hook_name, "whole_relu_num", whole_numerator)
                _add_to_hooked_matrix(hooked_data, hook_name, "relu_num_1", numerator_term_1)

    _add_to_hooked_matrix(hooked_data, hook_name, "preactivations", inputs.sum(dim=0)) # For debugging purposes


def apply_batch_einsum(*args) -> list[Tensor]:
    results = []
    for arg in args:
        result = torch.einsum('bij -> ij', arg)
        results.append(result)
    return results


def function_size_forward_hook_fn(
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
    rotate: bool,
    C_next_layer: Float[Tensor, "d_hidden d_hidden"],
) -> None:
    """Hook function for l2 norm of f^{l+1} (output of layer module).

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module.
        output: Output of the module. Handles modules with one or two outputs of varying d_hiddens
            and positional indices. If no positional indices, assumes one output.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
        rotate: Whether to rotate the functions to calculate l2 norm of fhat
        C_next_layer: Optional, provide for rotate=True case.
    """
    assert isinstance(data_key, list) or isinstance(data_key, str), "data_key must be a str or list of strings."
    outputs = output if isinstance(output, tuple) else (output,)
    outputs = torch.cat([x for x in outputs], dim=-1) # Concat over hidden dimension
    if rotate == True:
        outputs = outputs @ C_next_layer
    einsum_pattern = 'bpi ->' if outputs.dim() == 3 else 'bi->'
    # l2 norm of functions f^{l+1} - i.e. output of this layer
    _add_to_hooked_matrix(hooked_data, hook_name, "fn_size", torch.einsum(einsum_pattern, outputs.pow(2)))


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
    """Calculate C^l+1_scaled O^l W_hat^l f_hat^l.

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
    outputs = output if isinstance(output, tuple) else (output,)
    inputs = torch.cat([x.detach().clone() for x in inputs], dim=-1)
    batch_size = inputs.shape[0]
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
    W_hat_t_unsqueezed = repeat(W_hat.T, 'd_hidden_out d_hidden_trunc_in -> batch_size d_hidden_out d_hidden_trunc_in', batch_size=batch_size)
    C_O_W_hat: Float[Tensor, "batch d_hidden_trunc_curr d_hidden_trunc_next"] = W_hat_t_unsqueezed @ diag_operator_matrix @ C_next_layer_unsqueezed

    batch_size, d_hidden_trunc_curr, d_hidden_trunc_next = C_O_W_hat.shape
    rows_f_next_layer_hats = repeat(f_next_layer_hats, 'b d_hidden_trunc_next -> b d d_hidden_trunc_next', d=d_hidden_trunc_curr)
    cols_f_hats = repeat(f_hats, 'b d_hidden_trunc_curr -> b d_hidden_trunc_curr d', d=d_hidden_trunc_next)
    assert (cols_f_hats[:, :, 0] == f_hats).all()

    # E_ij = hat{f^{l+1}_i} * C_O_W_hat_ij * hat{f^l_j} where * denotes scalar product
    # Equivalent to Hadamard product of hat{f^{l+1}_i} repeated along rows with C_O_W_hat
    # Then Hadamard product with hat{f^l_j} repeated along columns
    # Finally, sum over batch dimension (don't forget divison by dataset size in accumulator function)
    edge_matrix = torch.einsum('bij -> ij', rows_f_next_layer_hats * C_O_W_hat * cols_f_hats)

    _add_to_hooked_matrix(hooked_data, hook_name, data_key, edge_matrix.detach())


# def clustered_relu_P_hook_fn(
#     module: torch.nn.Module,
#     inputs: Union[
#         tuple[Float[Tensor, "batch d_hidden"]],
#         tuple[Float[Tensor, "batch pos d_hidden"]],
#         tuple[Float[Tensor, "batch pos d_hidden1"],
#               Float[Tensor, "batch pos d_hidden2"]],
#     ],
#     output: Union[
#         Float[Tensor, "batch d_hidden"],
#         Float[Tensor, "batch pos d_hidden"],
#         tuple[Float[Tensor, "batch pos d_hidden1"],
#               Float[Tensor, "batch pos d_hidden2"]],
#     ],
#     hooked_data: dict[str, Any],
#     hook_name: str,
#     data_key: Union[str, list[str]],
#     C_next_layer: Float[Tensor, "d_hidden_out d_hidden_truncated"],
#     W_hat: Float[Tensor, "d_hidden_out d_hiddden_in"],
#     layer_cluster_idxs: list[Int[Tensor, "cluster_size"]],
# ) -> None:
#     """Do same thing as function above, but hook whole layer - get the P matrix from
#     notes. Calculate C^l+1 O^l W_hat^l f_hat^l.

#     Hooks activation layer (makes things easier for transformer).
#     """
#     assert isinstance(data_key, list) or isinstance(data_key, str), "data_key must be a str or list of strings."

#     inputs = torch.cat([x for x in inputs], dim=-1)

#     # next_layer_preactivations = module.linear(inputs)

#     output_is_tuple: bool = True if isinstance(output, tuple) else False
#     outputs = output if output_is_tuple else (output,)
#     outputs = torch.cat([x for x in outputs], dim=-1) # Concat over hidden dimension

#     # operator: Union[Float[Tensor, "batch d_hidden_out"], Float[Tensor, "batch pos d_hidden_out_concat"]] = torch.div(outputs, next_layer_preactivations)
#     # if operator.dim() == 3: # Combine token dimension with batch dimension to compute as if it were 2D
#     #     inputs = rearrange(inputs, "b p d_hidden_combined -> (b p) d_hidden_combined")
#     #     outputs = rearrange(outputs, "b p d_hidden_combined -> (b p) d_hidden_combined")
#     #     operator = rearrange(operator, 'b p d_hidden -> (b p) d_hidden')
#     # batch_size, d_hidden = operator.shape
#     # diag_operator_matrix: Float[Tensor, "batch d_hidden_out d_hidden_out"] = torch.diag_embed(operator)

#     # # ---------------- Now have edited output to pass to next layer, also hook vals in middle -----------------
#     # C_next_layer_unsqueezed = repeat(C_next_layer, "d_hidden_out d_hidden_trunc_next -> batch_size d_hidden_out d_hidden_trunc_next", batch_size=batch_size)
#     # W_hat_t_unsqueezed = repeat(W_hat.T, "d_hidden_out d_hidden_trunc_in -> batch_size d_hidden_out d_hidden_trunc_in", batch_size=batch_size)
#     # C_O_W_hat: Float[Tensor, "batch d_hidden_trunc_curr d_hidden_trunc_next"] = W_hat_t_unsqueezed @ diag_operator_matrix @ C_next_layer_unsqueezed

#     # batch_size, d_hidden_trunc_curr, d_hidden_trunc_next = C_O_W_hat.shape
#     # cols_f = repeat(inputs, "b d_hidden_trunc_curr -> b d_hidden_trunc_curr d", d=d_hidden_trunc_next)

#     P_dict = {}
#     for cluster_idxs in layer_cluster_idxs:
#         C_next_layer_cluster = C_next_layer[cluster_idxs, :] # Cols C in overleaf = rows C in code
#         W_hat_cluster = W_hat.T[:, cluster_idxs] # Rows W overleaf = cols C in code
#         P: Float[Tensor, "d_hidden_next_layer d_hidden"] = W_hat_cluster.T @ C_next_layer_cluster
#         P_dict[cluster_idxs] = P.detach().cpu()

#     _add_to_hooked_matrix(hooked_data, hook_name, data_key, P_dict)