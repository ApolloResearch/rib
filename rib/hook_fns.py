"""Defines hook functions that are used in a HookedModel.

All forward hook functions must contain "forward" in their function names, and all pre-forward
hook functions must contain "pre_forward" in their function names. This is done to ensure that
the correct type of hook is registered to the module.

By default, a HookedModel passes in the arguments `hooked_data`, `hook_name`, and `data_key` to
each hook function. Therefore, these arguments must be included in the signature of each hook.

Otherwise, the hook function operates like a regular pytorch hook function.
"""
from copy import deepcopy
from functools import partial
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from jaxtyping import Float, Int
from torch import Tensor
import gc

from rib.linalg import (
    calc_gram_matrix,
    edge_norm,
    integrated_gradient_trapezoidal_jacobian,
    integrated_gradient_trapezoidal_norm,
    get_local_jacobian,
)
from rib.models.utils import get_model_attr
from itertools import combinations


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
    dataset_size: int,
) -> None:
    """Hook function for calculating and updating the gram matrix.

    The tuple of outputs is concatenated over the hidden dimension.

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module (not used).
        output: Output of the module. Handles modules with one or two outputs of varying d_hiddens
            and positional indices. If no positional indices, assumes one output.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
        dataset_size: Size of the dataset. Used to normalize the gram matrix.

    """
    assert isinstance(data_key, str), "data_key must be a string."

    outputs = output if isinstance(output, tuple) else (output,)

    # Concat over the hidden dimension
    out_acts = torch.cat([x.detach().clone() for x in outputs], dim=-1)

    gram_matrix = calc_gram_matrix(out_acts, dataset_size=dataset_size)

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
    dataset_size: int,
) -> None:
    """Calculate the gram matrix for inputs with positional indices and add it to the global.

    The tuple of inputs is concatenated over the hidden dimension.

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module. Handles modules with one or two inputs of varying d_hiddens
            and positional indices. If no positional indices, assumes one input.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
        dataset_size: Size of the dataset. Used to normalize the gram matrix.
    """
    assert isinstance(data_key, str), "data_key must be a string."

    in_acts = torch.cat([x.detach().clone() for x in inputs], dim=-1)

    gram_matrix = calc_gram_matrix(in_acts, dataset_size=dataset_size)

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
    dataset_size: int,
    M_dtype: torch.dtype = torch.float64,
    Lambda_einsum_dtype: torch.dtype = torch.float64,
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
            to taking a point estimate at alpha == 0.5.
        dataset_size: Size of the dataset. Used to normalize the gram matrix.
        M_dtype: The data type to use for the M_dash matrix. Needs to be
            float64 for Pythia-14m (empirically). Defaults to float64.
        Lambda_einsum_dtype: The data type to use for the einsum computing batches for the
            Lambda_dash matrix. Does not affect the output, only used for the einsum itself.
            Needs to be float64 on CPU but float32 was fine on GPU. Defaults to float64.
    """
    assert isinstance(data_key, list), "data_key must be a list of strings."
    assert len(data_key) == 2, "data_key must be a list of length 2 to store M' and Lambda'."
    # Remove the pre foward hook to avoid recursion when calculating the jacobian
    module._forward_pre_hooks.popitem()
    assert not module._forward_hooks, "Module has multiple forward hooks"

    in_grads = integrated_gradient_trapezoidal_norm(
        module=module,
        inputs=inputs,
        C_out=C_out,
        n_intervals=n_intervals,
    )
    in_dtype = in_grads.dtype

    has_pos = inputs[0].dim() == 3

    einsum_pattern = "bpj,bpJ->jJ" if has_pos else "bj,bJ->jJ"
    normalization_factor = in_grads.shape[1] * dataset_size if has_pos else dataset_size

    with torch.inference_mode():
        M_dash = torch.einsum(
            einsum_pattern,
            in_grads.to(M_dtype) / normalization_factor,
            in_grads.to(M_dtype),
        )
        # Concatenate the inputs over the hidden dimension
        in_acts = torch.cat(inputs, dim=-1)
        Lambda_dash = torch.einsum(
            einsum_pattern,
            in_grads.to(Lambda_einsum_dtype) / normalization_factor,
            in_acts.to(Lambda_einsum_dtype),
        )
        Lambda_dash = Lambda_dash.to(in_dtype)

        _add_to_hooked_matrix(hooked_data, hook_name, data_key[0], M_dash)
        _add_to_hooked_matrix(hooked_data, hook_name, data_key[1], Lambda_dash)

    assert (
        Lambda_dash.std() > 0
    ), "Lambda_dash cannot be all zeros otherwise everything will be truncated"


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
    dataset_size: int,
) -> None:
    """Hook function for accumulating the edges (denoted E_hat) of the interaction graph.

    For calculating the Jacobian, we need to run the inputs through the module. Unfortunately,
    this causes an infinite recursion because the module has a hook which runs this function. To
    avoid this, we (hackily) remove the hook before running the module and then add it back after.

    The trapezoidal rule is used to approximate the integrated gradient. If n_intervals == 0, the
    integrated gradient effectively takes a point estimate for the integral at alpha == 0.5.

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
            to taking a point estimate at alpha == 0.5.
        dataset_size: Size of the dataset. Used to normalize the gradients.
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

    in_hidden_dims = [x.shape[-1] for x in inputs]

    # Compute f^{l+1}(x) to which the derivative is not applied.
    with torch.inference_mode():
        # f_in_hat @ C_in_pinv does not give exactly f due to C and C_in_pinv being truncated
        f_in_adjusted: Float[Tensor, "... in_hidden_combined_trunc"] = f_hat @ C_in_pinv
        input_tuples = torch.split(f_in_adjusted, in_hidden_dims, dim=-1)

        output_const = module(*tuple(x.detach().clone() for x in input_tuples))
        outputs_const = (output_const,) if isinstance(output_const, torch.Tensor) else output_const

    has_pos = f_hat.dim() == 3

    jac_out = hooked_data[hook_name][data_key]
    edge_norm_partial = partial(
        edge_norm,
        outputs_const=outputs_const,
        module=module,
        C_in_pinv=C_in_pinv,
        C_out=C_out,
        in_hidden_dims=in_hidden_dims,
        has_pos=has_pos,
    )

    integrated_gradient_trapezoidal_jacobian(
        fn=edge_norm_partial,
        x=f_hat,
        n_intervals=n_intervals,
        jac_out=jac_out,
        dataset_size=dataset_size,
    )


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
    detached_outputs = [x.detach().cpu() for x in outputs]
    # Store the output activations
    hooked_data[hook_name] = {data_key: detached_outputs}


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
        tuple[Float[Tensor, "batch d_hidden"],],
        tuple[Float[Tensor, "batch pos d_hidden"],],
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
    use_residual_stream: bool,
) -> None:
    """Hook function to store for ReLU metric numerators. The denominator is handled separately.

    TODO (nonurgent): fix 0/0 cases in type 0 (type 1 handles infs and nans with duct tape).

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module.
        output: Output of the module. Handles modules with one or two outputs of varying d_hiddens
            and positional indices. If no positional indices, assumes one output.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
        relu_metric_type: Determines which type of ReLU syncing metric is used. Explained in LaTeX doc.
        C_next_layer: Basis rotation matrix used only for metric type 2. Dimensions
        model: Pass whole model in to literally grab next layer for in_grads.
        module_name_next_layer: Also for getting next layer for in_grads.
        C_next_next_layer: Also for in_grads.
        n_intervals: Also for in_grads.
        use_residual_stream: Whether to use residual stream for ReLU clustering.
    """
    module = _remove_hooks(module)

    is_lm: bool = True if inputs[0].dim() == 3 else False
    outputs = output if isinstance(output, tuple) else (output,)
    raw_output = [x.clone() for x in outputs] # For passing into g_j finding function for metric 3
    out_hidden_dims = [x.shape[-1] for x in outputs]

    """Now fold the token into the batch to make it look like batch x height x width."""
    if is_lm and not use_residual_stream: # Don't want residual stream for operator syncing, so taken inputs[1] (last element of [MLP, resid_stream] tuple)
        inputs = rearrange(inputs[1], "b p d_hidden_combined -> (b p) d_hidden_combined")
        outputs = rearrange(outputs[1], "b p d_hidden_combined -> (b p) d_hidden_combined")
    elif is_lm and use_residual_stream:
        inputs = rearrange(torch.cat([x.detach().clone() for x in inputs], dim=-1), "b p d_hidden_combined -> (b p) d_hidden_combined")
        outputs = rearrange(torch.cat([x.detach().clone() for x in outputs], dim=-1), "b p d_hidden_combined -> (b p) d_hidden_combined")
    else:  # Inputs always tuple, and in this case we want to treat the residual stream and MLP in the same way OR we don't have LM
        inputs = torch.cat([x.detach().clone() for x in inputs], dim=-1)
        outputs = torch.cat([x.detach().clone() for x in outputs], dim=-1)

    operator: Float[Tensor, "batch d_hidden_out"] = torch.div(outputs, inputs)
    batch_size, d_hidden = operator.shape

    match relu_metric_type:
        case 0:
            """Compare every operator element to every other element - resulting matrix entries either 0 or 1."""
            operator_expanded_i: Float[Tensor, "batch 1 d_hidden"] = rearrange(operator, 'b d -> b 1 d')
            operator_expanded_j: Float[Tensor, "batch d_hidden 1"] = rearrange(operator, 'b d -> b d 1')
            numerator = torch.einsum('bik -> ik', operator_expanded_i == operator_expanded_j)

        case 1:
            with torch.no_grad():
                diag_values: Float[Tensor, "batch d_hidden"] = operator * inputs # Element-wise product O_j * p_j
                # TBH the line above could have been done by using f_j = O_j p_j, removing need to
                # calculate diag_values. I'm leaving this in for clarity.
                row_matrix = repeat(diag_values, 'b d -> b d_hidden d', d_hidden=d_hidden) # Repeat diag_values along rows - row_matrix_ij = O_j * p_j
                assert (row_matrix[0, 0, :].squeeze() == diag_values[0]).all()
                assert (row_matrix[:, 5, 7] == operator[:, 7] * inputs[:, 7]).all()

                outer_product: Float[Tensor, "batch d_hidden d_hidden"] = torch.bmm(rearrange(operator, 'b n -> b n 1'), rearrange(inputs, 'b n -> b 1 n'))  # O_i * p_j
                assert (outer_product[:, 3, 4] == operator[:, 3] * inputs[:, 4]).all()
                numerator = torch.einsum('bik -> ik', (outer_product - row_matrix).pow(2) ) # (O_i*p_j - O_j*p_j)^2

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

        case 3:
            """Note when running for any given metric, we are in layer l-1 for the lth layer metric!
            This is so we can hook the correct O^l(x) and p^l(x).
            Also makes calculating g_j^l a pain, as g_j^l depends on C^{l+1} and should be
            calculated for module l, not l-1. Pass in separate next module name and next_next_layer
            C^{l+1}.
            Note the gradient calculation is where position is treated differently to batch, so be
            careful to only pass in raw output into the function."""
            ## For first term of numerator (... indicates either "b" or "b p")
            next_layer_module = get_model_attr(unhooked_model, module_name_next_layer)
            # Deep copy module so we can pass it in to calculate g_j without worrying about existing
            # hooks on activation function which cause recursive hook calling when
            # forward method called inside integrated_gradient_trapezoidal_norm
            copy_next_layer_module = deepcopy(next_layer_module)
            if hasattr(copy_next_layer_module, 'activation') and copy_next_layer_module._forward_hooks:
                copy_next_layer_module.activation._forward_hooks.popitem()
                assert not copy_next_layer_module.activation._forward_hooks, "Module activation has multiple forward hooks"

            g_j_next_layer: Float[Tensor, "... d_hidden_combined"] = integrated_gradient_trapezoidal_norm(
                module=copy_next_layer_module,
                inputs=raw_output, # Inputs (ALWAYS TUPLE) to next layer are outputs of activation module -> may need to fix concat for transformers
                C_out=C_next_next_layer,
                n_intervals=n_intervals,
            )

            if is_lm:
                if not use_residual_stream:
                    g_j_next_layer = torch.split(g_j_next_layer, out_hidden_dims, dim=-1)[1]
                # Fold position into batch dimension
                g_j_next_layer = rearrange(g_j_next_layer, "b p d_hidden_combined -> (b p) d_hidden_combined")

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
                numerator: Float[Tensor, "batch d_hidden d_hidden"] = numerator_term_1 - numerator_term_2

                [numerator_term_2, numerator, numerator_term_1] = apply_batch_einsum(numerator_term_2, numerator, numerator_term_1)

    _add_to_hooked_matrix(hooked_data, hook_name, "relu_num", numerator)
    _add_to_hooked_matrix(hooked_data, hook_name, "preactivations", inputs.sum(dim=0)) # For debugging purposes


def apply_batch_einsum(*args: list[Tensor]) -> list[Tensor]:
    """For the einsum pattern 'bij -> ij', apply to every Tensor passed in."""
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
    module = _remove_hooks(module)

    outputs = output if isinstance(output, tuple) else (output,)
    outputs = torch.cat([x for x in outputs], dim=-1) # Concat over hidden dimension

    if rotate == True: outputs = outputs @ C_next_layer
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

    TODO: needs refactor
    """
    module = _remove_hooks(module)

    is_lm: bool = True if inputs[0].dim() == 3 else False
    if is_lm:
        next_layer_preactivations = module.forward(*inputs)
        next_layer_preactivations = torch.cat([x for x in next_layer_preactivations], dim=-1)

    outputs = output if isinstance(output, tuple) else (output,)
    inputs = torch.cat([x.detach().clone() for x in inputs], dim=-1)
    outputs = torch.cat([x.detach().clone() for x in outputs], dim=-1)

    if not is_lm: next_layer_preactivations = module.linear(inputs)

    operator: Union[Float[Tensor, "batch d_hidden_out"], Float[Tensor, "batch pos d_hidden_out_concat"]] = torch.div(outputs, next_layer_preactivations)
    if operator.dim() == 3: # Combine token dimension with batch dimension to compute as if it were 2D
        inputs = rearrange(inputs, "b p d_hidden_combined -> (b p) d_hidden_combined")
        outputs = rearrange(outputs, "b p d_hidden_combined -> (b p) d_hidden_combined")
        operator = rearrange(operator, 'b p d_hidden -> (b p) d_hidden')
    batch_size, d_hidden = operator.shape
    diag_operator_matrix: Float[Tensor, "batch d_hidden_out d_hidden_out"] = torch.diag_embed(operator)

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


def cluster_gram_forward_hook_fn(
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
    dataset_size: int,
    cluster_idxs: list[Int[Tensor, "d_hidden"]],
    use_residual_stream: bool,
) -> None:
    """Calculate gram matrix of each cluster, where cluster replacement has already been tested for
    100% accuracy retention.

    Status: this function only looked at forming a single gram matrix from the indexed functions
    in the unembed layer that are members of a given cluster.
    It is separate to the function below, which forms a gram for all the incoming functions in the
    embed layer elementwise multipled by the ReLU operator.

    Hook only valid for activation layer.
    """
    is_lm = True if inputs[0].dim() == 3 else False
    outputs = output if isinstance(output, tuple) else (output,)

    # Once again, fold in token dimension into batch
    if is_lm and not use_residual_stream:
        inputs = rearrange(inputs[1], "b p d_hidden_combined -> (b p) d_hidden_combined")
        outputs = rearrange(outputs[1], "b p d_hidden_combined -> (b p) d_hidden_combined")
    elif is_lm and use_residual_stream:
        inputs = rearrange(torch.cat([x for x in inputs], dim=-1), "b p d_hidden_combined -> (b p) d_hidden_combined")
        outputs = rearrange(torch.cat([x for x in outputs], dim=-1), "b p d_hidden_combined -> (b p) d_hidden_combined")
    else:  # Inputs always tuple, and in this case we don't have LM
        inputs = torch.cat([x.detach().clone() for x in inputs], dim=-1)
        outputs = torch.cat([x.detach().clone() for x in outputs], dim=-1)

    # Every cluster has its own gram matrix, which can be differentiated with `data_key` which tells
    # you the cluster number (the element of cluster_idxs it was taken from)
    for cluster_num, idxs in enumerate(cluster_idxs):
        cluster_inputs = inputs[..., idxs]
        cluster_outputs = outputs[..., idxs]
        cluster_operator = torch.div(cluster_outputs, cluster_inputs)
        gram_matrix = calc_gram_matrix(cluster_operator * cluster_inputs, dataset_size=dataset_size)
        output_gram_matrix = calc_gram_matrix(cluster_outputs, dataset_size=dataset_size)
        _add_to_hooked_matrix(hooked_data, hook_name, cluster_num, gram_matrix.detach())
        _add_to_hooked_matrix(hooked_data, hook_name, f"output_{cluster_num}", output_gram_matrix.detach())


# def cluster_gram_forward_hook_fn(
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
#     dataset_size: int,
#     cluster_idxs: list[Int[Tensor, "d_hidden"]],
#     use_residual_stream: bool,
# ) -> None:
#     """Status: temporary research function.

#     Calculate gram matrix of each cluster, where cluster replacement has already been tested for
#     100% accuracy retention.

#     Hook should be on section containing mlp_in layer and mlp_act layer in modadd transformer.
#     """
#     is_lm = True if inputs[0].dim() == 3 else False
#     outputs = output if isinstance(output, tuple) else (output,)

#     # Once again, fold in token dimension into batch
#     if is_lm and not use_residual_stream:
#         inputs = rearrange(inputs[1], "b p d_hidden_combined -> (b p) d_hidden_combined")
#         outputs = rearrange(outputs[1], "b p d_hidden_combined -> (b p) d_hidden_combined")
#     elif is_lm and use_residual_stream:
#         inputs = rearrange(torch.cat([x for x in inputs], dim=-1), "b p d_hidden_combined -> (b p) d_hidden_combined")
#         outputs = rearrange(torch.cat([x for x in outputs], dim=-1), "b p d_hidden_combined -> (b p) d_hidden_combined")
#     else:  # Inputs always tuple, and in this case we don't have LM
#         inputs = torch.cat([x.detach().clone() for x in inputs], dim=-1)
#         outputs = torch.cat([x.detach().clone() for x in outputs], dim=-1)

#     batch_size, d_hidden = inputs.shape

#     # Every cluster has its own gram matrix, which can be differentiated with `data_key` which tells
#     # you the cluster number (the element of cluster_idxs it was taken from)
#     for cluster_num, idxs in enumerate(cluster_idxs):
#         idx = idxs[0].item() # Arbitrarily pick first member of cluster
#         o_k = repeat(outputs[..., idx] > 0, 'b -> b d_hidden', d_hidden=d_hidden) # 1 If greater than zero else 0
#         gram_matrix = calc_gram_matrix(o_k * inputs, dataset_size=dataset_size)
#         _add_to_hooked_matrix(hooked_data, hook_name, cluster_num, gram_matrix.detach())

#     whole_layer_gram = calc_gram_matrix(inputs, dataset_size=dataset_size)
#     _add_to_hooked_matrix(hooked_data, hook_name, "whole layer", whole_layer_gram.detach())


def cluster_fn_pre_forward_hook_fn(
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
    cluster_idxs: list[Int[Tensor, "d_hidden"]],
    use_residual_stream: bool,
) -> None:
    """For post-ReLU activations of modadd tranformer (in this case in unembed layer)."""
    is_lm = True if inputs[0].dim() == 3 else False

    # Once again, fold in token dimension into batch
    if is_lm and not use_residual_stream:
        inputs = rearrange(inputs[1], "b p d_hidden_combined -> (b p) d_hidden_combined")
    elif is_lm and use_residual_stream:
        inputs = rearrange(torch.cat([x for x in inputs], dim=-1), "b p d_hidden_combined -> (b p) d_hidden_combined")
    else:  # Inputs always tuple, and in this case we don't have LM
        inputs = torch.cat([x.detach().clone() for x in inputs], dim=-1)

    for cluster_num, idxs in enumerate(cluster_idxs):
        cluster_outputs = torch.einsum('bi -> i', inputs[..., idxs])
        _add_to_hooked_matrix(hooked_data, hook_name, cluster_num, cluster_outputs.detach())


def collect_hessian_pre_forward_hook_fn(
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
    weight_module_names: list[str],
    copy_seq_model: nn.Module,
    C_list: list[Float[Tensor, "out_hidden_trunc in_hidden"]],
    use_residual_stream: bool,
    output_dim: int = 129,
) -> None:
    """Hook function for collecting output activations and passing into Jacobian.
    This hook function finds the Jacobian for all weight_modules and forms the Hessian from that.

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module (not used).
        output: Output of the module. Handles modules with one or two outputs of varying d_hiddens
            and positional indices. If no positional indices, assumes one output.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
        weight_module_names: Names of sections to form Hessian from i.e. these are the weight
            parameters the derivatives will be taken with respect to.
    """
    module_dict = {name: get_model_attr(copy_seq_model, name) for name in weight_module_names}
    # If the model is chosen to be separate to the one with hooks, no issues should arise, but
    # remove hooks just in case to prevent recursive hook calling in forward passes in Jacobian
    # function
    jac_modules = list(map(_remove_hooks, module_dict.values()))

    jacobians = {}
    for i, name in enumerate(weight_module_names):
        jac = get_local_jacobian(
            modules=jac_modules,
            weight_module=module_dict[name],
            inputs=inputs,
            C=C_list[i],
            use_residual_stream=use_residual_stream,
        )
        jacobians[f"derivative wrt {name}"] = jac

    J1 = jacobians["derivative wrt sections.section_0.0"]
    J2 = jacobians["derivative wrt sections.section_1.0"]
    H1 = torch.div(torch.matmul(J1.T, J1), 60000)
    H2 = torch.div(torch.matmul(J2.T, J2), 60000)
    M12 = torch.div(torch.matmul(J1.T, J2), 60000)
    M21 = torch.div(torch.matmul(J2.T, J1), 60000)

    top_row = torch.cat((H1, M12), dim=1)
    bottom_row = torch.cat((M21, H2), dim=1)
    H = torch.cat((top_row, bottom_row), dim=0)

    del H1, H2, M12, M21, top_row, bottom_row
    torch.cuda.empty_cache()
    gc.collect()

    _add_to_hooked_matrix(hooked_data, hook_name, f"hessian {hook_name}", H.detach().cpu())


def _remove_hooks(module: torch.nn.Module) -> torch.nn.Module:
    """Remove module hooks. Should be careful if these hooks are needed later in code, since this
    operates directly on the module and not a copy."""
    if module._forward_hooks:
        module._forward_hooks.popitem()
    if module._forward_pre_hooks:
        module._forward_pre_hooks.popitem()
    assert not module._forward_hooks, "Module has multiple forward hooks."
    assert not module._forward_pre_hooks, "Module has multiple pre-forward hooks."
    return module


def collect_hessian_forward_hook_fn(
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
    weight_module_names: list[str],
    copy_seq_model: nn.Module,
    C_list: list[Float[Tensor, "out_hidden_trunc in_hidden"]],
    use_residual_stream: bool,
    output_dim: int = 129,
) -> None:
    """Hook function for collecting output activations and passing into Jacobian.
    This hook function finds the Jacobian for all weight_modules and forms the Hessian from that.

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module (not used).
        output: Output of the module. Handles modules with one or two outputs of varying d_hiddens
            and positional indices. If no positional indices, assumes one output.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
        weight_module_names: Names of sections to form Hessian from i.e. these are the weight
            parameters the derivatives will be taken with respect to.
    """
    module_dict = {name: get_model_attr(copy_seq_model, name) for name in weight_module_names}
    module = _remove_hooks(module)
    DATASET_SIZE = 60000

    is_lm = True if inputs[0].dim() == 3 else False
    outputs = output if isinstance(output, tuple) else (output,)
    resid_stream_size = inputs[0].shape[-1]
    mlp_layer_size = outputs[1].shape[-1]
    # Calculate pre-activations
    for child in module.named_children():
        if child[0] == '0':
            preactivations = child[1].forward(*inputs)

    lm_einops_pattern = "b p d_hidden_combined -> (b p) d_hidden_combined"
    if is_lm and not use_residual_stream:
        inputs = rearrange(inputs[1], lm_einops_pattern)
        outputs = rearrange(outputs[1], lm_einops_pattern)
        preactivations = rearrange(preactivations[1], lm_einops_pattern)
        C_0, C_1 = C_list[0][resid_stream_size:,:], C_list[1][mlp_layer_size:,:]
    elif is_lm and use_residual_stream:
        inputs = rearrange(torch.cat([x for x in inputs], dim=-1), lm_einops_pattern)
        outputs = rearrange(torch.cat([x for x in outputs], dim=-1), lm_einops_pattern)
    else:  # Inputs always tuple, and in this case we don't have LM
        inputs = torch.cat([x.detach().clone() for x in inputs], dim=-1)
        outputs = torch.cat([x.detach().clone() for x in outputs], dim=-1)

    operator: Float[Tensor, "batch d_hidden"] = torch.div(outputs, preactivations)

    for name, param in module_dict["sections.section_1.0"].named_parameters():
        if "W" in name: w_out = param

    ## Calculate H1
    h1_unit = calc_gram_matrix(inputs, DATASET_SIZE)
    H1 = torch.block_diag(*([h1_unit]*output_dim))

    ## Calculate H2
    rotated_fn_1 = inputs @ C_0
    C_0_trunc_size = rotated_fn_1.shape[-1]
    h2_dim = C_0_trunc_size * mlp_layer_size
    mlp_layer_size = operator.shape[-1]
    C_0_trunc_size = rotated_fn_1.shape[-1]

    # w_out_mul_itself = (w_out.T @ w_out)
    # w_factor = repeat(w_out_mul_itself, 'h w -> (h d1) (w d2)', d1=h2_dim, d2=h2_dim)

    ## ====== Takes up a lot of space =======
    # indices = torch.arange(h2_dim) % C_0_trunc_size
    # operator_repeated = operator[:, indices % mlp_layer_size]
    # rotated_fn_repeated = rotated_fn_1[:, indices]
    # bit_in_sum = operator_repeated.unsqueeze(2) * operator_repeated.unsqueeze(1) * bit_in_sum * rotated_fn_repeated.unsqueeze(2) * rotated_fn_repeated.unsqueeze(1)
    # H2 = torch.einsum('bhw->hw', bit_in_sum)

    ## Middling ground: block for loop allows space-time tradeoff
    ## Apparently this still uses too much memory
    # H2 = torch.zeros((h2_dim, h2_dim))
    # for i in range(C_0_trunc_size):
    #     for j in range(C_0_trunc_size):
    #         indices_operator = torch.arange(mlp_layer_size)
    #         operator_i = repeat(operator[:, indices_operator], 'b d_hidden -> b d_hidden d', d=mlp_layer_size)
    #         operator_j = repeat(operator[:, indices_operator], 'b d_hidden -> b d d_hidden', d=mlp_layer_size)
    #         rotated_fn_i = repeat(rotated_fn_1[:, i], 'b -> b d', d=mlp_layer_size)
    #         rotated_fn_j = repeat(rotated_fn_1[:, j], 'b -> b d', d=mlp_layer_size)

    #         # Vectorized operation within the block
    #         bit_in_sum = operator_i * operator_j * rotated_fn_i.unsqueeze(2) * rotated_fn_j.unsqueeze(1)

    #         # Reduction for the block
    #         H2_block = torch.einsum('bhw->hw', bit_in_sum)

    #         # Place the computed block in the appropriate position of H2
    #         H2[i * mlp_layer_size:(i + 1) * mlp_layer_size, j * mlp_layer_size:(j + 1) * mlp_layer_size] = H2_block

    ## Middling ground: block for loop allows space-time tradeoff - take 2, using longer for loops
    H2 = torch.zeros((h2_dim, h2_dim))
    for i in range(mlp_layer_size):
        for j in range(mlp_layer_size):
            operator_i = repeat(operator[:, i], 'b -> b d', d=C_0_trunc_size)
            operator_j = repeat(operator[:, j], 'b -> b d', d=C_0_trunc_size)
            rotated_fn_i = repeat(rotated_fn_1, 'b d_C_0 -> b d_C_0 d', d=C_0_trunc_size)
            rotated_fn_j = repeat(rotated_fn_1, 'b d_C_0 -> b d d_C_0', d=C_0_trunc_size)

            # Vectorized operation within the block
            bit_in_sum = rotated_fn_i * rotated_fn_j * operator_i.unsqueeze(2) * operator_j.unsqueeze(1)

            # Reduction for the block
            H2_block = torch.einsum('bhw->hw', bit_in_sum)

            # Place the computed block in the appropriate position of H2
            H2[i * C_0_trunc_size:(i + 1) * C_0_trunc_size, j * C_0_trunc_size:(j + 1) * C_0_trunc_size] = H2_block

            # Debug check
            if i == 13 and j == 16:
                assert all(bit_in_sum[:, 13, 16] == operator[:, 13] * operator[:, 16] * rotated_fn_1[:, 13] * rotated_fn_1[:, 16])

    ## ====== Takes up a lot of time =======
    # H2 = torch.zeros((h2_dim, h2_dim))
    # for i in range(h2_dim):
    #     for j in range(h2_dim):
    #         w_factor = w_out_mul_itself[i % C_0_trunc_size, j % C_0_trunc_size]
    #         bit_in_sum = operator[:,i % mlp_layer_size] * operator[:,j % mlp_layer_size] * rotated_fn_1[:,i % C_0_trunc_size] * rotated_fn_1[:,j % C_0_trunc_size]
    #         H2[i, j] = torch.einsum('b -> ', bit_in_sum * w_factor)

    ## Calculate mixed off-diagonal block

    del H1, H2, M12
    torch.cuda.empty_cache()
    gc.collect()

    _add_to_hooked_matrix(hooked_data, hook_name, f"H1 {hook_name}", H1.detach().cpu())
    _add_to_hooked_matrix(hooked_data, hook_name, f"H2 {hook_name}", H2.detach().cpu())
    _add_to_hooked_matrix(hooked_data, hook_name, f"M12 {hook_name}", M12.detach().cpu())