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

import einops
import torch
from fancy_einsum import einsum
from jaxtyping import Float
from torch import Tensor

from rib.linalg import (
    calc_gram_matrix,
    integrated_gradient_trapezoidal_jacobian,
    integrated_gradient_trapezoidal_norm,
)
from rib.models.sequential_transformer.components import AttentionOut
from rib.utils import module_hat


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
        tuple[Float[Tensor, "batch pos d_hidden1"], Float[Tensor, "batch pos d_hidden2"]],
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


def attn_scores_pre_forward_hook(
    module: torch.nn.Module,
    inputs: tuple[Float[Tensor, "batch pos head_index_d_head"], ...],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
) -> None:
    """Calculate and store the attention scores.

    This should only be applied to the AttentionOut module.

    Note that this function overwrites hooked_data[hook_name] each time it is called since it is
    expected to only be used on a single batch.

    Args:
        module: Module that the hook is attached to.
        inputs: Inputs to the module. The first input is the residual, and the remaining inputs
            are the q, k, and v tensors.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
    """

    assert isinstance(module, AttentionOut), "This hook can only be applied to AttentionOut."
    _, q, k, v = inputs
    # Separate the last dimension into head_index and d_head (undo the operation from AttentionIn)
    q = einops.rearrange(
        q,
        "... pos (head_index d_head) -> ... pos head_index d_head",
        head_index=module.cfg.n_heads,
    )
    k = einops.rearrange(
        k,
        "... pos (head_index d_head) -> ... pos head_index d_head",
        head_index=module.cfg.n_heads,
    )
    v = einops.rearrange(
        v,
        "... pos (head_index d_head_v) -> ... pos head_index d_head_v",
        head_index=module.cfg.n_heads,
    )

    in_dtype = v.dtype

    if in_dtype not in [torch.float32, torch.float64]:
        # If using 16 bits, increase the precision to avoid numerical instabilities
        q = q.to(torch.float32)
        k = k.to(torch.float32)
    attn_scores = (
        einsum(
            "... query_pos head_index d_head, \
                    ... key_pos head_index d_head \
                    -> ... head_index query_pos key_pos",
            q,
            k,
        )
        / module.attn_scale
    )  # [..., head_index, query_pos, key_pos]

    attn_scores = module.apply_causal_mask(attn_scores)  # [..., head_index, query_pos, key_pos]

    hooked_data[hook_name] = {data_key: attn_scores}


def rotate_pre_forward_hook_fn(
    module: torch.nn.Module,
    inputs: tuple[Float[Tensor, "batch d_hidden"]],
    rotation_matrix: Float[Tensor, "d_hidden d_hidden"],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
) -> tuple[Float[Tensor, "batch d_hidden"], ...]:
    """Hook function for rotating the input tensor to a module.

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

    in_tuple_dims = [x.shape[-1] for x in inputs]

    # Compute f^{l+1}(x) to which the derivative is not applied.
    with torch.inference_mode():
        # f_in_hat @ C_in_pinv does not give exactly f due to C and C_in_pinv being truncated
        f_in_adjusted: Float[Tensor, "... in_hidden_combined_trunc"] = f_hat @ C_in_pinv
        input_tuples = torch.split(f_in_adjusted, in_tuple_dims, dim=-1)

        output_const = module(*tuple(x.detach().clone() for x in input_tuples))
        outputs_const = (output_const,) if isinstance(output_const, torch.Tensor) else output_const
        outputs_const = torch.cat(outputs_const, dim=-1)

    has_pos = f_hat.dim() == 3

    jac_out = hooked_data[hook_name][data_key]

    module_hat_partial = partial(
        module_hat,
        module=module,
        C_in_pinv=C_in_pinv,
        C_out=C_out,
        in_tuple_dims=in_tuple_dims,
    )

    integrated_gradient_trapezoidal_jacobian(
        module_hat=module_hat_partial,
        module=module,
        f_in_hat=f_hat,
        n_intervals=n_intervals,
        jac_out=jac_out,
        dataset_size=dataset_size,
        outputs_const=outputs_const,
        C_out=C_out,
        C_in_pinv=C_in_pinv,
        in_tuple_dims=in_tuple_dims,
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
