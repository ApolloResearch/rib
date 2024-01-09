"""Defines hook functions that are used in a HookedModel.

All forward hook functions must contain "forward" in their function names, and all pre-forward
hook functions must contain "pre_forward" in their function names. This is done to ensure that
the correct type of hook is registered to the module.

By default, a HookedModel passes in the arguments `hooked_data`, `hook_name`, and `data_key` to
each hook function. Therefore, these arguments must be included in the signature of each hook.

Otherwise, the hook function operates like a regular pytorch hook function.
"""

from typing import Any, Callable, Literal, Optional, Union

import torch
from jaxtyping import Float, Int
from torch import Tensor

from rib.linalg import (
    calc_edge_functional,
    calc_edge_squared,
    calc_gram_matrix,
    integrated_gradient_trapezoidal_norm,
)
from rib.models.sequential_transformer.components import AttentionOut
from rib.models.sequential_transformer.transformer import MultiSequential


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


def _append_to_hooked_list(
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: str,
    element_to_append: Any,
) -> None:
    """Append the given element to a hooked list. Creates the list if it doesn't exist.

    Args:
        hooked_data: Dictionary of hook data that will be updated.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
        element_to_append: Appended to hooked data.
    """
    hooked_data.setdefault(hook_name, {}).setdefault(data_key, [])
    hooked_data[hook_name][data_key].append(element_to_append)


InputActType = Union[
    tuple[Float[Tensor, "batch d_hidden"]],
    tuple[Float[Tensor, "batch pos d_hidden"]],
    tuple[Float[Tensor, "batch pos d_hidden1"], Float[Tensor, "batch pos d_hidden2"]],
]

OutputActType = Union[
    Float[Tensor, "batch d_hidden"],
    Float[Tensor, "batch pos d_hidden"],
    tuple[Float[Tensor, "batch pos d_hidden1"], Float[Tensor, "batch pos d_hidden2"]],
]


def _to_tuple(x: OutputActType) -> InputActType:
    return x if isinstance(x, tuple) else (x,)


def _get_bias_positions(
    module: torch.nn.Module, inputs: tuple[Tensor, ...]
) -> Int[Tensor, "segments"]:
    """
    This function finds the bias positions within a particular module's input.

    The code is super ugly, will be rendered obsolete by by fixing issue #231.
    """
    if isinstance(module, MultiSequential):
        next_module = list(module._modules.values())[0]
    else:
        next_module = module
    if isinstance(next_module, AttentionOut):
        raise NotImplementedError("there are many bias positions, could impliment if needed")

    cat_in_acts = torch.cat([x.detach().clone() for x in inputs], dim=-1)

    # if the inputs are of length [128, 128] bias positons might be 127 and/or 255
    segment_lens = torch.tensor([x.shape[-1] for x in inputs], device=cat_in_acts.device)
    potential_bias_positions = torch.cumsum(segment_lens, dim=0) - 1
    # Sometimes not all of our potential bias positons are guarenteed to be 1.
    # For instance, before Add one bias is 0. Before mlpact, one bias is act^{-1}(1).
    # We need to find at least a single bias position that is 1 for centering to work properly.
    # We thus filter potential bias positons for ones where the activation is 1.
    in_acts_at_bias = cat_in_acts[..., potential_bias_positions].mean(
        dim=(0 if cat_in_acts.ndim == 2 else (0, 1))
    )
    mean_acts_at_bias_pos_is_1 = (in_acts_at_bias - 1).abs() < 1e-3
    bias_positions = potential_bias_positions[mean_acts_at_bias_pos_is_1]
    return bias_positions


def dataset_mean_forward_hook_fn(
    module: torch.nn.Module,
    inputs: InputActType,
    output: OutputActType,
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    dataset_size: int,
) -> None:
    """Calculates the mean of the output activations and adds it to hooked_data.

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module (not used).
        output: Output of the module. Handles modules with one or two outputs of varying d_hiddens
            and positional indices.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
        dataset_size: Size of the dataset. Used to normalize the gram matrix.
    """

    assert isinstance(data_key, str), "data_key must be a string."
    out_acts = torch.cat([x.detach().clone() for x in _to_tuple(output)], dim=-1)
    out_acts_mean_contrib = out_acts.sum(dim=0) / dataset_size  # sum over batch
    if out_acts_mean_contrib.ndim == 2:
        out_acts_mean_contrib = out_acts_mean_contrib.mean(dim=0)  # mean over seqpos
    assert out_acts_mean_contrib.ndim == 1, f"mean must be 1D, shape={out_acts_mean_contrib.shape}"
    _add_to_hooked_matrix(hooked_data, hook_name, data_key, out_acts_mean_contrib)
    if "bias_positions" not in hooked_data[hook_name]:
        # bias positions is for the bias positions at the input, while this hook operates on the
        # output. It's also generally only called for the output of the model where there is no
        # bias position.
        hooked_data[hook_name]["bias_positions"] = torch.tensor(
            [], device=out_acts.device, dtype=torch.long
        )


def dataset_mean_pre_forward_hook_fn(
    module: torch.nn.Module,
    inputs: InputActType,
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    dataset_size: int,
) -> None:
    """Hook function for calculating the mean of the input activations.

    Adds activations/dataset_size into hooked_data[hook_name][data_key].

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Tuple of inputs to the module.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
        dataset_size: Size of the dataset. Used to normalize the means.
    """
    assert isinstance(data_key, str), "data_key must be a string."
    in_acts = torch.cat([x.detach().clone() for x in inputs], dim=-1)
    in_acts_mean_contrib = in_acts.sum(dim=0) / dataset_size  # sum over batch
    if in_acts_mean_contrib.ndim == 2:
        in_acts_mean_contrib = in_acts_mean_contrib.mean(dim=0)  # mean over seqpos
    assert in_acts_mean_contrib.ndim == 1, f"mean must be 1D, shape={in_acts_mean_contrib.shape}"
    _add_to_hooked_matrix(hooked_data, hook_name, data_key, in_acts_mean_contrib)
    if "bias_positions" not in hooked_data[hook_name]:
        hooked_data[hook_name]["bias_positions"] = _get_bias_positions(module=module, inputs=inputs)


def gram_forward_hook_fn(
    module: torch.nn.Module,
    inputs: InputActType,
    output: OutputActType,
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    dataset_size: int,
    shift: Optional[Float[Tensor, "d_hidden"]] = None,
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
        shift: added to the activations before gram matrix calculation. Used to center the data.
    """
    assert isinstance(data_key, str), "data_key must be a string."
    # Concat over the hidden dimension
    out_acts = torch.cat([x.detach().clone() for x in _to_tuple(output)], dim=-1)
    if shift is not None:
        out_acts += shift

    gram_matrix = calc_gram_matrix(out_acts, dataset_size=dataset_size)

    _add_to_hooked_matrix(hooked_data, hook_name, data_key, gram_matrix)


def gram_pre_forward_hook_fn(
    module: torch.nn.Module,
    inputs: InputActType,
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    dataset_size: int,
    shift: Optional[Float[Tensor, "d_hidden"]] = None,
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
        shift: added to the activations before gram matrix calculation. Used to center the data.
    """
    assert isinstance(data_key, str), "data_key must be a string."

    in_acts = torch.cat([x.detach().clone() for x in inputs], dim=-1)
    if shift is not None:
        in_acts += shift

    gram_matrix = calc_gram_matrix(in_acts, dataset_size=dataset_size)
    _add_to_hooked_matrix(hooked_data, hook_name, data_key, gram_matrix)


def rotate_pre_forward_hook_fn(
    module: torch.nn.Module,
    inputs: tuple[Float[Tensor, "batch d_hidden"]],
    rotation_matrix: Float[Tensor, "d_hidden d_hidden"],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: str,
    mode: Literal["modify", "cache"] = "modify",
) -> Optional[tuple[Float[Tensor, "batch d_hidden"], ...]]:
    """Hook function for rotating the input tensor to a module.

    The input is rotated by the specified rotation matrix.

    Handles multiple inputs by concatenating over the hidden dimension and then splitting the
    rotated tensor back into the original input sizes.

    Will either modify the activation within the forward pass (used for ablations) or cache the
    rotated result in hooked_data (used for get_rib_acts).

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module that we rotate.
        rotation_matrix: Rotation matrix to apply to the activations.
        hooked_data: Dictionary of hook data (not used for mode=modify).
        hook_name: Name of hook (not used for mode=modify).
        data_key: Name of data (not used for mode=modify).
        mode: if 'modify' return rotated inputs, if 'cache' store them but return None

    Returns:
        Rotated activations.
    """
    # Concatenate over the hidden dimension
    in_hidden_dims = [x.shape[-1] for x in inputs]
    in_acts = torch.cat(inputs, dim=-1)
    rotated = in_acts @ rotation_matrix
    if mode == "cache":
        _append_to_hooked_list(hooked_data, hook_name, data_key, rotated)
        return None
    else:
        assert mode == "modify"
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
    basis_formula: Literal["(1-alpha)^2", "(1-0)*alpha"] = "(1-0)*alpha",
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
        basis_formula: The formula to use for the integrated gradient. Must be one of
            "(1-alpha)^2" or "(1-0)*alpha". The former is the old (October) version while the
            latter is a new (November) version that should be used from now on. The latter makes
            sense especially in light of the new attribution (edge_formula="squared") but is
            generally good and does not change results much. Defaults to "(1-0)*alpha".
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
        basis_formula=basis_formula,
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
    module_hat: Callable[
        [Float[Tensor, "... in_hidden_trunc"], list[int]], Float[Tensor, "... out_hidden_trunc"]
    ],
    n_intervals: int,
    dataset_size: int,
    edge_formula: Literal["functional", "squared"] = "functional",
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
        module_hat: Partial function of rib.linalg.module_hat. Takes in f_in_hat and
            in_tuple_dims as arguments and calculates f_hat^{l} --> f_hat^{l+1}.
        n_intervals: Number of intervals to use for the trapezoidal rule. If 0, this is equivalent
            to taking a point estimate at alpha == 0.5.
        dataset_size: Size of the dataset. Used to normalize the gradients.
        edge_formula: The formula to use for the attribution. Must be one of "functional" or
            "squared". The former is the old (October) functional version, the latter is a new
            (November) version.
    """
    assert isinstance(data_key, str), "data_key must be a string."

    # Remove the pre-foward hook to avoid recursion when calculating the jacobian
    module._forward_pre_hooks.popitem()
    assert not module._forward_hooks, "Module has multiple forward hooks"

    in_tuple_dims = [x.shape[-1] for x in inputs]

    # We first concatenate the inputs over the hidden dimension
    # For each integral step, we calculate derivatives w.r.t alpha * in_acts @ C_in
    in_acts = torch.cat(inputs, dim=-1)
    f_hat = in_acts @ C_in
    edge = hooked_data[hook_name][data_key]

    if edge_formula == "functional":
        calc_edge_functional(
            module_hat=module_hat,
            f_in_hat=f_hat,
            in_tuple_dims=in_tuple_dims,
            edge=edge,
            dataset_size=dataset_size,
            n_intervals=n_intervals,
        )
    elif edge_formula == "squared":
        calc_edge_squared(
            module_hat=module_hat,
            f_in_hat=f_hat,
            in_tuple_dims=in_tuple_dims,
            edge=edge,
            dataset_size=dataset_size,
            n_intervals=n_intervals,
        )
    else:
        raise ValueError(
            f"edge_formula must be one of 'functional' or 'squared', got {edge_formula}"
        )


def acts_forward_hook_fn(
    module: torch.nn.Module,
    inputs: InputActType,
    output: OutputActType,
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
