"""Defines hook functions that are used in a HookedModel.

All forward hook functions must contain "forward" in their function names, and all pre-forward
hook functions must contain "pre_forward" in their function names. This is done to ensure that
the correct type of hook is registered to the module.

By default, a HookedModel passes in the arguments `hooked_data`, `hook_name`, and `data_key` to
each hook function. Therefore, these arguments must be included in the signature of each hook.

Otherwise, the hook function operates like a regular pytorch hook function.
"""

from typing import Any, Callable, Literal, Optional, Union

import einops
import torch
from jaxtyping import Float
from torch import Tensor

from rib.linalg import (
    calc_basis_integrated_gradient,
    calc_basis_jacobian,
    calc_edge_functional,
    calc_edge_squared,
    calc_edge_stochastic,
    calc_gram_matrix,
)


def _add_to_hooked_matrix(
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: str,
    hooked_matrix: Float[Tensor, "dim1 dim2"],
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
    tuple[Float[Tensor, "batch emb_in"]],
    tuple[Float[Tensor, "batch pos emb_in"]],
    tuple[Float[Tensor, "batch pos _"], ...],
]

OutputActType = Union[
    Float[Tensor, "batch emb_out"],
    Float[Tensor, "batch pos emb_out"],
    tuple[Float[Tensor, "batch pos _"], ...],
]


def _to_tuple(x: OutputActType) -> InputActType:
    return x if isinstance(x, tuple) else (x,)


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
        output: Output of the module. Handles modules with one or two outputs of varying sizes
            and with or without positional indices.
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


def gram_forward_hook_fn(
    module: torch.nn.Module,
    inputs: InputActType,
    output: OutputActType,
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    dataset_size: int,
    shift: Optional[Float[Tensor, "orig"]] = None,
) -> None:
    """Hook function for calculating and updating the gram matrix.

    The tuple of outputs is concatenated over the final dimension.

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module (not used).
        output: Output of the module. Handles modules with one or two outputs of varying sizes
            and with or without positional indices. If no positional indices, assumes one output.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
        dataset_size: Size of the dataset. Used to normalize the gram matrix.
        shift: added to the activations before gram matrix calculation. Used to center the data.
    """
    assert isinstance(data_key, str), "data_key must be a string."
    # Concat over the final dimension
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
    shift: Optional[Float[Tensor, "orig"]] = None,
) -> None:
    """Calculate the gram matrix for inputs with positional indices and add it to the global.

    The tuple of inputs is concatenated over the final dimension.

    Args:
        module: Module that the hook is attached to (not used).
        inputs: Inputs to the module. Handles modules with one or two inputs of varying sizes
            and with or without positional indices. If no positional indices, assumes one input.
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
    inputs: InputActType,
    rotation_matrix: Float[Tensor, "orig out"],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: str,
    mode: Literal["modify", "cache"] = "modify",
) -> Optional[
    Union[tuple[Float[Tensor, "batch _"], ...], tuple[Float[Tensor, "batch pos _"], ...]]
]:
    """Hook function for rotating the input tensor to a module.

    The input is rotated by the specified rotation matrix.

    Handles multiple inputs by concatenating over the final dimension and then splitting the
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
    # Concatenate over the embedding dimension
    in_emb_dims = [x.shape[-1] for x in inputs]
    in_acts = torch.cat(inputs, dim=-1)
    rotated = in_acts @ rotation_matrix
    if mode == "cache":
        _append_to_hooked_list(hooked_data, hook_name, data_key, rotated)
        return None
    else:
        assert mode == "modify"
        adjusted_inputs = tuple(torch.split(rotated, in_emb_dims, dim=-1))
        return adjusted_inputs


def get_acts_pre_forward_hook_fn(
    module: torch.nn.Module,
    inputs: InputActType,
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
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
    _add_to_hooked_matrix(hooked_data, hook_name, data_key, in_acts)


def edge_ablation_forward_hook_fn(
    module: torch.nn.Module,
    inputs: InputActType,
    output: OutputActType,
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    edge_mask: Float[Tensor, "out in"],
    in_C: Float[Tensor, "orig rib"],
    in_C_inv: Float[Tensor, "rib orig"],
    out_C: Float[Tensor, "orig rib"],
    out_C_inv: Float[Tensor, "rib orig"],
):
    """
    Intervenes on the forward pass of the model by zero-ablating some edges.

    In particular, calculates the output activations in the RIB basis. The activation in each output
    RIB direction will be computed separately by ablating some set of RIB directions in the input.

    Args:
        module: Module that the hook is attached to.
        inputs: Inputs to the module. Handles modules with one or two inputs of varying d_hiddens
            and positional indices. If no positional indices, assumes one input.
        output: Output of the module. Handles modules with one or two outputs of varying d_hiddens
            and positional indices. If no positional indices, assumes one output.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of 2nd-level keys to store in `hooked_data`.
        edge_mask: Mask of edges to ablate. Has shape (out_hidden, in_hidden).
        in_C: The C matrix for the pre-edge layer.
        in_C_inv: The inverse of the C matrix for the pre-edge layer.
        out_C: The C matrix for the post-edge layer.
        out_C_inv: The inverse of the C matrix for the post-edge layer.
    """
    # Remove this forward hook from the module to avoid recursion.
    module._forward_hooks.popitem()
    assert not module._forward_hooks, "Module has multiple forward hooks"

    # prep activations and matricies
    in_acts = torch.cat([x for x in inputs], dim=-1)
    orig_out_acts = torch.cat([x for x in _to_tuple(output)], dim=-1)

    in_shape = [part.shape[-1] for part in inputs]
    out_shape = [part.shape[-1] for part in _to_tuple(output)]

    in_rib_acts: Float[Tensor, "... rib"] = in_acts @ in_C
    orig_out_rib_acts = orig_out_acts @ out_C

    # We set all output RIB activations to zero and compute them one set at a time.
    new_out_rib_acts: Float[Tensor, "... rib"] = torch.zeros_like(orig_out_rib_acts)
    # In particular we can compute all output directions that share the same unablated input nodes.
    # Below we iterate over sets of input nodes (`in_mask`) and the output nodes that share this set
    # (`out_mask`). Worst case, we need to compute run this for every output node.
    unique_in_masks, out_node_to_in_mask_map = edge_mask.unique(dim=0, return_inverse=True)
    for i, in_mask in enumerate(unique_in_masks):
        # find output nodes that match in_mask
        out_mask = out_node_to_in_mask_map == i
        # project out some rib dirs in input
        ablated_in_rib_acts = torch.where(in_mask, in_rib_acts, 0.0)
        ablated_in_acts = ablated_in_rib_acts @ in_C_inv
        # pass through the hooked module
        raw_out = module(*ablated_in_acts.split(in_shape, dim=-1))
        ablated_out_acts = torch.cat(_to_tuple(raw_out), dim=-1)
        # rotate into rib and get the right component
        ablated_out_rib_acts_in_dir = ablated_out_acts @ out_C[:, out_mask]
        new_out_rib_acts[..., out_mask] = ablated_out_rib_acts_in_dir

    # rotate output RIB acts back to neuron basis
    new_out_acts: Float[Tensor, "... orig"] = new_out_rib_acts @ out_C_inv
    if isinstance(output, tuple):
        return new_out_acts.split(out_shape, dim=-1)
    else:
        return new_out_acts


def M_dash_and_Lambda_dash_pre_forward_hook_fn(
    module: torch.nn.Module,
    inputs: InputActType,
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    C_out: Optional[Float[Tensor, "orig_out rib_out"]],
    n_intervals: int,
    integration_method: Literal["trapezoidal", "gauss-legendre", "gradient"],
    dataset_size: int,
    M_dtype: torch.dtype = torch.float64,
    Lambda_einsum_dtype: torch.dtype = torch.float64,
    basis_formula: Literal["jacobian", "(1-alpha)^2", "(1-0)*alpha"] = "(1-0)*alpha",
    n_stochastic_sources_pos: Optional[int] = None,
    n_stochastic_sources_hidden: Optional[int] = None,
    out_dim_n_chunks: int = 1,
    out_dim_chunk_idx: int = 0,
) -> None:
    """Hook function for accumulating the M' and Lambda' matrices.

    Args:
        module: Module that the hook is attached to.
        inputs: Inputs to the module. Handles modules with one or two inputs of varying origs
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
        n_stochastic_sources: Stochastic sources for i and t. If None (default).

    """
    assert isinstance(data_key, list), "data_key must be a list of strings."
    assert len(data_key) == 2, "data_key must be a list of length 2 to store M' and Lambda'."
    # Remove the pre foward hook to avoid recursion when calculating the jacobian
    module._forward_pre_hooks.popitem()
    assert not module._forward_hooks, "Module has multiple forward hooks"

    if basis_formula == "(1-alpha)^2" or basis_formula == "(1-0)*alpha":
        if not (out_dim_n_chunks == 1 and out_dim_chunk_idx == 0):
            raise NotImplementedError

        in_grads = calc_basis_integrated_gradient(
            module=module,
            inputs=inputs,
            C_out=C_out,
            n_intervals=n_intervals,
            integration_method=integration_method,
            basis_formula=basis_formula,
        )
        in_dtype = in_grads.dtype

        has_pos = inputs[0].dim() == 3

        einsum_pattern = "bpj,bpJ->jJ" if has_pos else "bj,bJ->jJ"
        normalization_factor = in_grads.shape[1] * dataset_size if has_pos else dataset_size

        with torch.inference_mode():
            M_dash = torch.einsum(
                einsum_pattern,
                in_grads.to(M_dtype),
                in_grads.to(M_dtype),
            )
            M_dash /= normalization_factor
            # Concatenate the inputs over the final dimension
            in_acts = torch.cat(inputs, dim=-1)
            Lambda_dash = torch.einsum(
                einsum_pattern,
                in_grads.to(Lambda_einsum_dtype),
                in_acts.to(Lambda_einsum_dtype),
            )
            Lambda_dash /= normalization_factor
            Lambda_dash = Lambda_dash.to(in_dtype)

            _add_to_hooked_matrix(hooked_data, hook_name, data_key[0], M_dash)
            _add_to_hooked_matrix(hooked_data, hook_name, data_key[1], Lambda_dash)

            assert (
                Lambda_dash.std() > 0
            ), "Lambda_dash cannot be all zeros otherwise everything will be truncated"

    elif basis_formula == "jacobian":
        # in_grads.shape: batch, i (out_hidden), t (out_pos), s (in_pos), j/jprime (in_hidden)
        in_grads = calc_basis_jacobian(
            module=module,
            inputs=inputs,
            C_out=C_out,
            n_intervals=n_intervals,
            integration_method=integration_method,
            n_stochastic_sources_pos=n_stochastic_sources_pos,
            n_stochastic_sources_hidden=n_stochastic_sources_hidden,
            out_dim_n_chunks=out_dim_n_chunks,
            out_dim_chunk_idx=out_dim_chunk_idx,
        )
        has_pos = inputs[0].dim() == 3
        if has_pos:
            einsum_pattern = "r batch s j, r batch s jprime -> j jprime"
            in_pos_size = inputs[0].shape[1]
            normalization_factor = in_pos_size * dataset_size
            # It is intentional that normalization_factor is multiplied by both,
            # n_stochastic_sources_pos and n_stochastic_sources_hidden when both are present. The
            # total amount of stochastic sources in that case is the product.
            if n_stochastic_sources_pos is not None:
                normalization_factor *= n_stochastic_sources_pos
            if n_stochastic_sources_hidden is not None:
                normalization_factor *= n_stochastic_sources_hidden
        else:
            assert (
                n_stochastic_sources_pos is None and n_stochastic_sources_hidden is None
            ), "Stochastic sources only supported in case of has_pos=True"
            einsum_pattern = "batch i j, batch i jprime -> j jprime"
            in_pos_size = 1
            normalization_factor = dataset_size

        with torch.inference_mode():
            # M_dash.shape: j jprime
            M_dash = einops.einsum(in_grads.to(M_dtype), in_grads.to(M_dtype), einsum_pattern)
            M_dash /= normalization_factor
            # In the jacobian basis, Lambda is not computed here but from the M eigenvalues later.
            # Set a placeholder to maintain the same function signature.
            Lambda_dash = torch.tensor(torch.nan)

            _add_to_hooked_matrix(hooked_data, hook_name, data_key[0], M_dash)
            _add_to_hooked_matrix(hooked_data, hook_name, data_key[1], Lambda_dash)
    else:
        raise ValueError(
            f"basis_formula must be one of '(1-alpha)^2', '(1-0)*alpha', or 'jacobian', got {basis_formula}"
        )


def interaction_edge_pre_forward_hook_fn(
    module: torch.nn.Module,
    inputs: InputActType,
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    C_in: Float[Tensor, "orig_in rib_in"],
    module_hat: Callable[[Float[Tensor, "... rib_in"], list[int]], Float[Tensor, "... rib_out"]],
    n_intervals: int,
    integration_method: Literal["trapezoidal", "gauss-legendre", "gradient"],
    dataset_size: int,
    out_dim_start_idx: int,
    out_dim_end_idx: int,
    edge_formula: Literal["functional", "squared"] = "squared",
    n_stochastic_sources: Optional[int] = None,
) -> None:
    """Hook function for accumulating the edges (denoted E_hat) of the RIB graph.

    For calculating the Jacobian, we need to run the inputs through the module. Unfortunately,
    this causes an infinite recursion because the module has a hook which runs this function. To
    avoid this, we (hackily) remove the hook before running the module and then add it back after.

    The trapezoidal rule is used to approximate the integrated gradient. If n_intervals == 0, the
    integrated gradient effectively takes a point estimate for the integral at alpha == 0.5.

    Args:
        module: Module that the hook is attached to.
        inputs: Inputs to the module. Handles modules with one or two inputs of varying origs
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
        out_dim_start_idx: The index of the first output dimension to calculate.
        out_dim_end_idx: The index of the last output dimension to calculate.
        edge_formula: The formula to use for the attribution.
            - "functional" is the old (October 23) functional version
            - "squared" is the version which iterates over the output dim and output pos dim
        n_stochastic_sources: The number of stochastic sources for positional dimension
            (approximation). Defaults to None.
    """
    assert isinstance(data_key, str), "data_key must be a string."

    # Remove the pre-foward hook to avoid recursion when calculating the jacobian
    module._forward_pre_hooks.popitem()
    assert not module._forward_hooks, "Module has multiple forward hooks"

    in_tuple_dims = [x.shape[-1] for x in inputs]

    # We first concatenate the inputs over the final dimension
    # For each integral step, we calculate derivatives w.r.t alpha * in_acts @ C_in
    in_acts = torch.cat(inputs, dim=-1)
    f_hat = in_acts @ C_in
    edge = hooked_data[hook_name][data_key]

    tqdm_desc = f"Integration steps (alphas) for {hook_name}"
    if edge_formula == "functional":
        calc_edge_functional(
            module_hat=module_hat,
            f_in_hat=f_hat,
            in_tuple_dims=in_tuple_dims,
            edge=edge,
            dataset_size=dataset_size,
            n_intervals=n_intervals,
            integration_method=integration_method,
            tqdm_desc=tqdm_desc,
        )
    elif edge_formula == "squared":
        if n_stochastic_sources is None:
            calc_edge_squared(
                module_hat=module_hat,
                f_in_hat=f_hat,
                in_tuple_dims=in_tuple_dims,
                edge=edge,
                dataset_size=dataset_size,
                n_intervals=n_intervals,
                out_dim_start_idx=out_dim_start_idx,
                out_dim_end_idx=out_dim_end_idx,
                integration_method=integration_method,
                tqdm_desc=tqdm_desc,
            )
        else:
            assert f_hat.dim() == 3, "f_hat must have a position dimension to use stochastic noise."
            calc_edge_stochastic(
                module_hat=module_hat,
                f_in_hat=f_hat,
                in_tuple_dims=in_tuple_dims,
                edge=edge,
                dataset_size=dataset_size,
                n_intervals=n_intervals,
                integration_method=integration_method,
                n_stochastic_sources=n_stochastic_sources,
                out_dim_start_idx=out_dim_start_idx,
                out_dim_end_idx=out_dim_end_idx,
                tqdm_desc=tqdm_desc,
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
        output: Output of the module. Handles modules with one or two outputs of varying sizes
            and with or without positional indices. If no positional indices, assumes one output.
        hooked_data: Dictionary of hook data.
        hook_name: Name of hook. Used as a 1st-level key in `hooked_data`.
        data_key: Name of data. Used as a 2nd-level key in `hooked_data`.
    """
    assert isinstance(data_key, str), "data_key must be a string."
    outputs = output if isinstance(output, tuple) else (output,)
    detached_outputs = [x.detach().cpu() for x in outputs]
    # Store the output activations
    hooked_data[hook_name] = {data_key: detached_outputs}
