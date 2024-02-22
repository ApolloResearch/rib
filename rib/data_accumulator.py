"""Functions that apply hooks and accumulate data when passing batches through a model."""

from functools import partial
from typing import TYPE_CHECKING, Annotated, Literal, Optional, Union

import torch
from jaxtyping import Float
from pydantic import AfterValidator, BaseModel, ConfigDict
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from rib.distributed_utils import sum_across_processes
from rib.hook_fns import (
    M_dash_and_Lambda_dash_pre_forward_hook_fn,
    acts_forward_hook_fn,
    acts_pre_forward_hook_fn,
    dataset_mean_forward_hook_fn,
    dataset_mean_pre_forward_hook_fn,
    gram_forward_hook_fn,
    gram_pre_forward_hook_fn,
    interaction_edge_pre_forward_hook_fn,
    weighted_gram_backward_hook_fn,
    weighted_gram_pre_backward_hook_fn,
)
from rib.hook_manager import Hook, HookedModel
from rib.linalg import module_hat
from rib.log import logger
from rib.models.utils import get_model_attr
from rib.types import IntegrationMethod
from rib.utils import check_device_is_cpu, get_chunk_indices, lm_cross_entropy_loss

if TYPE_CHECKING:  # Prevent circular import to import type annotations
    from rib.interaction_algos import InteractionRotation


class Edges(BaseModel):
    """Stores a matrix of edges of shape (rib_out, rib_in) between two node layers."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    in_node_layer: str
    out_node_layer: str
    E_hat: Annotated[Float[Tensor, "rib_out rib_in"], AfterValidator(check_device_is_cpu)]


def run_dataset_through_model(
    hooked_model: HookedModel,
    dataloader: DataLoader,
    hooks: list[Hook],
    dtype: torch.dtype,
    device: str = "cuda",
    use_tqdm: bool = False,
    tqdm_desc: Optional[str] = None,
) -> None:
    """Simply pass all batches through a hooked model."""
    assert len(hooks) > 0, "Hooks have not been applied to this model."
    loader: Union[tqdm, DataLoader]
    if use_tqdm:
        desc = "Batches through entire model" if tqdm_desc is None else tqdm_desc
        loader = tqdm(dataloader, total=len(dataloader), desc=desc)
    else:
        loader = dataloader

    for batch in loader:
        data, _ = batch
        data = data.to(device=device)
        # Change the dtype unless the inputs are integers (e.g. like they are for LMs)
        if data.dtype not in [torch.int64, torch.int32]:
            data = data.to(dtype=dtype)

        hooked_model(data, hooks=hooks)


@torch.inference_mode()
def collect_dataset_means(
    hooked_model: HookedModel,
    module_names: list[str],
    data_loader: DataLoader,
    device: str,
    dtype: torch.dtype,
    collect_output_dataset_means: bool = True,
    hook_names: Optional[list[str]] = None,
) -> dict[str, Float[Tensor, "orig"]]:
    """Collect the mean input activation for each module on the dataset.

    Also returns the positions of the bias terms in each input activation. The mean should be
    one at these positions.

    Can also collect the mean output of the model if `collect_output_dataset_means` is True.

    Args:
        hooked_model: The hooked model.
        module_names: The names of the modules to collect gram matrices for. Often section ids.
        data_loader: The pytorch data loader.
        device: The device to run the model on.
        dtype: The data type to use for model computations.
        collect_output_dataset_means: Whether to collect the mean output of the final module.
        hook_names: Used to store the gram matrices in the hooked model. Often module ids.

    Returns:
        Dataset means, a dictionary from hook_names to mean tensors of shape (d_hidden,)
    """
    assert len(module_names) > 0, "No modules specified."
    if hook_names is not None:
        assert len(hook_names) == len(module_names), "Must specify a hook name for each module."
    else:
        hook_names = module_names

    if not hooked_model.model.has_folded_bias:
        logger.warning("model does not have folded bias, ")

    dataset_size = len(data_loader.dataset)  # type: ignore
    dataset_mean_hooks: list[Hook] = []
    # Add input hooks
    for module_name, hook_name in zip(module_names, hook_names):
        dataset_mean_hooks.append(
            Hook(
                name=hook_name,
                data_key="dataset_mean",
                fn=dataset_mean_pre_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={"dataset_size": dataset_size},
            )
        )
    if collect_output_dataset_means:
        # Add hook to collect model output
        dataset_mean_hooks.append(
            Hook(
                name="output",
                data_key="dataset_mean",
                fn=dataset_mean_forward_hook_fn,
                module_name=module_names[-1],
                fn_kwargs={"dataset_size": dataset_size},
            )
        )

    run_dataset_through_model(
        hooked_model, data_loader, dataset_mean_hooks, dtype=dtype, device=device, use_tqdm=True
    )

    dataset_mean: dict[str, Float[Tensor, "orig"]] = {
        hook_name: hooked_model.hooked_data[hook_name]["dataset_mean"]
        for hook_name in hooked_model.hooked_data
    }
    hooked_model.clear_hooked_data()

    expected_keys = (
        set(hook_names + ["output"]) if collect_output_dataset_means else set(hook_names)
    )
    assert set(dataset_mean.keys()) == expected_keys, (
        f"Gram matrix keys not the same as the module names that were hooked. "
        f"Expected: {expected_keys}, got: {set(dataset_mean.keys())}"
    )
    return dataset_mean


@torch.inference_mode()
def collect_gram_matrices(
    hooked_model: HookedModel,
    module_names: list[str],
    data_loader: DataLoader,
    device: str,
    dtype: torch.dtype,
    collect_output_gram: bool = True,
    hook_names: Optional[list[str]] = None,
    means: Optional[dict[str, Float[Tensor, "orig"]]] = None,
) -> dict[str, Float[Tensor, "orig orig"]]:
    """Collect gram matrices for the module inputs and optionally the output of the final module.

    We use pre_forward hooks for the input to each module. If `collect_output_gram` is True, we
    also collect the gram matrix for the output of the final module using a forward hook.

    Will collect correlation matrices (that is, gram matrices of centered activations) if `means` is
    provided. In this case, `bias_positions` must also be provided. The bias positions will not be
    centered.

    Args:
        hooked_model: The hooked model.
        module_names: The names of the modules to collect gram matrices for. Can be any valid
            pytorch module in hooked_model.model. These typically correspond to section_names (e.g.
            "sections.section_0") when the model is a SequentialTransformer or raw layers (e.g.
            "layers.2") when the model is an MLP.
        data_loader: The pytorch data loader.
        device: The device to run the model on.
        dtype: The data type to use for model computations.
        collect_output_gram: Whether to collect the gram matrix for the output of the final module.
        hook_names: Used to store the gram matrices in the hooked model.
        means: A dictionary of mean activations for each module. The keys are the hook names. If
            not none, will be used to center the activations when computing the gram matrices.

    Returns:
        A dictionary of gram matrices, where the keys are the hook names (a.k.a. node layer names)
    """
    assert len(module_names) > 0, "No modules specified."
    if hook_names is not None:
        assert len(hook_names) == len(module_names), "Must specify a hook name for each module."
    else:
        hook_names = module_names

    dataset_size = len(data_loader.dataset)  # type: ignore
    gram_hooks: list[Hook] = []
    # Add input hooks
    for module_name, hook_name in zip(module_names, hook_names):
        shift: Optional[Float[Tensor, "orig"]] = None
        if means is not None and hook_name in means:
            shift = -means[hook_name]
            shift[-1] = 0.0  # don't shift the final bias pos
        gram_hooks.append(
            Hook(
                name=hook_name,
                data_key="gram",
                fn=gram_pre_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={"dataset_size": dataset_size, "shift": shift},
            )
        )
    if collect_output_gram:
        # Add hook to collect model output
        gram_hooks.append(
            Hook(
                name="output",
                data_key="gram",
                fn=gram_forward_hook_fn,
                module_name=module_names[-1],
                fn_kwargs={
                    "dataset_size": dataset_size,
                    # we don't need to care about bias positions in the output
                    "shift": -means["output"] if means is not None else None,
                },
            )
        )

    run_dataset_through_model(
        hooked_model, data_loader, gram_hooks, dtype=dtype, device=device, use_tqdm=True
    )

    gram_matrices: dict[str, Float[Tensor, "orig orig"]] = {
        hook_name: hooked_model.hooked_data[hook_name]["gram"]
        for hook_name in hooked_model.hooked_data
    }
    hooked_model.clear_hooked_data()

    expected_gram_keys = set(hook_names + ["output"]) if collect_output_gram else set(hook_names)
    assert set(gram_matrices.keys()) == expected_gram_keys, (
        f"Gram matrix keys not the same as the module names that were hooked. "
        f"Expected: {expected_gram_keys}, got: {set(gram_matrices.keys())}"
    )

    return gram_matrices


def collect_weighted_gram_matrices(
    hooked_model: HookedModel,
    module_names: list[str],
    data_loader: DataLoader,
    device: str,
    dtype: torch.dtype,
    collect_output_gram: bool = True,
    hook_names: Optional[list[str]] = None,
    means: Optional[dict[str, Float[Tensor, "orig"]]] = None,
) -> dict[str, Float[Tensor, "orig orig"]]:
    """Collect gram matrices for the module inputs and optionally the output of the final module.

    We use pre_forward hooks for the input to each module. If `collect_output_gram` is True, we
    also collect the gram matrix for the output of the final module using a forward hook.

    Will collect correlation matrices (that is, gram matrices of centered activations) if `means` is
    provided. In this case, `bias_positions` must also be provided. The bias positions will not be
    centered.

    Args:
        hooked_model: The hooked model.
        module_names: The names of the modules to collect gram matrices for. Can be any valid
            pytorch module in hooked_model.model. These typically correspond to section_names (e.g.
            "sections.section_0") when the model is a SequentialTransformer or raw layers (e.g.
            "layers.2") when the model is an MLP.
        data_loader: The pytorch data loader.
        device: The device to run the model on.
        dtype: The data type to use for model computations.
        collect_output_gram: Whether to collect the gram matrix for the output of the final module.
        hook_names: Used to store the gram matrices in the hooked model.
        means: A dictionary of mean activations for each module. The keys are the hook names. If
            not none, will be used to center the activations when computing the gram matrices.

    Returns:
        A dictionary of gram matrices, where the keys are the hook names (a.k.a. node layer names)
    """
    assert len(module_names) > 0, "No modules specified."
    if hook_names is not None:
        assert len(hook_names) == len(module_names), "Must specify a hook name for each module."
    else:
        hook_names = module_names

    dataset_size = len(data_loader.dataset)  # type: ignore
    activation_hooks: list[Hook] = []
    gram_hooks: list[Hook] = []
    # Add input hooks
    for module_name, hook_name in zip(module_names, hook_names):
        shift: Optional[Float[Tensor, "orig"]] = None
        if means is not None and hook_name in means:
            shift = -means[hook_name]
            shift[-1] = 0.0  # don't shift the final bias pos
        activation_hooks.append(
            Hook(
                name=hook_name,
                data_key="activations",
                fn=acts_pre_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={},
            )
        )
        gram_hooks.append(
            Hook(
                name=hook_name,
                data_key="gram",
                fn=weighted_gram_backward_hook_fn,
                module_name=module_name,
                fn_kwargs={"dataset_size": dataset_size, "shift": shift},
            )
        )
    if collect_output_gram:
        # Add hook to collect model output
        activation_hooks.append(
            Hook(
                name="output",
                data_key="activations",
                fn=acts_forward_hook_fn,
                module_name=module_names[-1],
                fn_kwargs={},
            )
        )
        gram_hooks.append(
            Hook(
                name="output",
                data_key="gram",
                fn=weighted_gram_pre_backward_hook_fn,
                module_name=module_names[-1],
                fn_kwargs={
                    # we don't need to care about bias positions in the output
                    "shift": -means["output"] if means is not None else None
                },
            )
        )

    hooked_model.add_hooks(gram_hooks)

    desc = "Batches through entire model"
    loader = tqdm(data_loader, total=len(data_loader), desc=desc)
    for batch in loader:
        data, labels = batch
        data = data.to(device=device)
        labels = labels.to(device=device)
        # Change the dtype unless the inputs are integers (e.g. like they are for LMs)
        if data.dtype not in [torch.int64, torch.int32]:
            data = data.to(dtype=dtype)
        output = hooked_model(data, hooks=activation_hooks)[0]
        loss = lm_cross_entropy_loss(output, labels)
        loss.backward()
    hooked_model.remove_backward_hooks()

    gram_matrices: dict[str, Float[Tensor, "orig orig"]] = {
        hook_name: hooked_model.hooked_data[hook_name]["gram"]
        for hook_name in hooked_model.hooked_data
    }
    hooked_model.clear_hooked_data()

    expected_gram_keys = set(hook_names + ["output"]) if collect_output_gram else set(hook_names)
    assert set(gram_matrices.keys()) == expected_gram_keys, (
        f"Gram matrix keys not the same as the module names that were hooked. "
        f"Expected: {expected_gram_keys}, got: {set(gram_matrices.keys())}"
    )

    return gram_matrices


def collect_M_dash_and_Lambda_dash(
    C_out: Optional[Float[Tensor, "orig_out rib_out"]],
    hooked_model: HookedModel,
    n_intervals: int,
    integration_method: Literal["trapezoidal", "gauss-legendre", "gradient"],
    data_loader: DataLoader,
    module_name: str,
    dtype: torch.dtype,
    device: str,
    hook_name: Optional[str] = None,
    M_dtype: torch.dtype = torch.float64,
    Lambda_einsum_dtype: torch.dtype = torch.float64,
    basis_formula: Literal["jacobian", "(1-alpha)^2", "(1-0)*alpha"] = "(1-0)*alpha",
    n_stochastic_sources_pos: Optional[int] = None,
    n_stochastic_sources_hidden: Optional[int] = None,
    out_dim_n_chunks: int = 1,
    out_dim_chunk_idx: int = 0,
) -> tuple[Float[Tensor, "orig_in orig_in"], Float[Tensor, "orig_in orig_in"]]:
    """Collect the matrices M' and Lambda' for the input to the module specifed by `module_name`.

    We accumulate the matrices, M' and Lambda' for each batch. To do this, we apply
    a hook to the provided module. This hook will accumulate both matrices over the batches.

    Args:
        C_out: The rotation matrix for the next layer.
        hooked_model: The hooked model.
        n_intervals: The number of integrated gradient intervals to use.
        data_loader: The data loader.
        module_name: The name of the module whose inputs are the node layer we collect the matrices
            M' and Lambda' for.
        dtype: The data type to use for model computations.
        device: The device to run the model on.
        hook_name: The name of the hook to use to store the matrices in the hooked model.
        M_dtype: The data type to use for the M_dash matrix. Needs to be
            float64 for Pythia-14m (empirically). Defaults to float64.
        Lambda_einsum_dtype: The data type to use for the einsum computing batches for the
            Lambda_dash matrix. Does not affect the output, only used for the einsum within
            M_dash_and_Lambda_dash_pre_forward_hook_fn. Needs to be float64 on CPU but float32 was
            fine on GPU. Defaults to float64.
        basis_formula: The formula to use for the integrated gradient. Must be one of
            "(1-alpha)^2" or "(1-0)*alpha". The former is the old (October) version while the
            latter is a new (November) version that should be used from now on. The latter makes
            sense especially in light of the new attribution (edge_formula="squared") but is
            generally good and does not change results much. Defaults to "(1-0)*alpha".
    Returns:
        A tuple containing M' and Lambda'.
    """
    if hook_name is None:
        hook_name = module_name

    interaction_hook = Hook(
        name=hook_name,
        data_key=["M_dash", "Lambda_dash"],
        fn=M_dash_and_Lambda_dash_pre_forward_hook_fn,
        module_name=module_name,
        fn_kwargs={
            "C_out": C_out,
            "n_intervals": n_intervals,
            "integration_method": integration_method,
            "dataset_size": len(data_loader.dataset),  # type: ignore
            "M_dtype": M_dtype,
            "Lambda_einsum_dtype": Lambda_einsum_dtype,
            "basis_formula": basis_formula,
            "n_stochastic_sources_pos": n_stochastic_sources_pos,
            "n_stochastic_sources_hidden": n_stochastic_sources_hidden,
            "out_dim_n_chunks": out_dim_n_chunks,
            "out_dim_chunk_idx": out_dim_chunk_idx,
        },
    )

    run_dataset_through_model(
        hooked_model,
        data_loader,
        hooks=[interaction_hook],
        dtype=dtype,
        device=device,
        use_tqdm=True,
        tqdm_desc=f"Batches through model for {hook_name}",
    )

    M_dash = hooked_model.hooked_data[hook_name]["M_dash"]
    Lambda_dash = hooked_model.hooked_data[hook_name]["Lambda_dash"]
    hooked_model.clear_hooked_data()

    if out_dim_n_chunks > 1:
        M_dash = sum_across_processes(M_dash)
        Lambda_dash = sum_across_processes(Lambda_dash)

    return M_dash, Lambda_dash


def collect_interaction_edges(
    interaction_rotations: list["InteractionRotation"],
    hooked_model: HookedModel,
    n_intervals: int,
    integration_methods: list[IntegrationMethod],
    section_names: list[str],
    data_loader: DataLoader,
    dtype: torch.dtype,
    device: str,
    data_set_size: Optional[int] = None,
    edge_formula: Literal["functional", "squared"] = "squared",
    n_stochastic_sources: Optional[int] = None,
    out_dim_n_chunks: int = 1,
    out_dim_chunk_idx: int = 0,
) -> list[Edges]:
    """Collect interaction edges between each node layer in Cs.

    Note that there is no edge weight that uses the position of the final interaction matrix as a
    starting node. This means that, unless node_layers contained the model output, we ignore the
    final section name in section_names when calculating the edges.

    Args:
        interaction_rotations: InteractionRotation objects containing C, C_pinv, node_layer and
            orig_dim, order by node layer.
        hooked_model: The hooked model.
        n_intervals: The number of integrated gradient intervals to use.
        section_names: The names of the modules to apply the hooks to.
        data_loader: The pytorch data loader.
        dtype: The data type to use for model computations.
        device: The device to run the model on.
        data_set_size: the total size of the dataset, used to normalize. Defaults to
        `len(data_loader)`. Important to set when parallelizing over the dataset.
        edge_formula: The formula to use for the attribution.
            - "functional" is the old (October 23) functional version
            - "squared" is the version which iterates over the output dim and output pos dim
        n_stochastic_sources: The number of stochastic sources for positional dimension
            (approximation). Defaults to None.
        out_dim_n_chunks: The number of chunks to split the out_dim into. 1 unless the script is
            parallelized over the out_dim.
        out_dim_chunk_idx: The index of the current chunk. 0 unless the script is parallelized over
            the out_dim.
    Returns:
        A list of Edges objects, which contain a matrix of edges between two node layers.
    """
    assert hooked_model.model.has_folded_bias, "Biases must be folded in to calculate edges."

    edge_modules = (
        section_names if interaction_rotations[-1].node_layer == "output" else section_names[:-1]
    )
    assert (
        len(edge_modules) == len(interaction_rotations) - 1
    ), "Number of edge modules not the same as interaction_rotations - 1."

    edge_hooks: list[Hook] = []
    for idx, (interaction_rotation, module_name, integration_method) in enumerate(
        zip(interaction_rotations[:-1], edge_modules, integration_methods, strict=True)
    ):
        # C from the next node layer
        assert interaction_rotation.C is not None, "C matrix is None."
        assert interaction_rotation.C_pinv is not None, "C_pinv matrix is None."
        C_out = interaction_rotations[idx + 1].C
        if C_out is not None:
            C_out = C_out.to(device=device)

        module_hat_partial = partial(
            module_hat,
            module=get_model_attr(hooked_model.model, module_name),
            C_in_pinv=interaction_rotation.C_pinv.to(device=device),
            C_out=C_out,
        )
        # Get the output edge dimension from the next node layer
        C_out = interaction_rotations[idx + 1].C
        out_rib_dim = (
            C_out.shape[1] if C_out is not None else interaction_rotations[idx + 1].orig_dim
        )

        C_in = interaction_rotation.C
        assert C_in is not None, "C_in is None."
        # Get chunk indices in case we're parallelizing over the out_dim. Will be (0, out_rib_dim)
        # if not.
        out_dim_start_idx, out_dim_end_idx = get_chunk_indices(
            data_size=out_rib_dim,
            chunk_idx=out_dim_chunk_idx,
            n_chunks=out_dim_n_chunks,
        )
        chunk_size = out_dim_end_idx - out_dim_start_idx

        # Initialise the edge matrices to zeros(chunk_size, in_rib_dim). These get accumulated in
        # the forward hook.
        hooked_model.hooked_data[interaction_rotation.node_layer] = {
            "edge": torch.zeros(
                chunk_size,
                C_in.shape[1],
                dtype=dtype,
                device=device,
            )
        }

        edge_hooks.append(
            Hook(
                name=interaction_rotation.node_layer,
                data_key="edge",
                fn=interaction_edge_pre_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={
                    "C_in": interaction_rotation.C.to(
                        device=device
                    ),  # C from the current node layer
                    "module_hat": module_hat_partial,
                    "n_intervals": n_intervals,
                    "integration_method": integration_method,
                    "dataset_size": data_set_size or len(data_loader.dataset),  # type: ignore
                    "out_dim_start_idx": out_dim_start_idx,
                    "out_dim_end_idx": out_dim_end_idx,
                    "edge_formula": edge_formula,
                    "n_stochastic_sources": n_stochastic_sources,
                },
            )
        )

    run_dataset_through_model(
        hooked_model, data_loader, edge_hooks, dtype=dtype, device=device, use_tqdm=True
    )

    module_ids = [info.node_layer for info in interaction_rotations]
    all_edges: list[Edges] = []
    for start, end in zip(module_ids[:-1], module_ids[1:]):
        E_hat: Float[Tensor, "rib_out rib_in"] = hooked_model.hooked_data[start]["edge"]
        if torch.all(E_hat == 0.0):
            logger.warning(
                f"Edges for node layer {start}-{end} are still zero, must be an error somewhere."
            )
        all_edges.append(Edges(in_node_layer=start, out_node_layer=end, E_hat=E_hat.detach().cpu()))
    hooked_model.clear_hooked_data()

    return all_edges
