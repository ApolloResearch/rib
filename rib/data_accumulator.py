"""Functions that apply hooks and accumulate data when passing batches through a model."""

from functools import partial
from typing import TYPE_CHECKING, Literal, Optional, Union

import torch
from fancy_einsum import einsum
from jaxtyping import Float
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rib.hook_fns import (
    M_dash_and_Lambda_dash_pre_forward_hook_fn,
    gram_forward_hook_fn,
    gram_pre_forward_hook_fn,
    interaction_edge_pre_forward_hook_fn,
    linear_integrated_gradient_pre_forward_hook_fn,
)
from rib.hook_manager import Hook, HookedModel
from rib.log import logger
from rib.models.sequential_transformer.components import MLPIn, MLPOut
from rib.models.utils import get_model_attr

if TYPE_CHECKING:  # Prevent circular import to import type annotations
    from rib.interaction_algos import InteractionRotation


def run_dataset_through_model(
    hooked_model: HookedModel,
    dataloader: DataLoader,
    hooks: list[Hook],
    dtype: torch.dtype,
    device: str = "cuda",
    use_tqdm: bool = False,
) -> None:
    """Simply pass all batches through a hooked model."""
    assert len(hooks) > 0, "Hooks have not been applied to this model."
    loader: Union[tqdm, DataLoader]
    if use_tqdm:
        loader = tqdm(dataloader, total=len(dataloader), desc="Batches through entire model")
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
def collect_gram_matrices(
    hooked_model: HookedModel,
    module_names: list[str],
    data_loader: DataLoader,
    device: str,
    dtype: torch.dtype,
    collect_output_gram: bool = True,
    hook_names: Optional[list[str]] = None,
) -> dict[str, Float[Tensor, "d_hidden d_hidden"]]:
    """Collect gram matrices for the module inputs and optionally the output of the final module.

    We use pre_forward hooks for the input to each module. If `collect_output_gram` is True, we
    also collect the gram matrix for the output of the final module using a forward hook.

    Args:
        hooked_model: The hooked model.
        module_names: The names of the modules to collect gram matrices for.
        data_loader: The pytorch data loader.
        device: The device to run the model on.
        dtype: The data type to use for model computations.
        collect_output_gram: Whether to collect the gram matrix for the output of the final module.
        hook_names: Used to store the gram matrices in the hooked model.

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
        gram_hooks.append(
            Hook(
                name=hook_name,
                data_key="gram",
                fn=gram_pre_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={"dataset_size": dataset_size},
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
                fn_kwargs={"dataset_size": dataset_size},
            )
        )

    run_dataset_through_model(
        hooked_model, data_loader, gram_hooks, dtype=dtype, device=device, use_tqdm=True
    )

    gram_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]] = {
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
    C_out: Optional[Float[Tensor, "out_hidden out_hidden"]],
    hooked_model: HookedModel,
    n_intervals: int,
    data_loader: DataLoader,
    module_name: str,
    dtype: torch.dtype,
    device: str,
    hook_name: Optional[str] = None,
) -> tuple[Float[Tensor, "in_hidden in_hidden"], Float[Tensor, "in_hidden in_hidden"]]:
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
            "dataset_size": len(data_loader.dataset),  # type: ignore
        },
    )

    run_dataset_through_model(
        hooked_model,
        data_loader,
        hooks=[interaction_hook],
        dtype=dtype,
        device=device,
        use_tqdm=True,
    )

    M_dash = hooked_model.hooked_data[hook_name]["M_dash"]
    Lambda_dash = hooked_model.hooked_data[hook_name]["Lambda_dash"]
    hooked_model.clear_hooked_data()

    return M_dash, Lambda_dash


def collect_interaction_edges(
    Cs: list["InteractionRotation"],
    hooked_model: HookedModel,
    n_intervals: int,
    section_names: list[str],
    data_loader: DataLoader,
    dtype: torch.dtype,
    device: str,
    dataset_size: Optional[int] = None,
    use_analytic_integrad: bool = True,
) -> dict[str, Float[Tensor, "out_hidden_trunc in_hidden_trunc"]]:
    """Collect interaction edges between each node layer in Cs.

    Note that there is no edge weight that uses the position of the final interaction matrix as a
    starting node. This means that, unless node_layers contained the model output, we ignore the
    final section name in section_names when calculating the edges.

    Args:
        Cs: The interaction rotation matrix and its pseudoinverse, order by node layer.
        hooked_model: The hooked model.
        n_intervals: The number of integrated gradient intervals to use.
        section_names: The names of the modules to apply the hooks to.
        data_loader: The pytorch data loader.
        dtype: The data type to use for model computations.
        device: The device to run the model on.
        dataset_size: the total size of the dataset, used to normalize. Defaults to
            `len(data_loader)`. Important to set when parallelizing over the dataset.
        use_analytic_integrad: Whether to use the analytic edge calculation if the section supports
            it. Defaults to True.

    Returns:
        A dictionary of interaction edge matrices, keyed by the module name which the edge passes
        through.
    """
    assert hooked_model.model.has_folded_bias, "Biases must be folded in to calculate edges."

    dataset_size = dataset_size if dataset_size is not None else len(data_loader.dataset)  # type: ignore

    edge_modules = section_names if Cs[-1].node_layer_name == "output" else section_names[:-1]
    logger.info("Collecting edges for node layers: %s", [C.node_layer_name for C in Cs[:-1]])

    integrated_gradient_types: dict[str, Literal["linear", "numerical"]] = {}
    edge_hooks: list[Hook] = []
    for idx, (C_info, module_name) in enumerate(zip(Cs[:-1], edge_modules)):
        # C from the next node layer
        assert C_info.C is not None, "C matrix is None."
        assert C_info.C_pinv is not None, "C_pinv matrix is None."
        C_out = Cs[idx + 1].C
        if C_out is not None:
            C_out = C_out.to(device=device)

        # Get the list of modules in the section
        section = get_model_attr(hooked_model.model, module_name)

        if use_analytic_integrad:
            # Check if only a single module in the section
            if (isinstance(section, nn.Sequential) and len(section) == 1) or not isinstance(
                section, nn.Sequential
            ):
                module = section[0] if isinstance(section, nn.Sequential) else section
                if isinstance(module, (MLPIn, MLPOut)):
                    edge_hooks.append(
                        Hook(
                            name=C_info.node_layer_name,
                            data_key="f_hat_norm",
                            fn=linear_integrated_gradient_pre_forward_hook_fn,
                            module_name=module_name,
                            fn_kwargs={
                                "C_in": C_info.C.to(device=device),  # C from the current node layer
                                "dataset_size": dataset_size,
                            },
                        )
                    )
                    # Initialise f_hat_norm to (out_dim). This gets accumulated in the forward hook.
                    hooked_model.hooked_data[C_info.node_layer_name] = {
                        "f_hat_norm": torch.zeros(C_info.out_dim, dtype=dtype, device=device)
                    }
                    integrated_gradient_types[C_info.node_layer_name] = "linear"
        if C_info.node_layer_name not in integrated_gradient_types:
            # Haven't added an analytic integrated gradient hook, so add a numerical one
            edge_hooks.append(
                Hook(
                    name=C_info.node_layer_name,
                    data_key="edge",
                    fn=interaction_edge_pre_forward_hook_fn,
                    module_name=module_name,
                    fn_kwargs={
                        "C_in": C_info.C.to(device=device),  # C from the current node layer
                        "C_in_pinv": C_info.C_pinv.to(
                            device=device
                        ),  # C_pinv from current node layer
                        "C_out": C_out,
                        "n_intervals": n_intervals,
                        "dataset_size": dataset_size,
                    },
                )
            )
            # Initialise the edge matrices to zeros to (out_dim, in_dim). These get accumulated in
            # the forward hook.
            hooked_model.hooked_data[C_info.node_layer_name] = {
                "edge": torch.zeros(Cs[idx + 1].out_dim, C_info.out_dim, dtype=dtype, device=device)
            }
            integrated_gradient_types[C_info.node_layer_name] = "numerical"

    run_dataset_through_model(
        hooked_model, data_loader, edge_hooks, dtype=dtype, device=device, use_tqdm=True
    )

    # Ensure that we have the same number of edge_modules as we do data stored in the hooked model
    assert len(edge_modules) == len(hooked_model.hooked_data), (
        f"Number of edge modules ({len(edge_modules)}) not the same as the number of "
        f"hooked data ({len(hooked_model.hooked_data)})."
    )
    edges: dict[str, Float[Tensor, "out_hidden_trunc in_hidden_trunc"]] = {}
    for i, node_layer_name in enumerate(hooked_model.hooked_data):
        if integrated_gradient_types[node_layer_name] == "linear":
            section = get_model_attr(hooked_model.model, edge_modules[i])
            module = section[0] if isinstance(section, nn.Sequential) else section
            if isinstance(module, MLPIn):
                W_raw = module.W_in
            elif isinstance(module, MLPOut):
                W_raw = module.W_out
            else:
                raise ValueError(f"Module type {type(module)} not supported for linear type.")
            f_hat_norm: Float[Tensor, "in_hidden_extra_trunc"] = hooked_model.hooked_data[
                node_layer_name
            ]["f_hat_norm"]
            n_extra_dims = len(f_hat_norm) - W_raw.shape[0]

            # Create matrix ((I, 0), (0, W_raw)) where I is an identity matrix of size n_extra_dims
            # This handles the concatenated residual stream and other input stream
            W = torch.block_diag(torch.eye(n_extra_dims, dtype=dtype, device=device), W_raw)
            W_hat = einsum(
                "out out_trunc, in out, in_trunc in -> out_trunc in_trunc", C_out, W, C_info.C_pinv
            )
            edge = einsum(
                "out_trunc in_trunc, in_trunc -> out_trunc in_trunc", W_hat**2, f_hat_norm**2
            )
            edges[node_layer_name] = edge
        elif integrated_gradient_types[node_layer_name] == "numerical":
            edges[node_layer_name] = hooked_model.hooked_data[node_layer_name]["edge"]
        else:
            raise ValueError(
                f"Integrated gradient type {integrated_gradient_types[node_layer_name]} not "
                f"supported."
            )

    hooked_model.clear_hooked_data()

    # Ensure that the keys of the edges dict are the same as the node layer names without `output`
    if set(edges.keys()) != set([C.node_layer_name for C in Cs[:-1]]):
        logger.warning(
            "Edge keys not the same as node layer names. " "Expected: %s, got: %s",
            set([C.node_layer_name for C in Cs[:-1]]),
            set(edges.keys()),
        )
    return edges
