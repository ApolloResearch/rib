"""Functions that apply hooks and accumulate data when passing batches through a model."""

from typing import TYPE_CHECKING, Optional, Union

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from rib.hook_fns import (
    M_dash_and_Lambda_dash_pre_forward_hook_fn,
    gram_forward_hook_fn,
    gram_pre_forward_hook_fn,
    interaction_edge_pre_forward_hook_fn,
)
from rib.hook_manager import Hook, HookedModel
from rib.log import logger

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
    M_dtype: torch.dtype = torch.float64,
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
        M_dtype: The data type to use for the M_dash matrix. Needs to be
            float64 for Pythia-14m (empirically). Defaults to float64.

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
            "M_dtype": M_dtype,
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
    data_set_size: Optional[int] = None,
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
        data_set_size: the total size of the dataset, used to normalize. Defaults to
        `len(data_loader)`. Important to set when parallelizing over the dataset.

    Returns:
        A dictionary of interaction edge matrices, keyed by the module name which the edge passes
        through.
    """
    assert hooked_model.model.has_folded_bias, "Biases must be folded in to calculate edges."
    edge_modules = section_names if Cs[-1].node_layer_name == "output" else section_names[:-1]
    logger.info("Collecting edges for node layers: %s", [C.node_layer_name for C in Cs[:-1]])
    edge_hooks: list[Hook] = []
    for idx, (C_info, module_name) in enumerate(zip(Cs[:-1], edge_modules)):
        # C from the next node layer
        assert C_info.C is not None, "C matrix is None."
        assert C_info.C_pinv is not None, "C_pinv matrix is None."
        C_out = Cs[idx + 1].C
        if C_out is not None:
            C_out = C_out.to(device=device)
        edge_hooks.append(
            Hook(
                name=C_info.node_layer_name,
                data_key="edge",
                fn=interaction_edge_pre_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={
                    "C_in": C_info.C.to(device=device),  # C from the current node layer
                    "C_in_pinv": C_info.C_pinv.to(device=device),  # C_pinv from current node layer
                    "C_out": C_out,
                    "n_intervals": n_intervals,
                    "dataset_size": data_set_size if data_set_size is not None else len(data_loader.dataset),  # type: ignore
                },
            )
        )
        # Initialise the edge matrices to zeros to (out_dim, in_dim). These get added to in the
        # forward hook.
        hooked_model.hooked_data[C_info.node_layer_name] = {
            "edge": torch.zeros(Cs[idx + 1].out_dim, C_info.out_dim, dtype=dtype, device=device)
        }

    run_dataset_through_model(
        hooked_model, data_loader, edge_hooks, dtype=dtype, device=device, use_tqdm=True
    )

    edges: dict[str, Float[Tensor, "out_hidden_trunc in_hidden_trunc"]] = {
        node_layer_name: hooked_model.hooked_data[node_layer_name]["edge"]
        for node_layer_name in hooked_model.hooked_data
    }
    hooked_model.clear_hooked_data()

    # Ensure that the keys of the edges dict are the same as the node layer names without `output`
    if set(edges.keys()) != set([C.node_layer_name for C in Cs[:-1]]):
        logger.warning(
            "Edge keys not the same as node layer names. " "Expected: %s, got: %s",
            set([C.node_layer_name for C in Cs[:-1]]),
            set(edges.keys()),
        )
    return edges
