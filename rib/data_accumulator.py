"""Functions that apply hooks and accumulate data when passing batches through a model."""

from typing import TYPE_CHECKING, Literal, Optional, Union

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from rib.hook_fns import (
    M_dash_and_Lambda_dash_pre_forward_hook_fn,
    dataset_mean_forward_hook_fn,
    dataset_mean_pre_forward_hook_fn,
    gram_forward_hook_fn,
    gram_pre_forward_hook_fn,
    interaction_edge_pre_forward_hook_fn,
)
from rib.hook_manager import Hook, HookedModel
from rib.log import logger
from rib.models import SequentialTransformer

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
def collect_dataset_means(
    hooked_model: HookedModel,
    module_names: list[str],
    data_loader: DataLoader,
    device: str,
    dtype: torch.dtype,
    collect_output_dataset_means: bool = True,
    hook_names: Optional[list[str]] = None,
) -> dict[str, Float[Tensor, "d_hidden d_hidden"]]:
    """ """
    assert len(module_names) > 0, "No modules specified."
    if hook_names is not None:
        assert len(hook_names) == len(module_names), "Must specify a hook name for each module."
    else:
        hook_names = module_names

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
        print("collecting dataset_mean (pre) for", module_name)
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
        print("collecting dataset_mean for", module_name)

    run_dataset_through_model(
        hooked_model, data_loader, dataset_mean_hooks, dtype=dtype, device=device, use_tqdm=True
    )

    dataset_mean: dict[str, Float[Tensor, "d_hidden"]] = {
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
    Gamma_matrices: Optional[dict[str, Float[Tensor, "d_hidden d_hidden"]]] = None,
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
                fn_kwargs={
                    "dataset_size": dataset_size,
                    "Gamma_matrix": Gamma_matrices[hook_name] if Gamma_matrices else None,
                },
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
                    "Gamma_matrix": Gamma_matrices["output"] if Gamma_matrices else None,
                },
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
    Lambda_einsum_dtype: torch.dtype = torch.float64,
    basis_formula: Literal["(1-alpha)^2", "(1-0)*alpha"] = "(1-alpha)^2",
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
            "dataset_size": len(data_loader.dataset),  # type: ignore
            "M_dtype": M_dtype,
            "Lambda_einsum_dtype": Lambda_einsum_dtype,
            "basis_formula": basis_formula,
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
    edge_formula: Literal["functional", "squared"] = "functional",
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
        edge_formula: The formula to use for the attribution. Must be one of "functional" or
            "squared". The former is the old (October) functional version, the latter is a new
            (November) version.

    Returns:
        A dictionary of interaction edge matrices, keyed by the module name which the edge passes
        through.
    """
    assert hooked_model.model.has_folded_bias, "Biases must be folded in to calculate edges."

    if isinstance(hooked_model.model, SequentialTransformer):
        variable_position_dimension = hooked_model.model.last_pos_module_type == "add_resid1"
    else:
        variable_position_dimension = False

    edge_modules = section_names if Cs[-1].node_layer_name == "output" else section_names[:-1]
    print("edge_modules", edge_modules)
    print("Cs", [C.node_layer_name for C in Cs])
    assert (
        len(edge_modules) == len(Cs) - 1
    ), f"Number of edge modules not the same as Cs - 1. Num edge modules: {len(edge_modules)}, Cs - 1: {len(Cs) - 1}"

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
                    "edge_formula": edge_formula,
                    "variable_position_dimension": variable_position_dimension,
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
