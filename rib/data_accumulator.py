"""Functions that apply hooks and accumulate data when passing batches through a model."""

from functools import partial
from typing import TYPE_CHECKING, Literal, Optional, Union

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from rib.hook_fns import (
    Lambda_dash_pre_forward_hook_fn,
    M_dash_and_Lambda_dash_pre_forward_hook_fn,
    M_dash_pre_forward_hook_fn,
    acts_forward_hook_fn,
    gram_forward_hook_fn,
    gram_pre_forward_hook_fn,
    interaction_edge_pre_forward_hook_fn,
)
from rib.hook_manager import Hook, HookedModel
from rib.linalg import module_hat
from rib.log import logger
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
    M_dtype: torch.dtype = torch.float64,
    Lambda_einsum_dtype: torch.dtype = torch.float64,
    basis_formula: Literal["(1-alpha)^2", "(1-0)*alpha"] = "(1-alpha)^2",
    next_gradients: Optional[Float[Tensor, "batch out_hidden_combined_trunc"]] = None,
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
            "(1-alpha)^2", "(1-0)*alpha", or "g*f". The first is the old (October) version
            while the second is a new (November) version that should be used from now on. The second makes sense especially in light of the new attribution
            (edge_formula="squared") but is generally good and does not change results
            much. Defaults to "(1-0)*alpha".
        next_gradients: The integrated gradients of the next layer. Only used if
            basis_formula="g*f".
    Returns:
        A tuple containing M' and Lambda'.
    """
    if hook_name is None:
        hook_name = module_name

    data_key = ["M_dash", "Lambda_dash"]

    interaction_hook = Hook(
        name=hook_name,
        data_key=data_key,
        fn=M_dash_and_Lambda_dash_pre_forward_hook_fn,
        module_name=module_name,
        fn_kwargs={
            "C_out": C_out,
            "n_intervals": n_intervals,
            "dataset_size": len(data_loader.dataset),  # type: ignore
            "M_dtype": M_dtype,
            "Lambda_einsum_dtype": Lambda_einsum_dtype,
            "basis_formula": basis_formula,
            "next_gradients": next_gradients,
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

    edge_modules = section_names if Cs[-1].node_layer_name == "output" else section_names[:-1]
    assert len(edge_modules) == len(Cs) - 1, "Number of edge modules not the same as Cs - 1."

    logger.info("Collecting edges for node layers: %s", [C.node_layer_name for C in Cs[:-1]])
    edge_hooks: list[Hook] = []
    for idx, (C_info, module_name) in enumerate(zip(Cs[:-1], edge_modules)):
        # C from the next node layer
        assert C_info.C is not None, "C matrix is None."
        assert C_info.C_pinv is not None, "C_pinv matrix is None."
        C_out = Cs[idx + 1].C
        if C_out is not None:
            C_out = C_out.to(device=device)

        module_hat_partial = partial(
            module_hat,
            module=get_model_attr(hooked_model.model, module_name),
            C_in_pinv=C_info.C_pinv.to(device=device),
            C_out=C_out,
        )
        edge_hooks.append(
            Hook(
                name=C_info.node_layer_name,
                data_key="edge",
                fn=interaction_edge_pre_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={
                    "C_in": C_info.C.to(device=device),  # C from the current node layer
                    "module_hat": module_hat_partial,
                    "n_intervals": n_intervals,
                    "dataset_size": data_set_size if data_set_size is not None else len(data_loader.dataset),  # type: ignore
                    "edge_formula": edge_formula,
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


def collect_activations_single_batch(
    hooked_model: HookedModel,
    batch_loader: DataLoader,
    module_name: str,
    dtype: torch.dtype,
    device: str,
    hook_name: Optional[str] = "activations",
) -> tuple[Float[Tensor, "in_hidden in_hidden"], Float[Tensor, "in_hidden in_hidden"]]:
    """Collect the activations for a particular batch in the module specifed by `module_name`.

    Args:
        hooked_model: The hooked model.
        batch: The batch of data to pass through the model. This should be a dataloader with a single batch
        module_name: The name of the module whose inputs are the node layer we collect the
        activations for.
        dtype: The data type to use for model computations.
        device: The device to run the model on.
        hook_name: The name of the hook to use to store the matrices in the hooked model.

    Returns:

    """
    assert len(batch_loader) == 1, "Batch loader must have a single batch."

    if hook_name is None:
        hook_name = module_name

    data_key = ["activations"]

    interaction_hook = Hook(
        name=hook_name,
        data_key=data_key,
        fn=acts_forward_hook_fn,
        module_name=module_name,
        fn_kwargs={},
    )

    run_dataset_through_model(
        hooked_model,
        batch_loader,
        hooks=[interaction_hook],
        dtype=dtype,
        device=device,
        use_tqdm=True,
    )
    activations = hooked_model.hooked_data[hook_name]["activations"]

    # hooked_model.clear_hooked_data()
    return activations


def collect_M_dash(
    hooked_model: HookedModel,
    section_names: list[str],
    node_layers: list[str],
    n_intervals: int,
    data_loader: DataLoader,
    device: str,
    dtype: torch.dtype,
    M_dtype: torch.dtype = torch.float64,
    hook_names: Optional[list[str]] = None,
    basis_formula: Literal["g*f"] = "g*f",
) -> dict[str, Float[Tensor, "d_hidden d_hidden"]]:
    """Collect M matrices for global basis for the module inputs and the output of the
    final module.

    We collect activations using a forward hook.

    Args:
        hooked_model: The hooked model.
        section_names: The names of the sections to apply the hooks to.
        node_layers: The names of the node layers to apply the hooks to.
        n_intervals: The number of integrated gradient intervals to use.
        data_loader: The pytorch data loader.
        device: The device to run the model on.
        dtype: The data type to use for model computations.
        M_dtype: The data type to use for the M_dash matrix. Needs to be
            float64 for Pythia-14m (empirically). Defaults to float64.
        hook_names: Used to store the M_dash matrices. Defaults to the same as node_layers
        basis_formula: The formula to use for the integrated gradient. Must be one of "g*f".

    Returns:
        A dictionary of M matrices, where the keys are the hook names (a.k.a. node layer names)
    """
    module_names = node_layers
    assert len(module_names) > 0, "No modules specified."
    if hook_names is not None:
        assert len(hook_names) == len(module_names), "Must specify a hook name for each module."
    else:
        hook_names = module_names
    assert len(module_names) > 0, "No modules specified."
    if hook_names is not None:
        assert len(hook_names) == len(module_names), "Must specify a hook name for each module."
    else:
        hook_names = module_names

    # generate activation hooks to store activations for calculating integrated gradients
    activation_hooks: list[Hook] = []

    # Add input hooks
    for module_name, hook_name in zip(module_names, hook_names):
        activation_hooks.append(
            Hook(
                name=hook_name,
                data_key="activations",
                fn=acts_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={},
            )
        )
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

    # generate M_dash hooks
    M_dash_hooks: dict[str, Hook] = {}
    for module_name, hook_name in zip(module_names, hook_names):
        M_dash_hooks[module_name] = Hook(
            name=hook_name,
            data_keys=["M_dash", "integrated_gradients"],
            fn=M_dash_pre_forward_hook_fn,
            module_name=module_name,
            fn_kwargs={
                "n_intervals": n_intervals,
                "dataset_size": len(data_loader.dataset),  # type: ignore
                "basis_formula": basis_formula,
                "M_dtype": M_dtype,
                "basis_formula": basis_formula,
            },
        )

    for batch in tqdm(data_loader, desc="Global interaction rotations"):
        batch_dataset = TensorDataset(*batch)
        batch_loader = DataLoader(batch_dataset, batch_size=1, shuffle=False)

        run_dataset_through_model(
            hooked_model=hooked_model,
            data_loader=batch_loader,
            hooks=activation_hooks,
            dtype=dtype,
            device=device,
            use_tqdm=True,
        )

        # integrated_gradients: dict[str, Float[Tensor, "batch out_hidden_combined_trunc"]] = {}
        # # final layer gradients are the same as outputs
        # integrated_gradients[module_names[-1]] = hooked_model.hooked_data["output"]["activations"]

        for module, next_module in zip(module_names[-2::-1], module_names[:1:-1]):
            # get activations for layer
            assert (
                hooked_model.hooked_data[module]["activations"] is not None
            ), f"Activations for {module} are None."
            activations = hooked_model.hooked_data[module]["activations"]
            # assert (
            #     next_module in integrated_gradients
            # ), f"Integrated gradients for {next_module} are None."

            # next_gradients = integrated_gradients[next_module]

            interaction_hook = Hook(
                name=module,
                data_key="integrated_gradient",
                fn=M_dash_pre_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={
                    "next_hook_name": next_module,
                    "n_intervals": n_intervals,
                    "dataset_size": len(data_loader.dataset),  # type: ignore
                    "basis_formula": basis_formula,
                    # "next_gradients": next_gradients,
                },
            )
            interaction_hook = M_dash_hooks[module]
            run_dataset_through_model(
                hooked_model=hooked_model,
                data_loader=batch_loader,
                hooks=[interaction_hook],
                dtype=dtype,
                device=device,
                use_tqdm=True,
            )

    M_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]] = {
        hook_name: hooked_model.hooked_data[hook_name]["M_dash"]
        for hook_name in hooked_model.hooked_data
    }

    assert set(M_matrices.keys()) == set(hook_names)

    return M_matrices


def collect_Lambda_dash(
    C_out: Optional[Float[Tensor, "out_hidden out_hidden"]],
    hooked_model: HookedModel,
    n_intervals: int,
    data_loader: DataLoader,
    module_name: str,
    dtype: torch.dtype,
    device: str,
    hook_name: Optional[str] = None,
    Lambda_einsum_dtype: torch.dtype = torch.float64,
    basis_formula: Literal["(1-alpha)^2", "(1-0)*alpha"] = "(1-alpha)^2",
) -> Float[Tensor, "in_hidden in_hidden"]:
    """Collect the matrix Lambda' for the input to the module specifed by `module_name`.

    We accumulate the matrixLambda' for each batch. To do this, we apply
    a hook to the provided module. This hook will accumulate the matrix over the batches.

    Args:
        C_out: The rotation matrix for the next layer.
        hooked_model: The hooked model.
        n_intervals: The number of integrated gradient intervals to use.
        data_loader: The data loader.
        module_name: The name of the module whose inputs are the node layer we collect Lambda' for.
        dtype: The data type to use for model computations.
        device: The device to run the model on.
        hook_name: The name of the hook to use to store the matrices in the hooked model.
        Lambda_einsum_dtype: The data type to use for the einsum computing batches for the
            Lambda_dash matrix. Does not affect the output, only used for the einsum within
            M_dash_and_Lambda_dash_pre_forward_hook_fn. Needs to be float64 on CPU but float32 was
            fine on GPU. Defaults to float64.
        basis_formula: The formula to use for the integrated gradient used in the attribution
        formula (normally "(1-0)*alpha"). Defaults to "(1-0)*alpha". This is for a particular
        simplified way of writing Lambda when the attribution method is functional, that is used in
        the paper. This will need to be changed for squared attribution.
    Returns:
        Lambda'.
    """
    if hook_name is None:
        hook_name = module_name

    data_key = ["Lambda_dash"]

    interaction_hook = Hook(
        name=hook_name,
        data_key=data_key,
        fn=Lambda_dash_pre_forward_hook_fn,
        module_name=module_name,
        fn_kwargs={
            "C_out": C_out,
            "n_intervals": n_intervals,
            "dataset_size": len(data_loader.dataset),  # type: ignore
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
    Lambda_dash = hooked_model.hooked_data[hook_name]["Lambda_dash"]

    hooked_model.clear_hooked_data()

    return Lambda_dash
