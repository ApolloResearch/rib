"""Functions that apply hooks and accumulate data when passing batches through a model."""

from typing import TYPE_CHECKING, Optional, Union

import torch
from einops import repeat, rearrange, reduce
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader

from rib.hook_fns import (
    M_dash_and_Lambda_dash_pre_forward_hook_fn,
    gram_forward_hook_fn,
    gram_pre_forward_hook_fn,
    interaction_edge_pre_forward_hook_fn,
    relu_interaction_forward_hook_fn,
    acts_pre_forward_hook_fn,
)
from rib.hook_manager import Hook, HookedModel

if TYPE_CHECKING:  # Prevent circular import to import type annotations
    from rib.interaction_algos import InteractionRotation


def run_dataset_through_model(
    hooked_model: HookedModel,
    dataloader: DataLoader,
    hooks: list[Hook],
    dtype: torch.dtype,
    device: str = "cuda",
) -> None:
    """Simply pass all batches through a hooked model."""
    assert len(hooks) > 0, "Hooks have not been applied to this model."
    for batch in dataloader:
        data, _ = batch
        data = data.to(device=device)
        # Change the dtype unless the inputs are integers (e.g. like they are for LMs)
        if data.dtype not in [torch.int64, torch.int32]:
            data = data.to(dtype=dtype)

        hooked_model(data, hooks=hooks)


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
        assert len(hook_names) == len(
            module_names), "Must specify a hook name for each module."
    else:
        hook_names = module_names

    gram_hooks: list[Hook] = []
    # Add input hooks
    for module_name, hook_name in zip(module_names, hook_names):
        gram_hooks.append(
            Hook(
                name=hook_name,
                data_key="gram",
                fn=gram_pre_forward_hook_fn,
                module_name=module_name,
            )
        )
    if collect_output_gram:
        # Add output hook
        gram_hooks.append(
            Hook(
                name="output",
                data_key="gram",
                fn=gram_forward_hook_fn,
                module_name=module_names[-1],
            )
        )

    run_dataset_through_model(
        hooked_model, data_loader, gram_hooks, dtype=dtype, device=device)

    gram_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]] = {
        hook_name: hooked_model.hooked_data[hook_name]["gram"]
        for hook_name in hooked_model.hooked_data
    }
    hooked_model.clear_hooked_data()

    # Scale the gram matrix by the number of samples in the dataset.
    for hook_name in gram_matrices:
        gram_matrices[hook_name] /= len(data_loader.dataset)  # type: ignore

    # Ensure that the gram_matrix keys are the same as the module names (optionally with an
    # additional "output" if collect_output_gram is True).
    if collect_output_gram:
        assert set(gram_matrices.keys()) == set(hook_names + ["output"])
    else:
        assert set(gram_matrices.keys()) == set(module_names)

    return gram_matrices


def collect_M_dash_and_Lambda_dash(
    C_out: Float[Tensor, "out_hidden out_hidden"],
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
        },
    )

    run_dataset_through_model(
        hooked_model, data_loader, hooks=[
            interaction_hook], dtype=dtype, device=device
    )

    M_dash = hooked_model.hooked_data[hook_name]["M_dash"]
    Lambda_dash = hooked_model.hooked_data[hook_name]["Lambda_dash"]
    hooked_model.clear_hooked_data()

    # Scale the matrices by the number of samples in the dataset.
    len_dataset = len(data_loader.dataset)  # type: ignore
    M_dash = M_dash / len_dataset
    Lambda_dash = Lambda_dash / len_dataset

    return M_dash, Lambda_dash


def collect_interaction_edges(
    Cs: list["InteractionRotation"],
    hooked_model: HookedModel,
    n_intervals: int,
    module_names: list[str],
    data_loader: DataLoader,
    dtype: torch.dtype,
    device: str,
) -> dict[str, Float[Tensor, "out_hidden_trunc in_hidden_trunc"]]:
    """Collect interaction edges between each node layer in Cs.

    Recall that the node layers correspond to the positions at the input to each module specified in
    module_names, as well as the output of the final module.

    Args:
        Cs: The interaction rotation matrix and its pseudoinverse, order by node layer.
        hooked_model: The hooked model.
        n_intervals: The number of integrated gradient intervals to use.
        module_names: The names of the modules to apply the hooks to.
        data_loader: The pytorch data loader.
        dtype: The data type to use for model computations.
        device: The device to run the model on.

    Returns:
        A dictionary of interaction edge matrices, keyed by the module name which the edge passes
        through.
    """

    assert Cs[-1].node_layer_name == "output", "The last node layer name must be 'output'."
    edge_hooks: list[Hook] = []
    for idx, (C_info, module_name) in enumerate(zip(Cs[:-1], module_names)):
        edge_hooks.append(
            Hook(
                name=C_info.node_layer_name,
                data_key="edge",
                fn=interaction_edge_pre_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={
                    "C_in": C_info.C,  # C from the current node layer
                    "C_in_pinv": C_info.C_pinv,  # C_pinv from the current node layer
                    "C_out": Cs[idx + 1].C,  # C from the next node layer
                    "n_intervals": n_intervals,
                },
            )
        )

    run_dataset_through_model(
        hooked_model, data_loader, edge_hooks, dtype=dtype, device=device)

    edges: dict[str, Float[Tensor, "out_hidden_trunc in_hidden_trunc"]] = {
        node_layer_name: hooked_model.hooked_data[node_layer_name]["edge"]
        for node_layer_name in hooked_model.hooked_data
    }
    hooked_model.clear_hooked_data()

    # Scale the edges by the number of samples in the dataset
    for node_layer_name in edges:
        edges[node_layer_name] = edges[node_layer_name] / \
            len(data_loader.dataset)  # type: ignore

    # Ensure that the keys of the edges dict are the same as the node layer names without `output`
    assert set(edges.keys()) == set(
        [C.node_layer_name for C in Cs[:-1]]
    ), f"Edge keys not the same as node layer names. "
    return edges


def collect_relu_interactions(
    hooked_model: HookedModel,
    module_names: list[str],
    data_loader: DataLoader,
    dtype: torch.dtype,
    device: str,
    relu_metric_type: int,
    hook_names: Optional[str] = None
) -> dict[str, Union[Float[Tensor, "d_hidden d_hidden"], Float[Tensor, "d_hidden_concat d_hidden_concat"]]]:
    """Identify whether ReLUs are synchronising using basic checking of layer operators O(x) only (pointwise evaluations of ratio of input to output).

    This currently only works for piecewise linear functions and modules must be activation type modules.
    Recall that the node layers correspond to the positions at the input to each module specified in
    module_names, as well as the output of the final module.

    TODO: change ReLU metric type naming system to Enum rather than having integers floating around.

    Args:
        hooked_model: The hooked model.
        module_names: The names of the modules to collect gram matrices for.
        data_loader: The pytorch data loader.
        device: The device to run the model on.
        dtype: The data type to use for model computations.
        collect_output_gram: Whether to collect the gram matrix for the output of the final module.
        hook_names: Used to store the gram matrices in the hooked model.

    Returns:
        A dictionary of ReLU interaction matrices, where the keys are the hook names (a.k.a. node layer names)
    """
    assert len(module_names) > 0, "No modules specified."
    if hook_names is not None:
        assert len(hook_names) == len(
            module_names), "Must specify a hook name for each module."
    else:
        hook_names = module_names

    relu_interaction_hooks: list[Hook] = []
    for module_name, hook_name in zip(module_names, hook_names):
        relu_interaction_hooks.append(
            Hook(
                name=hook_name,
                data_key="relu_interaction",
                fn=relu_interaction_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={'relu_metric_type': relu_metric_type}
            )
        )

    run_dataset_through_model(
        hooked_model, data_loader, hooks=relu_interaction_hooks, dtype=dtype, device=device)

    # Collect ReLU interaction matrices and scale by size of dataset
    relu_interaction_matrices: dict[str, Union[Float[Tensor, "d_hidden d_hidden"], Float[Tensor, "d_hidden_concat d_hidden_concat"]]] = {
        hook_name: torch.div(hooked_model.hooked_data[hook_name]["relu_interaction"], len(data_loader)) for hook_name in hooked_model.hooked_data
    }
    hooked_model.clear_hooked_data()

    return relu_interaction_matrices


def collect_activations_and_rotate(
    Cs: list["InteractionRotation"],
    hooked_model: HookedModel,
    module_names: list[str],
    data_loader: DataLoader,
    dtype: torch.dtype,
    device: str,
    hook_names: Optional[str] = None
) -> list[Union[Float[Tensor, "batch d_hidden"], Float[Tensor, "batch pos d_hidden"]]]:
    """Use hooks to obtain inputs to modules (functions f(x) in that layer) and rotate them by the interaction rotation matrix for that layer."""
    assert len(module_names) > 0, "No modules specified."
    if hook_names is not None:
        assert len(hook_names) == len(
            module_names), "Must specify a hook name for each module."
    else:
        hook_names = module_names

    activation_hooks: list[Hook] = []
    for module_name, hook_name in zip(module_names, hook_names):
        activation_hooks.append(
            Hook(
                name=hook_name,
                data_key="activation",
                fn=acts_pre_forward_hook_fn,
                module_name=module_name,
            )
        )

    run_dataset_through_model(hooked_model, data_loader, hooks=activation_hooks, dtype=dtype, device=device)

    # Extra zero index when reading dictionary is because output is always stored as a tuple (useful
    activations: dict[str, Union[Float[Tensor, "batch d_hidden"], Float[Tensor, "batch pos d_hidden"]]] = {
        hook_name: hooked_model.hooked_data[hook_name]["activation"] for hook_name in hooked_model.hooked_data
    }

    rotated_activations_list = []
    for activation, C in zip(list(activations.values()), Cs):
        batch_size, d_hidden = activation.shape
        rotated_activations_list.append(torch.bmm(activation, repeat(C, 'd_hidden1 d_hidden2 -> batch_size d_hidden1 d_hidden2', batch_size=batch_size)))

    return activations