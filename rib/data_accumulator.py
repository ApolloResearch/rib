"""Functions that apply hooks and accumulate data when passing batches through a model."""

from typing import TYPE_CHECKING

from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader

from rib.hook_manager import Hook, HookedModel
from rib.types import TORCH_DTYPES

if TYPE_CHECKING:  # Prevent circular import to import type annotations
    from rib.interaction_algos import InteractionRotation


def run_dataset_through_model(
    hooked_model: HookedModel,
    dataloader: DataLoader,
    hooks: list[Hook],
    device: str = "cuda",
    dtype: str = "float32",
) -> None:
    """Simply pass all batches through a hooked model."""
    assert len(hooks) > 0, "Hooks have not been applied to this model."
    for batch in dataloader:
        data, _ = batch
        data = data.to(device=device, dtype=TORCH_DTYPES[dtype])

        hooked_model(data, hooks=hooks)


def collect_gram_matrices(
    hooked_model: HookedModel,
    module_names: list[str],
    data_loader: DataLoader,
    device: str,
    dtype: str = "float32",
    collect_output_gram: bool = True,
) -> dict[str, Float[Tensor, "d_hidden d_hidden"]]:
    """Collect gram matrices for the module inputs and optionally the output of the final module.

    We use pre_forward hooks for the input to each module. If `collect_output_gram` is True, we
    also collect the gram matrix for the output of the final module using a forward hook.

    Args:
        hooked_model: The hooked model.
        module_names: The names of the modules to collect gram matrices for.
        data_loader: The pytorch data loader.
        device: The device to run the model on.
        dtype: The data type to use.
        collect_output_gram: Whether to collect the gram matrix for the output of the final module.

    Returns:
        A dictionary of gram matrices, where the keys are the hook names (a.k.a. node layer names)
    """
    assert len(module_names) > 0, "No modules specified."

    gram_hooks: list[Hook] = []
    # Add input hooks
    for module_name in module_names:
        gram_hooks.append(
            Hook(
                name=module_name,
                data_key="gram",
                fn_name="gram_pre_forward_hook_fn",
                module_name=module_name,
            )
        )
    if collect_output_gram:
        # Add output hook
        gram_hooks.append(
            Hook(
                name="output",
                data_key="gram",
                fn_name="gram_forward_hook_fn",
                module_name=module_names[-1],
            )
        )

    run_dataset_through_model(hooked_model, data_loader, gram_hooks, device=device, dtype=dtype)

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
        assert set(gram_matrices.keys()) == set(module_names + ["output"])
    else:
        assert set(gram_matrices.keys()) == set(module_names)

    return gram_matrices


def collect_M_dash_and_Lambda_dash(
    C_out: Float[Tensor, "out_hidden out_hidden"],
    hooked_model: HookedModel,
    data_loader: DataLoader,
    module_name: str,
    device: str,
    dtype: str = "float32",
) -> tuple[Float[Tensor, "in_hidden in_hidden"], Float[Tensor, "in_hidden in_hidden"]]:
    """Collect the matrices M' and Lambda' for the input to the module specifed by `module_name`.

    We accumulate the matrices, M' and Lambda' for each batch. To do this, we apply
    a hook to the provided module. This hook will accumulate both matrices over the batches.

    Args:
        C_out: The rotation matrix for the next layer.
        hooked_model: The hooked model.
        data_loader: The data loader.
        module_name: The name of the module whose inputs are the node layer we collect the matrices
            M' and Lambda' for.
        device: The device to run the model on.
        dtype: The datatype to use.

    Returns:
        A tuple containing M' and Lambda'.
    """
    interaction_hook = Hook(
        name=module_name,
        data_key=["M_dash", "Lambda_dash"],
        fn_name=f"M_dash_and_Lambda_dash_forward_hook_fn",
        module_name=module_name,
        fn_kwargs={
            "C_out": C_out,
        },
    )

    run_dataset_through_model(
        hooked_model, data_loader, hooks=[interaction_hook], device=device, dtype=dtype
    )

    M_dash = hooked_model.hooked_data[module_name]["M_dash"]
    Lambda_dash = hooked_model.hooked_data[module_name]["Lambda_dash"]
    hooked_model.clear_hooked_data()

    # Scale the matrices by the number of samples in the dataset.
    len_dataset = len(data_loader.dataset)  # type: ignore
    M_dash = M_dash / len_dataset
    Lambda_dash = Lambda_dash / len_dataset

    return M_dash, Lambda_dash


def collect_interaction_edges(
    Cs: list["InteractionRotation"],
    hooked_model: HookedModel,
    data_loader: DataLoader,
    device: str,
    dtype: str = "float32",
) -> dict[str, Float[Tensor, "out_hidden_trunc in_hidden_trunc"]]:
    """Collect interaction edges between each node layer in Cs.

    Recall that the node layers correspond to the positions at the input to each module specified in
    module_names, as well as the output of the final module.

    Args:
        Cs: The interaction rotation matrix and its pseudoinverse, order by node layer.
        hooked_model: The hooked model.
        data_loader: The pytorch data loader.
        device: The device to run the model on.
        dtype: The data type to use.

    Returns:
        A dictionary of interaction edge matrices, keyed by the module name which the edge passes
        through.
    """

    assert Cs[-1].node_layer_name == "output", "The last node layer name must be 'output'."
    edge_hooks: list[Hook] = []
    for idx, C_info in enumerate(Cs[:-1]):
        edge_hooks.append(
            Hook(
                name=C_info.node_layer_name,
                data_key="edge",
                fn_name="interaction_edge_forward_hook_fn",
                module_name=C_info.node_layer_name,
                fn_kwargs={
                    "C_in": C_info.C,  # C from the current node layer
                    "C_in_pinv": C_info.C_pinv,  # C_pinv from the current node layer
                    "C_out": Cs[idx + 1].C,  # C from the next node layer
                },
            )
        )

    run_dataset_through_model(hooked_model, data_loader, edge_hooks, device=device, dtype=dtype)

    edges: dict[str, Float[Tensor, "out_hidden_trunc in_hidden_trunc"]] = {
        node_layer_name: hooked_model.hooked_data[node_layer_name]["edge"]
        for node_layer_name in hooked_model.hooked_data
    }
    hooked_model.clear_hooked_data()

    # Scale the edges by the number of samples in the dataset
    for node_layer_name in edges:
        edges[node_layer_name] = edges[node_layer_name] / len(data_loader.dataset)  # type: ignore

    # Ensure that the keys of the edges dict are the same as the node layer names without `output`
    assert set(edges.keys()) == set(
        [C.node_layer_name for C in Cs[:-1]]
    ), f"Edge keys not the same as node layer names. "
    return edges
