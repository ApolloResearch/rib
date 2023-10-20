"""Functions that apply hooks and accumulate data when passing batches through a model.

This file will tend to call on functions from rib.hook_fns and should act as an interface between
main code and these lower-level functions.
"""

from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader

from rib.hook_fns import (
    M_dash_and_Lambda_dash_pre_forward_hook_fn,
    acts_pre_forward_hook_fn,
    gram_forward_hook_fn,
    gram_pre_forward_hook_fn,
    interaction_edge_pre_forward_hook_fn,
    relu_interaction_forward_hook_fn,
    test_edges_forward_hook_fn,
    function_size_forward_hook_fn,
)
from rib.hook_fns_non_static import relu_swap_forward_hook_fn
from rib.hook_manager import Hook, HookedModel
from rib.utils import eval_model_accuracy, eval_model_metrics


if TYPE_CHECKING:  # Prevent circular import to import type annotations
    from rib.interaction_algos import InteractionRotation


def plot_temp(matrix: Float[Tensor, "d1 d2"], title: str) -> None:
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    out_dir = Path(__file__).parent / "relu_temp_debug"
    out_dir.mkdir(parents=True, exist_ok=True)


    if matrix.dim() == 1:
        aspect_ratio = 20 / matrix.shape[0]
        matrix = matrix.unsqueeze(-1)
    else:
        aspect_ratio = matrix.shape[1] / matrix.shape[0]
    # Set the vertical size, and let the horizontal size adjust based on the aspect ratio
    vertical_size = 7
    horizontal_size = vertical_size * aspect_ratio
    plt.figure(figsize=(horizontal_size, vertical_size))
    sns.heatmap(matrix.detach().cpu(), annot=False, cmap="YlGnBu", cbar=True, square=True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{title}.png")
    plt.close()


def run_dataset_through_model(
    hooked_model: HookedModel,
    dataloader: DataLoader,
    hooks: list[Hook],
    dtype: torch.dtype,
    device: str = "cuda",
) -> None:
    """Pass all batches through a hooked model, do not obtain output."""
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
        assert set(gram_matrices.keys()) == set(hook_names)

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
        A tuple containing M' and Lambda', as well as the integrated gradient term g_j.
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

    Note that there is no edge weight that uses the position of the final interaction matrix as a
    starting node. This means that, if we did not collect the output logits, we don't apply any
    hooks to the final module list in module_names.

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

    edge_modules = module_names if Cs[-1].node_layer_name == "output" else module_names[:-1]
    edge_hooks: list[Hook] = []
    for idx, (C_info, module_name) in enumerate(zip(Cs[:-1], edge_modules)):
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
    ), f"Edge keys not the same as node layer names."
    return edges


def collect_relu_interactions(
    hooked_model: HookedModel,
    module_names: list[str],
    data_loader: DataLoader,
    dtype: torch.dtype,
    device: str,
    relu_metric_type: int,
    Cs: list[Float[Tensor, "d_hidden1 d_hidden2"]],
    Lambda_dashes: list[Float[Tensor, "d_hidden d_hidden"]],
    layer_module_names: list[str],
    n_intervals: int,
    unhooked_model: nn.Module,
    hook_names: Optional[str] = None,
) -> tuple[dict[str, Float[Tensor, "d_hidden d_hidden"]], ...]:
    """Identify whether ReLUs are synchronising.

    This currently only works for piecewise linear functions and modules must be activation type modules.
    Recall that the node layers correspond to the positions at the input to each module specified in
    module_names, as well as the output of the final module.

    TODO: change ReLU metric type naming system to Enum rather than having integers floating around.

    Most of my code assumes hook_names = model_names.

    Args:
        hooked_model: The hooked model.
        module_names: The names of the modules to collect gram matrices for.
        data_loader: The pytorch data loader.
        device: The device to run the model on.
        dtype: The data type to use for model computations.
        relu_metric_type: Type of metric used to check whether ReLUs are synchronising.
        hook_names: Used to store the gram matrices in the hooked model.
        Cs: list of basis rotation matrices used only for metric type 2.
        Lambda_dash: Now pass in Lambda dashes as the first term in numerator for metric type 3.
        g_js: list of g_js as derivative of f_hat^{l+1} with respect to alpha*f_l.
        layer_module_names: non-activation layer list including last layer, to calculate functions
        sizes over.

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
    for i, (module_name, hook_name) in enumerate(zip(module_names, hook_names)):
        relu_interaction_hooks.append(
            Hook(
                name=hook_name,
                data_key="relu_num",
                fn=relu_interaction_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={"relu_metric_type": relu_metric_type, "C_next_layer": Cs[i+1].to(device), "unhooked_model": unhooked_model, "module_name_next_layer": layer_module_names[i+1], "C_next_next_layer": Cs[i+2], "n_intervals": n_intervals}
            )
        )

    run_dataset_through_model(
        hooked_model, data_loader, hooks=relu_interaction_hooks, dtype=dtype, device=device)

    relu_similarity_numerators: dict[str, Float[Tensor, "d_hidden d_hidden"]] = {
        hook_name: hooked_model.hooked_data[hook_name]["relu_num"].cpu()
        for hook_name in hooked_model.hooked_data
    }
    if relu_metric_type == 3:
        relu_similarity_whole_numerators = {
            hook_name: hooked_model.hooked_data[hook_name]["whole_relu_num"].cpu()
            for hook_name in hooked_model.hooked_data
        }
        relu_numerator_1s = {
            hook_name: hooked_model.hooked_data[hook_name]["relu_num_1"].cpu()
            for hook_name in hooked_model.hooked_data
        }
        my_Lambda_dashes = {
            hook_name: hooked_model.hooked_data[hook_name]["Lambda_dash"].cpu()
            for hook_name in hooked_model.hooked_data
        }
        for hook_name, mat in my_Lambda_dashes.items():
            plot_temp(mat, f"my_Lambda_dash_{hook_name}")
        for i, mat in enumerate(Lambda_dashes):
            plot_temp(mat, f"Lambda_dash_{i}")
    # Plot preactivations for debugging
    preactivations = {
        hook_name: torch.div(hooked_model.hooked_data[hook_name]["preactivations"], len(data_loader.dataset))
        for hook_name in hooked_model.hooked_data
    }
    for hook_name, mat in preactivations.items():
        plot_temp(mat, f"preact_{hook_name}")

    hooked_model.clear_hooked_data()

    match relu_metric_type:
        case 0 | 1 | 3:
            rotate = False
        case 2:
            rotate = True

    denominators: List[float] = collect_function_sizes(
        hooked_model=hooked_model,
        module_names=layer_module_names,
        data_loader=data_loader,
        dtype=dtype,
        device=device,
        hook_names=layer_module_names,
        rotate=rotate,
        Cs=Cs,
    )

    match relu_metric_type:
        case 0: relu_similarity_matrices = relu_similarity_numerators / len(data_loader.dataset)
        case 1 | 2: # We don't normalise either numerator or denominator by dataset size
            relu_similarity_matrices = {hook_name: torch.div(relu_similarity_numerators[hook_name], denominators[i]) for i, hook_name in enumerate(module_names)
            }
        case 3:
            """Todo: normalise by Lambda and fix g_j in last term."""
            num_2_shapes = [mat.shape for mat in list(relu_similarity_numerators.values())]
            lambda_shapes = [mat.shape for mat in Lambda_dashes]
            relu_similarity_matrices = {}
            for i, key in enumerate(module_names):
            ## Lambda code DOES work, but you'd need to multiply by dataset size here
            ## if not normalising by dataset size on both numerator and denominator
            #     vectorised_Lambda_dash = torch.diag(Lambda_dashes[i+1])
            #     d_hidden = vectorised_Lambda_dash.shape[0]
            #     numerator_term_1 = repeat(vectorised_Lambda_dash, 'd1 -> d1 d2', d2=d_hidden).cpu()
            #     numerator = numerator_term_1 - relu_similarity_numerators[key]

                numerator = relu_similarity_whole_numerators[key]
                relu_similarity_matrices[key] = numerator / denominators[i+1]

    return relu_similarity_matrices


def collect_function_sizes(
    hooked_model: HookedModel,
    module_names: list[str],
    data_loader: DataLoader,
    dtype: torch.dtype,
    device: str,
    rotate: bool,
    Cs: list[Float[Tensor, "d_hidden d_hidden"]],
    hook_names: Optional[str] = None,
) -> list[float]:
    """Calculate denominator for ReLU similarity metrics as l2 norm of function sizes in layer l+1."""
    assert len(module_names) > 0, "No modules specified."
    if hook_names is not None:
        assert len(hook_names) == len(
            module_names), "Must specify a hook name for each module."
    else:
        hook_names = module_names

    fn_size_hooks = []
    for i, (module_name, hook_name) in enumerate(zip(module_names, hook_names)):
        fn_size_hooks.append(
            Hook(
                name=hook_name,
                data_key="fn_size",
                fn=function_size_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={"rotate": rotate, "C_next_layer": Cs[i+1]}
            )
        )

    run_dataset_through_model(hooked_model, data_loader, hooks=fn_size_hooks, dtype=dtype, device=device)
    fn_sizes = [hooked_model.hooked_data[hook_name]["fn_size"].item() for hook_name in hooked_model.hooked_data]
    hooked_model.clear_hooked_data()

    return fn_sizes


def collect_test_edges(
    Cs_unscaled: list[Float[Tensor, "d_hidden1 d_hidden2"]],
    Cs: list[Float[Tensor, "d_hidden1 d_hidden2"]],
    W_hats: list[Float[Tensor, "d_hidden1 d_hidden2"]],
    hooked_model: HookedModel,
    module_names: list[str],
    data_loader: DataLoader,
    dtype: torch.dtype,
    device: str,
    hook_names: Optional[str] = None,
) -> Float[Tensor, ""]:
    """Collect test edge interactions with the C matrix between W and f edited to remove scaling."""
    assert len(module_names) > 0, "No modules specified."
    if hook_names is not None:
        assert len(hook_names) == len(
            module_names), "Must specify a hook name for each module."
    else:
        hook_names = module_names

    test_edges_hooks = []
    for i, (module_name, hook_name) in enumerate(zip(module_names, hook_names)):
        if i == len(module_names) - 1: # Can't collect test edges for output layer
            break
        test_edges_hooks.append(
            Hook(
                name=hook_name,
                data_key="test_edge",
                fn=test_edges_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={
                    'C_unscaled': Cs_unscaled[i].detach().cpu(),
                    'C_next_layer': Cs[i+1].detach().cpu(),
                    'W_hat': W_hats[i]
                }
            )
        )

    run_dataset_through_model(hooked_model, data_loader, hooks=test_edges_hooks, dtype=dtype, device=device)

    edges = {
        hook_name: torch.div(hooked_model.hooked_data[hook_name]["test_edge"], len(data_loader.dataset)) for hook_name in hooked_model.hooked_data
    }
    hooked_model.clear_hooked_data()

    return edges


def calculate_swapped_relu_loss(
    hooked_model: HookedModel,
    module_name: str,
    data_loader: DataLoader,
    dtype: torch.dtype,
    device: str,
    replacement_idx_list: list[int],
    num_replaced: int,
    hook_name: Optional[str] = None,
) -> tuple[list[float], ...]:
    """Calculate loss for unedited forward pass, and then on forward pass with specific ReLUs
    swapped."""
    if hook_name is None:
        hook_name = module_name

    relu_swap_hooks = []
    relu_swap_hooks.append(
        Hook(
            name=hook_name,
            data_key="relu_swap",
            fn=relu_swap_forward_hook_fn,
            module_name=module_name,
            fn_kwargs={"replacement_idx_list": replacement_idx_list}
        )
    )

    unhooked_accuracy, unhooked_loss = eval_model_metrics(
        hooked_model=hooked_model,
        dataloader=data_loader,
        hooks=None,
        dtype=dtype,
        device=device,
    )

    hooked_accuracy, hooked_loss = eval_model_metrics(
        hooked_model=hooked_model,
        dataloader=data_loader,
        hooks=relu_swap_hooks,
        dtype=dtype,
        device=device,
    )

    hooked_model.remove_hooks()

    random_relu_swap_hooks = []
    length = len(replacement_idx_list)
    random_idx_tensor = torch.arange(length)
    indices_to_replace = torch.randperm(length)[:num_replaced]
    random_idx_tensor[indices_to_replace] = torch.randint(0, length, (num_replaced,))
    random_relu_swap_hooks.append(
        Hook(
            name=hook_name,
            data_key="relu_swap",
            fn=relu_swap_forward_hook_fn,
            module_name=module_name,
            fn_kwargs={"replacement_idx_list": random_idx_tensor.tolist()}
        )
    )

    random_hooked_accuracy, random_hooked_loss = eval_model_metrics(
        hooked_model=hooked_model,
        dataloader=data_loader,
        hooks=random_relu_swap_hooks,
        dtype=dtype,
        device=device,
    )

    return unhooked_loss, hooked_loss, random_hooked_loss, unhooked_accuracy, hooked_accuracy, random_hooked_accuracy