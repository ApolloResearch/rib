"""Functions that apply hooks and accumulate data when passing batches through a model.

This file will tend to call on functions from rib.hook_fns and should act as an interface between
main code and these lower-level functions.
"""
from functools import partial
from typing import TYPE_CHECKING, Optional, Union, Literal

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange

from rib.hook_fns import (
    M_dash_and_Lambda_dash_pre_forward_hook_fn,
    cluster_gram_forward_hook_fn,
    function_size_forward_hook_fn,
    gram_forward_hook_fn,
    gram_pre_forward_hook_fn,
    interaction_edge_pre_forward_hook_fn,
    relu_interaction_forward_hook_fn,
    test_edges_forward_hook_fn,
    cluster_gram_forward_hook_fn,
    cluster_fn_pre_forward_hook_fn,
    collect_hessian_forward_hook_fn,
)
from rib.hook_fns_non_static import (
    delete_cluster_duplicate_forward_hook_fn,
    relu_swap_forward_hook_fn,
)
from rib.hook_manager import Hook, HookedModel
from rib.log import logger
from rib.linalg import module_hat
from rib.utils import eval_model_accuracy, eval_model_metrics, get_model_attr

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
    """Pass all batches through a hooked model, do not obtain output."""
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


def collect_relu_interactions(
    hooked_model: HookedModel,
    module_names: list[str],
    data_loader: DataLoader,
    dtype: torch.dtype,
    device: str,
    relu_metric_type: int,
    Cs_list: list[Float[Tensor, "d_hidden1 d_hidden2"]],
    layer_module_names: list[str],
    n_intervals: int,
    use_residual_stream: bool,
    unhooked_model: nn.Module,
    hook_names: Optional[str] = None,
    is_lm: Optional[bool] = True,
) -> tuple[dict[str, Float[Tensor, "d_hidden d_hidden"]], ...]:
    """Identify whether ReLUs are synchronising.

    This currently only works for piecewise linear functions and modules must be activation type modules.
    Recall that the node layers correspond to the positions at the input to each module specified in
    module_names, as well as the output of the final module.

    Most of my code assumes hook_names = model_names.

    Args:
        hooked_model: The hooked model.
        module_names: The names of the modules to collect relu interactions for. These should be
            activation layer types only.
        data_loader: The pytorch data loader.
        device: The device to run the model on.
        dtype: The data type to use for model computations.
        relu_metric_type: Type of metric used to check whether ReLUs are synchronising.
        hook_names: Used to store the gram matrices in the hooked model.
        Cs: list of basis rotation matrices used only for metric type 2.
        Lambda_dash: Now pass in Lambda dashes as the first term in numerator for metric type 3.
        g_js: list of g_js as derivative of f_hat^{l+1} with respect to alpha*f_l.
        layer_module_names: Non-activation layer list including last layer, to calculate functions
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
                fn_kwargs={
                    "relu_metric_type": relu_metric_type,
                    "C_next_layer": Cs_list[i+1].to(device),
                    "unhooked_model": unhooked_model,
                    "module_name_next_layer": layer_module_names[i+1],
                    "C_next_next_layer": Cs_list[i+2].to(device),
                    "n_intervals": n_intervals,
                    "use_residual_stream": use_residual_stream,
                }

            )
        )

    run_dataset_through_model(hooked_model, data_loader, hooks=relu_interaction_hooks, dtype=dtype, device=device)

    relu_similarity_numerators: dict[str, Float[Tensor, "d_hidden d_hidden"]] = {
        hook_name: hooked_model.hooked_data[hook_name]["relu_num"].cpu()
        for hook_name in hooked_model.hooked_data
    }
    hooked_model.clear_hooked_data()

    match relu_metric_type: # This is just used for the denominator, since some metrics divide by the rotated function L2 norm
        case 0 | 1 | 3: rotate = False
        case 2: rotate = True

    denominators: list[float] = collect_function_sizes(
        hooked_model=hooked_model,
        module_names=layer_module_names,
        data_loader=data_loader,
        dtype=dtype,
        device=device,
        hook_names=layer_module_names, # Changed hook layers to activation layers only
        rotate=rotate,
        Cs_list=Cs_list,
        is_lm=is_lm
    )

    match relu_metric_type:
        case 0: relu_similarity_matrices = relu_similarity_numerators / len(data_loader.dataset)
        case 1 | 2: # We don't normalise either numerator or denominator by dataset size
            relu_similarity_matrices = {hook_name: torch.div(relu_similarity_numerators[hook_name], denominators[i]) for i, hook_name in enumerate(module_names)}
        case 3:
            """Todo: normalise by Lambda and fix g_j in last term."""
            relu_similarity_matrices = {}
            for i, key in enumerate(module_names):
            ## Lambda code to check first term of numerator DOES work, but need to multiply by dataset size
            ## if not normalising by dataset size on both numerator and denominator
            #     vectorised_Lambda_dash = torch.diag(Lambda_dashes[i+1])
            #     d_hidden = vectorised_Lambda_dash.shape[0]
            #     numerator_term_1 = repeat(vectorised_Lambda_dash, 'd1 -> d1 d2', d2=d_hidden).cpu()
            #     numerator = numerator_term_1 - relu_similarity_numerators[key]

                numerator = relu_similarity_numerators[key]
                relu_similarity_matrices[key] = numerator / denominators[i+1]

    return relu_similarity_matrices


def collect_function_sizes(
    hooked_model: HookedModel,
    module_names: list[str],
    data_loader: DataLoader,
    dtype: torch.dtype,
    device: str,
    rotate: bool,
    Cs_list: list[Float[Tensor, "d_hidden d_hidden"]],
    hook_names: Optional[str] = None,
    is_lm: Optional[bool] = True,
) -> list[float]:
    """Calculate denominator for ReLU similarity metrics as l2 norm of function sizes in layer l+1.

    Args:
        rotate: Whether to rotate the functions to make f_hat.
    """
    fn_size_hooks = []
    if not is_lm:
        module_names = module_names[:-1]

    for i, (module_name, hook_name) in enumerate(zip(module_names[:-1], hook_names)):
        fn_size_hooks.append(
            Hook(
                name=hook_name,
                data_key="fn_size",
                fn=function_size_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={"rotate": rotate, "C_next_layer": Cs_list[i+1]}
            )
        )

    run_dataset_through_model(hooked_model, data_loader, hooks=fn_size_hooks, dtype=dtype, device=device)
    fn_sizes = [hooked_model.hooked_data[hook_name]["fn_size"].item()
                for hook_name in hooked_model.hooked_data]
    hooked_model.clear_hooked_data()

    return fn_sizes


def collect_test_edges(
    C_unscaled_list: list[Float[Tensor, "d_hidden1 d_hidden2"]],
    C_list: list[Float[Tensor, "d_hidden1 d_hidden2"]],
    W_hat_list: list[Float[Tensor, "d_hidden1 d_hidden2"]],
    hooked_model: HookedModel,
    module_names: list[str],
    data_loader: DataLoader,
    dtype: torch.dtype,
    device: str,
    hook_names: Optional[str] = None,
) -> Float[Tensor, ""]:
    """Collect test edge interactions with the C matrix between W and f edited to remove scaling."""
    if hook_names is not None:
        assert len(hook_names) == len(
            module_names), "Must specify a hook name for each module."
    else:
        hook_names = module_names

    test_edges_hooks = []
    for i, (module_name, hook_name) in enumerate(zip(module_names, hook_names)):
        # if i == len(module_names) - 1: # Can't collect test edges for output layer
        #     break
        test_edges_hooks.append(
            Hook(
                name=hook_name,
                data_key="test_edge",
                fn=test_edges_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={
                    'C_unscaled': C_unscaled_list[i],
                    'C_next_layer': C_list[i+1],
                    'W_hat': W_hat_list[i]
                }
            )
        )

    run_dataset_through_model(hooked_model, data_loader, hooks=test_edges_hooks, dtype=dtype, device=device)

    edges = {
        hook_name: torch.div(hooked_model.hooked_data[hook_name]["test_edge"], len(data_loader.dataset)).detach().cpu() for hook_name in hooked_model.hooked_data
    }
    hooked_model.clear_hooked_data()

    return edges


def calculate_swapped_relu_loss(
    hooked_model: HookedModel,
    module_name: str,
    data_loader: DataLoader,
    dtype: torch.dtype,
    device: str,
    replacement_idxs: Int[Tensor, "d_hidden"],
    num_replaced: list[int],
    hook_name: Optional[str] = None,
) -> tuple[float, ...]:
    """Calculate loss for unedited forward pass, and then on forward pass with specific ReLUs
    swapped.

    Act only on one layer. (See similar function below for one that a) iteratively replaces,
    b) acts on all layers.)
    """
    if hook_name is None: hook_name = module_name

    relu_swap_hooks = []
    relu_swap_hooks.append(
        Hook(
            name=hook_name,
            data_key="relu_swap",
            fn=relu_swap_forward_hook_fn,
            module_name=module_name,
            fn_kwargs={"replacement_idxs": replacement_idxs}
        )
    )

    unhooked_accuracy, unhooked_loss = eval_model_metrics(
        hooked_model=hooked_model,
        dataloader=data_loader,
        hooks=None,
        dtype=dtype,
        device=device,
    )

    # Pass hooks in relu_swap_hooks to forward pass evaluate with them in
    hooked_accuracy, hooked_loss = eval_model_metrics(
        hooked_model=hooked_model,
        dataloader=data_loader,
        hooks=relu_swap_hooks,
        dtype=dtype,
        device=device,
    )

    # Remove hooks above to now create one for case where indices to swap are randomised
    # Note that randomisation doesn't necessarily act as a perfect baseline if there is a high
    # degree of synchronisation
    hooked_model.remove_hooks()

    random_relu_swap_hooks = []
    length = len(replacement_idxs)
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


def calculate_all_swapped_relu_loss(
    hooked_model: HookedModel,
    module_names: list[str],
    data_loader: DataLoader,
    dtype: torch.dtype,
    device: str,
    replacement_idx_list: list[Int[Tensor, "d_hidden"]],
    num_replaced_list: list[list[int]],
    use_residual_stream: bool,
    hook_names: Optional[list[str]] = None,
) -> tuple[float, ...]:
    """Calculate loss for unedited forward pass, and then on forward pass with specific ReLUs
    swapped. Acts on all layers."""
    if hook_names is not None:
        assert len(hook_names) == len(
            module_names), "Must specify a hook name for each module."
    else:
        hook_names = module_names

    relu_swap_hooks = []
    for i, (module_name, hook_name) in enumerate(zip(module_names, hook_names)):
        relu_swap_hooks.append(
            Hook(
                name=hook_name,
                data_key="relu_swap",
                fn=relu_swap_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={
                    "replacement_idxs": replacement_idx_list[i],
                    "use_residual_stream": use_residual_stream
                }
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
    for i, (module_name, hook_name) in enumerate(zip(module_names, hook_names)):
        length = len(replacement_idx_list[i])
        total_num_swapped = sum(num_replaced_list[i])
        random_idx_tensor = torch.arange(length)
        indices_to_replace = torch.randperm(length)[:total_num_swapped]

        random_idx_tensor[indices_to_replace] = torch.randint(0, length, (total_num_swapped,))
        random_relu_swap_hooks.append(
            Hook(
                name=hook_name,
                data_key="relu_swap",
                fn=relu_swap_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={
                    "replacement_idxs": random_idx_tensor,
                    "use_residual_stream": use_residual_stream,
                }
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


def collect_clustered_relu_P_mats(
    module_names: list[str],
    C_list: list[Float[Tensor, "d_hidden_out d_hidden_truncated"]],
    W_hat_list: list[Float[Tensor, "d_hidden_out d_hiddden_in"]],
    all_cluster_idxs: list[list[Int[Tensor, "cluster_size"]]],
    hook_names: Optional[list[str]] = None,
) -> dict[str, dict[Int[Tensor, "cluster_size"], Float[Tensor, "d_hidden_next_layer d_hidden"]]]:
    """In expression for functions in next layer, obtain per-cluster P matrix for each cluster.

    This case only works for a transformation where a weight matrix is included with ReLU.
    Turns out this is not quite true for modadd.
    """
    if hook_names is not None:
        assert len(hook_names) == len(module_names), "Must specify a hook name for each module."
    else:
        hook_names = module_names

    P_dict = {}
    for i, (module_name, hook_name, layer_cluster_idxs) in enumerate(zip(module_names, hook_names, all_cluster_idxs)):
        layer_P_dict = {}
        for cluster_idxs in layer_cluster_idxs:
            C_next_layer_cluster = C_list[i][cluster_idxs, :] # Cols C in overleaf = rows C in code
            W_hat_t = W_hat_list[i].T
            W_hat_cluster = W_hat_t[:, cluster_idxs] # Rows W overleaf = cols C in code
            P: Float[Tensor, "d_hidden_next_layer d_hidden"] = W_hat_cluster @ C_next_layer_cluster
            layer_P_dict[f"{cluster_idxs}"] = P
        P_dict[module_name] = layer_P_dict

    return P_dict


def collect_clustered_relu_P_mats_no_W(
    module_names: list[str],
    C_list: list[Float[Tensor, "d_hidden_out d_hidden_truncated"]],
    all_cluster_idxs: list[list[Int[Tensor, "cluster_size"]]],
) -> dict[str, dict[Int[Tensor, "cluster_size"], Float[Tensor, "d_hidden_next_layer d_hidden"]]]:
    """In expression for functions in next layer, obtain per-cluster P matrix for each cluster.

    This case only works for a transformation where a weight matrix is included with ReLU.
    Turns out this is not quite true for modadd.
    """
    P_dict = {}
    for i, (module_name, layer_cluster_idxs) in enumerate(zip(module_names, all_cluster_idxs)):
        layer_P_dict = {}
        for cluster_idxs in layer_cluster_idxs:
            C_next_layer_cluster = C_list[i][cluster_idxs, :] # Cols C in overleaf = rows C in code
            layer_P_dict[f"{cluster_idxs}"] = C_next_layer_cluster
        P_dict[module_name] = layer_P_dict

    return P_dict


def collect_cluster_grams(
    hooked_model: HookedModel,
    module_names: list[str],
    data_loader: DataLoader,
    dtype: torch.dtype,
    device: str,
    all_cluster_idxs: list[list[Int[Tensor, "d_cluster"]]],
    use_residual_stream: bool,
    dataset_size: int,
    hook_names: Optional[str] = None,
) -> dict[str, list[Float[Tensor, "d_cluster d_cluster"]]]:
    """
    Args:
        module_names: The names of the modules to collect relu interactions for. These should be
            activation layer types only.
        hook_names: Used for saving in hook data and retrieving hook data. If not specified,
            automatically set to be module_names.
    """
    if hook_names is not None:
        assert len(hook_names) == len(module_names), "Must specify a hook name for each module."
    else:
        hook_names = module_names

    cluster_gram_hooks = []
    for i, (module_name, hook_name) in enumerate(zip(module_names, hook_names)):
        cluster_gram_hooks.append(
            Hook(
                name=hook_name,
                data_key="cluster_gram",
                fn=cluster_gram_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={"cluster_idxs": all_cluster_idxs[i], "use_residual_stream": use_residual_stream, "dataset_size": dataset_size}
            )
        )

    run_dataset_through_model(
        hooked_model=hooked_model,
        dataloader=data_loader,
        hooks=cluster_gram_hooks,
        dtype=dtype,
        device=device,
    )

    cluster_grams: dict[str, list[Float[Tensor, "d_cluster, d_cluster"]]] = {}
    for layer_idx, hook_name in enumerate(hook_names):
        cluster_grams[hook_name] = [hooked_model.hooked_data[hook_name][i] for i in range(len(all_cluster_idxs[layer_idx]))]
        # cluster_output_grams[hook_name] = [hooked_model.hooked_data[hook_name][f"output_{i}"] for
        # i in range(len(all_cluster_idxs[layer_idx]))]
        ## Whole layer gram - only use when using second hook function
        # whole_layer_gram = hooked_model.hooked_data[hook_name]["whole layer"]
    hooked_model.clear_hooked_data()

    return cluster_grams


def collect_cluster_fns(
    hooked_model: HookedModel,
    module_names: list[str],
    data_loader: DataLoader,
    dtype: torch.dtype,
    device: str,
    all_cluster_idxs: list[list[Int[Tensor, "d_cluster"]]],
    use_residual_stream: bool,
    hook_names: Optional[str] = None,
) -> dict[str, list[Float[Tensor, "d_cluster"]]]:
    if hook_names is not None:
        assert len(hook_names) == len(
            module_names), "Must specify a hook name for each module."
    else:
        hook_names = module_names

    cluster_fn_hooks = []
    for i, (module_name, hook_name) in enumerate(zip(module_names, hook_names)):
        cluster_fn_hooks.append(
            Hook(
                name=hook_name,
                data_key="relu_swap",
                fn=cluster_fn_pre_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={"cluster_idxs": all_cluster_idxs[i], "use_residual_stream": use_residual_stream}
            )
        )

    run_dataset_through_model(
        hooked_model=hooked_model,
        dataloader=data_loader,
        hooks=cluster_fn_hooks,
        dtype=dtype,
        device=device,
    )

    cluster_fns: dict[str, list[Float[Tensor, "d_cluster"]]] = {}
    for layer_idx, hook_name in enumerate(hook_names):
        cluster_fns[hook_name] = [torch.div(hooked_model.hooked_data[hook_name][i], len(data_loader.dataset)) \
                                  for i in range(len(all_cluster_idxs[layer_idx]))]
    hooked_model.clear_hooked_data()

    return cluster_fns


def calculate_delete_cluster_duplicate_loss(
    hooked_model: HookedModel,
    module_names: list[str],
    data_loader: DataLoader,
    dtype: torch.dtype,
    device: str,
    all_cluster_idxs: list[list[Int[Tensor, "d_cluster"]]],
    all_centroid_idxs: list[list[int]],
    use_residual_stream: bool,
    hook_names: Optional[list[str]] = None,
) -> tuple[float, ...]:
    if hook_names is not None:
        assert len(hook_names) == len(
            module_names), "Must specify a hook name for each module."
    else:
        hook_names = module_names

    cluster_delete_hooks = []
    for i, (module_name, hook_name) in enumerate(zip(module_names, hook_names)):
        cluster_delete_hooks.append(
            Hook(
                name=hook_name,
                data_key="relu_swap",
                fn=delete_cluster_duplicate_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={
                    "cluster_idxs": all_cluster_idxs[i],
                    "centroid_idxs": all_centroid_idxs[i],
                    "use_residual_stream": use_residual_stream}
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
        hooks=cluster_delete_hooks,
        dtype=dtype,
        device=device,
    )

    hooked_model.remove_hooks()

    return unhooked_loss, hooked_loss, unhooked_accuracy, hooked_accuracy


def collect_hessian(
    hooked_model: HookedModel,
    input_module_name: str,
    data_loader: DataLoader,
    dtype: torch.dtype,
    device: str,
    weight_module_names: list[str],
    copy_seq_model: nn.Module,
    C_list: list[Float[Tensor, "out_hidden_trunc in_hidden"]],
    use_residual_stream: Optional[bool] = False
) -> list[float]:
    """Calculate denominator for ReLU similarity metrics as l2 norm of function sizes in layer l+1.

    Args:
        rotate: Whether to rotate the functions to make f_hat.
        input_module_name: Module whose inputs are considered inputs for the model to end the
            backpropagation on. Used for applying the hook function as well, so should cover
            everything you want contained in hook.
        weight_module_names: Names of sections to form Hessian from i.e. these are the weight
            parameters the derivatives will be taken with respect to.
    """
    input_act_hook = Hook(
        name=input_module_name,
        data_key="hessian",
        fn=collect_hessian_forward_hook_fn,
        module_name=input_module_name,
        fn_kwargs={
            "weight_module_names": weight_module_names,
            "copy_seq_model": copy_seq_model,
            "C_list": C_list,
            "use_residual_stream": use_residual_stream,
        }
    )

    run_dataset_through_model(hooked_model, data_loader, hooks=[input_act_hook], dtype=dtype, device=device)
    H1 = hooked_model.hooked_data[input_module_name][f"H1 {input_module_name}"]
    H2 = hooked_model.hooked_data[input_module_name][f"H2 {input_module_name}"]
    M12 = hooked_model.hooked_data[input_module_name][f"M12 {input_module_name}"]
    hooked_model.clear_hooked_data()

    H2_flat = rearrange(H2, 'a b c d -> (a c) (b d)')
    M12_flat = rearrange(M12, 'a b c d -> (a d) (b c)')
    top_row = torch.cat((H1, M12_flat), dim=1)
    bottom_row = torch.cat((M12_flat.T, H2_flat), dim=1)
    H = torch.cat((top_row, bottom_row), dim=0)

    return H
