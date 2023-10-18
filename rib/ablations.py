from typing import Callable, Literal, Optional, Union

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from rib.hook_fns import rotate_pre_forward_hook_fn
from rib.hook_manager import Hook, HookedModel
from rib.linalg import calc_rotation_matrix
from rib.utils import calc_ablation_schedule

BasisVecs = Union[Float[Tensor, "d_hidden d_hidden_trunc"], Float[Tensor, "d_hidden d_hidden"]]
BasisVecsPinv = Union[Float[Tensor, "d_hidden_trunc d_hidden"], Float[Tensor, "d_hidden d_hidden"]]


def ablate_and_test(
    hooked_model: HookedModel,
    module_name: str,
    basis_vecs: BasisVecs,
    basis_vecs_pinv: BasisVecsPinv,
    test_loader: DataLoader,
    eval_fn: Callable,
    ablation_schedule: list[int],
    device: str,
    dtype: Optional[torch.dtype] = None,
    hook_name: Optional[str] = None,
    early_stopping_threshold: Optional[float] = None,
) -> dict[int, float]:
    """Ablate eigenvectors and test the model accuracy/loss.

    We start by ablating zero vectors, measuring the ce_loss/accuracy, and continue ablating vectors
    until the absolute value of result is `early_stopping_threshold` different than the original
    ce_loss/accuracy.

    Args:
        hooked_model: The hooked model.
        module_name: The name of the module whose inputs we want to rotate and ablate.
        basis_vecs: Matrix with basis vectors (rib or orthogonal) in the columns.
        basis_vecs_pinv: The pseudo-inverse of the basis_vecs matrix.
        hook_config: The config for the hook point.
        test_loader: The DataLoader for the test data.
        eval_fn: The function to use to evaluate the model.
        ablation_schedule: A list of the number of vectors to ablate at each step.
        device: The device to run the model on.
        dtype: The data type to cast the inputs to. Ignored if int32 or int64.
        hook_name: The name of the hook point to use. If None, defaults to `module_name`.
        early_stopping_threshold: The threshold to use for early stopping. If None, we don't use
            early stopping.

    Returns:
        Dictionary mapping the number of basis vectors remaining and the resulting accuracy.
    """

    if hook_name is None:
        hook_name = module_name

    base_result: Optional[float] = None

    eval_results: dict[int, float] = {}
    # Iterate through possible number of ablated vectors, starting from no ablated vectors
    for i, n_ablated_vecs in enumerate(
        tqdm(
            ablation_schedule[::-1],
            total=len(ablation_schedule),
            desc=f"Ablating {module_name}",
        )
    ):
        n_vecs_remaining = basis_vecs.shape[1] - n_ablated_vecs

        basis_vecs = basis_vecs.to(device)
        basis_vecs_pinv = basis_vecs_pinv.to(device)
        rotation_matrix = calc_rotation_matrix(
            vecs=basis_vecs,
            vecs_pinv=basis_vecs_pinv,
            n_ablated_vecs=n_ablated_vecs,
        )

        rotation_hook = Hook(
            name=hook_name,
            data_key="rotation",
            fn=rotate_pre_forward_hook_fn,
            module_name=module_name,
            fn_kwargs={"rotation_matrix": rotation_matrix},
        )

        eval_results_ablated = eval_fn(
            hooked_model, test_loader, hooks=[rotation_hook], dtype=dtype, device=device
        )
        eval_results[n_vecs_remaining] = eval_results_ablated

        if early_stopping_threshold is not None:
            if i == 0:
                base_result = eval_results_ablated
            else:
                # If the result is more than `early_stopping_threshold` different than the base result,
                # then we stop ablating vectors.
                if abs(eval_results_ablated - base_result) > early_stopping_threshold:
                    break

    return eval_results


@torch.inference_mode()
def run_ablations(
    basis_matrices: list[tuple[BasisVecs, BasisVecsPinv]],
    node_layers: list[str],
    hooked_model: HookedModel,
    data_loader: DataLoader,
    eval_fn: Callable,
    graph_module_names: list[str],
    ablate_every_vec_cutoff: Optional[int],
    exp_base: Optional[float],
    device: str,
    dtype: Optional[torch.dtype] = None,
    early_stopping_threshold: Optional[float] = None,
) -> dict[str, dict[int, float]]:
    """Rotate to and from a truncated basis and compare ablation accuracies/losses.

    Args:
        basis_matrices: List of basis vector matrices and their pseudoinverses. In the orthogonal
            basis case, the pseudoinverse is the transpose.
        node_layers: The names of the node layers to build the graph with.
        hooked_model: The hooked model.
        data_loader: The data loader to use for testing.
        eval_fn: The function to use to evaluate the model.
        graph_module_names: The names of the modules we want to build the graph around.
        ablate_every_vec_cutoff: The point in the ablation schedule to start ablating every vector.
        exp_base: The base of the exponential schedule.
        device: The device to run the model on.
        dtype: The data type to cast the inputs to. Ignored if int32 or int64.
        early_stopping_threshold: The threshold to use for early stopping. If None, we don't use
            early stopping.

    Returns:
        A dictionary mapping node layers to ablation accuracies/losses.
    """
    results: dict[str, dict[int, float]] = {}
    for hook_name, module_name, (basis_vecs, basis_vecs_pinv) in zip(
        node_layers, graph_module_names, basis_matrices
    ):
        ablation_schedule = calc_ablation_schedule(
            ablate_every_vec_cutoff=ablate_every_vec_cutoff,
            n_vecs=basis_vecs.shape[1],
            exp_base=exp_base,
        )
        ablation_eval_results: dict[int, float] = ablate_and_test(
            hooked_model=hooked_model,
            module_name=module_name,
            basis_vecs=basis_vecs,
            basis_vecs_pinv=basis_vecs_pinv,
            test_loader=data_loader,
            eval_fn=eval_fn,
            ablation_schedule=ablation_schedule,
            device=device,
            dtype=dtype,
            hook_name=hook_name,
            early_stopping_threshold=early_stopping_threshold,
        )
        results[hook_name] = ablation_eval_results

    return results


def load_basis_matrices(
    interaction_graph_info: dict,
    node_layers: list[str],
    ablation_type: Literal["rib", "orthogonal"],
    dtype: torch.dtype,
    device: str,
) -> list[tuple[BasisVecs, BasisVecsPinv]]:
    """Load the basis matrices and their pseudoinverses.

    Supports both rib and orthogonal basis matrices. Converts each matrix to the specified dtype
    and device.
    """
    if ablation_type == "rib":
        basis_matrix_key = "interaction_rotations"
    elif ablation_type == "orthogonal":
        basis_matrix_key = "eigenvectors"
    else:
        raise ValueError(f"ablation_type must be one of ['rib', 'orthogonal']")

    # Get the basis vecs and their pseudoinverses using the module_names as keys
    basis_matrices: list[tuple[BasisVecs, BasisVecsPinv]] = []
    for module_name in node_layers:
        for basis_info in interaction_graph_info[basis_matrix_key]:
            if basis_info["node_layer_name"] == module_name:
                if ablation_type == "rib":
                    basis_vecs = basis_info["C"].to(dtype=dtype, device=device)
                    basis_vecs_pinv = basis_info["C_pinv"].to(dtype=dtype, device=device)
                elif ablation_type == "orthogonal":
                    # Pseudoinverse of an orthonormal matrix is its transpose
                    basis_vecs = basis_info["U"].to(dtype=dtype, device=device)
                    basis_vecs_pinv = basis_vecs.T.detach().clone()
                basis_matrices.append((basis_vecs, basis_vecs_pinv))
                break
    assert len(basis_matrices) == len(
        node_layers
    ), f"Could not find all node_layer modules in the interaction graph config."
    return basis_matrices
