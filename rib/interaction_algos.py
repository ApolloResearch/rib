"""This module contains algorithms related to interaction rotations
"""

from dataclasses import dataclass
from typing import Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from rib.data_accumulator import collect_M_dash_and_Lambda_dash
from rib.hook_manager import HookedModel
from rib.linalg import eigendecompose, pinv_diag


@dataclass
class InteractionRotation:
    """Dataclass storing the interaction rotation matrix and its inverse for a node layer."""

    node_layer_name: str
    C: Optional[Float[Tensor, "d_hidden d_hidden_trunc"]] = None
    # pseudoinverse of C, not needed for the output node layer
    C_pinv: Optional[Float[Tensor, "d_hidden_trunc d_hidden"]] = None


@dataclass
class Eigenvectors:
    """Dataclass storing the eigenvectors of a node layer."""

    node_layer_name: str
    U: Optional[Float[Tensor, "d_hidden d_hidden"]] = None


def build_sorted_lambda_matrices(
    Lambda_abs: Float[Tensor, "d_hidden_trunc"],
    truncation_threshold: float,
) -> tuple[
    Float[Tensor, "d_hidden_trunc d_hidden_extra_trunc"],
    Float[Tensor, "d_hidden_extra_trunc d_hidden_trunc"],
]:
    """Build the sqrt sorted Lambda matrix and its pseudoinverse.

    We truncate the Lambda matrix to remove small values.

    Args:
        Lambda_abs: Vector of the absolute values of the lambdas.

    Returns:
        - The sqrt of the sorted Lambda matrix
        - The pseudoinverse of the sqrt sorted Lambda matrix

    """
    # Get the sort indices in descending order
    idxs: Int[Tensor, "d_hidden_trunc"] = torch.argsort(Lambda_abs, descending=True)

    # Get the number of values we will truncate
    n_small_lambdas: int = int(torch.sum(Lambda_abs < truncation_threshold).item())

    truncated_idxs: Int[Tensor, "d_hidden_extra_trunc"] = (
        idxs[:-n_small_lambdas] if n_small_lambdas > 0 else idxs
    )

    Lambda_abs_sqrt: Float[Tensor, "d_hidden_trunc"] = Lambda_abs.sqrt()
    # Create a matrix from lambda_vals with the sorted columns and removing n_small_lambdas cols
    lambda_matrix: Float[Tensor, "d_hidden_trunc d_hidden_extra_trunc"] = torch.diag(
        Lambda_abs_sqrt
    )[:, truncated_idxs]
    # We also need the pseudoinverse of this matrix. We sort and remove the n_small_lambdas rows
    lambda_matrix_pinv: Float[Tensor, "d_hidden_extra_trunc d_hidden_trunc"] = torch.diag(
        Lambda_abs_sqrt.reciprocal()
    )[truncated_idxs, :]

    assert not torch.any(torch.isnan(lambda_matrix_pinv)), "NaNs in the pseudoinverse."
    # (lambda_matrix @ lambda_matrix_pinv).diag() should contain d_hidden_extra_trunc 1s and
    # d_hidden_trunc - d_hidden_extra_trunc 0s
    assert torch.allclose(
        (lambda_matrix @ lambda_matrix_pinv).diag().sum(),
        torch.tensor(lambda_matrix.shape[0] - n_small_lambdas, dtype=lambda_matrix.dtype),
    )

    return lambda_matrix, lambda_matrix_pinv


def calculate_interaction_rotations(
    gram_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]],
    module_names: list[str],
    hooked_model: HookedModel,
    data_loader: DataLoader,
    dtype: torch.dtype,
    device: str,
    n_intervals: int,
    logits_node_layer: bool = True,
    truncation_threshold: float = 1e-5,
    rotate_final_node_layer: bool = True,
    hook_names: Optional[list[str]] = None,
) -> tuple[list[InteractionRotation], list[Eigenvectors]]:
    """Calculate the interaction rotation matrices (denoted C) and their psuedo-inverses.

    This function implements Algorithm 2 (Pseudocode for RIB in transformers) of the paper. We name
    the variables as they are named in the paper.

    We collect the interaction rotation matrices from the output layer backwards, as we need the
    next layer's rotation to compute the current layer's rotation. We reverse the resulting Cs and
    Us back to the original node order before returning.

    The output layer will either be the logits or the inputs to the final node_layer in the config,
    depending on whether collect_logits in the config is True or False, respectively.

    Args:
        gram_matrices: The gram matrices for each layer, keyed by layer name.
        module_names: The names of the modules to build the graph from, in order of appearance.
        hooked_model: The hooked model.
        data_loader: The data loader.
        dtype: The data type to use for model computations.
        device: The device to run the model on.
        n_intervals: The number of intervals to use for integrated gradients.
        truncation_threshold: Remove eigenvectors with eigenvalues below this threshold.
        logits_node_layer: Whether to build an extra output node layer for the logits.
        rotate_final_node_layer: Whether to rotate the output layer to its eigenbasis (which is
            equivalent to its interaction basis). Defaults to True.
        hook_names: Used to store the interaction rotation matrices in the hooked model.

    Returns:
        - A list of objects containing the interaction rotation matrices and their pseudoinverses,
        ordered by node layer appearance in model.
        - A list of objects containing the eigenvectors of each node layer, ordered by node layer
        appearance in model.
    """
    assert len(module_names) > 0, "No modules specified."
    if hook_names is not None:
        assert len(hook_names) == len(module_names), "Must specify a hook name for each module."
    else:
        hook_names = module_names

    # We start appending Us and Cs from the output layer and work our way backwards
    Us: list[Eigenvectors] = []
    Cs: list[InteractionRotation] = []

    # The C matrix for the final layer is either the eigenvectors U if rotate_final_node_layer is
    # True, and None otherwise
    final_node_layer = "output" if logits_node_layer else hook_names[-1]

    # If not rotating the final layer, we don't need U or C
    U_output: Optional[Float[Tensor, "d_hidden d_hidden"]] = (
        eigendecompose(gram_matrices[final_node_layer])[1] if rotate_final_node_layer else None
    )
    C_output: Optional[Float[Tensor, "d_hidden d_hidden"]] = (
        U_output.clone().detach() if U_output is not None else None
    )
    U_output = U_output.detach().cpu() if U_output is not None else None
    Us.append(Eigenvectors(node_layer_name=final_node_layer, U=U_output))
    Cs.append(InteractionRotation(node_layer_name=final_node_layer, C=C_output))

    module_and_hook_names = (
        zip(module_names[::-1], hook_names[::-1])
        if logits_node_layer
        else zip(module_names[:-1][::-1], hook_names[:-1][::-1])
    )
    for module_name, hook_name in tqdm(
        module_and_hook_names,
        total=len(module_names),
        desc="Interaction rotations",
    ):
        D_dash, U_dash = eigendecompose(gram_matrices[hook_name])

        n_small_eigenvals: int = int(torch.sum(D_dash < truncation_threshold).item())
        # Truncate the D matrix to remove small eigenvalues
        D: Float[Tensor, "d_hidden_trunc d_hidden_trunc"] = (
            torch.diag(D_dash)[:-n_small_eigenvals, :-n_small_eigenvals]
            if n_small_eigenvals > 0
            else torch.diag(D_dash)
        )
        # Truncate the columns of U to remove small eigenvalues
        U: Float[Tensor, "d_hidden d_hidden_trunc"] = (
            U_dash[:, :-n_small_eigenvals] if n_small_eigenvals > 0 else U_dash
        )
        Us.append(Eigenvectors(node_layer_name=hook_name, U=U.detach().cpu()))

        M_dash, Lambda_dash = collect_M_dash_and_Lambda_dash(
            C_out=Cs[-1].C,  # most recently stored interaction matrix
            hooked_model=hooked_model,
            n_intervals=n_intervals,
            data_loader=data_loader,
            module_name=module_name,
            dtype=dtype,
            device=device,
            hook_name=hook_name,
        )

        U_D_sqrt: Float[Tensor, "d_hidden d_hidden_trunc"] = U @ D.sqrt()
        M: Float[Tensor, "d_hidden_trunc d_hidden_trunc"] = U_D_sqrt.T @ M_dash @ U_D_sqrt
        _, V = eigendecompose(M)  # V has size (d_hidden_trunc, d_hidden_trunc)

        # Multiply U_D_sqrt with V, corresponding to $U D^{1/2} V$ in the paper.
        U_D_sqrt_V: Float[Tensor, "d_hidden d_hidden_trunc"] = U_D_sqrt @ V
        D_sqrt_pinv: Float[Tensor, "d_hidden_trunc d_hidden_trunc"] = pinv_diag(D.sqrt())
        U_D_sqrt_pinv_V: Float[Tensor, "d_hidden d_hidden_trunc"] = U @ D_sqrt_pinv @ V
        Lambda_abs: Float[Tensor, "d_hidden_trunc"] = (
            (U_D_sqrt_V.T @ Lambda_dash @ U_D_sqrt_pinv_V).diag().abs()
        )

        Lambda_abs_sqrt_trunc, Lambda_abs_sqrt_trunc_pinv = build_sorted_lambda_matrices(
            Lambda_abs, truncation_threshold
        )

        C: Float[Tensor, "d_hidden d_hidden_extra_trunc"] = U_D_sqrt_pinv_V @ Lambda_abs_sqrt_trunc
        C_pinv: Float[Tensor, "d_hidden_extra_trunc d_hidden"] = (
            Lambda_abs_sqrt_trunc_pinv @ U_D_sqrt_V.T
        )
        Cs.append(
            InteractionRotation(
                node_layer_name=hook_name, C=C.clone().detach(), C_pinv=C_pinv.clone().detach()
            )
        )

    return Cs[::-1], Us[::-1]
