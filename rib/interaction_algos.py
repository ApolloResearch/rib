"""This module contains algorithms related to interaction rotations
"""

from dataclasses import dataclass
from typing import Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader

from rib.data_accumulator import collect_M_dash_and_Lambda_dash
from rib.hook_manager import HookedModel
from rib.linalg import eigendecompose, pinv_diag


@dataclass
class InteractionRotation:
    """Dataclass storing the interaction rotation matrix and its inverse for a node layer."""

    node_layer_name: str
    C: Float[Tensor, "d_hidden d_hidden_trunc"]
    # pseudoinverse of C, not needed for the output node layer
    C_pinv: Optional[Float[Tensor, "d_hidden_trunc d_hidden"]] = None


def build_sorted_lambda_matrices(
    lambda_vals: Float[Tensor, "d_hidden_trunc"],
) -> tuple[
    Float[Tensor, "d_hidden_trunc d_hidden_trunc"], Float[Tensor, "d_hidden_trunc d_hidden_trunc"]
]:
    """Build the sorted Lambda matrix and its pseudoinverse.

    Args:
        lambda_vals: The vectors of lambda values

    Returns:
        - The sorted Lambda matrix
        - The pseudoinverse of the sorted Lambda matrix

    """
    # Get the sort indices in descending order
    idxs: Int[Tensor, "d_hidden_trunc"] = torch.argsort(lambda_vals, descending=True)
    # Create a matrix from lambda_vals with the columns sorted in descending order
    # of their values
    lambda_matrix: Float[Tensor, "d_hidden_trunc d_hidden_trunc"] = torch.diag(lambda_vals)[:, idxs]
    # We also need the pseudoinverse of this matrix
    lambda_matrix_pinv: Float[Tensor, "d_hidden_trunc d_hidden_trunc"] = torch.diag(
        lambda_vals.reciprocal()
    )[idxs, :]

    assert not torch.any(torch.isnan(lambda_matrix_pinv)), "NaNs in the pseudoinverse."
    assert torch.allclose(
        lambda_matrix @ lambda_matrix_pinv,
        torch.eye(lambda_matrix.shape[0], dtype=lambda_vals.dtype, device=lambda_vals.device),
        atol=1e-6,
    ), "Lambda matrix and its pseudoinverse are not inverses of each other."

    return lambda_matrix, lambda_matrix_pinv


def calculate_interaction_rotations(
    gram_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]],
    module_names: list[str],
    hooked_model: HookedModel,
    data_loader: DataLoader,
    dtype: torch.dtype,
    device: str,
    truncation_threshold: float = 1e-5,
    rotate_output: bool = True,
    hook_names: Optional[list[str]] = None,
) -> list[InteractionRotation]:
    """Calculate the interaction rotation matrices (denoted C) and their psuedo-inverses.

    This function implements Algorithm 1 of the paper. We name the variables as they are named in
    the paper.

    Recall that we have one more node layer than module layer, as we have a node layer for the
    output of the final module.

    We collect the interaction rotation matrices from the output layer backwards, as we need the
    next layer's rotation to compute the current layer's rotation. We reverse the list at the end.

    Args:
        gram_matrices: The gram matrices for each layer, keyed by layer name. Must include
            an `output` key for the output layer.
        module_names: The names of the modules to build the graph from, in order of appearance.
        hooked_model: The hooked model.
        data_loader: The data loader.
        dtype: The data type to use for model computations.
        device: The device to run the model on.
        truncation_threshold: Remove eigenvectors with eigenvalues below this threshold.
        rotate_output: Whether to rotate the output layer to its eigenbasis (which is equivalent
            to its interaction basis).
        hook_names: Used to store the interaction rotation matrices in the hooked model.

    Returns:
        A list of objects contain interaction rotation matrices and their pseudoinverses, ordered
        by node layer appearance in model.
    """
    assert "output" in gram_matrices, "Gram matrices must include an `output` key."
    assert len(module_names) > 0, "No modules specified."
    if hook_names is not None:
        assert len(hook_names) == len(module_names), "Must specify a hook name for each module."
    else:
        hook_names = module_names

    # We start appending Cs from the output layer and work our way backwards
    Cs: list[InteractionRotation] = []
    if rotate_output:
        _, U_output = eigendecompose(gram_matrices["output"])
        C_output: Float[Tensor, "d_hidden d_hidden"] = U_output
    else:
        C_output = torch.eye(
            gram_matrices["output"].shape[0],
            device=gram_matrices["output"].device,
            dtype=gram_matrices["output"].dtype,
        )
    Cs.append(InteractionRotation(node_layer_name="output", C=C_output.clone().detach()))

    for module_name, hook_name in zip(module_names[::-1], hook_names[::-1]):
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

        M_dash, Lambda_dash = collect_M_dash_and_Lambda_dash(
            C_out=Cs[-1].C,  # most recently stored interaction matrix
            hooked_model=hooked_model,
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
        U_D_sqrt_pinv_T_V: Float[Tensor, "d_hidden d_hidden_trunc"] = U @ D_sqrt_pinv.T @ V
        Lambda_raw: Float[Tensor, "d_hidden_trunc d_hidden_trunc"] = (
            U_D_sqrt_V.T @ Lambda_dash @ U_D_sqrt_pinv_T_V
        )
        # We only care about the sqrt of the absolute value of diagonal elements of Lambda_raw
        Lambda_diag_abs_sqrt: Float[Tensor, "d_hidden_trunc"] = Lambda_raw.diag().abs().sqrt()

        Lambda_abs_sqrt, Lambda_abs_sqrt_pinv = build_sorted_lambda_matrices(Lambda_diag_abs_sqrt)

        C: Float[Tensor, "d_hidden d_hidden_trunc"] = U_D_sqrt_pinv_T_V @ Lambda_abs_sqrt
        C_pinv: Float[Tensor, "d_hidden_trunc d_hidden"] = Lambda_abs_sqrt_pinv @ U_D_sqrt_V.T
        Cs.append(
            InteractionRotation(
                node_layer_name=hook_name, C=C.clone().detach(), C_pinv=C_pinv.clone().detach()
            )
        )

    return Cs[::-1]
