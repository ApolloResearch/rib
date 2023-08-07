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
from rib.linalg import eigendecompose, pinv_truncated_diag


@dataclass
class InteractionRotation:
    """Dataclass storing the interaction rotation matrix and its inverse for a node layer."""

    node_layer_name: str
    C: Float[Tensor, "d_hidden d_hidden_trunc"]
    # pseudoinverse of C, not needed for the output node layer
    C_pinv: Optional[Float[Tensor, "d_hidden_trunc d_hidden"]] = None


def calculate_interaction_rotations(
    gram_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]],
    module_names: list[str],
    hooked_model: HookedModel,
    data_loader: DataLoader,
    device: str,
    dtype: str = "float32",
    truncation_threshold: float = 1e-5,
    rotate_output: bool = True,
) -> list[InteractionRotation]:
    """Calculate the interaction rotation matrices (denoted C) and their inverses.

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
        device: The device to run the model on.
        dtype: The data type to run the model on.
        truncation_threshold: Remove eigenvectors with eigenvalues below this threshold.
        rotate_output: Whether to rotate the output layer to its eigenbasis (which is equivalent
            to its interaction basis).

    Returns:
        A list of objects contain interaction rotation matrices and their pseudoinverses, ordered
        by node layer appearance in model.
    """
    assert "output" in gram_matrices, "Gram matrices must include an `output` key."

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
    Cs.append(InteractionRotation(node_layer_name="output", C=C_output))

    for module_name in module_names[::-1]:
        D_dash, U = eigendecompose(gram_matrices[module_name])

        M_dash, Lambda_dash = collect_M_dash_and_Lambda_dash(
            C_out=Cs[-1].C,  # most recently stored interaction matrix
            hooked_model=hooked_model,
            data_loader=data_loader,
            module_name=module_name,
            device=device,
            dtype=dtype,
        )
        # Create sqaure matrix from eigenvalues then remove cols with vals < truncation_threshold
        n_small_eigenvals: int = int(torch.sum(D_dash < truncation_threshold).item())
        D: Float[Tensor, "d_hidden d_hidden_trunc"] = (
            torch.diag(D_dash)[:, :-n_small_eigenvals]
            if n_small_eigenvals > 0
            else torch.diag(D_dash)
        )
        U_D_sqrt: Float[Tensor, "d_hidden d_hidden_trunc"] = U @ D.sqrt()
        M: Float[Tensor, "d_hidden_trunc d_hidden_trunc"] = U_D_sqrt.T @ M_dash @ U_D_sqrt
        _, V = eigendecompose(M)  # V has size (d_hidden_trunc, d_hidden_trunc)

        # Multiply U_D_sqrt with V, corresponding to $U D^{1/2} V$ in the paper.
        U_D_sqrt_V: Float[Tensor, "d_hidden d_hidden_trunc"] = U_D_sqrt @ V
        Lambda_raw: Float[Tensor, "d_hidden_trunc d_hidden_trunc"] = (
            U_D_sqrt_V.T @ Lambda_dash @ U_D_sqrt_V
        )
        # We only care about the sqrt of the absolute value of diagonal elements of Lambda_raw
        Lambda_diag_abs_sqrt: Float[Tensor, "d_hidden_trunc"] = Lambda_raw.diag().abs().sqrt()
        # Get the sort indices in descending order
        Lambda_indices: Int[Tensor, "d_hidden_trunc"] = torch.argsort(
            Lambda_diag_abs_sqrt, descending=True
        )
        # Create a matrix from Lambda_diag_abs_sqrt with the columns sorted in descending order
        # of their values
        Lambda_abs_sqrt: Float[Tensor, "d_hidden_trunc d_hidden_trunc"] = torch.diag(
            Lambda_diag_abs_sqrt
        )[:, Lambda_indices]
        # We also need the pseudoinverse of this matrix
        Lambda_abs_sqrt_pinv: Float[Tensor, "d_hidden_trunc d_hidden_trunc"] = torch.diag(
            Lambda_diag_abs_sqrt.reciprocal()
        )[:, Lambda_indices]
        # Take the pseudoinverse of the sqrt of D. Can simply take the elementwise inverse
        # of the diagonal elements, since D is diagonal.
        D_sqrt_inv: Float[Tensor, "d_hidden_trunc d_hidden"] = pinv_truncated_diag(D.sqrt())

        C: Float[Tensor, "d_hidden d_hidden_trunc"] = U @ D_sqrt_inv.T @ V @ Lambda_abs_sqrt
        C_pinv: Float[Tensor, "d_hidden_trunc d_hidden"] = Lambda_abs_sqrt_pinv @ U_D_sqrt_V.T
        Cs.append(InteractionRotation(node_layer_name=module_name, C=C, C_pinv=C_pinv))

    return Cs[::-1]
