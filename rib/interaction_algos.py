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


@dataclass
class Eigenvectors:
    """Dataclass storing the eigenvectors of a node layer."""

    node_layer_name: str
    U: Float[Tensor, "d_hidden d_hidden"]


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

    # Verify various properties of the lambda matrices
    assert not torch.any(torch.isnan(lambda_matrix_pinv)), "NaNs in the pseudoinverse."

    lambda_lambda_pinv = lambda_matrix @ lambda_matrix_pinv
    # (lambda_matrix @ lambda_matrix_pinv).diag() should contain d_hidden_extra_trunc 1s and
    # d_hidden_trunc - d_hidden_extra_trunc 0s
    assert torch.allclose(
        lambda_lambda_pinv.diag().sum(),
        torch.tensor(lambda_matrix.shape[0] - n_small_lambdas, dtype=lambda_matrix.dtype),
    )
    # zero out diag els
    lambda_lambda_pinv[
        torch.arange(lambda_lambda_pinv.shape[0]), torch.arange(lambda_lambda_pinv.shape[0])
    ] = 0.0
    # assert all off-diagonal elements are zero
    assert torch.allclose(lambda_lambda_pinv, torch.zeros_like(lambda_lambda_pinv))

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
) -> tuple[list[InteractionRotation], list[Eigenvectors]]:
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
        - A list of objects containing the interaction rotation matrices and their pseudoinverses,
        ordered by node layer appearance in model.
        - A list of objects containing the eigenvectors of each node layer, ordered by node layer
        appearance in model.
    """
    assert "output" in gram_matrices, "Gram matrices must include an `output` key."
    assert len(module_names) > 0, "No modules specified."
    if hook_names is not None:
        assert len(hook_names) == len(module_names), "Must specify a hook name for each module."
    else:
        hook_names = module_names

    _, U_output = eigendecompose(gram_matrices["output"])
    Us: list[Eigenvectors] = [Eigenvectors(node_layer_name="output", U=U_output.detach().cpu())]

    # We start appending Us and Cs from the output layer and work our way backwards
    Cs: list[InteractionRotation] = []
    if rotate_output:
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
        # Set D to an identity matrix for testing
        # D = torch.eye(D.shape[0], dtype=D.dtype, device=D.device)
        # Set D to a diagonal matrix with floating point numbers between 1 and 2
        # D = torch.zeros(D.shape, dtype=D.dtype, device=D.device)
        # D[torch.arange(D.shape[0]), torch.arange(D.shape[0])] = (
        #     torch.rand(D.shape[0], dtype=D.dtype, device=D.device) * 0.0001
        # )

        # Truncate the columns of U to remove small eigenvalues
        U: Float[Tensor, "d_hidden d_hidden_trunc"] = (
            U_dash[:, :-n_small_eigenvals] if n_small_eigenvals > 0 else U_dash
        )
        Us.append(Eigenvectors(node_layer_name=hook_name, U=U.detach().cpu()))

        M_dash, Lambda_dash = collect_M_dash_and_Lambda_dash(
            C_out=Cs[-1].C,  # most recently stored interaction matrix
            hooked_model=hooked_model,
            data_loader=data_loader,
            module_name=module_name,
            dtype=dtype,
            device=device,
            hook_name=hook_name,
        )

        # # NOTE: For testing
        # if hook_name == "unembed":
        #     W_U = hooked_model.model.sections.section_2[0].W_U.detach().clone()
        #     M_dash = W_U @ W_U.T
        #     # Make M_dash be a random matrix
        #     # M_dash = torch.rand(M_dash.shape, dtype=M_dash.dtype, device=M_dash.device)
        #     # Add some random noise to the M_dash matrix
        #     # M_dash += torch.rand(M_dash.shape, dtype=M_dash.dtype, device=M_dash.device) * 0.1

        U_D_sqrt: Float[Tensor, "d_hidden d_hidden_trunc"] = U @ D.sqrt()
        temp1 = U @ pinv_diag(D.sqrt())
        temp2 = D.sqrt() @ U.T
        # M: Float[Tensor, "d_hidden_trunc d_hidden_trunc"] = U_D_sqrt.T @ M_dash @ U_D_sqrt
        # M_eigvals, V = eigendecompose(M)  # V has size (d_hidden_trunc, d_hidden_trunc)
        # # Make V be an identity to test that we get the same results
        # # V = torch.eye(V.shape[0], dtype=V.dtype, device=V.device)

        # # Multiply U_D_sqrt with V, corresponding to $U D^{1/2} V$ in the paper.
        # U_D_sqrt_V: Float[Tensor, "d_hidden d_hidden_trunc"] = U_D_sqrt @ V
        # D_sqrt_pinv: Float[Tensor, "d_hidden_trunc d_hidden_trunc"] = pinv_diag(D.sqrt())
        # U_D_sqrt_pinv_T_V: Float[Tensor, "d_hidden d_hidden_trunc"] = U @ D_sqrt_pinv.T @ V
        # Lambda_abs: Float[Tensor, "d_hidden_trunc"] = (
        #     (U_D_sqrt_V.T @ Lambda_dash @ U_D_sqrt_pinv_T_V).diag().abs()
        # )

        # Lambda_abs_sqrt_trunc, Lambda_abs_sqrt_trunc_pinv = build_sorted_lambda_matrices(
        #     Lambda_abs, truncation_threshold
        # )
        # Change all the non-zero values to 1s
        # Lambda_abs_sqrt_trunc[Lambda_abs_sqrt_trunc != 0.0] = 1.0
        # Lambda_abs_sqrt_trunc_pinv[Lambda_abs_sqrt_trunc_pinv != 0.0] = 1.0

        # Lambda_abs_sqrt_trunc = Lambda_abs.diag()
        # Lambda_abs_sqrt_trunc_pinv = Lambda_abs.reciprocal().diag()
        # # Set both lambda matrices to (truncated) identity matrices
        # Lambda_abs_sqrt_trunc = torch.zeros_like(Lambda_abs_sqrt_trunc)
        # # Set the leading diagonal to ones
        # Lambda_abs_sqrt_trunc[
        #     torch.arange(Lambda_abs_sqrt_trunc.shape[1]),
        #     torch.arange(Lambda_abs_sqrt_trunc.shape[1]),
        # ] = 1.0
        # Lambda_abs_sqrt_trunc_pinv = torch.zeros_like(Lambda_abs_sqrt_trunc_pinv)
        # Lambda_abs_sqrt_trunc_pinv[
        #     torch.arange(Lambda_abs_sqrt_trunc_pinv.shape[0]),
        #     torch.arange(Lambda_abs_sqrt_trunc_pinv.shape[0]),
        # ] = 1.0

        # C: Float[Tensor, "d_hidden d_hidden_extra_trunc"] = (
        #     U_D_sqrt_pinv_T_V @ Lambda_abs_sqrt_trunc
        # )
        # C_pinv: Float[Tensor, "d_hidden_extra_trunc d_hidden"] = (
        #     Lambda_abs_sqrt_trunc_pinv @ U_D_sqrt_V.T
        # )
        C = temp1
        C_pinv = temp2
        Cs.append(
            InteractionRotation(
                node_layer_name=hook_name, C=C.clone().detach(), C_pinv=C_pinv.clone().detach()
            )
        )

    return Cs[::-1], Us[::-1]
