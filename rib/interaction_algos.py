"""This module contains algorithms related to interaction rotations
"""

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from rib.data_accumulator import collect_M_dash_and_Lambda_dash
from rib.hook_manager import HookedModel
from rib.linalg import eigendecompose, move_const_dir_first, pinv_diag, shift_matrix
from rib.models.mlp import MLP
from rib.models.sequential_transformer.transformer import SequentialTransformer


@dataclass
class InteractionRotation:
    """Dataclass storing the interaction rotation matrix and its inverse for a node layer."""

    node_layer_name: str
    out_dim: int  # Equal to d_hidden_extra_trunc if C is not None and d_hidden otherwise
    C: Optional[Float[Tensor, "orig_coords rib_dir_idx"]] = None
    # pseudoinverse of C, not needed for the output node layer
    C_pinv: Optional[Float[Tensor, "rib_dir_idx orig_coords"]] = None

    def __post_init__(self):
        if self.C is not None:
            assert self.C.shape[1] == self.out_dim, f"Expected C to have shape (_, {self.out_dim})"


@dataclass
class Eigenvectors:
    """Dataclass storing the eigenvectors of a node layer."""

    node_layer_name: str
    out_dim: int  # Equal to d_hidden_trunc if U is not None and d_hidden otherwise
    U: Optional[Float[Tensor, "d_hidden d_hidden_trunc"]] = None

    def __post_init__(self):
        if self.U is not None:
            assert self.U.shape[1] == self.out_dim, f"Expected U to have shape (_, {self.out_dim})"


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
    section_names: list[str],
    node_layers: list[str],
    hooked_model: HookedModel,
    data_loader: DataLoader,
    dtype: torch.dtype,
    device: str,
    n_intervals: int,
    M_dtype: torch.dtype = torch.float64,
    Lambda_einsum_dtype: torch.dtype = torch.float64,
    truncation_threshold: float = 1e-5,
    rotate_final_node_layer: bool = True,
    basis_formula: Literal["(1-alpha)^2", "(1-0)*alpha", "svd", "neuron"] = "(1-alpha)^2",
    center: bool = False,
    means: Optional[dict[str, Float[Tensor, "d_hidden"]]] = None,
    bias_positions: Optional[dict[str, Int[Tensor, "sections"]]] = None,
) -> tuple[list[InteractionRotation], list[Eigenvectors]]:
    """Calculate the interaction rotation matrices (denoted C) and their psuedo-inverses.

    This function implements Algorithm 2 (Pseudocode for RIB in transformers) of the paper. We name
    the variables as they are named in the paper.

    We collect the interaction rotation matrices from the output layer backwards, as we need the
    next layer's rotation to compute the current layer's rotation. We reverse the resulting Cs and
    Us back to the original node order before returning.

    Args:
        gram_matrices: The gram matrices for each layer, keyed by layer name.
        section_names: The names of the sections to build the graph from, in order of appearance.
            Recall that each section is a pytorch module that can be hooked on. For MNIST, or other
            models without explicit sections, this will simply correspond to layer names.
        node_layers: Used as a key to store the interaction rotation matrices in the hooked model.
            May include an optional "output" for the final node layer.
        hooked_model: The hooked model.
        data_loader: The data loader.
        dtype: The data type to use for model computations.
        device: The device to run the model on.
        n_intervals: The number of intervals to use for integrated gradients.
        M_dtype: The data type to use for the M_dash and M matrices, including where the M is
            collected over the dataset in `M_dash_and_Lambda_dash_pre_forward_hook_fn`. Needs to be
            float64 for Pythia-14m (empirically). Defaults to float64.
        Lambda_einsum_dtype: The data type to use for the einsum computing batches for the
            Lambda_dash matrix. Does not affect the output, only used for the einsum within
            M_dash_and_Lambda_dash_pre_forward_hook_fn. Needs to be float64 on CPU but float32 was
            fine on GPU. Defaults to float64.
        truncation_threshold: Remove eigenvectors with eigenvalues below this threshold.
        rotate_final_node_layer: Whether to rotate the final layer to its eigenbasis (which is
            equivalent to its interaction basis). Defaults to True.
        basis_formula: The formula to use for the integrated gradient. Must be one of
            "(1-alpha)^2", "(1-0)*alpha", "neuron", or "svd".
             - "(1-alpha)^2" is the old (October) version based on the functional edge formula.
             - "(1-0)*alpha" is the new (November) version based on the squared edge formula. This
                is the default, and generally the best option.
             - "neuron" performs no rotation.
             - "svd" only decomposes the gram matrix and uses that as the basis. It is a good
                baseline. If `center=true` this becomes the "pca" basis.
        center: Whether to center the activations while computing the desired basis. Only supported
            for the "svd" basis formula.
        means: The means of the activations for each node layer. Only used if `center=true`.
        bias_positions: The positions of the biases for each node layer. Only used if `center=true`.
    Returns:
        - A list of objects containing the interaction rotation matrices and their pseudoinverses,
        ordered by node layer appearance in model.
        - A list of objects containing the eigenvectors of each node layer, ordered by node layer
        appearance in model.
    """
    assert hooked_model.model.has_folded_bias, "Biases must be folded in to calculate Cs."
    assert len(section_names) > 0, "No sections specified."

    non_output_node_layers = [node_layer for node_layer in node_layers if node_layer != "output"]
    assert len(non_output_node_layers) == len(
        section_names
    ), "Must specify a hook name for each section (except the output section)."

    if center and basis_formula == "neuron":
        raise NotImplementedError("centering is not currently implemented for the neuron basis.")

    # We start appending Us and Cs from the output layer and work our way backwards
    Us: list[Eigenvectors] = []
    Cs: list[InteractionRotation] = []

    # The C matrix for the final layer is either the eigenvectors U if rotate_final_node_layer is
    # True, and None otherwise
    U_output: Optional[Float[Tensor, "d_hidden d_hidden"]] = None
    C_output: Optional[Float[Tensor, "d_hidden d_hidden"]] = None
    if rotate_final_node_layer:
        # TODO: move bias pos to first pos
        U_output = eigendecompose(gram_matrices[node_layers[-1]])[1].detach()
        assert U_output is not None
        if center:
            assert means is not None
            assert bias_positions is not None
            assert node_layers[-1] in means
            Y = shift_matrix(-means[node_layers[-1]], bias_positions[node_layers[-1]])
            C_output = (Y @ U_output).detach().cpu()
        else:
            C_output = U_output.detach().cpu()
        U_output = U_output.cpu()

    if node_layers[-1] not in gram_matrices:
        # Technically we don't actually need the final node layer to be in gram_matrices if we're
        # not rotating it, but for now, our implementation assumes that it always is unless
        # final_node_layer is the logits (i.e. ="output").
        assert (
            node_layers[-1] == "output"
        ), f"Final node layer {node_layers[-1]} not in gram matrices."

        inner_model = hooked_model.model
        if isinstance(inner_model, MLP):
            final_node_dim = inner_model.output_size
        else:
            assert isinstance(inner_model, SequentialTransformer)
            final_node_dim = inner_model.cfg.d_vocab
    else:
        final_node_dim = gram_matrices[node_layers[-1]].shape[0]

    Us.append(
        Eigenvectors(
            node_layer_name=node_layers[-1],
            out_dim=final_node_dim,
            U=U_output,
        )
    )
    Cs.append(
        InteractionRotation(
            node_layer_name=node_layers[-1],
            out_dim=final_node_dim,
            C=C_output,
        )
    )
    if U_output is not None:
        assert U_output is not None
        assert C_output is not None
        assert (
            C_output.shape[1] == final_node_dim
        ), f"Expected C_output to have shape (_, {final_node_dim}). Got {C_output.shape}."
        assert (
            U_output.shape[1] == final_node_dim
        ), f"Expected U_output to have shape (_, {final_node_dim}). Got {U_output.shape}."

    # We only need to calculate C for the final section if there is no output node layer
    section_names_to_calculate = (
        section_names if node_layers[-1] == "output" else section_names[:-1]
    )

    assert (
        len(section_names_to_calculate) == len(node_layers) - 1
    ), "Must be a section name for all but the final node_layer which was already handled above."

    # Since we've already handled the last node layer, we can ignore it in the loop
    for node_layer, section_name in tqdm(
        zip(node_layers[-2::-1], section_names_to_calculate[::-1]),
        total=len(section_names_to_calculate),
        desc="Interaction rotations",
    ):
        if basis_formula == "neuron":
            # Use identity matrix as C and then progress to the next loop
            # TODO assert not rotate final
            width = gram_matrices[node_layer].shape[0]
            Id = torch.eye(width, dtype=dtype, device="cpu")
            Us.append(Eigenvectors(node_layer_name=node_layer, out_dim=width, U=Id))
            Cs.append(
                InteractionRotation(node_layer_name=node_layer, out_dim=width, C=Id, C_pinv=Id)
            )
            continue

        D_dash, U_dash = eigendecompose(gram_matrices[node_layer])
        if center:
            D_dash, U_dash = move_const_dir_first(D_dash, U_dash)

        # we trucate all directions with eigenvalues smaller than some threshold
        mask = D_dash > truncation_threshold  # true if we keep the direction
        D: Float[Tensor, "d_hidden_trunc d_hidden_trunc"] = D_dash[mask].diag()
        U: Float[Tensor, "d_hidden d_hidden_trunc"] = U_dash[:, mask]

        Us.append(Eigenvectors(node_layer_name=node_layer, out_dim=U.shape[1], U=U.detach().cpu()))

        # currently only used for svd basis, but will be used more in text PR
        if center:
            assert means is not None
            assert bias_positions is not None
            # Y (or Gamma) is a matrix that shifts the activations to be mean zero
            # with the exception of the bias positions which are still 1
            Y = shift_matrix(-means[node_layer], bias_positions[node_layer])
            Y_inv = shift_matrix(means[node_layer], bias_positions[node_layer])
        else:
            # if not centering, we set Y to be the identity matrix
            Y = torch.eye(U.shape[0], dtype=U.dtype, device=U.device)
            Y_inv = torch.eye(U.shape[0], dtype=U.dtype, device=U.device)

        if basis_formula == "svd":
            # We set C based on U (maybe centered as well)
            Cs.append(
                InteractionRotation(
                    node_layer_name=node_layer,
                    out_dim=U.shape[1],
                    C=(Y @ U).cpu(),
                    C_pinv=(U.T @ Y_inv).cpu(),
                )
            )
            continue

        # Most recently stored interaction matrix
        C_out = Cs[-1].C.to(device=device) if Cs[-1].C is not None else None
        M_dash, Lambda_dash = collect_M_dash_and_Lambda_dash(
            C_out=C_out,
            hooked_model=hooked_model,
            n_intervals=n_intervals,
            data_loader=data_loader,
            module_name=section_name,
            dtype=dtype,
            device=device,
            hook_name=node_layer,
            M_dtype=M_dtype,
            Lambda_einsum_dtype=Lambda_einsum_dtype,
            basis_formula=basis_formula,
        )

        Yinv_U_Dsqrt: Float[Tensor, "d_hidden d_hidden_trunc"] = Y_inv.T @ U @ D.sqrt()

        # Converts M to fp64
        M: Float[Tensor, "d_hidden_trunc d_hidden_trunc"] = (
            Yinv_U_Dsqrt.T.to(M_dtype) @ M_dash @ Yinv_U_Dsqrt.to(M_dtype)
        )
        if center:
            # we want to preserve having a unique constant direction in the RIB basis.
            # this constant direction is in the 0th position, so we eigendecompose the submatrix
            # excluding this direction
            sub_V = eigendecompose(M[1:, 1:])[1]
            V = torch.zeros_like(M)
            V[0, 0] = 1
            V[1:, 1:] = sub_V
        else:
            V = eigendecompose(M)[1]
        V = V.to(dtype)  # V has size (d_hidden_trunc, d_hidden_trunc)

        # left transform for lambda dash, first transform for C_pinv
        Vt_Dsqrt_Ut_Yinv = V.T @ D.sqrt() @ U.T @ Y_inv
        # right transform for lambda dash, first transform for C
        Y_U_Dsqrtpinv_V = Y @ U @ pinv_diag(D.sqrt()) @ V

        Lambda_abs: Float[Tensor, "d_hidden_trunc"] = (
            (Vt_Dsqrt_Ut_Yinv @ Lambda_dash @ Y_U_Dsqrtpinv_V).diag().abs()
        )

        Lambda_abs_sqrt_trunc, Lambda_abs_sqrt_trunc_pinv = build_sorted_lambda_matrices(
            Lambda_abs, truncation_threshold
        )

        C: Float[Tensor, "d_hidden d_hidden_extra_trunc"] = (
            (Y_U_Dsqrtpinv_V @ Lambda_abs_sqrt_trunc).detach().cpu()
        )
        C_pinv: Float[Tensor, "d_hidden_extra_trunc d_hidden"] = (
            (Lambda_abs_sqrt_trunc_pinv @ Vt_Dsqrt_Ut_Yinv).detach().cpu()
        )
        Cs.append(
            InteractionRotation(node_layer_name=node_layer, out_dim=C.shape[1], C=C, C_pinv=C_pinv)
        )

    return Cs[::-1], Us[::-1]
