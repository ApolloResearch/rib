"""This module contains algorithms related to interaction rotations."""

from typing import Literal, Optional

import torch
from jaxtyping import Float, Int
from pydantic import AfterValidator, BaseModel, ConfigDict, ValidationInfo
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Annotated

from rib.data_accumulator import collect_M_dash_and_Lambda_dash
from rib.hook_manager import HookedModel
from rib.linalg import centering_matrix, eigendecompose, move_const_dir_first, pinv_diag
from rib.models.mlp import MLP
from rib.models.transformer import SequentialTransformer
from rib.utils import check_device_is_cpu


def check_second_dim_is_out_dim(
    X: Optional[torch.Tensor], info: ValidationInfo
) -> Optional[torch.Tensor]:
    if X is not None:
        assert (
            X.shape[1] == info.data["out_dim"]
        ), f"Expected tensor to have shape (_, {info.data['out_dim']}). Got {X.shape}."
    return X


class InteractionRotation(BaseModel):
    """Stores an interaction rotation matrix and its pseudo-inverse for a node layer."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    node_layer_name: str
    out_dim: int  # Equal to 'rib' if C is not None and 'orig' otherwise
    # if centering was used, C[:, 0] is the constant direction
    C: Annotated[
        Optional[Float[Tensor, "orig rib"]],
        AfterValidator(check_second_dim_is_out_dim),
        AfterValidator(check_device_is_cpu),
    ] = None
    # pseudoinverse of C, not needed for the output node layer
    C_pinv: Annotated[
        Optional[Float[Tensor, "rib orig"]],
        AfterValidator(check_device_is_cpu),
    ] = None


class Eigenvectors(BaseModel):
    """Stores eigenvectors of a node layer."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    node_layer_name: str
    out_dim: int  # Equal to orig_trunc if U is not None and orig otherwise
    U: Annotated[
        Optional[Float[Tensor, "orig orig_trunc"]],
        AfterValidator(check_second_dim_is_out_dim),
        AfterValidator(check_device_is_cpu),
    ] = None


def build_sorted_lambda_matrices(
    Lambda_abs: Float[Tensor, "orig_trunc"],
    truncation_threshold: float,
) -> tuple[Float[Tensor, "orig_trunc rib"], Float[Tensor, "rib orig_trunc"],]:
    """Build the sqrt sorted Lambda matrix and its pseudoinverse.

    We truncate the Lambda matrix to remove small values.

    Args:
        Lambda_abs: Vector of the absolute values of the lambdas.

    Returns:
        - The sqrt of the sorted Lambda matrix
        - The pseudoinverse of the sqrt sorted Lambda matrix

    """
    # Get the sort indices in descending order
    idxs: Int[Tensor, "orig_trunc"] = torch.argsort(Lambda_abs, descending=True)

    # Get the number of values we will truncate
    n_small_lambdas: int = int(torch.sum(Lambda_abs < truncation_threshold).item())

    truncated_idxs: Int[Tensor, "rib"] = idxs[:-n_small_lambdas] if n_small_lambdas > 0 else idxs

    Lambda_abs_sqrt: Float[Tensor, "orig_trunc"] = Lambda_abs.sqrt()
    # Create a matrix from lambda_vals with the sorted columns and removing n_small_lambdas cols
    lambda_matrix: Float[Tensor, "orig_trunc rib"] = torch.diag(Lambda_abs_sqrt)[:, truncated_idxs]
    # We also need the pseudoinverse of this matrix. We sort and remove the n_small_lambdas rows
    lambda_matrix_pinv: Float[Tensor, "rib orig_trunc"] = torch.diag(Lambda_abs_sqrt.reciprocal())[
        truncated_idxs, :
    ]

    assert not torch.any(torch.isnan(lambda_matrix_pinv)), "NaNs in the pseudoinverse."
    # (lambda_matrix @ lambda_matrix_pinv).diag() should contain rib 1s and
    # orig_trunc - rib 0s
    assert torch.allclose(
        (lambda_matrix @ lambda_matrix_pinv).diag().sum(),
        torch.tensor(lambda_matrix.shape[0] - n_small_lambdas, dtype=lambda_matrix.dtype),
    )

    return lambda_matrix, lambda_matrix_pinv


def calculate_interaction_rotations(
    gram_matrices: dict[str, Float[Tensor, "orig orig"]],
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
    basis_formula: Literal["(1-alpha)^2", "(1-0)*alpha", "svd", "neuron"] = "(1-0)*alpha",
    center: bool = False,
    means: Optional[dict[str, Float[Tensor, "d_hidden"]]] = None,
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
            ["(1-alpha)^2","(1-0)*alpha", "neuron", "svd"]. Defaults to "(1-0)*alpha".
                - "(1-alpha)^2" is the old (October) version based on the functional edge formula.
                - "(1-0)*alpha" is the new (November) version based on the squared edge formula.
                    This is the default, and generally the best option.
                - "neuron" performs no rotation.
                - "svd" only decomposes the gram matrix and uses that as the basis. It is a good
                    baseline. If `center=true` this becomes the "pca" basis.
        center: Whether to center the activations while computing the desired basis. Only supported
            for the "svd" basis formula.
        means: The means of the activations for each node layer. Only used if `center=true`.
    Returns:
        - A list of objects containing the interaction rotation matrices and their pseudoinverses
            for each node layer.
        - A list of objects containing the eigenvectors of each node layer
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
    U_output: Optional[Float[Tensor, "orig orig"]] = None
    C_output: Optional[Float[Tensor, "orig orig"]] = None
    if rotate_final_node_layer:
        # We don't use D_output except in finding the index of the const_dir in U
        D_output, U_output = eigendecompose(gram_matrices[node_layers[-1]])
        assert U_output is not None
        if center:
            assert means is not None and node_layers[-1] in means
            mean = means[node_layers[-1]]
            D_output, U_output = move_const_dir_first(D_output, U_output)
            C_output = centering_matrix(mean) @ U_output
        else:
            C_output = U_output
        C_output = C_output.detach().cpu()
        U_output = U_output.detach().cpu()

    if node_layers[-1] not in gram_matrices:
        # Technically we don't actually need the final node layer to be in gram_matrices if we're
        # not rotating it, but for now, our implementation assumes that it always is unless the
        # final node_layer "output".
        assert (
            node_layers[-1] == "output"
        ), f"Final node layer {node_layers[-1]} not in gram matrices."

        if isinstance(hooked_model.model, MLP):
            out_dim = hooked_model.model.output_size
        else:
            assert isinstance(hooked_model.model, SequentialTransformer)
            out_dim = hooked_model.model.cfg.d_vocab
    else:
        out_dim = gram_matrices[node_layers[-1]].shape[0]

    Us.append(
        Eigenvectors(
            node_layer_name=node_layers[-1],
            out_dim=out_dim,
            U=U_output,
        )
    )
    Cs.append(
        InteractionRotation(
            node_layer_name=node_layers[-1],
            out_dim=out_dim,
            C=C_output,
        )
    )
    if U_output is not None:
        assert C_output is not None
        out_shape = (out_dim, out_dim)
        assert C_output.shape == out_shape, f"Expected shape {out_shape}. Got {C_output.shape}."
        assert U_output.shape == out_shape, f"Expected shape {out_shape}. Got {U_output.shape}."

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

        ### SVD ROTATION (U)
        D_dash, U_dash = eigendecompose(gram_matrices[node_layer])
        if center:
            assert means is not None and node_layer in means
            D_dash, U_dash = move_const_dir_first(D_dash, U_dash)

        # Trucate all directions with eigenvalues smaller than some threshold
        mask = D_dash > truncation_threshold  # true if we keep the direction
        D: Float[Tensor, "orig_trunc orig_trunc"] = D_dash[mask].diag()
        U: Float[Tensor, "orig orig_trunc"] = U_dash[:, mask]

        ### CENTERING MATRIX (Y)
        Y: Float[Tensor, "orig orig"]
        Y_inv: Float[Tensor, "orig orig"]

        if center:
            assert means is not None
            # Y uses the bias position to center the activations
            Y = centering_matrix(means[node_layer])
            Y_inv = centering_matrix(means[node_layer], inverse=True)
        else:
            # If no centering, Y is the identity matrix
            Id = torch.eye(U.shape[0], dtype=U.dtype, device=U.device)
            Y, Y_inv = Id, Id.detach().clone()

        Us.append(Eigenvectors(node_layer_name=node_layer, out_dim=U.shape[1], U=U.detach().cpu()))
        if basis_formula == "svd":
            # Use U as C, with centering matrix
            C_info = InteractionRotation(
                node_layer_name=node_layer,
                out_dim=U.shape[1],
                C=(Y @ U).cpu(),
                C_pinv=(U.T @ Y_inv).cpu(),
            )
            Cs.append(C_info)
            continue

        ### FIRST ROTATION MATRIX (R)
        # Combines Y, U, D. This centers (if Y is not an identity), then orthogonalizes and scales.
        R: Float[Tensor, "orig d_hidden_trunc"] = Y @ U @ pinv_diag(D.sqrt())
        R_inv: Float[Tensor, "d_hidden_trunc orig"] = D.sqrt() @ U.T @ Y_inv

        ### ROTATION TO SPARSIFY EDGES (V)
        # This is an orthogonal rotation that attempts to sparsify the edges
        last_C = Cs[-1].C.to(device=device) if Cs[-1].C is not None else None
        # Compute M_dash in the neuron basis
        M_dash, Lambda_dash = collect_M_dash_and_Lambda_dash(
            C_out=last_C,
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
        # Then convert it into the pca basis
        M: Float[Tensor, "orig_trunc orig_trunc"] = (
            R_inv.to(M_dash.dtype) @ M_dash @ R_inv.T.to(M_dash.dtype)
        )
        # and take it's eigenvector basis as V
        V: Float[Tensor, "orig_trunc orig_trunc"]
        if center:
            # We don't want to rotate the constant direction (in the 0th position).
            # We thus eigendecompose a submatrix ignoring the first row and col
            sub_V = eigendecompose(M[1:, 1:])[1]
            V = torch.zeros_like(M)
            V[0, 0] = 1
            V[1:, 1:] = sub_V
        else:
            V = eigendecompose(M)[1]
        V = V.to(dtype)

        ### SCALING MATRIX (Lambda)
        # Tranform lambda_dash (computed in the neuron basis) into our new basis with R and V
        Lambda_abs: Float[Tensor, "orig_trunc"] = (V.T @ R_inv @ Lambda_dash @ R @ V).diag().abs()
        # Build a matrix for scaling by sqrt(Lambda).
        # This function prunes directions with small Lambdas. This is our second trunctaion.
        L, L_inv = build_sorted_lambda_matrices(Lambda_abs, truncation_threshold)

        ### FINAL ROTATION MATRIX (C)
        C: Float[Tensor, "orig rib"] = (R @ V @ L).detach().cpu()
        C_pinv: Float[Tensor, "rib orig"] = (L_inv @ V.T @ R_inv).detach().cpu()
        Cs.append(
            InteractionRotation(node_layer_name=node_layer, out_dim=C.shape[1], C=C, C_pinv=C_pinv)
        )

    return Cs[::-1], Us[::-1]
