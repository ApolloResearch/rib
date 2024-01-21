"""This module contains algorithms related to interaction rotations."""

from typing import Callable, Literal, Optional

import torch
from jaxtyping import Float, Int
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationInfo
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Annotated

from rib.data_accumulator import collect_M_dash_and_Lambda_dash
from rib.hook_manager import HookedModel
from rib.linalg import centering_matrix, eigendecompose, move_const_dir_first, pinv_diag
from rib.models import MLP, SequentialTransformer
from rib.utils import check_device_is_cpu


def wrap_check_dim_is_orig_dim(
    dim: int,
) -> Callable[[Optional[torch.Tensor], ValidationInfo], Optional[torch.Tensor]]:
    """Returns a function that checks whether `dim` of a tensor is equal to orig_dim."""

    def check_dim_is_orig_dim(
        X: Optional[torch.Tensor], info: ValidationInfo
    ) -> Optional[torch.Tensor]:
        if X is not None:
            assert (
                X.shape[dim] == info.data["orig_dim"]
            ), f"Expected dim {dim} to be {info.data['orig_dim']}. Got {X.shape[dim]}."
        return X

    return check_dim_is_orig_dim


class InteractionRotation(BaseModel):
    """Stores useful matrices that are computed in `calculate_interaction_rotations`."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    node_layer: str = Field(..., description="The module_id representing the node layer.")
    orig_dim: int = Field(..., description="Size of the concatenated embeddings at node_layer")
    # if centering was used, C[:, 0] is the constant direction
    C: Annotated[
        Optional[Float[Tensor, "orig rib"]],
        AfterValidator(wrap_check_dim_is_orig_dim(dim=0)),
        AfterValidator(check_device_is_cpu),
    ] = None
    # pseudoinverse of C, not needed for the output node layer
    C_pinv: Annotated[
        Optional[Float[Tensor, "rib orig"]],
        AfterValidator(wrap_check_dim_is_orig_dim(dim=1)),
        AfterValidator(check_device_is_cpu),
    ] = None
    W: Annotated[
        Optional[Float[Tensor, "orig orig_trunc"]],
        AfterValidator(wrap_check_dim_is_orig_dim(dim=0)),
        AfterValidator(check_device_is_cpu),
    ] = None
    # pseudoinverse of W, not needed for the output node layer
    W_pinv: Annotated[
        Optional[Float[Tensor, "orig_trunc orig"]],
        AfterValidator(wrap_check_dim_is_orig_dim(dim=1)),
        AfterValidator(check_device_is_cpu),
    ] = None
    V: Annotated[
        Optional[Float[Tensor, "orig_trunc orig_trunc"]],
        AfterValidator(check_device_is_cpu),
    ] = None
    Lambda: Optional[Float[Tensor, "orig_trunc"]] = None


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
    # `(lambda_matrix @ lambda_matrix_pinv).diag()` should contain `rib_dim` 1s and
    # `orig_trunc - rib_dim` 0s
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
) -> list[InteractionRotation]:
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
        A list of InteractionRotation objects containing useful matrices.
    """
    assert hooked_model.model.has_folded_bias, "Biases must be folded in to calculate Cs."
    assert len(section_names) > 0, "No sections specified."

    non_output_node_layers = [node_layer for node_layer in node_layers if node_layer != "output"]
    assert len(non_output_node_layers) == len(
        section_names
    ), "Must specify a hook name for each section (except the output section)."

    if center and basis_formula == "neuron":
        raise NotImplementedError("centering is not currently implemented for the neuron basis.")

    # We start appending InteractionRotation from the output layer and work our way backwards
    interaction_rotations: list[InteractionRotation] = []

    # The C matrix for the final layer is either the centered, unscaled eigenvectors of the gram
    # matrix (i.e. W) if rotate_final_node_layer is True, and None otherwise
    W_output: Optional[Float[Tensor, "orig orig_trunc"]] = None
    if rotate_final_node_layer:
        # We don't use D_output except in finding the index of the const_dir in U
        D_dash_output, U_dash_output = eigendecompose(gram_matrices[node_layers[-1]])
        # Trucate all directions with eigenvalues smaller than some threshold
        mask = D_dash_output > truncation_threshold  # true if we keep the direction
        D_output: Float[Tensor, "orig_trunc orig_trunc"] = D_dash_output[mask].diag()
        U_output: Float[Tensor, "orig orig_trunc"] = U_dash_output[:, mask]
        if center:
            assert means is not None
            D_output, U_output = move_const_dir_first(D_output, U_output)
            Y_output = centering_matrix(means[node_layers[-1]])
        else:
            # If no centering, Y is the identity matrix
            Y_output = torch.eye(U_output.shape[0], dtype=U_output.dtype, device=U_output.device)

        ### ROTATION MATRIX (W) to eigenbasis
        # Combines Y and U. We don't include D as our best guess of importance here is the size of
        # the output activations. Thus there's no reason to scale.
        W_output = (Y_output @ U_output).cpu()

    # Get the out_dim of the final node layer
    if node_layers[-1] not in gram_matrices:
        assert (
            node_layers[-1] == "output"
        ), f"Final node layer {node_layers[-1]} not in gram matrices and not output."
        if isinstance(hooked_model.model, MLP):
            out_dim = hooked_model.model.output_size
        elif isinstance(hooked_model.model, SequentialTransformer):
            out_dim = hooked_model.model.cfg.d_vocab
        else:
            raise NotImplementedError(f"Unknown model type {type(hooked_model.model)}")
    else:
        out_dim = gram_matrices[node_layers[-1]].shape[0]

    interaction_rotations.append(
        InteractionRotation(
            node_layer=node_layers[-1],
            orig_dim=out_dim,
            C=W_output.detach().clone() if W_output is not None else None,
            W=W_output,
        )
    )

    # We only need to calculate C for the final section if there is a node layer at "output"
    # Otherwise, we've already handled the final node layer above
    section_names_to_calculate = (
        section_names if node_layers[-1] == "output" else section_names[:-1]
    )
    assert (
        len(section_names_to_calculate) == len(node_layers) - 1
    ), "Must be a section name for all but the final node_layer which was already handled above."

    for node_layer, section_name in tqdm(
        zip(node_layers[-2::-1], section_names_to_calculate[::-1]),
        total=len(section_names_to_calculate),
        desc="Interaction rotations",
    ):
        out_dim = gram_matrices[node_layer].shape[0]
        if basis_formula == "neuron":
            # Use identity matrix as C and W since we don't rotate
            interaction_rotations.append(
                InteractionRotation(
                    node_layer=node_layer,
                    orig_dim=out_dim,
                    C=torch.eye(out_dim, dtype=dtype),
                    C_pinv=torch.eye(out_dim, dtype=dtype),
                    W=torch.eye(out_dim, dtype=dtype),
                    W_pinv=torch.eye(out_dim, dtype=dtype),
                )
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
            Y = centering_matrix(means[node_layer])
            Y_inv = centering_matrix(means[node_layer], inverse=True)
        else:
            # If no centering, Y is the identity matrix
            Id = torch.eye(U.shape[0], dtype=U.dtype, device=U.device)
            Y, Y_inv = Id, Id.detach().clone()

        # First, W combines Y and U. This centers (if Y is not an identity), then orthogonalizes.
        W: Float[Tensor, "orig orig_trunc"] = Y @ U
        W_pinv: Float[Tensor, "orig_trunc orig"] = U.T @ Y_inv
        ### FIRST ROTATION MATRIX (R) to eigenbasis
        # Then scale by sqrt(D) to get R
        R: Float[Tensor, "orig orig_trunc"] = W @ pinv_diag(D.sqrt())
        R_pinv: Float[Tensor, "orig_trunc orig"] = D.sqrt() @ W_pinv

        if basis_formula == "svd":
            # Use W as C, with centering matrix
            interaction_rotations.append(
                InteractionRotation(
                    node_layer=node_layer,
                    orig_dim=out_dim,
                    C=W.cpu().detach().clone(),
                    C_pinv=W_pinv.cpu().detach().clone(),
                    W=W.cpu(),
                    W_pinv=W_pinv.cpu(),
                )
            )
            continue

        ### ROTATION TO SPARSIFY EDGES (V)
        # This is an orthogonal rotation that attempts to sparsify the edges
        last_C = (
            interaction_rotations[-1].C.to(device=device)
            if interaction_rotations[-1].C is not None
            else None
        )
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
            R_pinv.to(M_dash.dtype) @ M_dash @ R_pinv.T.to(M_dash.dtype)
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
        # Transform lambda_dash (computed in the neuron basis) into our new basis with R and V
        Lambda: Float[Tensor, "orig_trunc"] = (V.T @ R_pinv @ Lambda_dash @ R @ V).diag().abs()
        # Build a matrix for scaling by sqrt(Lambda).
        # This function prunes directions with small Lambdas. This is our second trunctaion.
        L, L_inv = build_sorted_lambda_matrices(Lambda, truncation_threshold)

        ### FINAL ROTATION MATRIX (C)
        C: Float[Tensor, "orig rib"] = (R @ V @ L).detach().cpu()
        C_pinv: Float[Tensor, "rib orig"] = (L_inv @ V.T @ R_pinv).detach().cpu()

        interaction_rotations.append(
            InteractionRotation(
                node_layer=node_layer,
                orig_dim=out_dim,
                C=C,
                C_pinv=C_pinv,
                W=W.cpu(),
                W_pinv=W_pinv.cpu(),
                V=V.detach().cpu(),
                Lambda=Lambda.detach().cpu(),
            )
        )

    return interaction_rotations[::-1]
