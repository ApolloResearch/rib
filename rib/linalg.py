from dataclasses import dataclass
from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor
from torch.func import jacrev, vmap


@dataclass
class EigenInfo:
    """Information about the eigendecomposition of a gram matrix."""

    hook_name: str
    eigenvals: Float[Tensor, "d_hidden"]
    eigenvecs: Float[Tensor, "d_hidden d_hidden"]  # Columns are eigenvectors


def eigendecompose(
    x: Float[Tensor, "d_hidden d_hidden"],
    descending: bool = True,
) -> tuple[Float[Tensor, "d_hidden"], Float[Tensor, "d_hidden d_hidden"]]:
    """Calculate eigenvalues and eigenvectors of a real symmetric matrix.

    Note that we hardcode the dtype to torch.float64 because lower dtypes tend to be very unstable.

    Args:
        x: A real symmetric matrix (e.g. the result of X^T @ X)
        descending: If True, sort eigenvalues and corresponding eigenvectors in descending order
            of eigenvalues.
        dtype: The precision in which to perform the eigendecomposition.
            Values below torch.float64 tend to be very unstable.
    Returns:
        eigenvalues: Diagonal matrix whose diagonal entries are the eigenvalues of x.
        eigenvectors: Matrix whose columns are the eigenvectors of x.
    """
    # We hardcode the dtype to torch.float64 because lower dtypes tend to be very unstable.
    dtype = torch.float64
    eigenvalues, eigenvectors = torch.linalg.eigh(x.to(dtype=dtype))
    if descending:
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors


def calc_rotation_matrix(
    vecs: Float[Tensor, "d_hidden d_hidden"],
    n_zero_vals: int = 0,
    n_ablated_vecs: int = 0,
) -> Float[Tensor, "d_hidden d_hidden"]:
    """Calculate the matrix to rotates into and out of the orthogonal basis with optional ablations.

    The formula for the rotation matrix is given by:

        rotation_matrix = vecs.T @ basis @ vecs

    where basis is the standard basis of size d_hidden with the final n_zero_vals or n_ablated_vecs
    rows/columns set to 0.

    If n_ablated_vecs > 0, we ignore the smallest n_ablated_vecs eigenvectors (regardless of
    the number of zero eigenvalues).

    If n_ablated_vecs == 0 and n_zero_vals > 0, we ignore the eigenvectors which correspond to zero
    eigenvalues (as given by `n_zero_vals`).

    Args:
        vecs: Matrix whose columns are the eigenvectors of the gram matrix.
        n_zero_vals: Number of zero eigenvalues. If > 0 and n_ablated_vecs == 0, we ignore the
            smallest n_zero_vals eigenvectors.
        n_ablated_vecs: Number of eigenvectors to ablate. If > 0, we ignore the smallest
            n_ablated_vecs eigenvectors.

    Returns:
        The rotation matrix with which to right multiply incoming activations to rotate them
        into the orthogonal basis.
    """
    assert not (
        n_zero_vals > 0 and n_ablated_vecs > 0
    ), "Cannot also ignore zero eigenvalues when ablating eigenvectors."
    assert (
        n_ablated_vecs <= vecs.shape[0] and n_zero_vals <= vecs.shape[0]
    ), "Cannot ablate more eigenvectors than there are."
    n_ignore = n_ablated_vecs if n_ablated_vecs > 0 else n_zero_vals
    basis = torch.eye(vecs.shape[0], dtype=vecs.dtype, device=vecs.device)
    basis[vecs.shape[0] - n_ignore :] = 0

    rotation_matrix = vecs @ basis @ vecs.T
    return rotation_matrix


def batched_jacobian(
    fn: Callable, x: Float[Tensor, "batch_size in_size"]
) -> Float[Tensor, "batch_size out_size in_size"]:
    """Calculate the Jacobian of a function fn with respect to input x.

    Makes use of the fact that x is batched, allowing for vectorized computation of the Jacobian
    across the batch using pytorch's vmap
    (https://pytorch.org/functorch/stable/generated/functorch.jacrev.html).

    Assumes that fn can take as input a non-batched x and return a non-batched output.
    """
    return vmap(jacrev(fn))(x)


def calc_interaction_matrix(
    module: torch.nn.Module,
    in_acts: Float[Tensor, "batch_size in_hidden"],
    out_rotation_matrix: Float[Tensor, "out_hidden out_hidden"],
    in_eigenvecs: Float[Tensor, "in_hidden in_hidden"],
    in_eigenvals: Float[Tensor, "in_hidden"],
    out_acts: Float[Tensor, "batch_size out_hidden"],
) -> Float[Tensor, "out_hidden out_hidden"]:
    """
    The interaction matrix for the input layer is calculated as:

        M = inner_product(sum_i(f_i^{l+1} O'_{i,j}^l), sum_{i'}(f_{i'}^{l+1} O'_{i',j'}^l))
    with f_i^{l+1} representing this module's outputs, and O'_{i,j}^l representing the jacobian of
    the layer w.r.t the inputs transformed by:
        O'^l = C^{l+1} O^l U^l.T {D^l}^{0.5}
    where l represents the input layer and l+1 the output layer, and:
        C^{l+1}: (out_hidden, out_hidden) the rotation matrix at the output
        O^l: (batch, out_hidden, in_hidden) the jacobian of the layer w.r.t the inputs
        U^l: (in_hidden, in_hidden) matrix with columns as the eigenvectors of the input
        D^l: (in_hidden) the eigenvalues of the input


    Note that the batch_jacobian must be calculated outside of inference_mode (otherwise it
    will be zero).

    Args:
        module: Module that the hook is attached to.
        in_acts: Input to the module.
        out_rotation_matrix: Rotation matrix for the output layer.
        in_eigenvecs: Matrix whose columns are the eigenvectors of the gram matrix of the input.
        in_eigenvals: Diagonal matrix whose diagonal entries are the eigenvalues of the gram matrix
            of the input.
        out_acts: Output of the module.

    Returns:
        The interaction matrix for the input layer.
    """

    jac: Float[Tensor, "batch out_hidden in_hidden"] = batched_jacobian(module, in_acts)
    with torch.inference_mode():
        o_dash = torch.einsum(
            "jJ,bji,iI,i->bJI", out_rotation_matrix, jac, in_eigenvecs, in_eigenvals.sqrt()
        )
        out_o_dash = torch.einsum("bj,bji->bi", out_acts, o_dash)
        M = torch.einsum("bi,bj->ij", out_o_dash, out_o_dash)
    return M
