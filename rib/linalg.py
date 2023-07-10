from typing import Tuple

import torch
from jaxtyping import Float
from torch import Tensor


def eigendecompose(
    x: Float[Tensor, "d_hidden d_hidden"],
    descending: bool = True,
) -> Tuple[Float[Tensor, "d_hidden"], Float[Tensor, "d_hidden d_hidden"]]:
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
