from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from jaxtyping import Float
from torch import Tensor

from rib.hooks import HookedModel


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
        eigenvalues: Eigenvalues of x.
        eigenvectors: Eigenvectors of x.
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
    """Calculate the matrix for rotating onto the orthogonal basis.

    The formula for the rotation matrix is given by:

        rotation_matrix = vecs.T @ basis @ vecs

    where basis is the standard basis of size d_hidden with the final n_zero_vals or n_ablated_vecs
    rows/columns set to 0.

    If n_ablated_vecs > 0, we ignore the smallest n_ablated_vecs eigenvectors (regardless of
    the number of zero eigenvalues).

    If n_ablated_vecs == 0 and n_zero_vals > 0, we ignore the eigenvectors which correspond to zero
    eigenvalues (as given by `n_zero_vals`).

    Args:
        vecs: Eigenvectors of a matrix.
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

    rotation_matrix = vecs.T @ basis @ vecs
    return rotation_matrix


@dataclass
class EigenInfo:
    """Eigenvalues, eigenvectors, and number of zero eigenvalues of a matrix.

    Attributes:
        vals: Eigenvalues of a matrix.
        vecs: Eigenvectors of a matrix.
        rotation_matrix: The matrix for rotating onto the orthogonal basis.
    """

    vals: Float[Tensor, "d_hidden"]
    vecs: Float[Tensor, "d_hidden d_hidden"]
    rotation_matrix: Float[Tensor, "d_hidden d_hidden"]


def calc_eigen_info(
    hooked_mlp: HookedModel,
    hook_points: list[str],
    matrix_key: str = "gram",
    zero_threshold: Optional[float] = 1e-13,
    n_ablated_vecs: int = 0,
) -> dict[str, EigenInfo]:
    """Calculate eigenvalues and eigenvectors of the gram matrices for each hook point.

    Args:
        hooked_mlp: The hooked model.
        hook_points: The hook points for which to calculate the eigenvalues and eigenvectors.
        matrix_key: The key used to store the gram matrix in the hooked data.
        zero_threshold: The threshold below which eigenvalues are considered to be zero. If None,
            no thresholding is performed.
        n_ablated_vecs: Number of eigenvectors with the smallest eigenvalues to ablate.

    Returns:
        A dictionary of EigenInfo objects, keyed by hook point.
    """
    eigens = {}
    for hook_point in hook_points:
        gram_matrix = hooked_mlp.hooked_data[hook_point][matrix_key]
        eigenvalues, eigenvectors = eigendecompose(gram_matrix, descending=True)
        n_zero_vals = (
            int(torch.sum(eigenvalues < zero_threshold).item()) if zero_threshold is not None else 0
        )
        rotation_matrix = calc_rotation_matrix(
            eigenvectors, n_zero_vals=n_zero_vals, n_ablated_vecs=n_ablated_vecs
        )
        eigens[hook_point] = EigenInfo(
            vals=eigenvalues, vecs=eigenvectors, rotation_matrix=rotation_matrix
        )

    return eigens
