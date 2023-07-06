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


@dataclass
class EigenInfo:
    """Eigenvalues, eigenvectors, and number of zero eigenvalues of a matrix.

    Attributes:
        vals: Eigenvalues of a matrix.
        vecs: Eigenvectors of a matrix.
        zero_vals: Number of eigenvalues that are zero.
    """

    vals: Float[Tensor, "d_hidden"]
    vecs: Float[Tensor, "d_hidden d_hidden"]
    zero_vals: Optional[int] = None


def calc_eigen_info(
    hooked_mlp: HookedModel,
    hook_points: list[str],
    matrix_key: str = "gram",
    zero_threshold: Optional[float] = 1e-13,
) -> dict[str, EigenInfo]:
    """Calculate eigenvalues and eigenvectors of the gram matrices for each hook point.

    Args:
        hooked_mlp: The hooked model.
        hook_points: The hook points for which to calculate the eigenvalues and eigenvectors.
        matrix_key: The key used to store the gram matrix in the hooked data.
        zero_threshold: The threshold below which eigenvalues are considered to be zero. If None,
            no thresholding is performed.

    Returns:
        A dictionary of EigenInfo objects, keyed by hook point.
    """
    eigens = {}
    for hook_point in hook_points:
        gram_matrix = hooked_mlp.hooked_data[hook_point][matrix_key]
        eigenvalues, eigenvectors = eigendecompose(gram_matrix, descending=True)
        zero_vals = (
            int(torch.sum(eigenvalues < zero_threshold).item())
            if zero_threshold is not None
            else None
        )
        eigens[hook_point] = EigenInfo(vals=eigenvalues, vecs=eigenvectors, zero_vals=zero_vals)

    return eigens
