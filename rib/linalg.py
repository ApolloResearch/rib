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
