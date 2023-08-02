from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor
from torch.func import jacrev, vmap


def eigendecompose(
    x: Float[Tensor, "d_hidden d_hidden"],
    descending: bool = True,
    dtype: torch.dtype = torch.float64,
) -> tuple[Float[Tensor, "d_hidden"], Float[Tensor, "d_hidden d_hidden"]]:
    """Calculate eigenvalues and eigenvectors of a real symmetric matrix.

    Eigenvectors are returned as columns of a matrix, sorted in descending order of eigenvalues.

    Note that we hardcode the dtype of the eigendecomposition calculation to torch.float64 because
    lower dtypes tend to be very unstable. We switch back to the original dtype after the operation.

    Eigendecomposition seems to be faster on the cpu. See
    https://discuss.pytorch.org/t/torch-linalg-eigh-is-significantly-slower-on-gpu/140818/12. We
    therefore convert to and from the cpu when performing the eigendecomposition.

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
    eigenvalues, eigenvectors = torch.linalg.eigh(x.to(dtype=dtype, device="cpu"))

    eigenvalues = eigenvalues.to(dtype=x.dtype, device=x.device)
    eigenvectors = eigenvectors.to(dtype=x.dtype, device=x.device)
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


def pinv_truncated_diag(x: Float[Tensor, "a b"]) -> Float[Tensor, "a b"]:
    """Calculate the pseudo-inverse of a truncated diagonal matrix.

    A truncated diagonal matrix is a diagonal matrix that isn't necessarily square. The
    pseudo-inverse of a truncated diagonal matrix is the same matrix with the diagonal elements
    replaced by their reciprocal.

    Args:
        x: A truncated diagonal matrix.

    Returns:
        Pseudo-inverse of x.
    """
    # Check that all non-diagonal entries are 0. Use a shortcut of comparing the sum of the
    # diagonal entries to the sum of all entries. They should be close
    assert torch.allclose(
        x.sum(), x.diagonal().sum()
    ), "It appears there are non-zero off-diagonal entries."
    res: Float[Tensor, "a b"] = torch.zeros_like(x)
    res.diagonal()[:] = x.diagonal().reciprocal()
    return res
