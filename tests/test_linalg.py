import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from rib.linalg import batched_jacobian, calc_rotation_matrix, eigendecompose


@pytest.mark.parametrize("descending", [True, False])
def test_eigendecompose(descending: bool) -> None:
    """Test the eigendecompose function.

    Compares the output of eigendecompose with the Singular Value Decomposition (SVD). Also verifies
    the order of the eigenvalues and eigenvectors depending on the 'descending' flag.

    Args:
        descending: If True, eigenvalues and eigenvectors should be sorted in descending order.
    """
    torch.manual_seed(0)
    # Generate a random symmetric matrix
    d_hidden = 20
    x = torch.randn(d_hidden, d_hidden)
    x = x @ x.T / x.shape[0]  # Ensure x is symmetric

    # Calculate eigenvalues and eigenvectors using eigendecompose
    eigenvalues, eigenvectors = eigendecompose(x, descending=descending)

    # Check the dtype of output matches the dtype of the input (the internal computation is done
    # in float64 to avoid numerical issues)
    assert eigenvalues.dtype == x.dtype

    # Compute SVD, u will be the eigenvectors, s the absolute values of eigenvalues
    u, s, _ = torch.linalg.svd(x.to(dtype=torch.float64))
    u = u.to(dtype=x.dtype)
    s = s.to(dtype=x.dtype)

    # Compare with the results of eigendecompose
    order = s.abs().sort(descending=descending)[1]
    assert torch.allclose(s[order], eigenvalues.abs(), atol=1e-4)
    assert torch.allclose(u[:, order].abs(), eigenvectors.T.abs(), atol=1e-4)

    # Check sorting
    if descending:
        assert torch.all(eigenvalues[:-1] >= eigenvalues[1:])
    else:
        assert torch.all(eigenvalues[:-1] <= eigenvalues[1:])


@pytest.mark.parametrize("n_zero_vals,n_ablated_vecs", [(0, 0), (2, 0), (0, 2)])
def test_calc_rotation_matrix(n_zero_vals: int, n_ablated_vecs: int) -> None:
    """Test the calc_rotation_matrix function.

    Checks if the rotation matrix has the correct dimensions and properties.

    We check that applying the rotation matrix is the same as:
        1. Rotating some activations acts into the eigenspace
        2. Zeroing out the last n dimensions of the activations in the eigenspace
        3. Rotating back into the original space

    Args:
        n_zero_vals: Number of zero eigenvalues.
        n_ablated_vecs: Number of eigenvectors to ablate.
    """
    torch.manual_seed(0)
    n_elements = 2
    d_hidden = 10
    dataset = torch.randn(n_elements, d_hidden)
    gram = dataset.T @ dataset / n_elements
    _, vecs = eigendecompose(gram)

    rotation_matrix = calc_rotation_matrix(
        vecs, n_zero_vals=n_zero_vals, n_ablated_vecs=n_ablated_vecs
    )

    # Check the dimensions of the rotation matrix
    assert rotation_matrix.shape == (d_hidden, d_hidden)

    # Check that the matrix is symmetric (a property of rotation matrices)
    assert torch.allclose(rotation_matrix, rotation_matrix.T, atol=1e-6)

    # Get a new set of activations
    acts = torch.randn(n_elements, d_hidden)

    # Transform eigenvectors with the rotation matrix
    rotated_vecs = acts @ rotation_matrix

    # See how this compares with rotating into the eigenspace, zeroing out the last n dimensions,
    # and rotating back
    acts_eigenspace = acts @ vecs.T
    n_ignore = n_ablated_vecs if n_ablated_vecs > 0 else n_zero_vals
    if n_ignore > 0:
        acts_eigenspace[:, -n_ignore:] = 0
    rotated_vecs_2 = acts_eigenspace @ vecs
    assert torch.allclose(rotated_vecs, rotated_vecs_2, atol=1e-6)


def test_batched_jacobian() -> None:
    """Test the batched_jacobian function.

    Checks if the batched_jacobian function returns the same result as
    torch.autograd.functional.jacobian after summing over the batch dimension and rearranging
    """
    torch.manual_seed(0)
    batch_size = 2
    d_hidden = 10
    x = torch.randn(batch_size, d_hidden)

    actual: Float[Tensor, "batch d_hidden d_hidden"] = batched_jacobian(torch.sin, x)

    torch_jac_summed: Float[Tensor, "d_hidden batch d_hidden"] = torch.autograd.functional.jacobian(
        torch.sin, x
    ).sum(dim=0)
    shuffled: Float[Tensor, "batch d_hidden d_hidden"] = torch_jac_summed.permute(1, 0, 2)

    assert torch.allclose(actual, shuffled, atol=1e-6)
