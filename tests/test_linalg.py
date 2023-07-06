import pytest
import torch

from rib.linalg import eigendecompose


@pytest.mark.parametrize("descending", [True, False])
def test_eigendecompose(descending: bool):
    """Test the eigendecompose function.

    Compares the output of eigendecompose with the Singular Value Decomposition (SVD).
    Also verifies the order of the eigenvalues and eigenvectors depending on the 'descending' flag.

    Args:
        descending: If True, eigenvalues and eigenvectors should be sorted in descending order.
    """
    # Generate a random symmetric matrix
    d_hidden = 20
    x = torch.randn(d_hidden, d_hidden)
    x = x @ x.T  # Ensure x is symmetric

    # Calculate eigenvalues and eigenvectors using eigendecompose
    eigenvalues, eigenvectors = eigendecompose(x, descending=descending)

    # Check the dtypes of output match the hardcoded dtype in the function
    assert eigenvalues.dtype == torch.float64
    assert eigenvectors.dtype == torch.float64

    # Compute SVD, u will be the eigenvectors, s the absolute values of eigenvalues
    u, s, _ = torch.linalg.svd(x.to(dtype=torch.float64))

    # Compare with the results of eigendecompose
    order = s.abs().sort(descending=descending)[1]
    assert torch.allclose(s[order], eigenvalues.abs(), atol=1e-4)
    assert torch.allclose(u[:, order].abs(), eigenvectors.abs(), atol=1e-4)

    # Check sorting
    if descending:
        assert torch.all(eigenvalues[:-1] >= eigenvalues[1:])
    else:
        assert torch.all(eigenvalues[:-1] <= eigenvalues[1:])
