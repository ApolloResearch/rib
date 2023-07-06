from unittest import mock

import pytest
import torch

from rib.linalg import EigenInfo, calc_eigen_info, eigendecompose


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


@pytest.mark.parametrize("zero_threshold", [None, 1e-1])
def test_calc_eigen_matrices(zero_threshold):
    """Test the calc_eigen_matrices function.

    Checks if the EigenInfo objects for each hook point are correctly computed.

    Args:
        zero_threshold: The threshold below which eigenvalues are considered to be zero.
    """
    # Define mock data
    d_hidden = 20
    gram_matrix = torch.randn(d_hidden, d_hidden)
    gram_matrix = gram_matrix @ gram_matrix.T  # Ensure gram_matrix is symmetric

    hooked_data = {"hook_point_1": {"gram": gram_matrix}, "hook_point_2": {"gram": gram_matrix}}

    # Create mock hooked model
    hooked_mlp = mock.Mock()
    hooked_mlp.hooked_data = hooked_data

    hook_points = list(hooked_data.keys())
    result = calc_eigen_info(hooked_mlp, hook_points, zero_threshold=zero_threshold)

    # Check if keys in the result match hook_points
    assert set(result.keys()) == set(hook_points)

    # Check the results
    for hook_point in hook_points:
        eigen_info = result[hook_point]

        assert isinstance(eigen_info, EigenInfo)
        assert eigen_info.vals.dtype == torch.float64
        assert eigen_info.vecs.dtype == torch.float64

        # If zero_threshold is provided, zero_vals should be computed
        if zero_threshold is not None:
            assert eigen_info.zero_vals == torch.sum(eigen_info.vals < zero_threshold).item()
        else:
            assert eigen_info.zero_vals is None
