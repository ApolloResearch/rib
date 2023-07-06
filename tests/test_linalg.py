from unittest import mock

import pytest
import torch

from rib.linalg import EigenInfo, calc_eigen_info, calc_rotation_matrix, eigendecompose


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
    x = x @ x.T / x.shape[0]  # Ensure x is symmetric

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


@pytest.mark.parametrize("n_zero_vals,n_ablated_vecs", [(0, 0), (2, 0), (0, 2)])
def test_calc_rotation_matrix(n_zero_vals, n_ablated_vecs):
    """Test the calc_rotation_matrix function.

    Checks if the rotation matrix has the correct dimensions and properties.

    We can validate whether the rotation is done correctly by checking that the transformed
    eigenvectors (except the ones set to zero) stay the same after the rotation, because a rotation
    matrix applied to its corresponding eigenvectors should return the eigenvectors themselves.

    Args:
        n_zero_vals: Number of zero eigenvalues.
        n_ablated_vecs: Number of eigenvectors to ablate.
    """
    n_elements = 2
    d_hidden = 10
    acts = torch.randn(n_elements, d_hidden).double()
    gram = acts.T @ acts / n_elements
    _, vecs = eigendecompose(gram)

    rotation_matrix = calc_rotation_matrix(
        vecs, n_zero_vals=n_zero_vals, n_ablated_vecs=n_ablated_vecs
    )

    # Check the dimensions of the rotation matrix
    assert rotation_matrix.shape == (d_hidden, d_hidden)

    # Check that the matrix is symmetric (a property of rotation matrices)
    assert torch.allclose(rotation_matrix, rotation_matrix.T, atol=1e-2)

    # Transform eigenvectors with the rotation matrix
    rotated_vecs = vecs @ rotation_matrix

    # Exclude the last n_ignore vectors
    n_ignore = n_ablated_vecs if n_ablated_vecs > 0 else n_zero_vals
    if n_ignore > 0:
        rotated_vecs = rotated_vecs[:-n_ignore]
        vecs = vecs[:-n_ignore]

    # Check if rotated vectors (except the ones set to zero) stay the same after the rotation
    assert torch.allclose(rotated_vecs, vecs, atol=1e-8)


@pytest.mark.parametrize("zero_threshold,n_ablated_vecs", [(None, 10), (1e-13, 0), (1e-6, 5)])
def test_calc_eigen_info(zero_threshold, n_ablated_vecs):
    """Test the calc_eigen_info function.

    Checks if the function correctly calculates and returns EigenInfo objects for all given hook
    points.
    """
    # Mock the hooked model
    hooked_mlp = mock.Mock()
    hooked_mlp.hooked_data = {
        "hook1": {"gram": torch.randn(10, 10)},
        "hook2": {"gram": torch.randn(10, 10)},
    }

    # Define hook points for calc_eigen_info
    hook_points = ["hook1", "hook2"]

    if zero_threshold is not None and n_ablated_vecs > 0:
        with pytest.raises(AssertionError):
            calc_eigen_info(
                hooked_mlp,
                hook_points,
                zero_threshold=zero_threshold,
                n_ablated_vecs=n_ablated_vecs,
            )
        return
    # Calculate eigen information
    eigen_infos = calc_eigen_info(
        hooked_mlp, hook_points, zero_threshold=zero_threshold, n_ablated_vecs=n_ablated_vecs
    )

    # Check the results
    assert len(eigen_infos) == len(hook_points)
    for hook_point in hook_points:
        eigen_info = eigen_infos[hook_point]

        # Assert instance of EigenInfo
        assert isinstance(eigen_info, EigenInfo)

        # Check dimensions of EigenInfo values
        assert eigen_info.vals.shape == (10,)
        assert eigen_info.vecs.shape == (10, 10)
        assert eigen_info.rotation_matrix.shape == (10, 10)
