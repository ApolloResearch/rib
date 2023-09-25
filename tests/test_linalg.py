import pytest
import torch

from rib.linalg import calc_rotation_matrix, eigendecompose, pinv_diag


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
    assert torch.allclose(u[:, order].abs(), eigenvectors.abs(), atol=1e-4)

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
    dataset = torch.randn(n_elements, d_hidden, dtype=torch.float64)
    gram = dataset.T @ dataset / n_elements
    _, vecs = eigendecompose(gram)

    rotation_matrix = calc_rotation_matrix(
        vecs=vecs, vecs_pinv=vecs.T, n_zero_vals=n_zero_vals, n_ablated_vecs=n_ablated_vecs
    )

    # Check the dimensions of the rotation matrix
    assert rotation_matrix.shape == (d_hidden, d_hidden)

    # Check that the matrix is symmetric (a property of rotation matrices)
    assert torch.allclose(rotation_matrix, rotation_matrix.T, atol=1e-6)

    # Get a new set of activations
    acts = torch.randn(n_elements, d_hidden, dtype=torch.float64)

    # Transform eigenvectors with the rotation matrix
    rotated_vecs = acts @ rotation_matrix

    # See how this compares with rotating into the eigenspace, zeroing out the last n dimensions,
    # and rotating back
    acts_eigenspace = acts @ vecs
    n_ignore = n_ablated_vecs if n_ablated_vecs > 0 else n_zero_vals
    if n_ignore > 0:
        acts_eigenspace[:, -n_ignore:] = 0
    rotated_vecs_2 = acts_eigenspace @ vecs.T
    assert torch.allclose(rotated_vecs, rotated_vecs_2, atol=1e-6)


@pytest.mark.parametrize(
    "x, expected",
    [
        (
            torch.diag(torch.tensor([1.0, 2.0, 3.0])),
            torch.diag(torch.tensor([1.0, 0.5, 1.0 / 3.0])),
        ),
    ],
)
def test_pinv_diag(x, expected):
    y = pinv_diag(x)
    assert torch.allclose(y, expected)


@pytest.mark.parametrize(
    "x",
    [
        (torch.tensor([[1.0, 0.5], [0.0, 2.0]])),
        (torch.diag(torch.tensor([1.0, 0.0, 3.0]))),
    ],
)
def test_pinv_diag_failure(x):
    with pytest.raises(AssertionError):
        y = pinv_diag(x)
