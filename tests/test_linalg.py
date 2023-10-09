import pytest
import torch

from rib.linalg import (
    calc_rotation_matrix,
    eigendecompose,
    integrated_gradient_norm,
    pinv_diag,
)


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


def test_intergrated_gradient_norm_linear():
    """Check independence of n_intervals for integrated gradient over a linear module without bias.

    This should be the case as we're integrating over alpha * inputs, which is linear in alpha.
    """
    torch.manual_seed(0)
    # Hyperparameters
    batch_size = 3
    in_hidden = 4
    out_hidden = 6
    out_hidden_trunc = 5

    # 1. Generate random tensors for `C_out` and `inputs`
    C_out = torch.randn(out_hidden, out_hidden_trunc)
    inputs = (torch.randn(batch_size, in_hidden),)

    # 2. Initialize a `torch.nn.Linear` module
    linear = torch.nn.Linear(in_hidden, out_hidden, bias=False)

    # 3. Call the `integrated_gradient_norm` function
    result_1 = integrated_gradient_norm(module=linear, inputs=inputs, C_out=C_out, n_intervals=1)
    result_5 = integrated_gradient_norm(module=linear, inputs=inputs, C_out=C_out, n_intervals=5)

    # 4. Assert that the results are close enough
    assert torch.allclose(result_1, result_5), "Results are not close enough"


def test_integrated_gradient_norm_non_linear():
    """A non-linear module should be sensitive to the number of intervals.

    This is because the integrated gradient is calculated by integrating over alpha * f(inputs),
    where f is the non-linear function. This is not linear in alpha between 0 and 1, so the number
    of intervals should matter.
    """

    torch.manual_seed(0)
    # Hyperparameters
    batch_size = 3
    hidden = 4
    hidden_trunc = 2

    # Torch module which is non-linear in alpha between 0 and 1
    sigmoid = torch.nn.Sigmoid()

    # 1. Generate random tensors for `C_out` and `inputs`
    C_out = torch.randn(hidden, hidden_trunc)
    inputs = (torch.randn(batch_size, hidden),)

    # 2. Call the `integrated_gradient_norm` function
    result_1 = integrated_gradient_norm(module=sigmoid, inputs=inputs, C_out=C_out, n_intervals=1)
    result_5 = integrated_gradient_norm(module=sigmoid, inputs=inputs, C_out=C_out, n_intervals=5)

    # 3. Assert that the results are not close enough
    assert not torch.allclose(result_1, result_5), "Results are close enough"
