import pytest
import torch

from rib.linalg import (
    calc_rotation_matrix,
    eigendecompose,
    integrated_gradient_trapezoidal_jacobian,
    integrated_gradient_trapezoidal_norm,
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


@pytest.mark.parametrize("n_ablated_vecs", [0, 2])
def test_calc_rotation_matrix(n_ablated_vecs: int) -> None:
    """Test the calc_rotation_matrix function.

    Checks if the rotation matrix has the correct dimensions and properties.

    We check that applying the rotation matrix is the same as:
        1. Rotating some activations acts into the eigenspace
        2. Zeroing out the last n dimensions of the activations in the eigenspace
        3. Rotating back into the original space

    Args:
        n_ablated_vecs: Number of eigenvectors to ablate.
    """
    torch.manual_seed(0)
    n_elements = 2
    d_hidden = 10
    dataset = torch.randn(n_elements, d_hidden, dtype=torch.float64)
    gram = dataset.T @ dataset / n_elements
    _, vecs = eigendecompose(gram)

    rotation_matrix = calc_rotation_matrix(
        vecs=vecs, vecs_pinv=vecs.T, n_ablated_vecs=n_ablated_vecs
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
    if n_ablated_vecs > 0:
        acts_eigenspace[:, -n_ablated_vecs:] = 0
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


def test_intergrated_gradient_trapezoidal_norm_linear():
    """Check integrated gradient values over a linear module without bias for different intervals.

    We check three cases:
        1. Point estimate (n_intervals=0), leads to alpha=1
        2. n_intervals=1
        3. n_intervals=5

    The point estimate should be double the value of n_intervals==1, and n_intervals==5 given that
    our function is linear between 0 and 1.
    """
    torch.manual_seed(0)
    batch_size = 3
    in_hidden = 4
    out_hidden = 6
    out_hidden_trunc = 5

    C_out = torch.randn(out_hidden, out_hidden_trunc)
    inputs = (torch.randn(batch_size, in_hidden),)

    linear = torch.nn.Linear(in_hidden, out_hidden, bias=False)

    result_point_estimate = integrated_gradient_trapezoidal_norm(
        module=linear, inputs=inputs, C_out=C_out, n_intervals=0
    )
    result_1 = integrated_gradient_trapezoidal_norm(
        module=linear, inputs=inputs, C_out=C_out, n_intervals=1
    )
    result_5 = integrated_gradient_trapezoidal_norm(
        module=linear, inputs=inputs, C_out=C_out, n_intervals=5
    )

    assert torch.allclose(
        result_point_estimate, result_1 * 1
    ), "Point estimate and double the n_intervals==1 estimate are not close enough"
    assert torch.allclose(
        result_1, result_5
    ), "n_intervals==1 and n_intervals==5 are not close enough"


def test_integrated_gradient_trapezoidal_norm_polynomial():
    """Show that our integrated gradient converges to the analytical solution for a polynomial.

    Assume we have a polynomial function f = x^3. Our normed function for the integrated gradient
    is then:
    f_norm = - integral_{0}^{1} day(((x^3 - (alpha * x)^3) @ C_out)^2) / day(alpha * x) d_alpha.
           = - integral_{0}^{1} 2 * (x^3 - (alpha * x)^3) @ C_out) * (3 * (alpha * x)^2) @ C_out) d_alpha
           = 6 * integral_{0}^{1} (x^3 - (alpha * x)^3)) ((alpha * x)^2)) d_alpha
           = 6 * integral_{0}^{1} (x^5 alpha^2 - alpha^5 x^5) d_alpha
           = [ 2 * x^5 alpha^3 - x^5 alpha^6 ]_{0}^{1}
           = x^5 C_out^2

    We show that this analytical solution is approached as n_intervals increases.
    """

    torch.manual_seed(0)
    batch_size = 2
    hidden = 3

    poly_module = torch.nn.Module()
    poly_module.forward = lambda x: x**3

    # Let C_out be a square identity matrix to avoid issues with partial derivative dimensions
    # TODO: Handle non-identity C_out
    C_out = torch.eye(hidden)
    inputs = (torch.randn(batch_size, hidden),)

    result_2 = integrated_gradient_trapezoidal_norm(
        module=poly_module, inputs=inputs, C_out=C_out, n_intervals=2
    )
    result_20 = integrated_gradient_trapezoidal_norm(
        module=poly_module, inputs=inputs, C_out=C_out, n_intervals=20
    )

    result_200 = integrated_gradient_trapezoidal_norm(
        module=poly_module, inputs=inputs, C_out=C_out, n_intervals=200
    )

    analytical_result = inputs[0] ** 5 @ C_out**2

    assert torch.allclose(
        result_200, analytical_result, atol=1e-2
    ), "Integrated grad norms are not close enough"

    # Check that the results approach inputs[0]**5 as n_intervals increases
    differences = [
        (result - analytical_result).sum().abs() for result in [result_2, result_20, result_200]
    ]  # Check that differences is decreasing
    assert (
        differences[0] > differences[1] > differences[2]
    ), "Integrated grad norms are not decreasing"


def test_integrated_gradient_trapezoidal_jacobian_linear():
    """Check independence of n_intervals for integrated gradient jacobian over a linear module
    without bias.

    This should be the case as we're integrating over alpha * inputs, which is linear in alpha.
    """
    torch.manual_seed(0)
    batch_size = 2
    in_hidden = 3
    out_hidden = 4

    in_tensor = torch.randn(batch_size, in_hidden)

    linear = torch.nn.Linear(in_hidden, out_hidden, bias=False)
    linear_edge_norm = lambda x, y: linear(y) - linear(x)

    result_point_estimate = integrated_gradient_trapezoidal_jacobian(
        fn=linear_edge_norm,
        in_tensor=in_tensor,
        n_intervals=0,
    )
    result_1 = integrated_gradient_trapezoidal_jacobian(
        fn=linear_edge_norm,
        in_tensor=in_tensor,
        n_intervals=1,
    )
    result_5 = integrated_gradient_trapezoidal_jacobian(
        fn=linear_edge_norm,
        in_tensor=in_tensor,
        n_intervals=2,
    )

    # Check that all results are close
    assert torch.allclose(
        result_point_estimate, result_1
    ), "Point estimate and n_intervals==1 are not close enough"
    assert torch.allclose(
        result_1, result_5
    ), "n_intervals==1 and n_intervals==5 are not close enough"
