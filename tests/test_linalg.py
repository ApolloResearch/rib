from functools import partial
from typing import Callable

import numpy as np
import pytest
import torch
from einops import rearrange
from fancy_einsum import einsum
from jaxtyping import Float
from torch import Tensor
from torch.func import jacrev

from rib.linalg import (
    IntegrationPoint,
    _calc_integration_points,
    _generate_phis_array,
    _generate_sources,
    calc_basis_integrated_gradient,
    calc_edge_functional,
    calc_edge_squared,
    calc_gram_matrix,
    calc_rotation_matrix,
    centering_matrix,
    eigendecompose,
    pinv_diag,
)
from rib.utils import set_seed
from tests.utils import assert_is_close


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
    dim = 20
    x = torch.randn(dim, dim)
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
    orig = 10
    dataset = torch.randn(n_elements, orig, dtype=torch.float64)
    gram = dataset.T @ dataset / n_elements
    _, vecs = eigendecompose(gram)

    rotation_matrix = calc_rotation_matrix(
        vecs=vecs, vecs_pinv=vecs.T, n_ablated_vecs=n_ablated_vecs
    )

    # Check the dimensions of the rotation matrix
    assert rotation_matrix.shape == (orig, orig)

    # Check that the matrix is symmetric (a property of rotation matrices)
    assert torch.allclose(rotation_matrix, rotation_matrix.T, atol=1e-6)

    # Get a new set of activations
    acts = torch.randn(n_elements, orig, dtype=torch.float64)

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


@pytest.mark.parametrize("integration_method", ["trapezoidal", "gauss-legendre"])
def test_calc_basis_integrated_gradient_linear(integration_method):
    """Check integrated gradient values over a linear module without bias for different intervals.

    We check three cases:
        1. Point estimate (n_intervals=0), leads to alpha=0.5
        2. n_intervals=1
        3. n_intervals=5

    The point estimate should be double the value of n_intervals==1, and n_intervals==5 given that
    our function is linear between 0 and 1.
    """
    torch.manual_seed(0)
    batch_size = 3
    in_dim = 4
    out_dim = 6
    rib_dim = 5

    C_out = torch.randn(out_dim, rib_dim)
    inputs = (torch.randn(batch_size, in_dim),)

    linear = torch.nn.Linear(in_dim, out_dim, bias=False)

    result_point_estimate = calc_basis_integrated_gradient(
        module=linear,
        inputs=inputs,
        C_out=C_out,
        n_intervals=0,
        integration_method=integration_method,
    )
    result_1 = calc_basis_integrated_gradient(
        module=linear,
        inputs=inputs,
        C_out=C_out,
        n_intervals=1,
        integration_method=integration_method,
    )
    result_5 = calc_basis_integrated_gradient(
        module=linear,
        inputs=inputs,
        C_out=C_out,
        n_intervals=5,
        integration_method=integration_method,
    )

    assert torch.allclose(
        result_point_estimate, result_1 * 1
    ), "Point estimate and n_intervals==1 estimate are not close enough"
    assert torch.allclose(
        result_1, result_5
    ), "n_intervals==1 and n_intervals==5 are not close enough"


@pytest.mark.parametrize("integration_method", ["trapezoidal", "gauss-legendre"])
def test_calc_basis_integrated_gradient_polynomial(integration_method):
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
    dim = 3

    poly_module = torch.nn.Module()
    poly_module.forward = lambda x: x**3

    # Let C_out be a square identity matrix to avoid issues with partial derivative dimensions
    # TODO: Handle non-identity C_out
    C_out = torch.eye(dim)
    inputs = (torch.randn(batch_size, dim),)
    analytical_result = inputs[0] ** 5 @ C_out**2

    result_2 = calc_basis_integrated_gradient(
        module=poly_module,
        inputs=inputs,
        C_out=C_out,
        n_intervals=2,
        integration_method=integration_method,
    )
    result_20 = calc_basis_integrated_gradient(
        module=poly_module,
        inputs=inputs,
        C_out=C_out,
        n_intervals=20,
        integration_method=integration_method,
    )

    result_200 = calc_basis_integrated_gradient(
        module=poly_module,
        inputs=inputs,
        C_out=C_out,
        n_intervals=200,
        integration_method=integration_method,
    )
    differences = [(r - analytical_result).abs().sum() for r in [result_2, result_20, result_200]]

    if integration_method == "trapezoidal":
        # Trapezoidal rule check 1% accuracy
        assert torch.allclose(
            result_200, analytical_result, atol=1e-2
        ), "Integrated grad norms are not close enough"
        # Check that the results approach inputs[0]**5 as n_intervals increases
        assert (
            differences[0] > differences[1] > differences[2]
        ), f"Integrated grad norms are not decreasing {[d.item() for d in differences]}"
    else:
        # In the gauss-legendre method the integrals with >5 samples have negligible numerical
        # errors (around 1e-6). These are no longer strictly decreasing, but we allow this as the
        # accuracy is sufficient.
        assert differences[0] < 1e-2
        assert differences[1] < 1e-5
        assert differences[2] < 1e-5


def _integrate(f, points: list[IntegrationPoint]):
    xs = np.array([p.alpha for p in points])
    weights = np.array([p.weight for p in points])
    return np.sum(weights * f(xs))


@pytest.mark.parametrize(
    "rule, n_intervals, error_lim",
    [("trapezoidal", 50, 0.0003), ("gauss-legendre", 5, 0.0002), ("gauss-legendre", 50, 1e-7)],
)
def test_integration_methods_complicated_function(rule, n_intervals, error_lim):
    f = lambda x: np.exp(x) + x ** np.sin(x)
    true_result = 2.5096861280  # says wolfram-alpha
    result = _integrate(f, _calc_integration_points(n_intervals=n_intervals, rule=rule))
    assert_is_close(result, true_result, atol=error_lim, rtol=0)


def test_integration_methods_hard_function():
    """Trapezoidal rule cannot integrate 1/sqrt(x) to appreciable accuracy, Gauss-Legendre is
    insanely better"""
    f = lambda x: 1 / np.sqrt(x)
    true_result = 2
    # Trapezoidal error @ 5 samples > 50%
    trapezoidal_result_50 = _integrate(f, _calc_integration_points(50, "trapezoidal"))
    with pytest.raises(AssertionError):
        assert_is_close(trapezoidal_result_50, true_result, atol=0, rtol=0.5)
    # Gauss-Legendre error @ 5 samples < 20%
    gl_result_5 = _integrate(f, _calc_integration_points(5, "gauss-legendre"))
    assert_is_close(gl_result_5, true_result, atol=0, rtol=0.2)
    # Gauss-Legendre error @ 50 samples < 2%
    gl_result_50 = _integrate(f, _calc_integration_points(50, "gauss-legendre"))
    assert_is_close(gl_result_50, true_result, atol=0, rtol=0.02)


@pytest.mark.parametrize("integration_method", ["trapezoidal", "gauss-legendre"])
@pytest.mark.xfail(reason="We are making this assumption again in (1-0)*alpha IG for basis")
def test_calc_basis_integrated_gradient_offset_polynomial(integration_method):
    """Show that our integrated gradient of our norm function converges to the analytical
    solution for a polynomial, with the special feature that act(0) != 0. Earlier code made this
    assumption, and this test checks that our new code also holds without the assumption

    Assume we have a polynomial function f = x^3. Our normed function for the integrated gradient
    is then:
    f_norm = integral_{0}^{1} day((((alpha * x)^3 + 1) @ C_out)^2) / day(alpha * x) d_alpha.
           = (x^5 + 2 * x^2) C_out^2

    We show that this analytical solution is approached as n_intervals increases.
    """

    torch.manual_seed(0)
    batch_size = 2
    dim = 3

    poly_module = torch.nn.Module()
    poly_module.forward = lambda x: x**3 + 1

    # Let C_out be a square identity matrix to avoid issues with partial derivative dimensions
    # TODO: Handle non-identity C_out
    C_out = torch.eye(dim)
    inputs = (torch.randn(batch_size, dim),)

    result_2 = calc_basis_integrated_gradient(
        module=poly_module,
        inputs=inputs,
        C_out=C_out,
        n_intervals=2,
        integration_method=integration_method,
    )
    result_20 = calc_basis_integrated_gradient(
        module=poly_module,
        inputs=inputs,
        C_out=C_out,
        n_intervals=20,
        integration_method=integration_method,
    )

    result_200 = calc_basis_integrated_gradient(
        module=poly_module,
        inputs=inputs,
        C_out=C_out,
        n_intervals=200,
        integration_method=integration_method,
    )

    analytical_result = inputs[0] ** 5 @ C_out**2
    # Old formula gave (inputs[0] ** 5 + 2 * inputs[0] ** 2) @ C_out**2

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


@pytest.mark.parametrize("integration_method", ["trapezoidal", "gauss-legendre"])
def test_calc_edge_n_intervals(integration_method, edge_formula="squared"):
    """Check independence of n_intervals for integrated gradient jacobian over a linear module
    without bias.

    This should be the case as we're integrating over alpha * inputs, which is linear in alpha.
    """
    if edge_formula == "functional":
        calc_edge = calc_edge_functional
    elif edge_formula == "squared":
        calc_edge = partial(calc_edge_squared, out_dim_start_idx=0, out_dim_end_idx=4)
    else:
        raise ValueError(f"edge_formula {edge_formula} not recognized")

    torch.manual_seed(0)
    batch_size = 2
    in_dim = 3
    out_dim = 4

    in_tensor = torch.randn(batch_size, in_dim, requires_grad=True)

    linear = torch.nn.Linear(in_dim, out_dim, bias=False)
    # Need to account for module_hat taking both inputs and an in_tuple_dims argument
    linear_partial = lambda x, _: linear(x)

    result_point_estimate: Float[Tensor, "out_dim in_dim"] = torch.zeros(out_dim, in_dim)

    calc_edge(
        module_hat=linear_partial,
        f_in_hat=in_tensor,
        in_tuple_dims=(in_dim,),
        n_intervals=0,
        integration_method=integration_method,
        edge=result_point_estimate,
        dataset_size=batch_size,
    )
    result_1: Float[Tensor, "out_dim in_dim"] = torch.zeros(out_dim, in_dim)
    calc_edge(
        module_hat=linear_partial,
        f_in_hat=in_tensor,
        in_tuple_dims=(in_dim,),
        n_intervals=1,
        integration_method=integration_method,
        edge=result_1,
        dataset_size=batch_size,
    )

    result_5: Float[Tensor, "out_dim in_dim"] = torch.zeros(out_dim, in_dim)
    calc_edge(
        module_hat=linear_partial,
        f_in_hat=in_tensor,
        in_tuple_dims=(in_dim,),
        n_intervals=5,
        integration_method=integration_method,
        edge=result_5,
        dataset_size=batch_size,
    )

    # Check that all results are close
    assert torch.allclose(
        result_point_estimate, result_1
    ), "Point estimate and n_intervals==1 are not close enough"
    assert torch.allclose(
        result_1, result_5
    ), "n_intervals==1 and n_intervals==5 are not close enough"


def _integrated_gradient_jacobian_with_jacrev(
    module_hat: Callable, x: Float[Tensor, "batch in_dim"], n_intervals: int, dataset_size: int
) -> Float[Tensor, "out_dim in_dim"]:
    """Compute the integrated gradient jacobian using jacrev."""
    edge = None
    for point in _calc_integration_points(n_intervals, rule="trapezoidal"):
        alpha_x = point.alpha * x
        fn_norm = lambda f: ((module_hat(x) - module_hat(f)) ** 2).sum(
            dim=0
        )  # Sum over batch dimension
        alpha_edge = jacrev(fn_norm)(alpha_x)  # [out_dim, batch_size, in_dim]
        # No pos dim for this test
        E = torch.einsum("ibj,bj->ij", alpha_edge / dataset_size, x)
        if edge is None:
            edge = -E * point.weight
        else:
            edge -= E * point.weight

    return edge


@pytest.mark.parametrize("edge_formula", ["functional", "squared"])
def test_calc_edge_jacrev(edge_formula):
    """Check that our custom jacobian in the integrated gradient matches torch.func.jacrev."""
    if edge_formula == "functional":
        calc_edge = calc_edge_functional
    elif edge_formula == "squared":
        pytest.skip("squared edge formula not implemented in the jacrev test")
    else:
        raise ValueError(f"edge_formula {edge_formula} not recognized")

    torch.manual_seed(0)
    batch_size = 2
    in_dim = 3
    out_dim = 4

    in_tensor = torch.randn(batch_size, in_dim, requires_grad=True)

    linear = torch.nn.Linear(in_dim, out_dim, bias=False)
    # Need to account for module_hat taking both inputs and an in_tuple_dims argument
    linear_partial = lambda x, _: linear(x)

    result_ours: Float[Tensor, "out_dim in_dim"] = torch.zeros(out_dim, in_dim)
    calc_edge(
        module_hat=linear_partial,
        f_in_hat=in_tensor,
        in_tuple_dims=(in_dim,),
        n_intervals=5,
        integration_method="trapezoidal",
        edge=result_ours,
        dataset_size=batch_size,
    )
    result_jacrev: Float[Tensor, "out_dim in_dim"] = _integrated_gradient_jacobian_with_jacrev(
        module_hat=linear,
        x=in_tensor,
        n_intervals=5,
        dataset_size=batch_size,
    )
    assert torch.allclose(result_ours, result_jacrev), "calc_edge and jacrev are not close enough"


def test_calc_integration_points_gauss_legendre():
    points = _calc_integration_points(
        4,
        rule="gauss-legendre",
        lower=-1,
        upper=1,
    )
    x, w = np.polynomial.legendre.leggauss(5)
    assert np.allclose(x, [p.alpha for p in points])
    assert np.allclose(w, [p.weight for p in points])


def test_calc_integration_points_gradient():
    points = _calc_integration_points(
        0,
        rule="gradient",
    )
    assert points == [IntegrationPoint(alpha=1.0, weight=1.0)]


@pytest.mark.parametrize(
    "n_intervals,integral_boundary_relative_epsilon,expected_alphas,expected_weights",
    [
        # Testing for n_intervals=0
        (0, 1e-3, [0.5], [1]),
        # Testing for n_intervals=1
        (1, 1e-3, [5e-4, 1 - 5e-4], [0.5, 0.5]),
        # Testing for n_intervals=2 and small epsilon
        (2, 1e-4, [1e-4 / 3, 0.5, 1 - 1e-4 / 3], [0.25, 0.5, 0.25]),
    ],
)
def test_calc_integration_points_trapezoidal(
    n_intervals, integral_boundary_relative_epsilon, expected_alphas, expected_weights
):
    points = _calc_integration_points(
        n_intervals,
        rule="trapezoidal",
        integral_boundary_relative_epsilon=integral_boundary_relative_epsilon,
    )

    # Assert that the returned alphas are close to the expected values
    alphas = [p.alpha for p in points]
    assert np.allclose(alphas, expected_alphas), f"alphas: {alphas} != {expected_alphas}"

    # Assert that the returned interval_size is close to the expected value
    weights = [p.weight for p in points]
    assert np.allclose(weights, expected_weights), f"interval_size: {weights} != {expected_weights}"


@pytest.mark.parametrize(
    "input_tensor, dataset_size, expected_output",
    [
        # Tensor without positional indices
        (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 2, torch.tensor([[5.0, 7.0], [7.0, 10.0]])),
        # Tensor with positional indices (scaled by number of positions = 2)
        (
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]),
            4,
            torch.tensor([[2.5, 3.5], [3.5, 5.0]]),
        ),
    ],
)
def test_calc_gram_matrix(input_tensor, dataset_size, expected_output):
    gram_matrix = calc_gram_matrix(input_tensor, dataset_size)

    # Check if the output tensor matches the expected tensor
    assert torch.allclose(
        gram_matrix, expected_output
    ), f"gram_matrix: {gram_matrix} != {expected_output}"


@pytest.mark.parametrize("dim1, dim2", [(2, 1), (3, 3), (1, 5)])
def test_centering_matrix(dim1: int, dim2: int) -> None:
    """Test that centering_matrix satisfies `x @ S = x - mean` except for the last position.

    Also checks that calling centering_matrix with inverse=True will add the mean back to the
    centered matrix.
    """
    set_seed(0)
    x = torch.randn(dim1, dim2)
    x[:, -1] = 1.0  # Final position should be a bias position that will remain unchanged
    mean = x.mean(dim=0)
    S = centering_matrix(mean)
    centered_S = x @ S
    centered_mean = x - mean
    assert torch.allclose(centered_S[:, :-1], centered_mean[:, :-1])
    assert torch.allclose(centered_S[:, -1], x[:, -1])  # Final position should be unchanged

    # Check that the inverse centered matrix correctly adds the mean back
    S_inv = centering_matrix(mean, inverse=True)
    assert torch.allclose(x, centered_S @ S_inv)


def test_generate_sources():
    # Simple setup: Just replace one dimension with stochastic sources
    torch.manual_seed(0)
    batch_size = 1000
    n_stochastic_sources = 10
    dim_actual = 100

    phi = torch.empty(batch_size, n_stochastic_sources, dim_actual)
    phi = _generate_sources(phi)

    out = einsum("batch source dim1, batch source dim2 -> batch dim1 dim2", phi, phi)
    out /= n_stochastic_sources

    diagonal_of_out = torch.diagonal(out, dim1=1, dim2=2)
    deviation_from_identity = out - torch.eye(dim_actual)

    assert torch.allclose(diagonal_of_out, torch.ones_like(diagonal_of_out))
    assert deviation_from_identity.mean() < 0.001
    assert deviation_from_identity.std() < 0.4  # Empirically around 0.3147 for these dims


def test_generate_phis_array():
    n_batch, n_pos, n_hid = 2, 3, 4
    r_pos, r_hid = 5, 6

    get_phis = partial(
        _generate_phis_array,
        batch_size=n_batch,
        out_pos_size=n_pos,
        out_hat_hidden_size=n_hid,
        out_dim_n_chunks=1,
        out_dim_chunk_idx=0,
        device=torch.device("cpu"),
    )

    # No stochastic sources
    phis = get_phis(n_stochastic_sources_pos=None, n_stochastic_sources_hidden=None)
    assert ((phis == 0) | (phis == 1)).all()
    assert phis.shape == (n_pos * n_hid, n_batch, n_pos, n_hid)
    assert phis.sum() == n_pos * n_hid * n_batch
    phis_5d = rearrange(phis, "(npos nhid) batch pos hid -> npos nhid batch pos hid", npos=n_pos)
    assert (phis_5d.diagonal(dim1=0, dim2=3).diagonal(dim1=0, dim2=2) == 1).all()

    # Stochastic sources over position
    phis = get_phis(n_stochastic_sources_pos=r_pos, n_stochastic_sources_hidden=None)
    assert ((phis == 0) | (phis == 1) | (phis == -1)).all()
    assert phis.shape == (r_pos * n_hid, n_batch, n_pos, n_hid)
    assert (phis**2).sum() == r_pos * n_hid * n_batch * n_pos
    phis_5d = rearrange(phis, "(rpos nhid) batch pos hid -> rpos nhid batch pos hid", rpos=r_pos)
    assert (phis_5d.diagonal(dim1=1, dim2=4) != 0).all()

    # Stochastic sources over hidden
    phis = get_phis(n_stochastic_sources_pos=None, n_stochastic_sources_hidden=r_hid)
    assert ((phis == 0) | (phis == 1) | (phis == -1)).all()
    assert phis.shape == (n_pos * r_hid, n_batch, n_pos, n_hid)
    assert (phis**2).sum() == n_pos * r_hid * n_batch * n_hid
    phis_5d = rearrange(phis, "(npos rhid) batch pos hid -> npos rhid batch pos hid", npos=n_pos)
    assert (phis_5d.diagonal(dim1=0, dim2=3) != 0).all()

    # Stochastic sources over both
    phis = get_phis(n_stochastic_sources_pos=r_pos, n_stochastic_sources_hidden=r_hid)
    assert ((phis == 1) | (phis == -1)).all()
    assert phis.shape == (r_pos * r_hid, n_batch, n_pos, n_hid)
