from itertools import product
from typing import Callable, Literal, NamedTuple, Optional, Union

import numpy as np
import torch
from einops import rearrange
from fancy_einsum import einsum
from jaxtyping import Float, Int8
from torch import Tensor
from tqdm import tqdm

from rib.types import TORCH_DTYPES, StrDtype
from rib.utils import get_chunk_indices


def eigendecompose(
    x: Float[Tensor, "orig orig"],
    descending: bool = True,
    dtype: StrDtype = "float64",
) -> tuple[Float[Tensor, "orig"], Float[Tensor, "orig orig"]]:
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
        eigenvalues: Vector of the eigenvalues of x.
        eigenvectors: Matrix whose columns are the eigenvectors of x.
    """
    in_dtype = x.dtype
    x = x.to(dtype=TORCH_DTYPES[dtype])
    eigenvalues, eigenvectors = torch.linalg.eigh(x)

    eigenvalues = eigenvalues.to(dtype=in_dtype).abs()
    eigenvectors = eigenvectors.to(dtype=in_dtype)
    if descending:
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors


def move_const_dir_first(
    D_dash: Float[Tensor, "d_hidden"],
    U_dash: Float[Tensor, "d_hidden d_hidden"],
) -> tuple[Float[Tensor, "d_hidden"], Float[Tensor, "d_hidden d_hidden"]]:
    """
    Finds the constant direction in D and U and moves it to the first position.

    When performing centered RIB we expect there to be a unique direction with constant activation
    that encodes the mean activation. We special-handle this RIB direction in various places and
    it's thus convenient to ensure it's first in our basis.

    This function finds that direction, asserts it's unique, and rearranges U, D to put it first.

    We use the eigenvalues as sometimes there are directions with non-zero bias component but
    very small eigenvalues. We don't want these to trigger the assert.

    Args:
        D_dash: Eigenvalues of gram matrix.
        U_dash: Eigenvectors of gram matrix.
    """
    # we expect the const dir to have non-zero component in the bias dir and nonzero eigenvalue
    threshold = 1e-6
    nonzero_in_bias_component = U_dash[-1, :].abs() > threshold
    nonzero_eigenval = D_dash.abs() > threshold
    is_const_dir = nonzero_in_bias_component & nonzero_eigenval
    assert is_const_dir.any(), "No const direction found"
    assert is_const_dir.sum() == 1, "More than one const direction found"
    const_dir_idx = is_const_dir.nonzero()[0, 0].item()
    # move the const dir to the first position
    order = torch.tensor(
        [const_dir_idx] + [i for i in range(D_dash.shape[0]) if i != const_dir_idx]
    )
    D_dash = D_dash[order]
    U_dash = U_dash[:, order]
    return D_dash, U_dash


def calc_rotation_matrix(
    vecs: Float[Tensor, "orig orig_trunc"],
    vecs_pinv: Float[Tensor, "orig_trunc orig"],
    n_ablated_vecs: int,
) -> Float[Tensor, "orig orig"]:
    """Calculate the matrix to rotates into and out of a new basis with optional ablations.

    This can be used for rotating to and from an eigenbasis or interaction basis (or any other
    basis). The basis vectors are given in the columns of the `vecs` matrix.

    The formula for the rotation matrix is given by:

        rotation_matrix = vecs_ablated @ vecs_pinv

    where vecs_ablated is the matrix with the final n_ablated_vecs set to 0.

    Args:
        vecs: Matrix whose columns are the basis vectors.
        vecs_pinv: Pseudo-inverse of vecs. This will be the transpose if vecs is orthonormal.
        n_ablated_vecs: Number of vectors to ablate, starting from the last column.

    Returns:
        The rotation matrix with which to right multiply incoming activations to rotate them
        into the new basis.
    """
    assert n_ablated_vecs >= 0, "n_ablated_vecs must be positive."
    assert n_ablated_vecs <= vecs.shape[1], "n_ablated_vecs must be less than the number of vecs."
    if n_ablated_vecs == 0:
        # No ablations, so we use an identity rotation. Note that this will be slightly different
        # from multiplying by vecs @ vecs_pinv because of the truncation.
        rotation_matrix = torch.eye(vecs.shape[0], dtype=vecs.dtype, device=vecs.device)
    elif n_ablated_vecs == vecs.shape[1]:
        # Completely zero out the matrix
        rotation_matrix = torch.zeros(
            (vecs.shape[0], vecs.shape[0]), dtype=vecs.dtype, device=vecs.device
        )
    else:
        vecs_ablated = vecs.clone().detach()
        # Zero out the final n_ignore vectors
        vecs_ablated[:, -n_ablated_vecs:] = 0
        rotation_matrix = vecs_ablated @ vecs_pinv
    return rotation_matrix


def _fold_jacobian_pos_recursive(
    x: Union[Float[Tensor, "batch out_pos orig_out in_pos orig_in"], tuple]
) -> Union[Float[Tensor, "batch pos_orig_out pos_orig_in"], tuple]:
    """Recursively fold the pos dimension into the final dimension."""
    if isinstance(x, torch.Tensor):
        out = rearrange(
            x,
            "batch out_pos orig_out in_pos orig_in -> batch (out_pos orig_out) (in_pos orig_in)",
        )
        return out
    elif isinstance(x, tuple):
        return tuple(_fold_jacobian_pos_recursive(y) for y in x)
    else:
        raise TypeError(f"Unsupported type {type(x)}")


def pinv_diag(x: Float[Tensor, "a a"]) -> Float[Tensor, "a a"]:
    """Calculate the pseudo-inverse of a diagonal matrix.

    Simply take the reciprocal of the diagonal entries.

    We check that all non-diagonal entries are 0 using a shortcut of comparing the sum of the
    absolute values of the diagonal entries to the sum of the absolute values of all entries.

    Args:
        x: A truncated diagonal matrix.

    Returns:
        Pseudo-inverse of x.
    """
    # Check that all non-diagonal entries are 0. Use a shortcut of comparing the sum of the
    # diagonal entries to the sum of all entries. They should be close
    assert torch.allclose(
        x, torch.diag(x.diag())
    ), "It appears there are non-zero off-diagonal entries. "

    res = torch.diag(x.diag().reciprocal())

    # Check if there are any 'inf' values in the result
    assert not torch.isinf(res).any(), "The resulting matrix contains 'inf' values."

    return res


def module_hat(
    f_in_hat: Float[Tensor, "... rib_in"],
    in_tuple_dims: list[int],
    module: torch.nn.Module,
    C_in_pinv: Float[Tensor, "rib_in orig_in"],
    C_out: Optional[Float[Tensor, "orig_out rib_out"]],
) -> Float[Tensor, "... rib_out"]:
    """Run a module in the RIB basis, f_hat^{l} --> f_hat^{l+1}.

    This converts the input f_in_hat to f_in (using C_in_pinv), splits it into a tuple (using
    in_tuple_dims), runs the module, concatenates the outputs, and converts it back to f_out_hat
    (using C_out).

    Args:
        module: The module to run.
        in_tuple_dims: The dimensions of the input tuple
        f_in_hat: The input in the RIB basis.
        C_in_pinv: The pseudo-inverse of input RIB rotation.
        C_out: The output RIB rotation.

    Returns:
        f_out_hat: The module output in the RIB basis.
    """
    f_in: Float[Tensor, "... orig_in"] = f_in_hat @ C_in_pinv
    f_in_tuple = torch.split(f_in, in_tuple_dims, dim=-1)
    f_out_tuple = module(*f_in_tuple)
    f_out = f_out_tuple if isinstance(f_out_tuple, torch.Tensor) else torch.cat(f_out_tuple, dim=-1)

    f_out_hat: Float[Tensor, "... rib_out"] = f_out @ C_out if C_out is not None else f_out

    return f_out_hat


def _gauss_legendre_weights(
    n_intervals: int, lower: float = 0, upper: float = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the weights and points for Gauss-Legendre quadrature.

    See https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature.
    """
    # Get the roots and weights of the Legendre polynomial of degree n_points
    x, w = np.polynomial.legendre.leggauss(n_intervals + 1)
    # Scale the x values from [-1, 1] to [lower_bound, upper_bound]
    x = (x + 1) * (upper - lower) / 2 + lower
    # Scale the weights to account for the change in x
    w = w * (upper - lower) / 2
    # Make sure weights sum to upper-lower
    assert np.allclose(w.sum(), upper - lower), f"Weights don't sum to {upper-lower}"

    return w, x


def _trapezoidal_weights(
    n_intervals: int,
    integral_boundary_relative_epsilon: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns n_intervals + 1 evenly spaced integration points, half weigted at the end points.

    Samples points between [eps, 1-eps], where
    `eps = integral_boundary_relative_epsilon / (n_intervals + 1)`. This ensures that eps goes to 0
    as n_intervals goes to infinity.
    """
    if n_intervals == 0:
        alphas = np.array([0.5])
        weights = np.array([1.0])
    else:
        # Set the integration boundaries to avoid numerical problems near 0 and 1, but still
        # scale with n_alphas to keep the desirable (?) property that n_alphas -> infinity
        # gives the exact integration result.
        integral_boundary_epsilon = integral_boundary_relative_epsilon / (n_intervals + 1)
        n_alphas = n_intervals + 1
        alphas = np.linspace(integral_boundary_epsilon, 1 - integral_boundary_epsilon, n_alphas)
        # Set weights to 1/n_intervals. We divide by n_intervals rather than n_alphas to account
        # for the endpoints being down-weighted by 0.5. Note: We previously set weights to
        # diff(alphas) and then accounted for the epsilons, but this is equivalent.
        weights = np.ones_like(alphas) / n_intervals
        weights[0] /= 2
        weights[-1] /= 2
        # Basic checks
        assert np.allclose(np.diff(alphas), alphas[1] - alphas[0]), "alphas must be equally spaced."
        assert np.allclose(weights.sum(), 1), f"Weights don't sum to 1"
    return weights, alphas


class IntegrationPoint(NamedTuple):
    alpha: float
    weight: float


def _calc_integration_points(
    n_intervals: int,
    rule: Literal["gauss-legendre", "trapezoidal", "gradient"] = "gauss-legendre",
    **kwargs,
) -> list[IntegrationPoint]:
    """Calculate the integration steps and weights for n_intervals.

    Args:
        n_intervals: The number of intervals to use for the integral approximation.
        rule: The integration method to use. One of:
            - gauss-legendre: choose some generically good points to evaluate the integrand at
                https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature
            - trapezoidal: choose equally spaced points in [ε, 1-ε], with endpoints half-weighted.
            - gradient: use only one point at α=1. This replaces Integrated Gradients with
                normal gradients.

    Returns:
        A list of IntegrationPoint objects, each defining an alpha and weight.
    """
    if rule == "gradient":
        assert n_intervals == 0
        assert kwargs == {}, f"Got unexpected arguments {kwargs}"
        weights = np.array([1.0])
        alphas = np.array([1.0])
    elif rule == "gauss-legendre":
        weights, alphas = _gauss_legendre_weights(n_intervals, **kwargs)
    elif rule == "trapezoidal":
        weights, alphas = _trapezoidal_weights(n_intervals, **kwargs)
    else:
        raise ValueError(f"Unknown rule {rule}")

    pts = [IntegrationPoint(alpha=alpha, weight=weight) for alpha, weight in zip(alphas, weights)]
    return pts


def calc_edge_functional(
    module_hat: Callable[[Float[Tensor, "... rib_in"], list[int]], Float[Tensor, "... rib_out"]],
    f_in_hat: Float[Tensor, "... rib_out"],
    in_tuple_dims: list[int],
    edge: Float[Tensor, "rib_out rib_in"],
    dataset_size: int,
    n_intervals: int,
    integration_method: Literal["trapezoidal", "gauss-legendre", "gradient"],
    tqdm_desc: str = "Integration steps (alphas)",
) -> None:
    """Calculate the interaction attribution (edge) for module_hat. Updates the edge in-place.

    Args:
        module_hat: Partial function of rib.linalg.module_hat. Takes in f_in_hat and
            in_tuple_dims as arguments and calculates f_hat^{l} --> f_hat^{l+1}.
        f_in_hat: The inputs to the module. May or may not include a position dimension.
        in_tuple_dims: The final dimensions of the inputs to the module.
        edge: The edge between f_in_hat and f_out_hat. This is modified in-place for each batch.
        dataset_size: The size of the dataset. Used for normalizing the gradients.
        n_intervals: The number of intervals to use for the integral approximation. If 0, take a
            point estimate at alpha=0.5 instead of using the trapezoidal rule.
        tqdm_desc: The description to use for the tqdm progress bar.
    """
    has_pos = f_in_hat.ndim == 3

    f_in_hat.requires_grad_(True)

    if has_pos:
        einsum_pattern = "batch in_pos in_dim, batch in_pos in_dim -> in_dim"
    else:
        einsum_pattern = "batch in_dim, batch in_dim -> in_dim"

    # Compute f^{l+1}(x) to which the derivative is not applied.
    with torch.inference_mode():
        f_out_hat_const = module_hat(f_in_hat, in_tuple_dims)

    normalization_factor = f_in_hat.shape[1] * dataset_size if has_pos else dataset_size

    int_points = _calc_integration_points(n_intervals, integration_method)
    for point in tqdm(int_points, desc=tqdm_desc, leave=False):
        # Need to define alpha_f_in_hat for autograd
        alpha_f_in_hat = point.alpha * f_in_hat
        f_out_hat_alpha = module_hat(alpha_f_in_hat, in_tuple_dims)

        f_out_hat_norm: Float[Tensor, "... rib_out"] = (f_out_hat_const - f_out_hat_alpha) ** 2
        if has_pos:
            # Sum over the position dimension
            f_out_hat_norm = f_out_hat_norm.sum(dim=1)

        # Sum over the batch dimension
        f_out_hat_norm = f_out_hat_norm.sum(dim=0)

        assert f_out_hat_norm.ndim == 1, f"f_out_hat_norm should be 1d, got {f_out_hat_norm.ndim}"
        for i in tqdm(
            range(len(f_out_hat_norm)),
            total=len(f_out_hat_norm),
            desc="Iteration over output dims (functional edges)",
            leave=False,
        ):
            # Get the derivative of the ith output element w.r.t alpha_f_in_hat
            i_grad = torch.autograd.grad(f_out_hat_norm[i], alpha_f_in_hat, retain_graph=True)[0]
            i_grad *= point.weight / normalization_factor
            with torch.inference_mode():
                E = einsum(einsum_pattern, i_grad, f_in_hat)
                # Note that edge is initialised to zeros in
                # `rib.data_accumulator.collect_interaction_edges`
                edge[i] -= E


def calc_edge_squared(
    module_hat: Callable[[Float[Tensor, "... rib_in"], list[int]], Float[Tensor, "... rib_out"]],
    f_in_hat: Float[Tensor, "... rib_out"],
    in_tuple_dims: list[int],
    edge: Float[Tensor, "rib_out rib_in"],
    dataset_size: int,
    n_intervals: int,
    out_dim_start_idx: int,
    out_dim_end_idx: int,
    integration_method: Literal["trapezoidal", "gauss-legendre", "gradient"],
    tqdm_desc: str = "Integration steps (alphas)",
) -> None:
    """Calculate the interaction attribution (edge) for module_hat, updating edges in-place.

    Args:
        module_hat: Partial function of rib.linalg.module_hat. Takes in f_in_hat and
            in_tuple_dims as arguments and calculates f_hat^{l} --> f_hat^{l+1}.
        f_in_hat: The inputs to the module. May or may not include a position dimension.
        in_tuple_dims: The final dimensions of the inputs to the module.
        edge: The edge between f_in_hat and f_out_hat. This is modified in-place for each batch.
        dataset_size: The size of the dataset. Used for normalizing the gradients.
        out_dim_start_idx: The index of the first output dimension to calculate.
        out_dim_end_idx: The index of the last output dimension to calculate.
        n_intervals: The number of intervals to use for the integral approximation.
        integration_method: Method to choose integration points.
        tqdm_desc: The description to use for the tqdm progress bar.
    """
    has_pos = f_in_hat.ndim == 3

    f_in_hat.requires_grad_(True)

    # Get sizes for intermediate result storage
    batch_size = f_in_hat.shape[0]
    rib_in_size = edge.shape[1]

    chunk_size = out_dim_end_idx - out_dim_start_idx

    if has_pos:
        # Just run the model to see what the output pos size is
        with torch.inference_mode():
            out_pos_size = module_hat(f_in_hat, in_tuple_dims).shape[1]

    # Accumulate integral results for all x (batch) and t (out position) values.
    # We store values because we need to square the integral result before summing.
    J_hat = (
        torch.zeros(batch_size, out_pos_size, chunk_size, rib_in_size, device=f_in_hat.device)
        if has_pos
        else torch.zeros(batch_size, chunk_size, rib_in_size, device=f_in_hat.device)
    )

    normalization_factor = f_in_hat.shape[1] * dataset_size if has_pos else dataset_size

    int_points = _calc_integration_points(n_intervals, integration_method)
    for point in tqdm(int_points, desc=tqdm_desc, leave=False):
        # We have to compute inputs from f_hat to make autograd work
        alpha_f_in_hat = point.alpha * f_in_hat
        f_out_alpha_hat = module_hat(alpha_f_in_hat, in_tuple_dims)

        # Take the derivative of the (i, t) element (output dim and output pos) of the output.
        for idx_in_chunk, out_dim in tqdm(
            enumerate(range(out_dim_start_idx, out_dim_end_idx)),
            total=chunk_size,
            desc=f"Iteration over output dims (+token dims if has_pos). Chunk_idxs: {out_dim_start_idx}-{out_dim_end_idx}",
            leave=False,
        ):
            if has_pos:
                for output_pos_idx in range(out_pos_size):
                    # autograd gives us the derivative w.r.t. b (batch dim), p (input pos) and
                    # j (input dim).
                    # The sum over the batch dimension before passing to autograd is just a trick
                    # to get the grad for every batch index vectorized.
                    i_grad = torch.autograd.grad(
                        f_out_alpha_hat[:, output_pos_idx, out_dim].sum(dim=0),
                        alpha_f_in_hat,
                        retain_graph=True,
                    )[0]
                    i_grad *= point.weight

                    with torch.inference_mode():
                        # Element-wise multiply with f_in_hat and sum over the input pos
                        J_hat[:, output_pos_idx, idx_in_chunk, :] -= einsum(
                            "batch in_pos in_dim, batch in_pos in_dim -> batch in_dim",
                            i_grad,
                            f_in_hat,
                        )
            else:
                i_grad = torch.autograd.grad(
                    f_out_alpha_hat[:, out_dim].sum(dim=0), alpha_f_in_hat, retain_graph=True
                )[0]
                i_grad *= point.weight

                with torch.inference_mode():
                    # Element-wise multiply with f_in_hat
                    J_hat[:, idx_in_chunk, :] -= einsum(
                        "batch in_dim, batch in_dim -> batch in_dim", i_grad, f_in_hat
                    )

    # Square, and sum over batch size and output pos (if applicable)
    edge += (J_hat**2 / normalization_factor).sum(dim=(0, 1) if has_pos else 0)


def _generate_sources(like_tensor: Tensor):
    """Generate a tensor of ±1s, with shape and dtype given by `like_tensor`."""
    bools = torch.rand(size=like_tensor.size(), device=like_tensor.device) > 0.5
    return torch.where(bools, 1, -1).to(like_tensor)


def _generate_phis_array(
    batch_size: int,
    out_pos_size: int,
    out_hat_hidden_size: int,
    n_stochastic_sources_pos: Optional[int],
    n_stochastic_sources_hidden: Optional[int],
    out_dim_n_chunks: int,
    out_dim_chunk_idx: int,
    device: torch.device,
) -> Int8[Tensor, "n_phis batch out_pos out_hidden"]:
    """Makes array representing the output pos and emb compoents to weight in the jacobian.

    The output tesors can be thought of as a collection of phi arrays, where each phi is a single
    vector to take a jacobian-vector product with. Each phi is shape (batch out_pos out_hidden)
    to match the output activations.
    - If no stochasticity is used, we iterate over (pos, hidden) pairs (t, i), making phi with
        - phi[:, t, i] = 1
        - phi zero everywhere else
    - If stocasticity is over position only, we iterate over (pos_sources, hidden) pairs (r_p, i):
        - phi[:, :, i] = ±1
        - phi zero everywhere else
    - If stocasticity is over hidden only, we iterate over (pos, hidden_sources) pairs (t, r_h):
        - phi[:, t, :] = ±1
        - phi zero everywhere else
    - If stocasticity is over both, we iterate over (pos_sources, hidden_sources) pairs (r_p, r_h):
        - phi = ±1 everywhere

    All values will be in {-1, 0, 1}. We use int8 to save memory.

    When running our basis calculation distributed over multiple processes, we only return one
    chunk of the phis per process. The outputs are combined by summing the resulting M_dash
    tensors in `rib.data_accumulator.collect_M_dash_and_Lambda_dash`.
    """
    phis = []
    all_pos_hid_idxs = list(
        product(
            range(n_stochastic_sources_pos or out_pos_size),
            range(n_stochastic_sources_hidden or out_hat_hidden_size),
        )
    )
    subset_pod_hid_idxs = all_pos_hid_idxs[
        slice(
            *get_chunk_indices(
                len(all_pos_hid_idxs), n_chunks=out_dim_n_chunks, chunk_idx=out_dim_chunk_idx
            )
        )
    ]
    for t, i in subset_pod_hid_idxs:
        phi = torch.zeros(
            (batch_size, out_pos_size, out_hat_hidden_size), dtype=torch.int8, device=device
        )
        if n_stochastic_sources_pos is None and n_stochastic_sources_hidden is None:
            phi[:, t, i] = 1
        else:
            p_slice = t if n_stochastic_sources_pos is None else slice(None)
            h_slice = i if n_stochastic_sources_hidden is None else slice(None)
            phi[:, p_slice, h_slice] = _generate_sources(phi[:, p_slice, h_slice])  # type: ignore[index]
        phis.append(phi)
    return torch.stack(phis, dim=0)


def calc_basis_jacobian(
    module: torch.nn.Module,
    inputs: Union[
        tuple[Float[Tensor, "batch in_hidden"]],
        tuple[Float[Tensor, "batch pos _"], ...],
    ],
    C_out: Optional[Float[Tensor, "out_hidden out_hidden_trunc"]],
    n_intervals: int,
    integration_method: Literal["trapezoidal", "gauss-legendre", "gradient"],
    n_stochastic_sources_pos: Optional[int] = None,
    n_stochastic_sources_hidden: Optional[int] = None,
    out_dim_n_chunks: int = 1,
    out_dim_chunk_idx: int = 0,
) -> Float[Tensor, "out_hidden_or_sources batch in_pos in_hidden"]:
    # Ensure that the inputs have requires_grad=True
    for x in inputs:
        x.requires_grad_(True)

    int_points = _calc_integration_points(n_intervals, integration_method)

    has_pos = inputs[0].ndim == 3
    batch_size = inputs[0].shape[0]
    in_pos_size = inputs[0].shape[1] if has_pos else None
    in_hidden_size = sum(x.shape[-1] for x in inputs)
    # Compute out sizes
    with torch.inference_mode():
        f_out_dummy = module(*inputs)
        f_out_dummy = (f_out_dummy,) if isinstance(f_out_dummy, torch.Tensor) else f_out_dummy

    out_hidden_size = sum(x.shape[-1] for x in f_out_dummy)
    out_hat_hidden_size = out_hidden_size if C_out is None else C_out.shape[1]
    out_pos_size = f_out_dummy[0].shape[1] if has_pos else None
    if C_out is not None:
        assert out_hidden_size == C_out.shape[0], "C_out has wrong shape"
    assert batch_size == f_out_dummy[0].shape[0], "batch size mismatch"

    if has_pos:
        phis: Int8[Tensor, "n_phis batch out_pos out_hidden"] = _generate_phis_array(
            batch_size=batch_size,
            out_pos_size=out_pos_size,
            out_hat_hidden_size=out_hat_hidden_size,
            n_stochastic_sources_pos=n_stochastic_sources_pos,
            n_stochastic_sources_hidden=n_stochastic_sources_hidden,
            out_dim_n_chunks=out_dim_n_chunks,
            out_dim_chunk_idx=out_dim_chunk_idx,
            device=inputs[0].device,
        )
        if out_dim_n_chunks == 1:
            assert phis.shape[0] == (n_stochastic_sources_pos or out_pos_size) * (
                n_stochastic_sources_hidden or out_hat_hidden_size
            )

        # in_grads.shape: batch, i (out_hidden), t (out_pos), s (in_pos), j (in_hidden)
        assert in_pos_size is not None  # needed for mypy
        in_grads: Float[Tensor, "n_phis batch in_pos in_hidden"] = torch.zeros(
            phis.shape[0],
            batch_size,
            in_pos_size,
            in_hidden_size,
            dtype=inputs[0].dtype,
            device=inputs[0].device,
        )

        for point in tqdm(
            int_points, desc="Integration steps (alphas)", leave=False, disable=len(int_points) == 1
        ):
            # Compute f^{l+1}(f^l(alpha x))
            f_in_alpha = tuple(point.alpha * x for x in inputs)
            outputs_alpha = module(*f_in_alpha)
            f_out_alpha = (
                outputs_alpha
                if isinstance(outputs_alpha, torch.Tensor)
                else torch.cat(outputs_alpha, dim=-1)
            )
            f_out_hat_alpha = f_out_alpha @ C_out if C_out is not None else f_out_alpha

            for r, phi in tqdm(enumerate(phis), desc="Iteration over sources", leave=False):
                # Torch supports taking jacobian-vector products (where our vector is given by phi)
                # It's possible to pass in all of phis at once with `is_grads_batched=True` and it
                # will be a bit faster due to vmap but the memory usage is much higher.

                # phi_in_grads is a tuple of the same shape as inputs
                phi_in_grads = torch.autograd.grad(
                    outputs=f_out_hat_alpha, inputs=f_in_alpha, grad_outputs=phi, retain_graph=True
                )
                in_grads[r] += torch.cat(phi_in_grads, dim=-1) * point.weight
    else:
        if not (out_dim_n_chunks == 1 and out_dim_chunk_idx == 0):
            raise NotImplementedError

        # in_grads.shape: batch, i (out_hidden), j (in_hidden)
        in_grads = torch.zeros(
            batch_size,
            out_hat_hidden_size,
            in_hidden_size,
            dtype=inputs[0].dtype,
            device=inputs[0].device,
        )

        for point in tqdm(int_points, desc="Integration steps (alphas)", leave=False):
            # Compute f^{l+1}(f^l(alpha x))
            f_in_alpha = tuple(point.alpha * x for x in inputs)
            outputs_alpha = module(*f_in_alpha)
            f_out_alpha = (
                outputs_alpha
                if isinstance(outputs_alpha, torch.Tensor)
                else torch.cat(outputs_alpha, dim=-1)
            )
            f_out_hat_alpha = f_out_alpha @ C_out if C_out is not None else f_out_alpha

            # Calculate the grads, numerator index i
            for i in range(out_hat_hidden_size):
                # Need to retain_graph because we call autpgrad on f_out_hat_alpha multiple
                # times (for each i in a for loop) aka we're doing a jacobian.
                # Sum over batch is a trick to get the grad for every batch index vectorized.
                alpha_in_grads = torch.cat(
                    torch.autograd.grad(
                        f_out_hat_alpha[:, i].sum(dim=0), f_in_alpha, retain_graph=True
                    ),
                    dim=-1,
                )
                in_grads[:, i] += alpha_in_grads * point.weight

    return in_grads


def calc_edge_stochastic(
    module_hat: Callable[[Float[Tensor, "... rib_in"], list[int]], Float[Tensor, "... rib_out"]],
    f_in_hat: Float[Tensor, "... pos rib_out"],
    in_tuple_dims: list[int],
    edge: Float[Tensor, "rib_out rib_in"],
    dataset_size: int,
    n_intervals: int,
    integration_method: Literal["trapezoidal", "gauss-legendre", "gradient"],
    n_stochastic_sources: int,
    out_dim_start_idx: int,
    out_dim_end_idx: int,
    tqdm_desc: str = "Integration steps (alphas)",
) -> None:
    """Calculate the interaction attribution (edge) for module_hat using the stochastic method.

    Note that this method can only be run with models that have a position dimension.

    Uses the trapezoidal rule for integration. Updates the edge in-place.

    Args:
        module_hat: Partial function of rib.linalg.module_hat. Takes in f_in_hat and
            in_tuple_dims as arguments and calculates f_hat^{l} --> f_hat^{l+1}.
        f_in_hat: The inputs to the module. May or may not include a position dimension.
        in_tuple_dims: The final dimensions of the inputs to the module.
        edge: The edge between f_in_hat and f_out_hat. This is modified in-place for each batch.
        dataset_size: The size of the dataset. Used for normalizing the gradients.
        n_intervals: The number of intervals to use for the integral approximation.
        integration_method: Method to choose integration points.
        n_stochastic_sources: The number of stochastic sources to add to each input.
        out_dim_start_idx: The index of the first output dimension to calculate.
        out_dim_end_idx: The index of the last output dimension to calculate.
        tqdm_desc: The description to use for the tqdm progress bar.
    """

    f_in_hat.requires_grad_(True)

    # Get sizes for intermediate result storage
    batch_size = f_in_hat.shape[0]
    rib_in_size = edge.shape[1]

    chunk_size = out_dim_end_idx - out_dim_start_idx

    # Just run the model to see what the output pos size is
    with torch.inference_mode():
        out_pos_size = module_hat(f_in_hat, in_tuple_dims).shape[1]

    # Accumulate integral results for all x (batch) and r (stochastic dim) values.
    # We store values because we need to square the integral result before summing.
    J_hat = torch.zeros(
        batch_size, n_stochastic_sources, chunk_size, rib_in_size, device=f_in_hat.device
    )

    # Create phis that are -1 or 1 with equal probability
    phi_shape = (batch_size, n_stochastic_sources, out_pos_size)
    phi = torch.where(
        torch.randn(phi_shape) < 0.0, -1 * torch.ones(phi_shape), torch.ones(phi_shape)
    ).to(dtype=f_in_hat.dtype, device=f_in_hat.device)

    normalization_factor = f_in_hat.shape[1] * dataset_size * n_stochastic_sources

    int_points = _calc_integration_points(n_intervals, integration_method)
    for point in tqdm(int_points, desc=tqdm_desc, leave=False):
        # We have to compute inputs from f_hat to make autograd work
        alpha_f_in_hat = point.alpha * f_in_hat
        f_out_alpha_hat = module_hat(alpha_f_in_hat, in_tuple_dims)

        # Take the derivative of the (i, t) element (output dim and output pos) of the output.
        for idx_in_chunk, out_dim in tqdm(
            enumerate(range(out_dim_start_idx, out_dim_end_idx)),
            total=chunk_size,
            desc=f"Iteration over output dims and stochastic sources. Chunk_idxs: {out_dim_start_idx}-{out_dim_end_idx}",
            leave=False,
        ):
            for r in range(n_stochastic_sources):
                # autograd gives us the derivative w.r.t. in_batch, in_pos, in_dim.
                # The sum over the out_batch dim before passing to autograd is just a trick
                # to get the grad for every batch index vectorized.
                phi_f_out_alpha_hat = einsum(
                    "out_batch out_pos, out_batch out_pos ->",
                    phi[:, r, :],
                    f_out_alpha_hat[:, :, out_dim],
                )
                i_grad = torch.autograd.grad(
                    phi_f_out_alpha_hat, alpha_f_in_hat, retain_graph=True
                )[0]
                i_grad *= point.weight

                with torch.inference_mode():
                    # Element-wise multiply with f_in_hat and sum over the input pos
                    J_hat[:, r, idx_in_chunk, :] -= einsum(
                        "in_batch in_pos in_dim, in_batch in_pos in_dim -> in_batch in_dim",
                        i_grad,
                        f_in_hat,
                    )

    # Square, and sum over batch size and output pos
    edge += (J_hat**2 / normalization_factor).sum(dim=(0, 1))


def calc_basis_integrated_gradient(
    module: torch.nn.Module,
    inputs: Union[
        tuple[Float[Tensor, "batch emb_in"]],
        tuple[Float[Tensor, "batch pos emb_in"]],
        tuple[Float[Tensor, "batch pos _"], ...],
    ],
    C_out: Optional[Float[Tensor, "orig_out rib_out"]],
    n_intervals: int,
    integration_method: Literal["trapezoidal", "gauss-legendre", "gradient"],
    basis_formula: Literal["(1-alpha)^2", "(1-0)*alpha"] = "(1-0)*alpha",
) -> Float[Tensor, "... orig_in"]:
    """Calculate the integrated gradient of the norm of the output of a module w.r.t its inputs,
    following the definition of e.g. g() in equation (3.27) of the paper. This means we compute the
    derivative of f^{l+1}(x) - f^{l+1}(f^l(alpha x)) where module(·) is f^{l+1}(·).

    Uses the trapezoidal rule to approximate the integral between 0+eps and 1-eps.

    Unlike in the integrated gradient calculation for the edge weights, this function takes the norm
    of the output of the module, condensing the output to a single number which we can run backward
    on. (Thus we do not need to use jacrev.)

    Args:
        module: The module to calculate the integrated gradient of.
        inputs: The inputs to the module. May or may not include a position dimension.
        C_out: The truncated interaction rotation matrix for the module's outputs.
        n_intervals: The number of intervals to use for the integral approximation.
        integration_method: Method to choose integration points.
        basis_formula: The formula to use for the integrated gradient. Must be one of
            "(1-alpha)^2" or "(1-0)*alpha". The former is the old (October) version while the
            latter is a new (November) version that should be used from now on. The latter makes
            sense especially in light of the new attribution (edge_formula="squared") but is
            generally good and does not change results much. Defaults to "(1-0)*alpha".

    Returns:
        The integrated gradient of the norm of the module output w.r.t. its inputs.
    """
    # Compute f^{l+1}(x) to which the derivative is not applied.
    with torch.inference_mode():
        output_const = module(*tuple(x.detach().clone() for x in inputs))
        # Concatenate the outputs over the final dimension
        out_acts_const = (
            output_const
            if isinstance(output_const, torch.Tensor)
            else torch.cat(output_const, dim=-1)
        )

    # Ensure that the inputs have requires_grad=True from now on
    for x in inputs:
        x.requires_grad_(True)

    in_grads = torch.zeros_like(torch.cat(inputs, dim=-1))

    for point in _calc_integration_points(n_intervals, integration_method):
        # Compute f^{l+1}(f^l(alpha x))
        alpha_inputs = tuple(point.alpha * x for x in inputs)
        output_alpha = module(*alpha_inputs)
        # Concatenate the outputs over the final dimension
        out_acts_alpha = (
            output_alpha
            if isinstance(output_alpha, torch.Tensor)
            else torch.cat(output_alpha, dim=-1)
        )

        if basis_formula == "(1-alpha)^2":
            # Subtract to get f^{l+1}(x) - f^{l+1}(f^l(alpha x))
            f_hat_1_alpha = (
                (out_acts_const - out_acts_alpha) @ C_out
                if C_out is not None
                else (out_acts_const - out_acts_alpha)
            )
            # Note that the below also sums over the batch dimension. Mathematically, this is equivalent
            # to taking the gradient of each output element separately, but it lets us simply use
            # backward() instead of more complex (and probably less efficient) vmap operations.
            # Note the minus sign here. In the paper this minus is in front of the integral, but
            # for generality we put it here.
            f_hat_norm = -(f_hat_1_alpha**2).sum()
        elif basis_formula == "(1-0)*alpha":
            # clone out_acts_alpha and out_acts_const as they were created in torch.inference_mode
            f_hat_alpha = out_acts_alpha @ C_out if C_out is not None else out_acts_alpha.clone()
            f_hat_1_0 = out_acts_const @ C_out if C_out is not None else out_acts_const.clone()
            f_hat_norm = (f_hat_alpha * f_hat_1_0).sum()
        else:
            raise ValueError(
                f"Unexpected basis formula {basis_formula} != '(1-alpha)^2' or '(1-0)*alpha'"
            )

        # Accumulate the grad of f_hat_norm w.r.t the input tensors
        f_hat_norm.backward(inputs=alpha_inputs, retain_graph=True)

        alpha_in_grads = torch.cat([x.grad for x in alpha_inputs], dim=-1)  # type: ignore
        in_grads += alpha_in_grads * point.weight

        for x in alpha_inputs:
            assert x.grad is not None, "Input grad should not be None."
            x.grad.zero_()

    return in_grads


def calc_gram_matrix(
    acts: Union[
        Float[Tensor, "batch pos orig"],
        Float[Tensor, "batch orig"],
    ],
    dataset_size: int,
) -> Float[Tensor, "orig orig"]:
    """Calculate the gram matrix for a given tensor.

    The gram is normalized by the number of positions if the tensor has a position dimension.

    Note that the inputs must contain a batch dimension, otherwise the normalization will not be
    correct.

    Args:
        acts: The tensor to calculate the gram matrix for. May or may not have a position dimension.
        dataset_size: The size of the dataset. Used for scaling the gram matrix.

    Returns:
        The gram matrix.
    """
    if acts.dim() == 3:  # tensor with pos dimension
        einsum_pattern = "bpi, bpj -> ij"
        normalization_factor = acts.shape[1] * dataset_size
    elif acts.dim() == 2:  # tensor without pos dimension
        einsum_pattern = "bi, bj -> ij"
        normalization_factor = dataset_size
    else:
        raise ValueError("Unexpected tensor rank")

    return torch.einsum(einsum_pattern, acts / normalization_factor, acts)


def centering_matrix(
    mean: Float[torch.Tensor, "emb"],
    inverse: bool = False,
) -> Float[torch.Tensor, "emb emb"]:
    """
    Returns a matrix S such that `x @ S = x - mean` (everywhere except the last position of x)

    If inverse=True, instead returns the inverse (a matrix that adds the mean back to x)
    Example:
        >>> mean = torch.tensor([2., 2., 4., 1.])
        >>> shift_matrix(mean, inverse=False)
        tensor([[1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [-2., -2., -4., 1.]])
    """
    assert mean.ndim == 1, "mean must be 1d"
    assert (mean[-1].abs() - 1).abs() < 1e-6, "last element of mean must be 1 or -1"
    S = torch.eye(mean.shape[0], dtype=mean.dtype, device=mean.device)
    shift = mean.clone() if inverse else -mean.clone()
    S[-1, :-1] = shift[:-1]  # don't shift the bias position
    return S
