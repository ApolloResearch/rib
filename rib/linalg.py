from typing import Callable, Literal, Optional, Union

import numpy as np
import torch
from einops import rearrange
from fancy_einsum import einsum
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from rib.types import TORCH_DTYPES, StrDtype


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


def _calc_integration_intervals(
    n_intervals: int,
    integral_boundary_relative_epsilon: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """Calculate the integration steps for n_intervals between 0+eps and 1-eps.

    Args:
        n_intervals: The number of intervals to use for the integral approximation. If 0, take a
            point estimate at alpha=0.5 instead of using the trapezoidal rule.
        integral_boundary_relative_epsilon: Rather than integrating from 0 to 1, we integrate from
            integral_boundary_epsilon to 1 - integral_boundary_epsilon, to avoid issues with
            ill-defined derivatives at 0 and 1.
            integral_boundary_epsilon = integral_boundary_relative_epsilon/(n_intervals+1).

    Returns:
        alphas: The integration steps.
        interval_size: The size of each integration step, including a correction factor to account
            for integral_boundary_epsilon.
    """
    # Scale accuracy of the integral boundaries with the number of intervals
    integral_boundary_epsilon = integral_boundary_relative_epsilon / (n_intervals + 1)
    # Integration samples
    if n_intervals == 0:
        alphas = np.array([0.5])
        interval_size = 1.0
        n_alphas = 1
    else:
        # Integration steps for n_intervals intervals
        n_alphas = n_intervals + 1
        alphas = np.linspace(integral_boundary_epsilon, 1 - integral_boundary_epsilon, n_alphas)
        assert np.allclose(np.diff(alphas), alphas[1] - alphas[0]), "alphas must be equally spaced."
        # Multiply the interval sizes by (1 + 2 eps) to balance out the smaller integration interval
        interval_size = (alphas[1] - alphas[0]) / (1 - 2 * integral_boundary_epsilon)
        assert np.allclose(
            n_intervals * interval_size,
            1,
        ), f"n_intervals * interval_size ({n_intervals * interval_size}) != 1"
    return alphas, interval_size


def calc_edge_functional(
    module_hat: Callable[[Float[Tensor, "... rib_in"], list[int]], Float[Tensor, "... rib_out"]],
    f_in_hat: Float[Tensor, "... rib_out"],
    in_tuple_dims: list[int],
    edge: Float[Tensor, "rib_out rib_in"],
    dataset_size: int,
    n_intervals: int,
    tqdm_desc: str = "Integration steps (alphas)",
) -> None:
    """Calculate the interaction attribution (edge) for module_hat using the functional method.

    Uses the trapezoidal rule for integration. Updates the edge in-place.

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

    alphas, interval_size = _calc_integration_intervals(n_intervals)

    # Compute f^{l+1}(x) to which the derivative is not applied.
    with torch.inference_mode():
        f_out_hat_const = module_hat(f_in_hat, in_tuple_dims)

    normalization_factor = f_in_hat.shape[1] * dataset_size if has_pos else dataset_size

    # Integration with the trapzoidal rule
    for alpha_idx, alpha in tqdm(enumerate(alphas), total=len(alphas), desc=tqdm_desc, leave=False):
        # As per the trapezoidal rule, multiply endpoints by 1/2 (unless taking a point estimate at
        # alpha=0.5) and multiply by the interval size.
        if n_intervals > 0 and (alpha_idx == 0 or alpha_idx == n_intervals):
            trapezoidal_scaler = 0.5 * interval_size
        else:
            trapezoidal_scaler = interval_size

        # Need to define alpha_f_in_hat for autograd
        alpha_f_in_hat = alpha * f_in_hat
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
            i_grad = (
                torch.autograd.grad(f_out_hat_norm[i], alpha_f_in_hat, retain_graph=True)[0]
                / normalization_factor
                * trapezoidal_scaler
            )
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
    tqdm_desc: str = "Integration steps (alphas)",
) -> None:
    """Calculate the interaction attribution (edge) for module_hat using the squared method.

    Uses the trapezoidal rule for integration. Updates the edge in-place.

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

    alphas, interval_size = _calc_integration_intervals(n_intervals)

    # Get sizes for intermediate result storage
    batch_size = f_in_hat.shape[0]
    rib_out_size, rib_in_size = edge.shape
    if has_pos:
        # Just run the model to see what the output pos size is
        with torch.inference_mode():
            out_pos_size = module_hat(f_in_hat, in_tuple_dims).shape[1]

    # Accumulate integral results for all x (batch) and t (out position) values.
    # We store values because we need to square the integral result before summing.
    J_hat = (
        torch.zeros(batch_size, out_pos_size, rib_out_size, rib_in_size, device=f_in_hat.device)
        if has_pos
        else torch.zeros(batch_size, rib_out_size, rib_in_size, device=f_in_hat.device)
    )

    normalization_factor = f_in_hat.shape[1] * dataset_size if has_pos else dataset_size

    # Integration with the trapzoidal rule
    for alpha_idx, alpha in tqdm(enumerate(alphas), total=len(alphas), desc=tqdm_desc, leave=False):
        # As per the trapezoidal rule, multiply endpoints by 1/2 (unless taking a point estimate at
        # alpha=0.5) and multiply by the interval size.
        if n_intervals > 0 and (alpha_idx == 0 or alpha_idx == n_intervals):
            trapezoidal_scaler = 0.5 * interval_size
        else:
            trapezoidal_scaler = interval_size

        # We have to compute inputs from f_hat to make autograd work
        alpha_f_in_hat = alpha * f_in_hat
        f_out_alpha_hat = module_hat(alpha_f_in_hat, in_tuple_dims)

        # Take the derivative of the (i, t) element (output dim and output pos) of the output.
        for out_dim in tqdm(
            range(rib_out_size),
            total=rib_out_size,
            desc="Iteration over output dims (+token dims if has_pos)",
            leave=False,
        ):
            if has_pos:
                for output_pos_idx in range(out_pos_size):
                    # autograd gives us the derivative w.r.t. b (batch dim), p (input pos) and
                    # j (input dim).
                    # The sum over the batch dimension before passing to autograd is just a trick
                    # to get the grad for every batch index vectorized.
                    i_grad = (
                        torch.autograd.grad(
                            f_out_alpha_hat[:, output_pos_idx, out_dim].sum(dim=0),
                            alpha_f_in_hat,
                            retain_graph=True,
                        )[0]
                        * trapezoidal_scaler
                    )

                    with torch.inference_mode():
                        # Element-wise multiply with f_in_hat and sum over the input pos
                        J_hat[:, output_pos_idx, out_dim, :] -= einsum(
                            "batch in_pos in_dim, batch in_pos in_dim -> batch in_dim",
                            i_grad,
                            f_in_hat,
                        )
            else:
                i_grad = (
                    torch.autograd.grad(
                        f_out_alpha_hat[:, out_dim].sum(dim=0), alpha_f_in_hat, retain_graph=True
                    )[0]
                    * trapezoidal_scaler
                )
                with torch.inference_mode():
                    # Element-wise multiply with f_in_hat
                    J_hat[:, out_dim, :] -= einsum(
                        "batch in_dim, batch in_dim -> batch in_dim", i_grad, f_in_hat
                    )

    # Square, and sum over batch size and output pos (if applicable)
    edge += (J_hat**2 / normalization_factor).sum(dim=(0, 1) if has_pos else 0)


def _generate_sources(shape, like_tensor):
    """Generate a tensor of -1 or 1 with equal probability.

    Args:
        shape: The shape of the output tensor.
        like_tensor: A tensor with the desired dtype and device (not modified).

    Returns:
        A tensor of shape `shape` with values -1 or 1 with equal probability.
    """
    return torch.where(torch.rand(shape) > 0.5, 1, -1).to(like_tensor)


def calc_basis_jacobian(
    module: torch.nn.Module,
    inputs: Union[
        tuple[Float[Tensor, "batch in_hidden"]],
        tuple[Float[Tensor, "batch pos _"], ...],
    ],
    C_out: Optional[Float[Tensor, "out_hidden out_hidden_trunc"]],
    n_intervals: int,
    n_stochastic_sources: Optional[int] = None,
    dim_stochastic_sources: Optional[Literal["both", "out_pos", "out_hidden"]] = None,
) -> Float[Tensor, "batch out_hidden_trunc out_pos in_pos in_hidden"]:
    # Ensure that the inputs have requires_grad=True
    for x in inputs:
        x.requires_grad_(True)

    alphas, interval_size = _calc_integration_intervals(n_intervals)

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
        if n_stochastic_sources is None:
            n_sources_A = out_hat_hidden_size
            n_sources_B = out_pos_size
        elif dim_stochastic_sources == "both":
            n_sources_A = n_stochastic_sources
            n_sources_B = 1
        elif dim_stochastic_sources == "out_pos":
            n_sources_A = out_hat_hidden_size
            n_sources_B = n_stochastic_sources
        elif dim_stochastic_sources == "out_hidden":
            n_sources_A = n_stochastic_sources
            n_sources_B = out_pos_size
        else:
            raise ValueError("dim_stochastic_sources must be 'both', 'out_pos', or 'out_hidden'.")

        # in_grads.shape: batch, i (out_hidden), t (out_pos), s (in_pos), j (in_hidden)
        assert in_pos_size is not None  # needed for mypy
        in_grads = torch.zeros(
            batch_size,
            n_sources_A,
            n_sources_B,
            in_pos_size,
            in_hidden_size,
            dtype=inputs[0].dtype,
            device=inputs[0].device,
        )

        phi_shape = (batch_size, n_sources_A, n_sources_B, out_hat_hidden_size, out_pos_size)
        if n_stochastic_sources is None:
            # Full dimensions
            phi = torch.zeros(phi_shape, dtype=in_grads.dtype, device=in_grads.device)
            for i in range(out_hat_hidden_size):
                for t in range(out_pos_size):
                    phi[:, i, t, i, t] = 1.0
        elif dim_stochastic_sources == "both":
            # Introduce stochastic sources: Don't iterate over all i and/or t, but a random
            # direction in the i and t space. Sources = -1 or 1 with equal probability.
            phi = _generate_sources(phi_shape, like_tensor=in_grads)
        elif dim_stochastic_sources == "out_pos":
            phi = torch.zeros(phi_shape, dtype=in_grads.dtype, device=in_grads.device)
            for i in range(out_hat_hidden_size):
                phi[:, i, :, i, :] = _generate_sources(
                    (batch_size, n_stochastic_sources, out_pos_size),
                    like_tensor=in_grads,
                )
        elif dim_stochastic_sources == "out_hidden":
            phi = torch.zeros(phi_shape, dtype=in_grads.dtype, device=in_grads.device)
            for t in range(out_pos_size):
                phi[:, :, t, :, t] = _generate_sources(
                    (batch_size, n_stochastic_sources, out_hat_hidden_size),
                    like_tensor=in_grads,
                )
        else:
            raise ValueError(
                "dim_stochastic_sources must be 'both', 'out_pos', or 'out_hidden'."
            )
        for alpha_index, alpha in tqdm(
            enumerate(alphas), total=len(alphas), desc="Integration steps (alphas)", leave=False
        ):
            # As per the trapezoidal rule, multiply endpoints by 1/2 (unless taking a
            # point estimate at alpha=0.5) and multiply by the interval size.
            if n_intervals > 0 and (alpha_index == 0 or alpha_index == n_intervals):
                trapezoidal_scaler = 0.5 * interval_size
            else:
                trapezoidal_scaler = interval_size
            # Compute f^{l+1}(f^l(alpha x))
            f_in_alpha = tuple(alpha * x for x in inputs)
            outputs_alpha = module(*f_in_alpha)
            f_out_alpha = (
                outputs_alpha
                if isinstance(outputs_alpha, torch.Tensor)
                else torch.cat(outputs_alpha, dim=-1)
            )
            f_out_hat_alpha = f_out_alpha @ C_out if C_out is not None else f_out_alpha

            # Shapes of inputs and outputs, that lead to in_grads.shape = batch, i, t, s, j/jprime
            # f_out_hat_alpha.shape: batch t i
            # f_in_alpha: batch, s, j


            for r_A, r_B in tqdm(
                np.ndindex(n_sources_A, n_sources_B),
                total=n_sources_A * n_sources_B,
                desc="Iteration over sources",
                leave=False,
            ):
                # phi_shape = (batch_size, n_sources_A, n_sources_B, out_hat_hidden_size, out_pos_size)
                phi_f_out_hat_alpha = einsum(
                    "batch out_hidden out_pos, batch out_pos out_hidden -> batch",
                    phi[:, r_A, r_B, :, :],
                    f_out_hat_alpha,
                )
                # Need to retain_graph because we call autograd on f_out_hat_alpha multiple
                # times (for each i and t, in a for loop) aka we're doing a jacobian.
                # Sum over batch is a trick to get the grad for every batch index vectorized.
                alpha_in_grads = torch.cat(
                    torch.autograd.grad(
                        phi_f_out_hat_alpha.sum(dim=0), f_in_alpha, retain_graph=True
                    ),
                    dim=-1,
                )
                in_grads[:, r_A, r_B, :, :] += alpha_in_grads * trapezoidal_scaler
                # Contraction over batch, i, t, s happens after the integral, but we cannot
                # contract earlier because we have to finish the integral sum (+=) first.
    else:
        # in_grads.shape: batch, i (out_hidden), j (in_hidden)
        in_grads = torch.zeros(
            batch_size,
            out_hat_hidden_size,
            in_hidden_size,
            dtype=inputs[0].dtype,
            device=inputs[0].device,
        )

        for alpha_index, alpha in enumerate(alphas):
            # As per the trapezoidal rule, multiply endpoints by 1/2 (unless taking a
            # point estimate at alpha=0.5) and multiply by the interval size.
            if n_intervals > 0 and (alpha_index == 0 or alpha_index == n_intervals):
                trapezoidal_scaler = 0.5 * interval_size
            else:
                trapezoidal_scaler = interval_size
            # Compute f^{l+1}(f^l(alpha x))
            f_in_alpha = tuple(alpha * x for x in inputs)
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
                in_grads[:, i] += alpha_in_grads * trapezoidal_scaler

    return in_grads


def calc_edge_stochastic(
    module_hat: Callable[[Float[Tensor, "... rib_in"], list[int]], Float[Tensor, "... rib_out"]],
    f_in_hat: Float[Tensor, "... pos rib_out"],
    in_tuple_dims: list[int],
    edge: Float[Tensor, "rib_out rib_in"],
    dataset_size: int,
    n_intervals: int,
    n_stochastic_sources: int,
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
        n_intervals: The number of intervals to use for the integral approximation. If 0, take a
            point estimate at alpha=0.5 instead of using the trapezoidal rule.
        n_stochastic_sources: The number of stochastic sources to add to each input.
        tqdm_desc: The description to use for the tqdm progress bar.
    """

    f_in_hat.requires_grad_(True)

    alphas, interval_size = _calc_integration_intervals(n_intervals)

    # Get sizes for intermediate result storage
    batch_size = f_in_hat.shape[0]
    rib_out_size, rib_in_size = edge.shape

    # Just run the model to see what the output pos size is
    with torch.inference_mode():
        out_pos_size = module_hat(f_in_hat, in_tuple_dims).shape[1]

    # Accumulate integral results for all x (batch) and r (stochastic dim) values.
    # We store values because we need to square the integral result before summing.
    J_hat = torch.zeros(
        batch_size, n_stochastic_sources, rib_out_size, rib_in_size, device=f_in_hat.device
    )

    # Create phis that are -1 or 1 with equal probability
    phi_shape = (batch_size, n_stochastic_sources, out_pos_size)
    phi = torch.where(
        torch.randn(phi_shape) < 0.0, -1 * torch.ones(phi_shape), torch.ones(phi_shape)
    ).to(dtype=f_in_hat.dtype, device=f_in_hat.device)

    normalization_factor = f_in_hat.shape[1] * dataset_size * n_stochastic_sources

    # Integration with the trapzoidal rule
    for alpha_idx, alpha in tqdm(enumerate(alphas), total=len(alphas), desc=tqdm_desc, leave=False):
        # As per the trapezoidal rule, multiply endpoints by 1/2 (unless taking a point estimate at
        # alpha=0.5) and multiply by the interval size.
        if n_intervals > 0 and (alpha_idx == 0 or alpha_idx == n_intervals):
            trapezoidal_scaler = 0.5 * interval_size
        else:
            trapezoidal_scaler = interval_size

        # We have to compute inputs from f_hat to make autograd work
        alpha_f_in_hat = alpha * f_in_hat
        f_out_alpha_hat = module_hat(alpha_f_in_hat, in_tuple_dims)

        # Take derivative of the (i, r) element (output dim and stochastic noise dim) of the output.
        for out_dim in tqdm(
            range(rib_out_size),
            total=rib_out_size,
            desc="Iteration over output dims (+stochastic sources)",
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
                i_grad = (
                    torch.autograd.grad(phi_f_out_alpha_hat, alpha_f_in_hat, retain_graph=True)[0]
                    * trapezoidal_scaler
                )

                with torch.inference_mode():
                    # Element-wise multiply with f_in_hat and sum over the input pos
                    J_hat[:, r, out_dim, :] -= einsum(
                        "in_batch in_pos in_dim, in_batch in_pos in_dim -> in_batch in_dim",
                        i_grad,
                        f_in_hat,
                    )

    # Square, and sum over batch size and output pos
    J_hat = J_hat**2 / normalization_factor
    edge += J_hat.sum(dim=(0, 1))


def integrated_gradient_trapezoidal_norm(
    module: torch.nn.Module,
    inputs: Union[
        tuple[Float[Tensor, "batch emb_in"]],
        tuple[Float[Tensor, "batch pos emb_in"]],
        tuple[Float[Tensor, "batch pos _"], ...],
    ],
    C_out: Optional[Float[Tensor, "orig_out rib_out"]],
    n_intervals: int,
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
        n_intervals: The number of intervals to use for the integral approximation. If 0, take a
            point estimate at alpha=0.5 instead of using the trapezoidal rule.
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

    alphas, interval_size = _calc_integration_intervals(n_intervals)

    for alpha_idx, alpha in enumerate(alphas):
        # Compute f^{l+1}(f^l(alpha x))
        alpha_inputs = tuple(alpha * x for x in inputs)
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
                f"Unexpected integrated gradient formula {basis_formula} != '(1-alpha)^2' or '(1-0)*alpha'"
            )

        # Accumulate the grad of f_hat_norm w.r.t the input tensors
        f_hat_norm.backward(inputs=alpha_inputs, retain_graph=True)

        alpha_in_grads = torch.cat([x.grad for x in alpha_inputs], dim=-1)
        # As per the trapezoidal rule, multiply the endpoints by 1/2 (unless we're taking a point
        # estimate at alpha=0.5)
        if n_intervals > 0 and (alpha_idx == 0 or alpha_idx == n_intervals):
            alpha_in_grads = 0.5 * alpha_in_grads

        in_grads += alpha_in_grads * interval_size

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
