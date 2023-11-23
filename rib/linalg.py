from typing import Callable, Optional, Union

import numpy as np
import torch
from einops import rearrange
from fancy_einsum import einsum
from jaxtyping import Float
from torch import Tensor, nn
from tqdm import tqdm

from rib.types import TORCH_DTYPES


def eigendecompose(
    x: Float[Tensor, "d_hidden d_hidden"],
    descending: bool = True,
    dtype: str = "float64",
) -> tuple[Float[Tensor, "d_hidden"], Float[Tensor, "d_hidden d_hidden"]]:
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
        eigenvalues: Diagonal matrix whose diagonal entries are the eigenvalues of x.
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


def calc_rotation_matrix(
    vecs: Float[Tensor, "d_hidden d_hidden_trunc"],
    vecs_pinv: Float[Tensor, "d_hidden_trunc d_hidden"],
    n_ablated_vecs: int,
) -> Float[Tensor, "d_hidden d_hidden"]:
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
    x: Union[Float[Tensor, "batch out_pos out_hidden in_pos in_hidden"], tuple]
) -> Union[Float[Tensor, "batch out_pos_hidden in_pos_hidden"], tuple]:
    """Recursively fold the pos dimension into the hidden dimension."""
    if isinstance(x, torch.Tensor):
        out = rearrange(
            x,
            "batch out_pos out_hidden in_pos in_hidden -> batch (out_pos out_hidden) (in_pos in_hidden)",
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


def edge_norm(
    alpha_f_in_hat: Float[Tensor, "... in_hidden_combined"],
    outputs_const: tuple[Float[Tensor, "... out_hidden_combined"]],
    module: nn.Module,
    C_in_pinv: Float[Tensor, "in_hidden_trunc in_hidden"],
    C_out: Optional[Float[Tensor, "out_hidden out_hidden_trunc"]],
    in_hidden_dims: list[int],
    has_pos: bool,
) -> Float[Tensor, "out_hidden_combined_trunc"]:
    """Calculates the norm (square, and sum over batch and position) of the difference
    `module(alpha_f_in_hat @ C_in_pinv) - outputs_const`.

    Since the module may take a tuple of inputs, we need to split the `x` tensor into a tuple
    based on the `in_hidden_dims` of each input.

    Args:
        alpha_f_in_hat: The alpha-adjusted concatenated inputs to the model.
            i.e. alpha * in_acts @ C_in (included in grad)
        outputs_const: The non-adjusted outputs of the module, i.e.
            module(in_acts) = module(f_in_hat @ C_in_pinv). Not in RIB basis.
        module: The model to pass the f_in_hat through.
        C_in_pinv: The pseudoinverse of C_in.
        C_out: The truncated interaction rotation matrix for the output node layer.
        in_hidden_dims: The hidden dimension of the original inputs to the module.
        has_pos: Whether the module has a position dimension.

    Returns:
        The norm (over batch and position) of the output of the module for every
            out-(RIB)-dimension i.
    """

    # Compute f^{l+1}(f^l(alpha x))
    alpha_f_in_adjusted: Float[Tensor, "... in_hidden_combined_trunc"] = alpha_f_in_hat @ C_in_pinv
    alpha_input_tuples = torch.split(alpha_f_in_adjusted, in_hidden_dims, dim=-1)

    output_alpha = module(*alpha_input_tuples)
    outputs_alpha = (output_alpha,) if isinstance(output_alpha, torch.Tensor) else output_alpha

    # Subtract to get f^{l+1}(x) - f^{l+1}(f^l(alpha x))
    outputs = tuple(a - b for a, b in zip(outputs_const, outputs_alpha))

    # Concatenate the outputs over the hidden dimension
    out_acts = torch.cat(outputs, dim=-1)

    f_out_hat: Float[Tensor, "... out_hidden_combined_trunc"] = (
        out_acts @ C_out if C_out is not None else out_acts
    )

    f_out_hat_norm: Float[Tensor, "... out_hidden_combined_trunc"] = f_out_hat**2
    if has_pos:
        # f_out_hat is shape (batch, pos, hidden)
        assert f_out_hat.dim() == 3, f"f_out_hat should have 3 dims, got {f_out_hat.dim()}"
        f_out_hat_norm = f_out_hat_norm.sum(dim=1)

    # Sum over the batch dimension
    f_out_hat_norm = f_out_hat_norm.sum(dim=0)

    return f_out_hat_norm


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


def integrated_gradient_trapezoidal_jacobian(
    fn: Callable,
    x: Float[Tensor, "... out_hidden_combined_trunc"],
    n_intervals: int,
    jac_out: Float[Tensor, "out_hidden_combined_trunc in_hidden_combined_trunc"],
    dataset_size: int,
) -> None:
    """Calculate the integrated gradient of the jacobian of a function w.r.t its input.

    Args:
        fn: The function to calculate the jacobian of.
        x: The input to the function.
        n_intervals: The number of intervals to use for the integral approximation. If 0, take a
            point estimate at alpha=0.5 instead of using the trapezoidal rule.
        jac_out: The output of the jacobian calculation. This is modified in-place.
        dataset_size: The size of the dataset. Used for normalizing the gradients.

    """
    einsum_pattern = "bpj,bpj->j" if x.ndim == 3 else "bj,bj->j"

    alphas, interval_size = _calc_integration_intervals(
        n_intervals, integral_boundary_relative_epsilon=1e-3
    )

    x.requires_grad_(True)

    # Normalize by the dataset size and the number of positions (if the input has a position dim)
    has_pos = x.ndim == 3
    normalization_factor = x.shape[1] * dataset_size if has_pos else dataset_size

    for alpha_index, alpha in tqdm(
        enumerate(alphas), total=len(alphas), desc="Integration steps (alphas)", leave=False
    ):
        alpha_f_in_hat = alpha * x
        f_out_hat_norm = fn(alpha_f_in_hat)

        assert f_out_hat_norm.ndim == 1, f"f_out_hat_norm should be 1d, got {f_out_hat_norm.ndim}"
        # As per the trapezoidal rule, multiply the endpoints by 1/2 (unless we're taking a point
        # estimate at alpha=0.5)
        scaler = 0.5 if n_intervals > 0 and (alpha_index == 0 or alpha_index == n_intervals) else 1
        for i in tqdm(
            range(len(f_out_hat_norm)), total=len(f_out_hat_norm), desc="Output idxs", leave=False
        ):
            # Get the derivative of the ith output element w.r.t alpha_f_in_hat
            i_grad = (
                torch.autograd.grad(f_out_hat_norm[i], alpha_f_in_hat, retain_graph=True)[0]
                / normalization_factor
                * interval_size
                * scaler
            )
            with torch.inference_mode():
                E = torch.einsum(einsum_pattern, i_grad, x)
                # We have a minus sign in front of the IG integral, see e.g. the definition of g_j
                # in equation (3.27)
                # Note that jac_out is initialised to zeros in
                # `rib.data_accumulator.collect_interaction_edges`
                jac_out[i] -= E


def integrated_gradient_trapezoidal_norm(
    module: nn.Module,
    inputs: Union[
        tuple[Float[Tensor, "batch in_hidden"]],
        tuple[Float[Tensor, "batch pos _"], ...],
    ],
    C_out: Optional[Float[Tensor, "out_hidden out_hidden_trunc"]],
    n_intervals: int,
    integral_boundary_relative_epsilon: float = 1e-3,
) -> Float[Tensor, "... in_hidden_combined"]:
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
        integral_boundary_relative_epsilon: Rather than integrating from 0 to 1, we integrate from
            integral_boundary_epsilon to 1 - integral_boundary_epsilon, to avoid issues with
            ill-defined derivatives at 0 and 1.
            integral_boundary_epsilon = integral_boundary_relative_epsilon/(n_intervals+1).
    """
    # Compute f^{l+1}(x) to which the derivative is not applied.
    with torch.inference_mode():
        output_const = module(*tuple(x.detach().clone() for x in inputs))
        outputs_const = (output_const,) if isinstance(output_const, torch.Tensor) else output_const

    # Ensure that the inputs have requires_grad=True from now on
    for x in inputs:
        x.requires_grad_(True)

    in_grads = torch.zeros_like(torch.cat(inputs, dim=-1))

    alphas, interval_size = _calc_integration_intervals(
        n_intervals, integral_boundary_relative_epsilon
    )

    for alpha_index, alpha in enumerate(alphas):
        # Compute f^{l+1}(f^l(alpha x))
        alpha_inputs = tuple(alpha * x for x in inputs)
        output_alpha = module(*alpha_inputs)
        outputs_alpha = (output_alpha,) if isinstance(output_alpha, torch.Tensor) else output_alpha

        # Subtract to get f^{l+1}(x) - f^{l+1}(f^l(alpha x))
        outputs = tuple(a - b for a, b in zip(outputs_const, outputs_alpha))

        # Concatenate the outputs over the hidden dimension
        out_acts = torch.cat(outputs, dim=-1)

        f_hat = out_acts @ C_out if C_out is not None else out_acts

        # Note that the below also sums over the batch dimension. Mathematically, this is equivalent
        # to taking the gradient of each output element separately, but it lets us simply use
        # backward() instead of more complex (and probably less efficient) vmap operations.
        f_hat_norm = (f_hat**2).sum()

        # Accumulate the grad of f_hat_norm w.r.t the input tensors
        f_hat_norm.backward(inputs=alpha_inputs, retain_graph=True)

        alpha_in_grads = torch.cat([x.grad for x in alpha_inputs], dim=-1)
        # As per the trapezoidal rule, multiply the endpoints by 1/2 (unless we're taking a point
        # estimate at alpha=0.5)
        if n_intervals > 0 and (alpha_index == 0 or alpha_index == n_intervals):
            alpha_in_grads = 0.5 * alpha_in_grads

        in_grads += alpha_in_grads * interval_size

        for x in alpha_inputs:
            assert x.grad is not None, "Input grad should not be None."
            x.grad.zero_()

    # Add the minus sign in front of the IG integral, see e.g. the definition of g_j in equation (3.27)
    in_grads *= -1

    return in_grads


def calc_gram_matrix(
    acts: Union[
        Float[Tensor, "batch pos d_hidden"],
        Float[Tensor, "batch d_hidden"],
    ],
    dataset_size: int,
) -> Float[Tensor, "d_hidden d_hidden"]:
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


def calc_linear_edge_analytic(
    W_raw: Float[Tensor, "in out"],
    f_hat_norm: Float[Tensor, "in_hidden_combined_trunc"],
    in_dim: int,
    C_in_pinv: Float[Tensor, "in_trunc in"],
    C_out: Optional[Float[Tensor, "out out_trunc"]],
    dtype: torch.dtype,
    device: str,
) -> Float[Tensor, "out_trunc in_trunc"]:
    """Calculate the edges for a purely linear module analytically.

    Args:
        W_raw: The raw weight matrix.
        f_hat_norm: The norm of the output of the module containing the weight matrix.
        in_dim: The dimension of the input to the module containing the weight matrix.
        C_in_pinv: The pseudoinverse of C_in.
        C_out: The interaction basis for the output node layer.
        dtype: The dtype to use for the calculation.
        device: The device to use for the calculation.

    Returns:
        The edge weights across the linear module containing W_raw.
    """

    # Account for cases where our function layer is the concatenation of multiple streams
    n_extra_dims = in_dim - W_raw.shape[0]

    # Create matrix ((I, 0), (0, W_raw)) where I is an identity matrix of size n_extra_dims
    # This handles the concatenated residual stream and other input stream
    W = torch.block_diag(torch.eye(n_extra_dims, dtype=dtype, device=device), W_raw)

    W_C_pinv = einsum("in out, in_trunc in -> out in_trunc", W, C_in_pinv)
    if C_out is None:
        W_hat = W_C_pinv
    else:
        W_hat = einsum("out out_trunc, out in_trunc -> out_trunc in_trunc", C_out, W_C_pinv)
    edge = einsum("out_trunc in_trunc, in_trunc -> out_trunc in_trunc", W_hat**2, f_hat_norm**2)
    return edge
