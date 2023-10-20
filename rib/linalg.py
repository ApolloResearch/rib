from typing import Callable, Optional, Union

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
from torch.func import jacrev, vmap

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
    eigenvalues, eigenvectors = torch.linalg.eigh(x.to(dtype=TORCH_DTYPES[dtype]))

    eigenvalues = eigenvalues.to(dtype=x.dtype).abs()
    eigenvectors = eigenvectors.to(dtype=x.dtype)
    if descending:
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors


def calc_rotation_matrix(
    vecs: Float[Tensor, "d_hidden d_hidden_trunc"],
    vecs_pinv: Float[Tensor, "d_hidden_trunc d_hidden"],
    n_ablated_vecs: int = 0,
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
        n_ablated_vecs: Number of vectors to ablate, starting from the last column. If > 0, we
        ignore the smallest n_ablated_vecs eigenvectors.

    Returns:
        The rotation matrix with which to right multiply incoming activations to rotate them
        into the new basis.
    """

    vecs_ablated = vecs.clone().detach()
    # Zero out the final n_ignore vectors
    if n_ablated_vecs > 0:
        vecs_ablated[:, -n_ablated_vecs:] = 0
    return vecs_ablated @ vecs_pinv


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
    f_in_hat: Float[Tensor, "... in_hidden_combined"],
    module: torch.nn.Module,
    C_in_pinv: Float[Tensor, "in_hidden_trunc in_hidden"],
    C_out: Optional[Float[Tensor, "out_hidden out_hidden_trunc"]],
    in_hidden_dims: list[int],
    has_pos: bool = False,
) -> Float[Tensor, "batch pos in_hidden_combined_trunc out_hidden_combined_trunc"]:
    """Calculates the norm of the alpha * in_acts @ C_in @ C_in_pinv when passed through the model.

    Since the module may take a tuple of inputs, we need to split the `x` tensor into a tuple
    based on the `in_hidden_dims` of each input.

    Note that f_in_hat may be a GradTensor resulting from a vmap operation over the batch dim.
    If this is the case, it will not have a batch dimension.

    Args:
        f_in_hat: The alpha-adjusted concatenated inputs to the model.
            i.e. alpha * in_acts @ C_in
        module: The model to pass the f_in_hat through.
        C_in_pinv: The pseudoinverse of C_in.
        C_out: The truncated interaction rotation matrix for the output node layer.
        in_hidden_dims: The hidden dimension of the original inputs to the module.

    """
    # f_in_hat @ C_in_pinv does not give exactly f due to C and C_in_pinv being truncated
    f_in_adjusted: Float[Tensor, "... in_hidden_combined_trunc"] = f_in_hat @ C_in_pinv
    input_tuples = torch.split(f_in_adjusted, in_hidden_dims, dim=-1)

    output = module(*input_tuples)

    outputs = (output,) if isinstance(output, torch.Tensor) else output
    # Concatenate the outputs over the hidden dimension
    out_acts = torch.cat(outputs, dim=-1)

    f_out_hat: Float[Tensor, "... out_hidden_combined_trunc"] = (
        out_acts @ C_out if C_out is not None else out_acts
    )

    # Calculate the square and sum over the pos dimension if it exists.
    f_out_hat_norm: Float[Tensor, "... out_hidden_combined_trunc"] = f_out_hat**2
    if has_pos:
        # f_out_hat is shape (pos, hidden) if vmapped or (batch, pos, hidden) otherwise
        assert (
            f_out_hat.dim() == 2 or f_out_hat.dim() == 3
        ), f"f_out_hat should have 2 or 3 dims, got {f_out_hat.dim()}"
        pos_dim = 0 if f_out_hat.dim() == 2 else 1
        f_out_hat_norm = f_out_hat_norm.sum(dim=pos_dim)

    return f_out_hat_norm


def integrated_gradient_trapezoidal_norm(
    module: torch.nn.Module,
    inputs: Union[
        tuple[Float[Tensor, "batch in_hidden"]],
        tuple[Float[Tensor, "batch pos _"], ...],
    ],
    C_out: Optional[Float[Tensor, "out_hidden out_hidden_trunc"]],
    n_intervals: int,
) -> Float[Tensor, "... in_hidden_combined"]:
    """Calculate the integrated gradient of the norm of the output of a module w.r.t its inputs.

    Uses the trapezoidal rule to approximate the integral between 0 and 1.

    Unlike in the integrated gradient calculation for the edge weights, this function takes the norm
    of the output of the module, condensing the output to a single number which we can run backward
    on.

    Args:
        module: The module to calculate the integrated gradient of.
        inputs: The inputs to the module. May or may not include a position dimension.
        C_out: The truncated interaction rotation matrix for the module's outputs.
        n_intervals: The number of intervals to use for the integral approximation. If 0, take a
            point estimate at alpha=1 instead of using the trapezoidal rule.
        concat_input: Whether to concatenate the input along the last dimension (used for
        transformer functionality)
    """
    # Ensure that the inputs have requires_grad=True
    for x in inputs:
        x.requires_grad_(True)

    # Use an interval size of 1/n_intervals (unless n_intervals == 0, in which case we use 1 so
    # that our multiplication by interval_size below doesn't change the result)
    interval_size = 1.0 / max(n_intervals, 1)
    in_grads = torch.zeros_like(torch.cat(inputs, dim=-1))

    alphas = np.array([1]) if n_intervals == 0 else np.arange(0, 1 + interval_size, interval_size)

    for alpha in alphas:
        alpha_inputs = tuple(alpha * x for x in inputs)
        output = module(*alpha_inputs)
        outputs = (output,) if isinstance(output, torch.Tensor) else output

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
        # estimate at alpha=1)
        if alpha == 0 or (alpha == 1 and n_intervals > 0):
            alpha_in_grads = 0.5 * alpha_in_grads

        in_grads += alpha_in_grads

        for x in alpha_inputs:
            assert x.grad is not None, "Input grad should not be None."
            x.grad.zero_()

    in_grads *= interval_size

    return in_grads


def integrated_gradient_trapezoidal_jacobian(
    fn: Callable[[Float[Tensor, "... in_hidden"]], Float[Tensor, "... out_hidden"]],
    in_tensor: Float[Tensor, "... in_hidden"],
    n_intervals: int,
) -> Float[Tensor, "... in_hidden out_hidden"]:
    """Calculate the integrated gradient of the batched jacobian of a function w.r.t its input.

    Uses the trapezoidal rule to approximate the integral between 0 and 1.

    Args:
        fn: The function to calculate the integrated gradient of.
        in_tensor: The input to the function.
        n_intervals: The number of intervals to use for the integral approximation.
    """

    # Use an interval size of 1/n_intervals (unless n_intervals == 0, in which case we use 1 so
    # that our multiplication by interval_size below doesn't change the result)
    interval_size = 1.0 / max(n_intervals, 1)
    jac_out: Optional[
        Union[
            Float[Tensor, "batch out_hidden_trunc in_hidden_trunc"],
            Float[Tensor, "batch out_hidden_trunc pos in_hidden_trunc"],
        ]
    ] = None
    alphas = np.array([1]) if n_intervals == 0 else np.arange(0, 1 + interval_size, interval_size)
    for alpha in alphas:
        # Need to detach the output to avoid a memory leak
        alpha_jac_out = vmap(jacrev(fn))(alpha * in_tensor).detach()

        # As per the trapezoidal rule, multiply the endpoints by 1/2 (unless we're taking a point
        # estimate at alpha=1)
        if alpha == 0 or (alpha == 1 and n_intervals > 0):
            alpha_jac_out = 0.5 * alpha_jac_out

        if jac_out is None:
            jac_out = alpha_jac_out
        else:
            jac_out += alpha_jac_out

    assert jac_out is not None, "jac_out should not be None."
    jac_out *= interval_size

    return jac_out
