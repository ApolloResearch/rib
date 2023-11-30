from typing import Callable, Literal, Optional, Union

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from rib.types import TORCH_DTYPES, StrDtype


def eigendecompose(
    x: Float[Tensor, "d_hidden d_hidden"],
    descending: bool = True,
    dtype: StrDtype = "float64",
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


def module_hat(
    f_in_hat: Float[Tensor, "... in_hidden_trunc"],
    module: torch.nn.Module,
    C_in_pinv: Float[Tensor, "in_hidden_trunc in_hidden"],
    C_out: Optional[Float[Tensor, "out_hidden out_hidden_trunc"]],
    in_tuple_dims: list[int],
):
    """Run a module in the RIB basis, f_hat^{l} --> f_hat^{l+1}.

    This converts the input f_in_hat to f_in (using C_in_pinv), splits it into a tuple (using
    in_hidden_dims), runs the module, concatenates the outputs, and converts it back to f_out_hat
    (using C_out).

    Args:
        module: The module to run.
        f_in_hat: The input in the RIB basis.
        C_in_pinv: The pseudo-inverse of input RIB rotation.
        C_out: The output RIB rotation.
        in_hidden_dims: The dimensions of the input tuple


    Returns:
        f_out_hat: The module output in the RIB basis.
    """
    f_in: Float[Tensor, "... in_hidden"] = f_in_hat @ C_in_pinv
    f_in_tuple = torch.split(f_in, in_tuple_dims, dim=-1)
    f_out_tuple = module(*f_in_tuple)
    f_out = f_out_tuple if isinstance(f_out_tuple, torch.Tensor) else torch.cat(f_out_tuple, dim=-1)

    f_out_hat: Float[Tensor, "... out_hidden_trunc"] = f_out @ C_out if C_out is not None else f_out

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


def integrated_gradient_trapezoidal_jacobian_functional(
    module_hat: Callable,
    f_in_hat: Float[Tensor, "... out_hidden_combined_trunc"],
    jac_out: Float[Tensor, "out_hidden_combined_trunc in_hidden_combined_trunc"],
    dataset_size: int,
    n_intervals: int,
) -> None:
    """Calculate the interaction attribution (edges) for module_hat with inputs f_in_hat.

    Args:
        module_hat: The RIB-wrapped module to calculate edges for.
        f_in_hat: The inputs to the module. May or may not include a position dimension.
        jac_out: The output of the jacobian calculation. This is modified in-place.
        dataset_size: The size of the dataset. Used for normalizing the gradients.
        n_intervals: The number of intervals to use for the integral approximation. If 0, take a
            point estimate at alpha=0.5 instead of using the trapezoidal rule.
    """
    has_pos = f_in_hat.ndim == 3
    # Ensure inputs require grads
    f_in_hat.requires_grad_(True)

    einsum_pattern = "bpj,bpj->j" if f_in_hat.ndim == 3 else "bj,bj->j"

    # Prepare integral
    alphas, interval_size = _calc_integration_intervals(n_intervals)

    # Compute f^{l+1}(x) to which the derivative is not applied.
    with torch.inference_mode():
        f_out_hat_const = module_hat(f_in_hat)

    for alpha_index, alpha in tqdm(
        enumerate(alphas), total=len(alphas), desc="Integration steps (alphas)", leave=False
    ):
        # As per the trapezoidal rule, multiply the endpoints by 1/2
        # (unless we're taking a point estimate at alpha=0.5)
        scaler = 0.5 if n_intervals > 0 and (alpha_index == 0 or alpha_index == n_intervals) else 1

        # Normalize by the dataset size and the number of positions (if the input has a position dim)
        normalization_factor = f_in_hat.shape[1] * dataset_size if has_pos else dataset_size
        # Need to define alpha_f_in_hat for autograd
        alpha_f_in_hat = alpha * f_in_hat
        f_out_hat_alpha = module_hat(alpha_f_in_hat)

        f_out_hat_norm: Float[Tensor, "... out_hidden_combined_trunc"] = (
            f_out_hat_const - f_out_hat_alpha
        ) ** 2
        if has_pos:
            # Sum over the position dimension
            f_out_hat_norm = f_out_hat_norm.sum(dim=1)

        # Sum over the batch dimension
        f_out_hat_norm = f_out_hat_norm.sum(dim=0)

        assert f_out_hat_norm.ndim == 1, f"f_out_hat_norm should be 1d, got {f_out_hat_norm.ndim}"
        for i in tqdm(
            range(len(f_out_hat_norm)),
            total=len(f_out_hat_norm),
            desc="Output idxs",
            leave=False,
        ):
            # Get the derivative of the ith output element w.r.t alpha_f_in_hat
            i_grad = (
                torch.autograd.grad(f_out_hat_norm[i], alpha_f_in_hat, retain_graph=True)[0]
                / normalization_factor
                * interval_size
                * scaler
            )
            with torch.inference_mode():
                E = torch.einsum(einsum_pattern, i_grad, f_in_hat)
                # We have a minus sign in front of the IG integral, see e.g. the definition of g_j
                # in equation (3.27)
                # Note that jac_out is initialised to zeros in
                # `rib.data_accumulator.collect_interaction_edges`
                jac_out[i] -= E


def integrated_gradient_trapezoidal_jacobian_squared(
    module_hat: Callable,
    f_in_hat: Float[Tensor, "... out_hidden_combined_trunc"],
    jac_out: Float[Tensor, "out_hidden_combined_trunc in_hidden_combined_trunc"],
    dataset_size: int,
    n_intervals: int,
    variable_position_dimension: bool = False,
) -> None:
    """Calculate the interaction attribution (edges) for module_hat with inputs f_in_hat.

    Args:
        module_hat: The RIB-wrapped module to calculate edges for.
        f_in_hat: The inputs to the module. May or may not include a position dimension.
        jac_out: The output of the jacobian calculation. This is modified in-place.
        dataset_size: The size of the dataset. Used for normalizing the gradients.
        n_intervals: The number of intervals to use for the integral approximation. If 0, take a
            point estimate at alpha=0.5 instead of using the trapezoidal rule.
        variable_position_dimension: If True, the size of the position dimension may vary between
            input and output of a module. Applies only to mod add currently.
    """
    has_pos = f_in_hat.ndim == 3
    # Ensure inputs require grads
    f_in_hat.requires_grad_(True)

    # Prepare integral
    alphas, interval_size = _calc_integration_intervals(n_intervals)

    # Get sizes for intermediate resullt storage
    batch_size = f_in_hat.shape[0]
    out_hidden_size_comb_trunc, in_hidden_size_comb_trunc = jac_out.shape
    if has_pos:
        # out_pos_size and in_pos_size are the same except in mod add where we throw
        # away all but one position dimension at some point
        if not variable_position_dimension:
            out_pos_size = f_in_hat.shape[1]
        else:
            # Just run the model to see what the output pos size is
            with torch.inference_mode():
                f_out_hat_const = module_hat(f_in_hat)
            out_pos_size = f_out_hat_const.shape[1]

    # Accumulate integral results for all x (batch) and t (out position) values,
    # store values because we need to square the integral result before summing
    # This term is the content of the brackets before the square, i.e. the sum over tprime
    inner_token_sums = (
        torch.zeros(
            batch_size,
            out_pos_size,
            out_hidden_size_comb_trunc,
            in_hidden_size_comb_trunc,
            device=f_in_hat.device,
        )
        if has_pos
        else torch.zeros(batch_size, out_hidden_size_comb_trunc, in_hidden_size_comb_trunc)
    )

    # Integral
    for alpha_index, alpha in tqdm(
        enumerate(alphas), total=len(alphas), desc="Integration steps (alphas)", leave=False
    ):
        # As per the trapezoidal rule, multiply the endpoints by 1/2
        # (unless we're taking a point estimate at alpha=0.5)
        scaler = 0.5 if n_intervals > 0 and (alpha_index == 0 or alpha_index == n_intervals) else 1

        # We have to compute inputs from f_hat to make autograd work
        alpha_f_in_hat = alpha * f_in_hat
        f_out_alpha_hat = module_hat(alpha_f_in_hat)

        normalization_factor = f_in_hat.shape[1] * dataset_size if has_pos else dataset_size

        # Take the derivative of the (i, t) element (output dim and output pos) of the output
        # Note that t (output pos) is different from p (tprime, input pos)
        for out_dim in range(out_hidden_size_comb_trunc):
            if has_pos:
                for token_index in range(out_pos_size):
                    # autograd gives us the derivative w.r.t. j (input dim) and p (tprime, input pos).
                    # We sum over p (tprime) != token_index (t) according to Lucius' formula.
                    # The sum is just a trick to get the grad for every batch index vectorized.
                    i_grad = (
                        torch.autograd.grad(
                            f_out_alpha_hat[:, token_index, out_dim].sum(dim=0),
                            alpha_f_in_hat,
                            retain_graph=True,
                        )[0]
                        / normalization_factor
                        * interval_size
                        * scaler
                    )

                    # Sum over tprime (p, input pos) as per Lucius' formula (A.18)
                    with torch.inference_mode():
                        inner_token_sum = torch.einsum(
                            "bpj,bpj->bj", i_grad * interval_size * scaler, f_in_hat
                        )
                        # We have a minus sign in front of the IG integral
                        inner_token_sums[:, token_index, out_dim, :] -= inner_token_sum
            else:
                i_grad = (
                    torch.autograd.grad(
                        f_out_alpha_hat[:, out_dim].sum(dim=0),
                        alpha_f_in_hat,
                        retain_graph=True,
                    )[0]
                    / normalization_factor
                    * interval_size
                    * scaler
                )
                with torch.inference_mode():
                    # The einsum is actually just an elementwise multiplication here
                    inner_token_sum = torch.einsum(
                        "bj,bj->bj",
                        i_grad,
                        f_in_hat,
                    ).to(inner_token_sums.device)
                    # We have a minus sign in front of the IG integral
                    inner_token_sums[:, out_dim, :] -= inner_token_sum

    # Finished alpha integral, integral result present in inner_token_sums
    # Square, and sum over batch size and t (not tprime)
    inner_token_sums = inner_token_sums**2
    if has_pos:
        jac_out[:, :] = inner_token_sums.sum(dim=(0, 1))
    else:
        jac_out[:, :] = inner_token_sums.sum(dim=0)


def integrated_gradient_trapezoidal_norm(
    module: torch.nn.Module,
    inputs: Union[
        tuple[Float[Tensor, "batch in_hidden"]],
        tuple[Float[Tensor, "batch pos _"], ...],
    ],
    C_out: Optional[Float[Tensor, "out_hidden out_hidden_trunc"]],
    n_intervals: int,
    basis_formula: Literal["(1-alpha)^2", "(1-0)*alpha"] = "(1-0)*alpha",
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
        basis_formula: The formula to use for the integrated gradient. Must be one of
            "(1-alpha)^2" or "(1-0)*alpha". The former is the old (October) version while the
            latter is a new (November) version that should be used from now on. The latter makes
            sense especially in light of the new attribution (edge_formula="squared") but is
            generally good and does not change results much. Defaults to "(1-0)*alpha".
    """
    # Compute f^{l+1}(x) to which the derivative is not applied.
    with torch.inference_mode():
        output_const = module(*tuple(x.detach().clone() for x in inputs))
        # Concatenate the outputs over the hidden dimension
        out_acts_const = (
            output_const
            if isinstance(output_const, torch.Tensor)
            else torch.cat(output_const, dim=-1)
        )
        if basis_formula == "(1-0)*alpha":
            output_zero = module(*tuple(torch.zeros_like(x) for x in inputs))
            # Concatenate the outputs over the hidden dimension
            out_acts_zero = (
                output_zero
                if isinstance(output_zero, torch.Tensor)
                else torch.cat(output_zero, dim=-1)
            )

    # Ensure that the inputs have requires_grad=True from now on
    for x in inputs:
        x.requires_grad_(True)

    in_grads = torch.zeros_like(torch.cat(inputs, dim=-1))

    alphas, interval_size = _calc_integration_intervals(n_intervals)

    for alpha_index, alpha in enumerate(alphas):
        # Compute f^{l+1}(f^l(alpha x))
        alpha_inputs = tuple(alpha * x for x in inputs)
        output_alpha = module(*alpha_inputs)
        # Concatenate the outputs over the hidden dimension
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
            f_hat_alpha = out_acts_alpha @ C_out if C_out is not None else out_acts_alpha
            f_hat_1_0 = (
                (out_acts_const - out_acts_zero) @ C_out
                if C_out is not None
                else (out_acts_const - out_acts_zero)
            )
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
        if n_intervals > 0 and (alpha_index == 0 or alpha_index == n_intervals):
            alpha_in_grads = 0.5 * alpha_in_grads

        in_grads += alpha_in_grads * interval_size

        for x in alpha_inputs:
            assert x.grad is not None, "Input grad should not be None."
            x.grad.zero_()

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
