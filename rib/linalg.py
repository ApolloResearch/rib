from typing import Callable, Optional, Union

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
    eigenvalues, eigenvectors = torch.linalg.eigh(x.to(dtype=TORCH_DTYPES[dtype], device="cpu"))

    eigenvalues = eigenvalues.to(dtype=x.dtype, device=x.device).abs()
    eigenvectors = eigenvectors.to(dtype=x.dtype, device=x.device)
    if descending:
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors


def calc_rotation_matrix(
    vecs: Float[Tensor, "d_hidden d_hidden_trunc"],
    vecs_pinv: Float[Tensor, "d_hidden_trunc d_hidden"],
    n_zero_vals: int = 0,
    n_ablated_vecs: int = 0,
) -> Float[Tensor, "d_hidden d_hidden"]:
    """Calculate the matrix to rotates into and out of a new basis with optional ablations.

    This can be used for rotating to and from an eigenbasis or interaction basis (or any other
    basis). The basis vectors are given in the columns of the `vecs` matrix.

    The formula for the rotation matrix is given by:

        rotation_matrix = vecs_ablated @ vecs_pinv

    where vecs_ablated is the matrix with the final n_ablated_vecs or n_zero_vals columns set to 0

    If n_ablated_vecs > 0, we ignore the smallest n_ablated_vecs vectors. Cannot have both
    n_ablated_vecs > 0 and n_zero_vals > 0.

    If n_ablated_vecs == 0 and n_zero_vals > 0, we ignore the last n_zero_vals vectors.

    Args:
        vecs: Matrix whose columns are the basis vectors.
        vecs_pinv: Pseudo-inverse of vecs. This will be the transpose if vecs is orthonormal.
        n_zero_vals: Number of vectors to zero out, starting from the last column. If > 0 and
        n_ablated_vecs == 0, we ignore the smallest n_zero_vals basis vectors.
        n_ablated_vecs: Number of vectors to ablate, starting from the last column. If > 0, we
        ignore the smallest n_ablated_vecs eigenvectors.

    Returns:
        The rotation matrix with which to right multiply incoming activations to rotate them
        into the new basis.
    """
    assert not (
        n_zero_vals > 0 and n_ablated_vecs > 0
    ), "Cannot also ignore a given n_zero_vals when ablating basis vectors."
    assert (
        n_ablated_vecs <= vecs.shape[1] and n_zero_vals <= vecs.shape[1]
    ), "Cannot ablate more basis vectors than there are."
    n_ignore = n_ablated_vecs if n_ablated_vecs > 0 else n_zero_vals
    vecs_ablated = vecs.clone().detach()
    # Zero out the final n_ignore vectors
    if n_ignore > 0:
        vecs_ablated[:, -n_ignore:] = 0
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


def batched_jacobian(
    fn: Callable,
    inputs: Union[
        tuple[Float[Tensor, "batch in_hidden"]],
        tuple[Float[Tensor, "batch pos _"], ...],
    ],
    out_tuple_len: int = 0,
) -> Float[Tensor, "batch_size out_size in_size"]:
    """Calculate the Jacobian of a function fn with respect to input x.

    Makes use of the fact that x is batched, allowing for vectorized computation of the Jacobian
    across the batch using pytorch's vmap
    (https://pytorch.org/functorch/stable/generated/functorch.jacrev.html).

    Handles the case where the inputs and output contains one or more tensors.

    Assumes that fn can take as input a tensor without a batch dimension.

    If we have a pos dimension (as in an LM), we concatenate the pos and hidden dimensions.

    If we have multiple outputs, we concatenate them over their hidden dimension. We do the same
    for multiple inputs.

    Args:
        fn: The function to calculate the Jacobian of.
        inputs: The input to fn. Can be a tuple of tensors. If > 1 input, we concatenate their
            hidden dimensions after calculating the jacobian.
        out_tuple_len: The number of outputs of fn. If > 1, we concatenate their hidden dimensions
            after calculating the jacobian. If 0, we assume that fn returns a single tensor.

    Returns:
        The Jacobian of the (concatenated) outputs of fn w.r.t its (concatenated) inputs, folding
            the pos dimension into the hidden dimension if it exists.
    """
    in_tuple_len = len(inputs)
    has_pos_dim = any(x.dim() == 3 for x in inputs)

    argnums = tuple(range(in_tuple_len)) if in_tuple_len > 1 else 0
    jac_out = vmap(jacrev(fn, argnums=argnums))(*inputs)

    # If there is a pos dimension, we concatenate the pos and hidden dimensions for all tensors
    if has_pos_dim:
        jac_out = _fold_jacobian_pos_recursive(jac_out)

    # If there are multiple inputs, the innermost tuple will correspond to the jacobians w.r.t
    # each input. We concatenate over in_hidden (i.e. the last dimension)
    if in_tuple_len > 1:
        assert isinstance(jac_out, tuple), "jac_out should be a tuple if multiple inputs."
        # Check whether there is also an inner tuple (indicating out_tuple_len > 0)
        if isinstance(jac_out[0], tuple):
            assert out_tuple_len > 0, "out_tuple_len should be > 0 if there is an inner tuple."
            # Iterate over the inner tuples and concatenate over the hidden dimension
            jac_out = tuple(torch.cat(jac_out_inner, dim=-1) for jac_out_inner in jac_out)
        else:
            assert out_tuple_len == 0, "out_tuple_len should be 0 if there is no inner tuple."
            jac_out = torch.cat(jac_out, dim=-1)

    # If out_tuple_len > 0, concatenate over out_hidden (i.e. the second last dimension)
    if isinstance(jac_out, tuple):
        # There should now be only max one tuple, depending on whether there are multiple outputs
        assert all(isinstance(jac_out_inner, torch.Tensor) for jac_out_inner in jac_out)
        assert out_tuple_len > 0, "out_tuple_len should be > 0 if there is still a tuple here."
        jac_out = torch.cat(jac_out, dim=-2)

    assert isinstance(jac_out, torch.Tensor), "jac_out should be a tensor at this point."
    assert jac_out.ndim == 3, "jac_out should have 3 dimensions at this point."

    return jac_out


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
    return res
