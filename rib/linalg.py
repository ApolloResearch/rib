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


def _fold_jacobian_pos(
    x: Float[Tensor, "batch out_pos out_hidden in_pos in_hidden"]
) -> Float[Tensor, "batch out_pos_hidden in_pos_hidden"]:
    out = rearrange(
        x,
        "batch out_pos out_hidden in_pos in_hidden -> batch (out_pos out_hidden) (in_pos in_hidden)",
    )
    return out


def _get_jac_output_type(
    in_tuple_len: int, out_tuple_len: int, JacType: Float[Tensor, "*dims"]
) -> Union[
    Float[Tensor, "*dims"],
    tuple[Float[Tensor, "*dims"]],
    tuple[Float[Tensor, "*dims"], Float[Tensor, "*dims"]],
    tuple[
        tuple[Float[Tensor, "*dims"], Float[Tensor, "*dims"]],
        tuple[Float[Tensor, "*dims"], Float[Tensor, "*dims"]],
    ],
]:
    """Deduce the output type of the vmapped jacrev function.

    The jacrev function will return a tuple of jacobians if there are multiple outputs, and an
    (inner) tuple of jacobians if there are multiple inputs. We need to handle all combinations of
    these.

    Args:
        in_tuple_len: The number of inputs to the function.
        out_tuple_len: The number of outputs of the function. If 0, we assume that the function
            returns a single tensor as opposed to a tuple of tensors.
        JacType: The type of a single element of the jacobian output (i.e. the output of the
            vmapped jacrev function if only one output and input).

    Returns:
        The output type of the vmapped jacrev function
    """
    # Output shapes of the entire jacobian function which depends on in_tuple_len and out_tuple_len
    if out_tuple_len == 0 and in_tuple_len == 1:
        JacOutType = JacType
    elif (out_tuple_len == 0 and in_tuple_len == 2) or (out_tuple_len == 2 and in_tuple_len == 1):
        JacOutType = tuple[JacType, JacType]
    elif out_tuple_len == 1 and in_tuple_len == 1:
        JacOutType = tuple[JacType]
    elif out_tuple_len == 1 and in_tuple_len == 2:
        JacOutType = tuple[tuple[JacType, JacType]]
    elif out_tuple_len == 2 and in_tuple_len == 2:
        JacOutType = tuple[tuple[JacType, JacType], tuple[JacType, JacType]]
    else:
        raise ValueError(
            f"Unsupported combination of in_tuple_len and out_tuple_len ({in_tuple_len}, {out_tuple_len})"
        )

    return JacOutType


def batched_jacobian(
    fn: Callable,
    inputs: Union[
        tuple[Float[Tensor, "batch in_hidden"]],
        tuple[Float[Tensor, "batch pos in_hidden"]],
        tuple[Float[Tensor, "batch pos in_hidden1"], Float[Tensor, "batch pos in_hidden2"]],
    ],
    out_tuple_len: Optional[int] = 0,
) -> Float[Tensor, "batch_size out_size in_size"]:
    """Calculate the Jacobian of a function fn with respect to input x.

    Makes use of the fact that x is batched, allowing for vectorized computation of the Jacobian
    across the batch using pytorch's vmap
    (https://pytorch.org/functorch/stable/generated/functorch.jacrev.html).

    Handles the case where the input is a tuple of tensors, as well as when the output is a tuple
    of tensors.

    Assumes that fn can take as input a non-batched x and return a non-batched output.

    If we have a pos dimension, we concatenate the pos and hidden dimensions.

    If we have multiple outputs, We element-wise sum their jacobians (handling the case where we
    have one or two inputs).

    If we have multiple inputs, we concatenate them over their hidden dimension.

    Args:
        fn: The function to calculate the Jacobian of.
        inputs: The input to fn. Can be a tuple of tensors. If > 1 input, we concatenate their
            hidden dimensions after calculating the jacobian.
        out_tuple_len: The number of outputs of fn. If > 1, we concatenate their hidden dimensions
            after calculating the jacobian. If 0, we assume that fn returns a single tensor.

    Returns
    """
    in_tuple_len = len(inputs)
    assert out_tuple_len in [0, 1, 2], "Currently only outputs of 0, 1 or 2 are supported."
    has_pos_dim = any(x.dim() == 3 for x in inputs)

    if has_pos_dim:
        jac_out_dims = "batch out_pos out_hidden in_pos in_hidden"
    else:
        jac_out_dims = "batch out_hidden in_hidden"

    # The type of a single element of the jacobian output (i.e. if only one output and input)
    JacType = Float[Tensor, jac_out_dims]

    JacOutType = _get_jac_output_type(in_tuple_len, out_tuple_len, JacType)

    # Make tuple of n_inputs to get jacobians for each input. E.g. if n_inputs = 2, argnums = (0, 1)
    argnums = tuple(range(in_tuple_len))
    jac_out: JacOutType = vmap(jacrev(fn, argnums=argnums))(*inputs)

    # If there are multiple outputs, we element-wise sum the jacobians of each output
    if out_tuple_len == 2 and in_tuple_len == 1:
        jac_out: JacType = sum(jac_out)
    elif out_tuple_len == 2 and in_tuple_len == 2:
        jac_out: tuple[JacType, JacType] = tuple(sum(x) for x in zip(*jac_out))

    # If there is a pos dimension, we concatenate the pos and hidden dimensions
    if has_pos_dim:
        if isinstance(jac_out, torch.Tensor):
            jac_out: Float[Tensor, "batch out_size in_size"] = _fold_jacobian_pos(jac_out)
        elif isinstance(jac_out, tuple):
            jac_out: Union[
                tuple[Float[Tensor, "batch out_pos_hidden in_pos_hidden"]],
                tuple[
                    Float[Tensor, "batch out_pos_hidden in_pos_hidden"],
                    Float[Tensor, "batch out_pos_hidden in_pos_hidden"],
                ],
            ] = tuple(_fold_jacobian_pos(x) for x in jac_out)

    # Concatenate the inputs over the hidden dimensions (i.e. the final dimension)
    if isinstance(jac_out, tuple):
        jac_out: Float[Tensor, "batch out_size in_size"] = torch.cat(jac_out, dim=-1)

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
