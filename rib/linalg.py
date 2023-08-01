from dataclasses import dataclass
from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor
from torch.func import jacrev, vmap


@dataclass
class EigenInfo:
    """Information about the eigendecomposition of a gram matrix."""

    hook_name: str
    eigenvals: Float[Tensor, "d_hidden"]
    eigenvecs: Float[Tensor, "d_hidden d_hidden"]  # Columns are eigenvectors


def eigendecompose(
    x: Float[Tensor, "d_hidden d_hidden"],
    descending: bool = True,
    dtype: torch.dtype = torch.float64,
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
    eigenvalues, eigenvectors = torch.linalg.eigh(x.to(dtype=dtype, device="cpu"))

    eigenvalues = eigenvalues.to(dtype=x.dtype, device=x.device)
    eigenvectors = eigenvectors.to(dtype=x.dtype, device=x.device)
    if descending:
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors


def calc_rotation_matrix(
    vecs: Float[Tensor, "d_hidden d_hidden"],
    n_zero_vals: int = 0,
    n_ablated_vecs: int = 0,
) -> Float[Tensor, "d_hidden d_hidden"]:
    """Calculate the matrix to rotates into and out of the orthogonal basis with optional ablations.

    The formula for the rotation matrix is given by:

        rotation_matrix = vecs.T @ basis @ vecs

    where basis is the standard basis of size d_hidden with the final n_zero_vals or n_ablated_vecs
    rows/columns set to 0.

    If n_ablated_vecs > 0, we ignore the smallest n_ablated_vecs eigenvectors (regardless of
    the number of zero eigenvalues).

    If n_ablated_vecs == 0 and n_zero_vals > 0, we ignore the eigenvectors which correspond to zero
    eigenvalues (as given by `n_zero_vals`).

    Args:
        vecs: Matrix whose columns are the eigenvectors of the gram matrix.
        n_zero_vals: Number of zero eigenvalues. If > 0 and n_ablated_vecs == 0, we ignore the
            smallest n_zero_vals eigenvectors.
        n_ablated_vecs: Number of eigenvectors to ablate. If > 0, we ignore the smallest
            n_ablated_vecs eigenvectors.

    Returns:
        The rotation matrix with which to right multiply incoming activations to rotate them
        into the orthogonal basis.
    """
    assert not (
        n_zero_vals > 0 and n_ablated_vecs > 0
    ), "Cannot also ignore zero eigenvalues when ablating eigenvectors."
    assert (
        n_ablated_vecs <= vecs.shape[0] and n_zero_vals <= vecs.shape[0]
    ), "Cannot ablate more eigenvectors than there are."
    n_ignore = n_ablated_vecs if n_ablated_vecs > 0 else n_zero_vals
    basis = torch.eye(vecs.shape[0], dtype=vecs.dtype, device=vecs.device)
    basis[vecs.shape[0] - n_ignore :] = 0

    rotation_matrix = vecs @ basis @ vecs.T
    return rotation_matrix


def batched_jacobian(
    fn: Callable, x: Float[Tensor, "batch_size in_size"]
) -> Float[Tensor, "batch_size out_size in_size"]:
    """Calculate the Jacobian of a function fn with respect to input x.

    Makes use of the fact that x is batched, allowing for vectorized computation of the Jacobian
    across the batch using pytorch's vmap
    (https://pytorch.org/functorch/stable/generated/functorch.jacrev.html).

    Assumes that fn can take as input a non-batched x and return a non-batched output.
    """
    return vmap(jacrev(fn))(x)


def calc_interaction_matrix(
    module: torch.nn.Module,
    in_acts: Float[Tensor, "batch_size in_hidden"],
    out_rotation_matrix: Float[Tensor, "out_hidden out_hidden"],
    in_gram_eigenvecs: Float[Tensor, "in_hidden in_hidden"],
    in_gram_eigenvals: Float[Tensor, "in_hidden"],
    out_acts: Float[Tensor, "batch_size out_hidden"],
) -> Float[Tensor, "out_hidden out_hidden"]:
    """
    NOTE: UNTESTED STUB FUNCTION: The interaction matrix for the input layer is calculated as:

        M = inner_product(sum_i(f_i^{l+1} O'_{i,j}^l), sum_{i'}(f_{i'}^{l+1} O'_{i',j'}^l))
    with f_i^{l+1} representing this module's outputs, and O'_{i,j}^l representing the jacobian of
    the layer w.r.t the inputs transformed by:
        O'^l = C^{l+1} O^l U^l.T {D^l}^{0.5}
    where l represents the input layer and l+1 the output layer, and:
        C^{l+1}: (out_hidden, out_hidden) the rotation matrix at the output
        O^l: (batch, out_hidden, in_hidden) the jacobian of the layer w.r.t the inputs
        U^l: (in_hidden, in_hidden) matrix with columns as the eigenvectors of the input gram matrix
        D^l: (in_hidden) the eigenvalues of the input


    Note that the batch_jacobian must be calculated outside of inference_mode (otherwise it
    will be zero).

    Args:
        module: Module that the hook is attached to.
        in_acts: Input to the module.
        out_rotation_matrix: Rotation matrix for the output layer.
        in_gram_eigenvecs: Matrix whose columns are the eigenvecs of the input layer's gram matrix
        in_gram_eigenvals: Diagonal matrix whose diagonal entries are the eigenvalues of the gram
            matrix of the input layer's activations.
        out_acts: Output of the module.

    Returns:
        The interaction matrix for the input layer.
    """

    jac: Float[Tensor, "batch out_hidden in_hidden"] = batched_jacobian(module, in_acts)
    with torch.inference_mode():
        D: Float[Tensor, "in_hidden in_hidden"] = torch.diag(in_gram_eigenvals.sqrt())
        O_dash = torch.einsum("jJ,bji,iI,Ik->bJk", out_rotation_matrix, jac, in_gram_eigenvecs, D)
        out_o_dash = torch.einsum("bj,bjk->bk", out_acts, O_dash)
        M = torch.einsum("bi,bj->ij", out_o_dash, out_o_dash)
    return M


def calc_cap_lambda(
    next_layer_rotation_matrix: Float[Tensor, "out_hidden out_hidden"],
    edge_weights: Float[Tensor, "out_hidden out_hidden in_hidden in_hidden"],
    curr_layer_gram_eigeninfo: EigenInfo,
    interaction_eigenvecs: Float[Tensor, "out_hidden out_hidden"],
) -> Float[Tensor, "out_hidden out_hidden"]:
    """NOTE: UNTESTED STUB FUNCTION: Calculate the cap lambda matrix.

    The cap lambda matrix is given at the top of page 5 in the paper.

    Args:
        next_layer_rotation_matrix: Rotation matrix for the output layer.
        edge_weights: Weight matrix for the edges between the current layer and the next layer.
        curr_layer_gram_eigeninfo: Eigenvalues and eigenvectors of the gram matrix of the current
            layer's activations.
        interaction_eigenvecs: Matrix whose columns are the eigenvectors of the interaction matrix.

    Returns:
        The cap lambda matrix.
    """
    cap_lambda = torch.einsum(
        "ij,jJkK,kl,lm->iJ",
        next_layer_rotation_matrix,
        edge_weights,
        curr_layer_gram_eigeninfo.eigenvecs,
        interaction_eigenvecs,
    )
    return cap_lambda


def calc_interaction_rotation_matrix(
    cap_lambda: Float[Tensor, "out_hidden out_hidden"],
    interaction_eigenvecs: Float[Tensor, "out_hidden out_hidden"],
    curr_layer_gram_eigeninfo: EigenInfo,
) -> Float[Tensor, "out_hidden out_hidden"]:
    """NOTE: UNTESTED STUB FUNCTION: Calculate the interaction rotation matrix.

    The interaction rotation matrix is given by:
        C^l = |\Lambda^l|^{1/2} V^l (D^l^{1/2})^{\+} U^l
    where:
        |\Lambda^l|: (out_hidden, out_hidden) the element-wise absolute value of cap_lambda
        V^l: (out_hidden, out_hidden) matrix with columns as the eigenvectors of the interaction
            matrix
        (D^l^{1/2})^{\+}: (in_hidden, in_hidden) the Moore-Penrose pseudo-inverse of D^l^{1/2}
        U^l: (in_hidden, in_hidden) matrix with columns as the eigenvectors of the input gram matrix

    Args:
        cap_lambda: The Lambda^l matrix.
        interaction_eigenvecs: Matrix whose columns are the eigenvectors of the interaction matrix.
            Denoted as V^l in the equation above.
        curr_layer_gram_eigeninfo: Eigenvalues and eigenvectors of the gram matrix of the current
            layer's activations. Denoted as U^l and D^l in the equation above.

    Returns:
        The interaction rotation matrix.
    """
    cap_lambda_abs_sqrt = torch.abs(cap_lambda).sqrt()
    gram_eigenvals_pinv = torch.diag(curr_layer_gram_eigeninfo.eigenvals.sqrt()).pinverse()
    interaction_rotation_matrix = torch.einsum(
        "ij,jk,kl,lm->im",
        cap_lambda_abs_sqrt,
        interaction_eigenvecs,
        gram_eigenvals_pinv,
        curr_layer_gram_eigeninfo.eigenvecs,
    )
    return interaction_rotation_matrix


def pinv_truncated_diag(x: Float[Tensor, "a b"]) -> Float[Tensor, "a b"]:
    """Calculate the pseudo-inverse of a truncated diagonal matrix.

    A truncated diagonal matrix is a diagonal matrix that isn't necessarily square. The
    pseudo-inverse of a truncated diagonal matrix is the same matrix with the diagonal elements
    replaced by their reciprocal.

    Args:
        x: A truncated diagonal matrix.

    Returns:
        Pseudo-inverse of x.
    """
    # Check that all non-diagonal entries are 0. Use a shortcut of comparing the sum of the
    # diagonal entries to the sum of all entries.
    assert x.sum() == x.diagonal().sum(), "It appears there are non-zero off-diagonal entries."
    res: Float[Tensor, "a b"] = torch.zeros_like(x)
    res.diagonal()[:] = x.diagonal().reciprocal()
    return res
