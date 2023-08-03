"""Calculate the interaction graph of an MLP trained on MNIST.

The full algorithm is Algorithm 1 of https://www.overleaf.com/project/6437d0bde0eaf2e8c8ac3649
The process is as follows:
    1. Load an MLP trained on MNIST and a test set of MNIST images.
    2. Collect gram matrices at each node layer.
    3. Calculate the interaction rotation matrices (labelled C in the paper) for each node layer. A
        node layer is positioned at the input of each module_name specified in the config, as well
        as at the output of the final module.
    4. Calculate the edges of the interaction graph between each node layer.

We often use variable names from the paper in this script. For example, we refer to the interaction
rotation matrix as C, the gram matrix as G, and the edge weights as E.

This algorithm makes use of pytorch hooks to extract the relevant data from the modules during
forward passes. The hooks are all forward hooks. When setting a hook to a module, the inputs to
that module will correspond to $f^l(X)$ in the paper, and the outputs $f^{l+1}(X)$. This means that
we must treat the output layer differently, as there is no module we can set a hook point to which
will have inputs corresponding to the model output. To cater for this, we add an extra hook to the
final `module_name` and collect the gram matrices from that module's output, as opposed to its
inputs as is done for the other modules.

Beware, when calculating the jacobian, if torch.inference_mode() is set, the jacobian will output
zeros. This is because torch.inference_mode() disables autograd, which is required for calculating
the jacobian. Setting requires_grad=True on the inputs and outputs of the jacobian calculation
DOES NOT fix this issue.

Usage:
    python build_interaction_graph.py <path/to/yaml_config_file>

"""

from dataclasses import dataclass
from typing import Optional

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader

from rib.data_accumulator import collect_M_dash_and_Lambda_dash
from rib.hook_manager import HookedModel
from rib.linalg import eigendecompose, pinv_truncated_diag


@dataclass
class InteractionRotation:
    """Dataclass storing the interaction rotation matrix and its inverse for a node layer."""

    node_layer_name: str
    C: Float[Tensor, "d_hidden d_hidden_trunc"]
    # pseudoinverse of C, not needed for the output node layer
    C_pinv: Optional[Float[Tensor, "d_hidden_trunc d_hidden"]] = None


def calculate_interaction_rotations(
    gram_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]],
    module_names: list[str],
    hooked_model: HookedModel,
    data_loader: DataLoader,
    device: str,
    truncation_threshold: float = 1e-5,
) -> list[InteractionRotation]:
    """Calculate the interaction rotation matrices (denoted C) and their inverses.

    Recall that we have one more node layer than module layer, as we have a node layer for the
    output of the final module.

    Note that the variable names refer to the notation used in Algorithm 1 in the paper.

    We collect the interaction rotation matrices from the output layer backwards, as we need the
    next layer's rotation to compute the current layer's rotation. We reverse the list at the end.

    Args:
        gram_matrices: The gram matrices for each layer, keyed by layer name. Must include
            an `output` key for the output layer.
        module_names: The names of the modules to build the graph from, in order of appearance.
        hooked_model: The hooked model.
        data_loader: The data loader.
        device: The device to use for the calculations.

    Returns:
        A list of objects contain interaction rotation matrices and their pseudoinverses, ordered
        by node layer appearance in model.
    """
    assert "output" in gram_matrices, "Gram matrices must include an `output` key."

    # We start appending Cs from the output layer and work our way backwards
    Cs: list[InteractionRotation] = []
    # Cs: list[tuple[str, Float[Tensor, "d_hidden d_hidden_trunc"]]] = []
    _, U_output = eigendecompose(gram_matrices["output"])
    Cs.append(InteractionRotation(node_layer_name="output", C=U_output))

    for module_name in module_names[::-1]:
        D_dash, U = eigendecompose(gram_matrices[module_name])

        M_dash, Lambda_dash = collect_M_dash_and_Lambda_dash(
            C_out=Cs[-1].C,  # most recently stored interaction matrix
            hooked_model=hooked_model,
            data_loader=data_loader,
            module_name=module_name,
            device=device,
        )
        # Create sqaure matrix from eigenvalues then remove cols with vals < truncation_threshold
        n_small_eigenvals: int = int(torch.sum(D_dash < truncation_threshold).item())
        D: Float[Tensor, "d_hidden d_hidden_trunc"] = (
            torch.diag(D_dash)[:, :-n_small_eigenvals]
            if n_small_eigenvals > 0
            else torch.diag(D_dash)
        )
        U_sqrt_D: Float[Tensor, "d_hidden d_hidden_trunc"] = U @ D.sqrt()
        M: Float[Tensor, "d_hidden_trunc d_hidden_trunc"] = torch.einsum(
            "kj,jJ,JK->kK", U_sqrt_D.T, M_dash, U_sqrt_D
        )
        _, V = eigendecompose(M)  # V has size (d_hidden_trunc, d_hidden_trunc)
        # Multiply U_sqrt_D with V, corresponding to $U D^{1/2} V$ in the paper.
        U_sqrt_D_V: Float[Tensor, "d_hidden d_hidden_trunc"] = U_sqrt_D @ V
        Lambda_raw: Float[Tensor, "d_hidden_trunc d_hidden_trunc"] = torch.einsum(
            "kj,jJ,JK->kK", U_sqrt_D_V.T, Lambda_dash, U_sqrt_D_V
        )
        # Zero out non-diagonal elements of Lambda_raw
        Lambda = torch.diag(Lambda_raw.diag())
        # Take the pseudoinverse of the sqrt of D. Can simply take the elementwise inverse
        # of the diagonal elements, since D is diagonal.
        D_sqrt_inv: Float[Tensor, "d_hidden d_hidden_trunc"] = pinv_truncated_diag(D.sqrt())
        U_D_sqrt_inv_V: Float[Tensor, "d_hidden d_hidden_trunc"] = U @ D_sqrt_inv @ V
        Lambda_abs_sqrt = Lambda.abs().sqrt()
        C: Float[Tensor, "d_hidden d_hidden_trunc"] = U_D_sqrt_inv_V @ Lambda_abs_sqrt
        C_pinv: Float[Tensor, "d_hidden_trunc d_hidden"] = (
            pinv_truncated_diag(Lambda_abs_sqrt) @ U_D_sqrt_inv_V.T
        )
        Cs.append(InteractionRotation(node_layer_name=module_name, C=C, C_pinv=C_pinv))

    return Cs[::-1]
