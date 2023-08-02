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

import json
from pathlib import Path

import fire
import torch
import yaml
from jaxtyping import Float
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rib.data_accumulator import collect_gram_matrices, collect_M_dash_and_Lambda_dash
from rib.hook_manager import HookedModel
from rib.linalg import eigendecompose, pinv_truncated_diag
from rib.log import logger
from rib.models import MLP
from rib.utils import REPO_ROOT, load_config


class Config(BaseModel):
    exp_name: str
    mlp_path: Path
    batch_size: int
    seed: int
    truncation_threshold: float
    module_names: list[str]


def load_mlp(config_dict: dict, mlp_path: Path) -> MLP:
    mlp = MLP(
        hidden_sizes=config_dict["model"]["hidden_sizes"],
        input_size=784,
        output_size=10,
        activation_fn=config_dict["model"]["activation_fn"],
        bias=config_dict["model"]["bias"],
        fold_bias=config_dict["model"]["fold_bias"],
    )
    mlp.load_state_dict(torch.load(mlp_path))
    return mlp


def load_mnist_dataloader(train: bool = False, batch_size: int = 64) -> DataLoader:
    transform = transforms.ToTensor()
    test_data = datasets.MNIST(
        root=REPO_ROOT / ".data", train=train, download=True, transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return test_loader


def calculate_interaction_rotations(
    gram_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]],
    module_names: list[str],
    hooked_model: HookedModel,
    data_loader: DataLoader,
    device: str,
    truncation_threshold: float = 1e-5,
) -> list[tuple[str, Float[Tensor, "d_hidden d_hidden"]]]:
    """Calculate the interaction rotation matrices (denoted C in the paper) for each node layer.

    Recall that we have one more node layer than module layer, as we have a node layer for the
    output of the final module.

    Note that the variable names refer to the notation used in Algorithm 1 in the paper.

    Args:
        gram_matrices: The gram matrices for each layer, keyed by layer name. Must include
            an `output` key for the output layer.
        module_names: The names of the modules to build the graph from, in order of appearance.
        hooked_model: The hooked model.
        data_loader: The data loader.
        device: The device to use for the calculations.

    Returns:
        A list of (node_layer, rotation_matrix) tuples, one for each node layer in the graph.
    """
    assert "output" in gram_matrices, "Gram matrices must include an `output` key."

    # We start appending Cs from the output layer and work our way backwards (we reverse the list
    # at the end)
    Cs: list[tuple[str, Float[Tensor, "d_hidden d_hidden_trunc"]]] = []
    _, U_output = eigendecompose(gram_matrices["output"])
    Cs.append(("output", U_output))

    for module_name in module_names[::-1]:
        D_dash, U = eigendecompose(gram_matrices[module_name])

        M_dash, Lambda_dash = collect_M_dash_and_Lambda_dash(
            next_layer_C=Cs[-1][1],  # most recently stored interaction matrix
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
        Lambda = torch.einsum("kj,jJ,JK->kK", U_sqrt_D_V.T, Lambda_dash, U_sqrt_D_V)
        # Take the pseudoinverse of the sqrt of D. Can simply take the elementwise inverse
        # of the diagonal elements, since D is diagonal.
        D_sqrt_inv: Float[Tensor, "d_hidden d_hidden_trunc"] = pinv_truncated_diag(D.sqrt())
        C = (module_name, torch.einsum("jJ,Jm,mn,nk->jk", U, D_sqrt_inv, V, Lambda.abs().sqrt()))
        Cs.append(C)

    Cs.reverse()
    return Cs


def main(config_path_str: str) -> None:
    """Implement the main algorithm and store the graph to disk."""
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)
    torch.manual_seed(config.seed)

    with open(config.mlp_path.parent / "config.yaml", "r") as f:
        model_config_dict = yaml.safe_load(f)

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_interaction_matrix_file = out_dir / f"{config.exp_name}_interaction_matrix.pt"
    if out_interaction_matrix_file.exists():
        logger.error("Output file %s already exists. Exiting.", out_interaction_matrix_file)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = load_mlp(model_config_dict, config.mlp_path)
    mlp.eval()
    mlp.to(device)
    hooked_mlp = HookedModel(mlp)

    test_loader = load_mnist_dataloader(train=False, batch_size=config.batch_size)

    gram_matrices = collect_gram_matrices(
        hooked_model=hooked_mlp,
        module_names=config.module_names,
        data_loader=test_loader,
        device=device,
        collect_output_gram=True,
    )

    Cs = calculate_interaction_rotations(
        gram_matrices=gram_matrices,
        module_names=config.module_names,
        hooked_model=hooked_mlp,
        data_loader=test_loader,
        device=device,
        truncation_threshold=config.truncation_threshold,
    )

    results = {
        "exp_name": config.exp_name,
        "interaction_rotations": Cs,
        "config": json.loads(config.model_dump_json()),
        "model_config_dict": model_config_dict,
    }

    # Save the results (which include torch tensors) to file
    torch.save(results, out_interaction_matrix_file)
    logger.info("Saved results to %s", out_interaction_matrix_file)


if __name__ == "__main__":
    fire.Fire(main)
