"""Calculate the interaction graph for each layer of an MLP trained on MNIST.

The full algorithm is Algorithm 1 of https://www.overleaf.com/project/6437d0bde0eaf2e8c8ac3649
The process is as follows:
    1. Load an MLP trained on MNIST and a test set of MNIST images.
    2. Collect gram matrices at the given hook points.
    3. Calculate the interaction rotation matrices (labelled C in the paper) for each layer.
    4. Calculate the edges of the interaction graph for each layer.

We often use variable names from the paper in this script. For example, we refer to the interaction
rotation matrix as C, the gram matrix as G, and the edge weights as E.

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
from rib.hook_manager import HookConfig, HookedModel
from rib.linalg import eigendecompose, pinv_truncated_diag
from rib.log import logger
from rib.models import MLP
from rib.utils import REPO_ROOT, load_config


class Config(BaseModel):
    mlp_name: str
    mlp_path: Path
    batch_size: int
    seed: int
    truncation_threshold: float
    hook_configs: list[HookConfig]  # List of hook configs ordered by model appearance


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
    hook_configs: list[HookConfig],
    hooked_model: HookedModel,
    data_loader: DataLoader,
    device: str,
    truncation_threshold: float = 1e-5,
) -> list[tuple[str, Float[Tensor, "d_hidden d_hidden"]]]:
    """Calculate the interaction rotation matrices (denoted C in the paper) for each layer.

    Note that the variable names refer to the notation used in Algorithm 1 in the paper.

    Args:
        gram_matrices: The gram matrices for each layer, keyed by layer name.
        hook_configs: The configs for each hook point.
        hooked_model: The hooked model.
        data_loader: The data loader.
        device: The device to use for the calculations.

    Returns:
        A list of (hook_name, rotation_matrix) tuples, in order of appearance in the model.
    """

    # We start appending Cs from the final layer and work our way backwards (we reverse the list
    # at the end)
    Cs: list[tuple[str, Float[Tensor, "d_hidden d_hidden"]]] = []
    # Cs: dict[int, Float[Tensor, "d_hidden d_hidden"]] = {}
    for layer_idx in range(len(hook_configs) - 1, -1, -1):
        hook_name = hook_configs[layer_idx].hook_name
        D_dash, U = eigendecompose(gram_matrices[hook_name])

        if layer_idx == len(hook_configs) - 1:
            # Final layer
            C = (hook_name, U)
        else:
            M_dash, Lambda_dash = collect_M_dash_and_Lambda_dash(
                next_layer_C=Cs[-1][1],  # most recently stored interaction matrix
                hooked_model=hooked_model,
                data_loader=data_loader,
                hook_config=hook_configs[layer_idx],
                device=device,
            )
            # Create a matrix from the eigenvalues, removing cols with vals < truncation_threshold
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
            # Right multiply U_sqrt_D with V, corresponding to $U D^{1/2} V$ in the paper.
            U_sqrt_D_V: Float[Tensor, "d_hidden d_hidden_trunc"] = U_sqrt_D @ V
            Lambda = torch.einsum("kj,jJ,JK->kK", U_sqrt_D_V.T, Lambda_dash, U_sqrt_D_V)
            # Take the pseudoinverse of the sqrt of D. Can simply take the elementwise inverse
            # of the diagonal elements, since D is diagonal.
            D_sqrt_inv: Float[Tensor, "d_hidden d_hidden_trunc"] = pinv_truncated_diag(D.sqrt())
            C = (hook_name, torch.einsum("jJ,Jm,Mn,nk->jk", U, D_sqrt_inv, V, Lambda.abs().sqrt()))
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
    out_interaction_matrix_file = out_dir / f"{config.mlp_name}_interaction_matrix.pt"
    if out_interaction_matrix_file.exists():
        logger.error("Output file %s already exists. Exiting.", out_interaction_matrix_file)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = load_mlp(model_config_dict, config.mlp_path)
    mlp.eval()
    mlp.to(device)
    hooked_mlp = HookedModel(mlp)

    test_loader = load_mnist_dataloader(train=False, batch_size=config.batch_size)

    gram_matrices = collect_gram_matrices(hooked_mlp, config.hook_configs, test_loader, device)

    Cs = calculate_interaction_rotations(
        gram_matrices=gram_matrices,
        hook_configs=config.hook_configs,
        hooked_model=hooked_mlp,
        data_loader=test_loader,
        device=device,
        truncation_threshold=config.truncation_threshold,
    )

    results = {
        "mlp_name": config.mlp_name,
        "interaction_rotations": Cs,
        # "interaction_rotations": [asdict(rotation_info) for rotation_info in rotation_infos],
        "config": json.loads(config.model_dump_json()),
        "model_config_dict": model_config_dict,
    }

    # Save the results (which include torch tensors) to file
    torch.save(results, out_interaction_matrix_file)
    logger.info("Saved results to %s", out_interaction_matrix_file)


if __name__ == "__main__":
    fire.Fire(main)
