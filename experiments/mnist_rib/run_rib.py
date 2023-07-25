"""Calculate the interaction basis for each layer of an MLP trained on MNIST.

The process is as follows:
    1. Load an MLP trained on MNIST and a test set of MNIST images.
    2. Calculate gram matrices at the given hook points.
    3. Eigendecompose the gram matrices. For the final layer, the interaction basis is equal to the
        eigenbasis of the gram matrix.
    4. For the other layers, we calculate the interaction matrix using the algorithm described in
       `rib/linalg.calc_interaction_matrix`.

Beware, when calculating the jacobian, if torch.inference_mode() is set, the jacobian will output
zeros. This is because torch.inference_mode() disables autograd, which is required for calculating
the jacobian. Setting requires_grad=True on the inputs and outputs of the jacobian calculation
DOES NOT fix this issue.

Usage:
    python run_rib.py <path/to/yaml_config_file>

"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import fire
import torch
import yaml
from jaxtyping import Float
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing_extensions import Literal

from rib.hook_manager import Hook, HookedModel
from rib.linalg import EigenInfo, eigendecompose
from rib.log import logger
from rib.models import MLP
from rib.utils import REPO_ROOT, load_config, run_dataset_through_model


class HookConfig(BaseModel):
    hook_name: str
    module_name: str  # The module to hook into
    hook_type: Literal["forward", "pre_forward"]
    layer_size: int  # The size of the data at the hook point


class Config(BaseModel):
    mlp_name: str
    mlp_path: Path
    batch_size: int
    seed: int
    hook_configs: list[HookConfig]


@dataclass
class InteractionInfo:
    """Information about the interaction matrix of a layer."""

    hook_name: str
    interaction_matrix: Float[Tensor, "d_hidden d_hidden"]


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


def collect_layer_interaction_matrix(
    hooked_mlp: HookedModel,
    data_key: str,
    curr_layer_name: str,
    next_layer_module_name: str,
    next_layer_interaction_matrix: Float[Tensor, "d_hidden d_hidden"],
    curr_layer_eigeninfo: EigenInfo,
    data_loader: DataLoader,
    device: str,
) -> None:
    """Calculate the interaction matrix for a layer.

    Adds a forward hook to the next_layer module to calculate the jacobian and other components
    needed to accumulate the interaction matrix.

    The interaction matrix will be stored in hooked_mlp.hooked_data[curr_layer_name][data_key].

    Args:
        hooked_mlp: The hooked model.
        data_key: The key to use to store the data in the hook.
        curr_layer_name: The name of the current layer.
        next_layer_module_name: The next layer module name we apply the forward hook to.
        next_layer_interaction_matrix: The interaction matrix for the next layer.
        curr_layer_eigeninfo: The eigen information for the current layer.
        data_loader: The pytorch data loader.
        device: The device to run the model on.
    """
    interaction_hook = Hook(
        name=curr_layer_name,
        data_key=data_key,
        fn_name="interaction_forward_hook_fn",
        module_name=next_layer_module_name,
        fn_kwargs={
            "input_eigen_info": curr_layer_eigeninfo,
            "output_interaction_matrix": next_layer_interaction_matrix,
        },
    )
    run_dataset_through_model(hooked_mlp, data_loader, hooks=[interaction_hook], device=device)


def collect_interaction_matrices(
    hooked_mlp: HookedModel,
    hook_configs: list[HookConfig],
    eigen_infos: list[EigenInfo],
    data_loader: DataLoader,
    device: str,
) -> list[InteractionInfo]:
    """Collect the interaction matrices for each layer specified in hook_configs.

    Importantly, we assume that no computation happens between the hooked modules. I.e.
    the input to the 2nd layer is the output of the 1st layer, and so on. This assumption makes
    it much easier to calculate the interaction matrix for each layer, and also allows for creating
    connected graphs with all of the interaction bases.

    This means that the only hook which can be a pre_forward hook is the first hook. All other hooks
    must be forward hooks.

    Args:
        hooked_mlp: The hooked model.
        hook_configs: The configs for the hook points.
        eigen_infos: The eigen information for each layer.
        data_loader: The pytorch data loader.
        device: The device to run the model on.

    Returns:
        A list of objects containing the interaction info (hook name and matrix) for each layer.
    """

    assert all(
        cfg.hook_type == "forward" for cfg in hook_configs[1:]
    ), "All hooks except the first hook must be forward hooks. "
    interaction_hooks: list[Hook] = []
    for hook_config in hook_configs:
        assert hook_config.hook_type in ["forward", "pre_forward"]
        interaction_hook_fn_name = f"gram_{hook_config.hook_type}_hook_fn"
        interaction_hooks.append(
            Hook(
                name=hook_config.hook_name,
                data_key="gram",
                fn_name=interaction_hook_fn_name,
                module_name=hook_config.module_name,
            )
        )

    # Key for storing the data in the hook
    data_key = "interaction"

    interaction_infos: list[InteractionInfo] = []
    # Iterate through the configs in backwards order
    for i in range(len(hook_configs) - 1, -1, -1):
        if i == len(hook_configs) - 1:
            # The interaction matrix for the final layer is equal to the eigenbasis.
            interaction_matrix = eigen_infos[i].eigenvecs
        else:
            collect_layer_interaction_matrix(
                hooked_mlp=hooked_mlp,
                data_key=data_key,
                curr_layer_name=hook_configs[i].hook_name,
                next_layer_module_name=hook_configs[i + 1].module_name,
                next_layer_interaction_matrix=interaction_infos[-1].interaction_matrix,
                curr_layer_eigeninfo=eigen_infos[i],
                data_loader=data_loader,
                device=device,
            )
            interaction_matrix = hooked_mlp.hooked_data[hook_configs[i].hook_name][data_key]
        interaction_infos.append(
            InteractionInfo(
                hook_name=hook_configs[i].hook_name, interaction_matrix=interaction_matrix
            )
        )
    return interaction_infos


def collect_gram_matrices(
    hooked_mlp: HookedModel,
    hook_configs: list[HookConfig],
    data_loader: DataLoader,
    device: str,
) -> None:
    """Calculate gram matrices for each hook config and store it in `hooked_mlp.hooked_data`.

    Args:
        hooked_mlp: The hooked model.
        hook_configs: The configs for the hook points.
        data_loader: The pytorch data loader.
        device: The device to run the model on.
    """
    gram_hooks: list[Hook] = []
    for hook_config in hook_configs:
        assert hook_config.hook_type in ["forward", "pre_forward"]
        gram_hook_fn_name = f"gram_{hook_config.hook_type}_hook_fn"
        gram_hooks.append(
            Hook(
                name=hook_config.hook_name,
                data_key="gram",
                fn_name=gram_hook_fn_name,
                module_name=hook_config.module_name,
            )
        )

    run_dataset_through_model(hooked_mlp, data_loader, gram_hooks, device=device)
    len_dataset = len(data_loader.dataset)  # type: ignore

    for hook_config in hook_configs:
        # Scale the gram matrix by the number of samples in the dataset.
        hooked_mlp.hooked_data[hook_config.hook_name]["gram"] = (
            hooked_mlp.hooked_data[hook_config.hook_name]["gram"] / len_dataset
        )


def run_interaction_algorithm(
    model_config_dict: dict, mlp_path: Path, hook_configs: list[HookConfig], batch_size: int
) -> list[InteractionInfo]:
    """Calculate the interaction basis for each layer of an MLP trained on MNIST.

    First, we collect and run an eigendecomposition algorithm on the gram matrices for each
    layer specified in the hook_configs.
    Then, we use this eigen information, along with raw activations, to calculate the interaction
    matrix for each layer.

    Args:
        model_config_dict: The config dictionary used for training the mlp.
        mlp_path: The path to the saved mlp.
        hook_configs: Information about the hook points.
        batch_size: The batch size to use for the data loader.

    Returns:
        A list of objects containing the interaction matrix for each hook point.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = load_mlp(model_config_dict, mlp_path)
    mlp.eval()
    mlp.to(device)
    hooked_mlp = HookedModel(mlp)

    test_loader = load_mnist_dataloader(train=False, batch_size=batch_size)

    collect_gram_matrices(hooked_mlp, hook_configs, test_loader, device)

    eigen_infos: list[EigenInfo] = []
    for hook_config in hook_configs:
        eigenvals, eigenvecs = eigendecompose(hooked_mlp.hooked_data[hook_config.hook_name]["gram"])
        eigen_infos.append(
            EigenInfo(hook_name=hook_config.hook_name, eigenvals=eigenvals, eigenvecs=eigenvecs)
        )

    interaction_info = collect_interaction_matrices(
        hooked_mlp=hooked_mlp,
        hook_configs=hook_configs,
        eigen_infos=eigen_infos,
        data_loader=test_loader,
        device=device,
    )
    return interaction_info


def main(config_path_str: str) -> None:
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

    interaction_infos = run_interaction_algorithm(
        model_config_dict=model_config_dict,
        mlp_path=config.mlp_path,
        hook_configs=config.hook_configs,
        batch_size=config.batch_size,
    )
    results = {
        "mlp_name": config.mlp_name,
        "interaction_infos": [asdict(interaction_info) for interaction_info in interaction_infos],
        "config": json.loads(config.model_dump_json()),
        "model_config_dict": model_config_dict,
    }

    # Save the results (which include torch tensors) to file
    torch.save(results, out_interaction_matrix_file)
    logger.info("Saved results to %s", out_interaction_matrix_file)


if __name__ == "__main__":
    fire.Fire(main)
