"""Calculate the interaction graph of an MLP trained on MNIST.

The full algorithm is Algorithm 1 of https://www.overleaf.com/project/6437d0bde0eaf2e8c8ac3649
The process is as follows:
    1. Load an MLP trained on MNIST and a test set of MNIST images.
    2. Collect gram matrices at each node layer.
    3. Calculate the interaction rotation matrices (labelled C in the paper) for each node layer. A
        node layer is positioned at the input of each module_name specified in the config, as well
        as at the output of the final module.
    4. Calculate the edges of the interaction graph between each node layer.

Usage:
    python run_mnist_rib_build.py <path/to/yaml_config_file>

"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union

import fire
import torch
import yaml
from pydantic import BaseModel, Field, field_validator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rib.data_accumulator import collect_gram_matrices, collect_interaction_edges
from rib.hook_manager import HookedModel
from rib.interaction_algos import calculate_interaction_rotations
from rib.loader import load_mlp
from rib.log import logger
from rib.models import MLP
from rib.types import TORCH_DTYPES
from rib.utils import REPO_ROOT, check_outfile_overwrite, load_config, set_seed


class Config(BaseModel):
    exp_name: str
    force_overwrite_output: Optional[bool] = Field(
        False, description="Don't ask before overwriting the output file."
    )
    mlp_path: Path
    batch_size: int
    seed: int
    truncation_threshold: float  # Remove eigenvectors with eigenvalues below this threshold.
    rotate_final_node_layer: bool  # Whether to rotate the output layer to its eigenbasis.
    n_intervals: int  # The number of intervals to use for integrated gradients.
    dtype: str  # Data type of all tensors (except those overriden in certain functions).
    node_layers: list[str]

    @field_validator("dtype")
    def dtype_validator(cls, v):
        assert v in TORCH_DTYPES, f"dtype must be one of {TORCH_DTYPES}"
        return v


def load_mnist_dataloader(train: bool = False, batch_size: int = 64) -> DataLoader:
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(
        root=REPO_ROOT / ".data", train=train, download=True, transform=transform
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


def main(config_path_or_obj: Union[str, Config], force: bool = False) -> None:
    """Implement the main algorithm and store the graph to disk."""
    config = load_config(config_path_or_obj, config_model=Config)
    set_seed(config.seed)

    with open(config.mlp_path.parent / "config.yaml", "r") as f:
        model_config_dict = yaml.safe_load(f)

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{config.exp_name}_rib_graph.pt"
    if not check_outfile_overwrite(out_file, config.force_overwrite_output or force, logger=logger):
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = TORCH_DTYPES[config.dtype]
    mlp = load_mlp(model_config_dict, config.mlp_path, device=device)
    assert mlp.has_folded_bias
    mlp.eval()
    mlp.to(device=torch.device(device), dtype=TORCH_DTYPES[config.dtype])
    hooked_mlp = HookedModel(mlp)

    train_loader = load_mnist_dataloader(train=True, batch_size=config.batch_size)

    non_output_node_layers = [layer for layer in config.node_layers if layer != "output"]
    # Only need gram matrix for logits if we're rotating the final node layer
    collect_output_gram = config.node_layers[-1] == "output" and config.rotate_final_node_layer

    gram_matrices = collect_gram_matrices(
        hooked_model=hooked_mlp,
        module_names=non_output_node_layers,
        data_loader=train_loader,
        dtype=dtype,
        device=device,
        collect_output_gram=collect_output_gram,
    )

    Cs, Us, _, _, _, _, _, = calculate_interaction_rotations(
        gram_matrices=gram_matrices,
        section_names=non_output_node_layers,
        node_layers=config.node_layers,
        hooked_model=hooked_mlp,
        data_loader=train_loader,
        dtype=dtype,
        device=device,
        n_intervals=config.n_intervals,
        truncation_threshold=config.truncation_threshold,
        rotate_final_node_layer=config.rotate_final_node_layer,
    )

    E_hats = collect_interaction_edges(
        Cs=Cs,
        hooked_model=hooked_mlp,
        n_intervals=config.n_intervals,
        section_names=config.node_layers,
        data_loader=train_loader,
        dtype=dtype,
        device=device,
    )

    # Move interaction matrices to the cpu and store in dict
    interaction_rotations = []
    for C_info in Cs:
        info_dict = asdict(C_info)
        info_dict["C"] = info_dict["C"].cpu() if info_dict["C"] is not None else None
        info_dict["C_pinv"] = info_dict["C_pinv"].cpu() if info_dict["C_pinv"] is not None else None
        interaction_rotations.append(info_dict)

    eigenvectors = [asdict(U_info) for U_info in Us]

    results = {
        "exp_name": config.exp_name,
        "gram_matrices": {k: v.cpu() for k, v in gram_matrices.items()},
        "interaction_rotations": interaction_rotations,
        "eigenvectors": eigenvectors,
        "edges": [(module, E_hats[module].cpu()) for module in E_hats],
        "config": json.loads(config.model_dump_json()),
        "model_config_dict": model_config_dict,
    }

    # Save the results (which include torch tensors) to file
    torch.save(results, out_file)
    logger.info("Saved results to %s", out_file)


if __name__ == "__main__":
    fire.Fire(main)