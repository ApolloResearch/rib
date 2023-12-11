"""Calculate the interaction graph of an MLP trained on MNIST or CIFAR.

The full algorithm is Algorithm 1 of https://www.overleaf.com/project/6437d0bde0eaf2e8c8ac3649
The process is as follows:
    1. Load an MLP trained on MNIST and a test set of MNIST images.
    2. Collect gram matrices at each node layer.
    3. Calculate the interaction rotation matrices (labelled C in the paper) for each node layer. A
        node layer is positioned at the input of each module_name specified in the config, as well
        as at the output of the final module.
    4. Calculate the edges of the interaction graph between each node layer.

Usage:
    python run_mlp_rib_build.py <path/to/yaml_config_file>

"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional, Union

import fire
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import DataLoader

from rib.data import VisionDatasetConfig
from rib.data_accumulator import (
    collect_dataset_means,
    collect_gram_matrices,
    collect_interaction_edges,
)
from rib.hook_manager import HookedModel
from rib.interaction_algos import calculate_interaction_rotations
from rib.loader import load_dataset, load_mlp
from rib.log import logger
from rib.models.mlp import MLPConfig
from rib.types import TORCH_DTYPES, RibBuildResults, RootPath, StrDtype
from rib.utils import check_outfile_overwrite, load_config, set_seed


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    exp_name: str
    mlp_path: RootPath
    batch_size: int
    seed: Optional[int] = 0
    truncation_threshold: float  # Remove eigenvectors with eigenvalues below this threshold.
    rotate_final_node_layer: bool  # Whether to rotate the output layer to its eigenbasis.
    n_intervals: int  # The number of intervals to use for integrated gradients.
    dtype: StrDtype  # Data type of all tensors (except those overriden in certain functions).
    node_layers: list[str]
    basis_formula: Literal["(1-alpha)^2", "(1-0)*alpha", "svd"] = Field(
        "(1-0)*alpha",
        description="The integrated gradient formula to use to calculate the basis.",
    )
    edge_formula: Literal["functional", "squared"] = Field(
        "functional",
        description="The attribution method to use to calculate the edges.",
    )
    out_dir: Optional[RootPath] = Field(
        Path(__file__).parent / "out",
        description="Directory for the output files. Defaults to `./out/`. If None, no output "
        "is written. If a relative path, it is relative to the root of the rib repo.",
    )
    dataset: VisionDatasetConfig = VisionDatasetConfig()
    centre: bool = Field(
        False,
        description="Whether to centre the activations before performing rib. Currently only"
        "supported for basis_formula='svd', which gives the 'pca' basis.",
    )


def main(config_path_or_obj: Union[str, Config], force: bool = False) -> RibBuildResults:
    """Implement the main algorithm and store the graph to disk."""
    config = load_config(config_path_or_obj, config_model=Config)
    set_seed(config.seed)

    with open(config.mlp_path.parent / "config.yaml", "r") as f:
        model_config_dict = yaml.safe_load(f)

    if config.out_dir is not None:
        config.out_dir.mkdir(parents=True, exist_ok=True)
        out_file = config.out_dir / f"{config.exp_name}_rib_graph.pt"
        if not check_outfile_overwrite(out_file, force):
            raise FileExistsError("Not overwriting output file")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = TORCH_DTYPES[config.dtype]
    mlp_config = MLPConfig(**model_config_dict["model"])
    mlp = load_mlp(mlp_config, config.mlp_path, fold_bias=True, device=device)
    assert mlp.has_folded_bias, "MLP must have folded bias to run RIB"

    all_possible_node_layers = [f"layers.{i}" for i in range(len(mlp.layers))] + ["output"]
    assert "|".join(config.node_layers) in "|".join(all_possible_node_layers), (
        f"config.node_layers must be a subsequence of {all_possible_node_layers} for a plain MLP, "
        f"otherwise our algorithm will be invalid because we require that the output of a "
        f"node layer is the input to the next node layer."
    )

    mlp.eval()
    mlp.to(device=torch.device(device), dtype=TORCH_DTYPES[config.dtype])
    hooked_mlp = HookedModel(mlp)

    dataset = load_dataset(config.dataset, "train")
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    non_output_node_layers = [layer for layer in config.node_layers if layer != "output"]
    # Only need gram matrix for logits if we're rotating the final node layer
    collect_output_gram = config.node_layers[-1] == "output" and config.rotate_final_node_layer

    if config.basis_formula == "pca":
        logger.info("Collecting dataset means")
        means, bias_positions = collect_dataset_means(
            hooked_model=hooked_mlp,
            module_names=non_output_node_layers,
            data_loader=train_loader,
            dtype=dtype,
            device=device,
            collect_output_dataset_means=collect_output_gram,
        )
    else:
        means, bias_positions = None, None

    gram_matrices = collect_gram_matrices(
        hooked_model=hooked_mlp,
        module_names=non_output_node_layers,
        data_loader=train_loader,
        dtype=dtype,
        device=device,
        collect_output_gram=collect_output_gram,
        means=means,
        bias_positions=bias_positions,
    )

    Cs, Us = calculate_interaction_rotations(
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
        basis_formula=config.basis_formula,
        means=means,
        bias_positions=bias_positions,
    )

    E_hats = collect_interaction_edges(
        Cs=Cs,
        hooked_model=hooked_mlp,
        n_intervals=config.n_intervals,
        section_names=non_output_node_layers,
        data_loader=train_loader,
        dtype=dtype,
        device=device,
        edge_formula=config.edge_formula,
    )

    # Move interaction matrices to the cpu and store in dict
    interaction_rotations = []
    for C_info in Cs:
        info_dict = asdict(C_info)
        info_dict["C"] = info_dict["C"].cpu() if info_dict["C"] is not None else None
        info_dict["C_pinv"] = info_dict["C_pinv"].cpu() if info_dict["C_pinv"] is not None else None
        interaction_rotations.append(info_dict)

    eigenvectors = [asdict(U_info) for U_info in Us]

    results: RibBuildResults = {
        "exp_name": config.exp_name,
        "gram_matrices": {k: v.cpu() for k, v in gram_matrices.items()},
        "interaction_rotations": interaction_rotations,
        "eigenvectors": eigenvectors,
        "edges": [(module, E_hats[module].cpu()) for module in E_hats],
        "config": json.loads(config.model_dump_json()),
        "model_config_dict": model_config_dict,
    }

    # Save the results (which include torch tensors) to file
    if config.out_dir is not None:
        torch.save(results, out_file)
        logger.info("Saved results to %s", out_file)
    return results


if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: "")
