"""Calculate the interaction graph of an hand-coded modular DNN."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional, Union

import fire
import torch
import yaml
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader

from rib.data import VisionDatasetConfig
from rib.data_accumulator import collect_gram_matrices, collect_interaction_edges
from rib.hook_manager import HookedModel
from rib.interaction_algos import calculate_interaction_rotations
from rib.log import logger
from rib.models.mlp import MLPConfig
from rib.models.modular_dnn import (
    BlockVectorDataset,
    BlockVectorDatasetConfig,
    ModularDNN,
    ModularDNNConfig,
)
from rib.types import TORCH_DTYPES, RibBuildResults, RootPath, StrDtype
from rib.utils import check_outfile_overwrite, load_config, set_seed


class Config(BaseModel):
    exp_name: str
    batch_size: int
    seed: Optional[int] = 0
    truncation_threshold: float  # Remove eigenvectors with eigenvalues below this threshold.
    rotate_final_node_layer: bool  # Whether to rotate the output layer to its eigenbasis.
    n_intervals: int  # The number of intervals to use for integrated gradients.
    dtype: StrDtype = "float64"
    node_layers: list[str]
    basis_formula: Literal["(1-alpha)^2", "(1-0)*alpha", "svd"] = Field(
        "(1-0)*alpha",
        description="The integrated gradient formula to use to calculate the basis. If 'svd', will"
        "use Us as Cs, giving the eigendecomposition of the gram matrix.",
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
    dataset: BlockVectorDatasetConfig = Field(
        BlockVectorDatasetConfig(),
        description="The dataset to use.",
    )
    model: ModularDNNConfig = Field(
        ModularDNNConfig(),
        description="The model to use.",
    )


def main(config_path_or_obj: Union[str, Config], force: bool = False) -> RibBuildResults:
    """Implement the main algorithm and store the graph to disk."""
    config = load_config(config_path_or_obj, config_model=Config)
    model_config = config.model
    dataset_config = config.dataset

    dnn = ModularDNN(
        dnn_config=model_config,
        dtype=config.dtype,
        seed=config.seed,
    )

    dataset = BlockVectorDataset(
        data_config=dataset_config,
        dtype=config.dtype,
        seed=config.seed,
    )

    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = TORCH_DTYPES[config.dtype]

    assert dnn.has_folded_bias, "MLP must have folded bias to run RIB"

    all_possible_node_layers = [f"layers.{i}" for i in range(len(dnn.layers))] + ["output"]
    assert "|".join(config.node_layers) in "|".join(all_possible_node_layers), (
        f"config.node_layers must be a subsequence of {all_possible_node_layers} for a plain MLP, "
        f"otherwise our algorithm will be invalid because we require that the output of a "
        f"node layer is the input to the next node layer."
    )  # TODO Is this related to why things to wrong when we try to rotate the final node layer?

    if config.out_dir is not None:
        config.out_dir.mkdir(parents=True, exist_ok=True)
        out_file = config.out_dir / f"{config.exp_name}_rib_graph.pt"
        if not check_outfile_overwrite(out_file, force):
            raise FileExistsError("Not overwriting output file")

    dnn.eval()
    dnn.to(device=torch.device(device), dtype=TORCH_DTYPES[config.dtype])
    hooked_dnn = HookedModel(dnn)

    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    non_output_node_layers = [layer for layer in config.node_layers if layer != "output"]
    # Only need gram matrix for logits if we're rotating the final node layer
    collect_output_gram = config.node_layers[-1] == "output" and config.rotate_final_node_layer

    gram_matrices = collect_gram_matrices(
        hooked_model=hooked_dnn,
        module_names=non_output_node_layers,
        data_loader=train_loader,
        dtype=dtype,
        device=device,
        collect_output_gram=collect_output_gram,
    )

    Cs, Us = calculate_interaction_rotations(
        gram_matrices=gram_matrices,
        section_names=non_output_node_layers,
        node_layers=config.node_layers,
        hooked_model=hooked_dnn,
        data_loader=train_loader,
        dtype=dtype,
        device=device,
        n_intervals=config.n_intervals,
        truncation_threshold=config.truncation_threshold,
        rotate_final_node_layer=config.rotate_final_node_layer,
        basis_formula=config.basis_formula,
    )

    E_hats = collect_interaction_edges(
        Cs=Cs,
        hooked_model=hooked_dnn,
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
        "model_config_dict": {},
    }

    # Save the results (which include torch tensors) to file
    if config.out_dir is not None:
        torch.save(results, out_file)
        logger.info("Saved results to %s", out_file)
    return results


if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: "")
