"""This script builds a RIB graph for a language model.
We build the graph using a SequentialTransformer model, with weights ported over from a
transformer-lens model.

Steps to build the graph:
1. Load a model from transformerlens (either from_pretrained or via ModelConfig).
2. Fold in the biases into the weights.
3. Convert the model to a SequentialTransformer model, which has nn.Modules corresponding to each
    node layer.
5. Collect the gram matrices at each node layer.
6. Calculate the interaction basis matrices (labelled C in the paper) for each node layer, starting
    from the final node layer and working backwards.
7. Calculate the edges of the interaction graph between each node layer.

Usage:
    python lm_build_rib_graph.py <path/to/config.yaml>

The config.yaml should contain the `node_layers` field. This describes the sections of the
graph that will be built: A graph layer will be built on the inputs to each specified node layer,
as well as the output of the final node layer. For example, if `node_layers` is ["attn.0",
"mlp_act.0"], this script will build the following graph layers:
- One on the inputs to the "attn.0" node layer. This will include the residual stream concatenated
    with the output of ln1.0.
- One on the input to "mlp_act.0". This will include the residual stream concatenated with the
    output of "mlp_in.0".
- One on the output of "mlp_act.0". This will include the residual stream concatenated with the
    output of "mlp_act.0".
"""
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Optional

import fire
import torch
from pydantic import BaseModel, Field, field_validator, model_validator

from rib.data_accumulator import collect_gram_matrices, collect_interaction_edges
from rib.hook_manager import HookedModel
from rib.interaction_algos import calculate_interaction_rotations
from rib.loader import (
    create_modular_arithmetic_data_loader,
    load_sequential_transformer,
)
from rib.log import logger
from rib.types import TORCH_DTYPES
from rib.utils import eval_model_accuracy, load_config, overwrite_output, set_seed


class Config(BaseModel):
    exp_name: str = Field(..., description="The name of the experiment")
    seed: int = Field(..., description="The random seed value for reproducibility")
    tlens_pretrained: Optional[Literal["gpt2"]] = Field(
        None, description="Pretrained transformer lens model."
    )
    tlens_model_path: Optional[Path] = Field(
        None, description="Path to saved transformer lens model."
    )
    node_layers: list[str] = Field(
        ..., description="Names of the node layers to build the graph with."
    )
    dataset: Literal["modular_arithmetic", "wikitext"] = Field(
        ...,
        description="The dataset to use to build the graph.",
    )
    batch_size: int = Field(..., description="The batch size to use when building the graph.")
    truncation_threshold: float = Field(
        ...,
        description="Remove eigenvectors with eigenvalues below this threshold.",
    )
    rotate_output: bool = Field(
        ...,
        description="Whether to rotate the output layer to its eigenbasis.",
    )
    last_pos_module_type: Optional[Literal["add_resid1", "unembed"]] = Field(
        None,
        description="Module type in which to only output the last position index.",
    )

    dtype: str = Field(..., description="The dtype to use when building the graph.")

    @field_validator("dtype")
    def dtype_validator(cls, v):
        assert v in TORCH_DTYPES, f"dtype must be one of {TORCH_DTYPES}"
        return v

    @model_validator(mode="after")
    def verify_model_info(self) -> "Config":
        if sum(1 for val in [self.tlens_pretrained, self.tlens_model_path] if val is not None) != 1:
            raise ValueError(
                "Exactly one of [tlens_pretrained, tlens_model_path] must be specified"
            )
        return self


def main(config_path_str: str) -> Optional[dict[str, Any]]:
    """Build the interaction graph and store it on disk."""
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)
    set_seed(config.seed)

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_interaction_graph_file = out_dir / f"{config.exp_name}_interaction_graph.pt"
    if out_interaction_graph_file.exists() and not overwrite_output(out_interaction_graph_file):
        logger.info("Exiting.")
        return None

    assert (
        config.tlens_pretrained is None and config.tlens_model_path is not None
    ), "Currently can't build graphs for pretrained models due to memory limits."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = TORCH_DTYPES[config.dtype]

    seq_model, tlens_cfg_dict = load_sequential_transformer(
        node_layers=config.node_layers,
        last_pos_module_type=config.last_pos_module_type,
        tlens_pretrained=config.tlens_pretrained,
        tlens_model_path=config.tlens_model_path,
        dtype=dtype,
        device=device,
    )
    seq_model.eval()
    seq_model.to(device=torch.device(device), dtype=dtype)
    seq_model.fold_bias()
    hooked_model = HookedModel(seq_model)

    assert config.dataset == "modular_arithmetic", "Currently only supports modular arithmetic."

    # Importantly, use the same dataset as was used for training
    train_loader = create_modular_arithmetic_data_loader(
        shuffle=True,
        return_set="train",
        tlens_model_path=config.tlens_model_path,
        batch_size=config.batch_size,
    )

    # Test model accuracy before graph building, ta be sure
    accuracy = eval_model_accuracy(hooked_model, train_loader, dtype=dtype, device=device)
    logger.info("Model accuracy on dataset: %.2f%%", accuracy * 100)

    # Don't build the graph for the section of the model before the first node layer
    graph_module_names = [f"sections.{sec}" for sec in seq_model.sections if sec != "pre"]

    gram_matrices = collect_gram_matrices(
        hooked_model=hooked_model,
        module_names=graph_module_names,
        data_loader=train_loader,
        dtype=dtype,
        device=device,
        collect_output_gram=True,
        hook_names=config.node_layers,
    )

    Cs, Us = calculate_interaction_rotations(
        gram_matrices=gram_matrices,
        module_names=graph_module_names,
        hooked_model=hooked_model,
        data_loader=train_loader,
        dtype=dtype,
        device=device,
        truncation_threshold=config.truncation_threshold,
        rotate_output=config.rotate_output,
        hook_names=config.node_layers,
    )

    E_hats = collect_interaction_edges(
        Cs=Cs,
        hooked_model=hooked_model,
        module_names=graph_module_names,
        data_loader=train_loader,
        dtype=dtype,
        device=device,
    )

    # Move interaction matrices to the cpu and store in dict
    interaction_rotations = []
    for C_info in Cs:
        info_dict = asdict(C_info)
        info_dict["C"] = info_dict["C"].cpu()
        if info_dict["C_pinv"] is not None:
            info_dict["C_pinv"] = info_dict["C_pinv"].cpu()
        else:
            info_dict["C_pinv"] = None
        interaction_rotations.append(info_dict)

    eigenvectors = [asdict(U_info) for U_info in Us]

    results = {
        "exp_name": config.exp_name,
        "gram_matrices": {k: v.cpu() for k, v in gram_matrices.items()},
        "interaction_rotations": interaction_rotations,
        "eigenvectors": eigenvectors,
        "edges": [(node_layer, E_hats[node_layer].cpu()) for node_layer in config.node_layers],
        "config": json.loads(config.model_dump_json()),
        "model_config_dict": tlens_cfg_dict,
    }

    # Save the results (which include torch tensors) to file
    torch.save(results, out_interaction_graph_file)
    logger.info("Saved results to %s", out_interaction_graph_file)
    return results


if __name__ == "__main__":
    fire.Fire(main)
