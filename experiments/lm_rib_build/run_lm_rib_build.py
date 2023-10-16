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
    python run_lm_rib_build.py <path/to/config.yaml>

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
from typing import Literal, Optional, Union

import fire
import torch
from pydantic import BaseModel, Field, field_validator, model_validator

from rib.data import ModularArithmeticDatasetConfig, WikitextConfig
from rib.data_accumulator import collect_gram_matrices, collect_interaction_edges
from rib.hook_manager import HookedModel
from rib.interaction_algos import calculate_interaction_rotations
from rib.loader import create_data_loader, load_dataset, load_sequential_transformer
from rib.log import logger
from rib.types import TORCH_DTYPES
from rib.utils import eval_model_accuracy, load_config, overwrite_output, set_seed


class Config(BaseModel):
    exp_name: str = Field(..., description="The name of the experiment")
    seed: int = Field(..., description="The random seed value for reproducibility")
    tlens_pretrained: Optional[Literal["gpt2", "pythia-14m"]] = Field(
        None, description="Pretrained transformer lens model."
    )
    tlens_model_path: Optional[Path] = Field(
        None, description="Path to saved transformer lens model."
    )
    node_layers: list[str] = Field(
        ..., description="Names of the modules whose inputs correspond to node layers in the graph."
    )
    logits_node_layer: bool = Field(
        ...,
        description="Whether to build an extra output node layer for the logits.",
    )
    rotate_final_node_layer: bool = Field(
        ...,
        description="Whether to rotate the final node layer to its eigenbasis or not.",
    )
    dataset: Union[ModularArithmeticDatasetConfig, WikitextConfig] = Field(
        ...,
        discriminator="name",
        description="The dataset to use to build the graph.",
    )
    batch_size: int = Field(..., description="The batch size to use when building the graph.")
    truncation_threshold: float = Field(
        ...,
        description="Remove eigenvectors with eigenvalues below this threshold.",
    )
    last_pos_module_type: Optional[Literal["add_resid1", "unembed"]] = Field(
        None,
        description="Module type in which to only output the last position index.",
    )

    n_intervals: int = Field(
        ...,
        description="The number of intervals to use for the integrated gradient approximation.",
    )

    dtype: str = Field(..., description="The dtype to use when building the graph.")

    eps: float = Field(
        1e-5,
        description="The epsilon value to use for numerical stability in layernorm layers.",
    )
    calculate_edges: bool = Field(
        True,
        description="Whether to calculate the edges of the interaction graph.",
    )

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


def main(config_path_str: str):
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = TORCH_DTYPES[config.dtype]

    seq_model, tlens_cfg_dict = load_sequential_transformer(
        node_layers=config.node_layers,
        last_pos_module_type=config.last_pos_module_type,
        tlens_pretrained=config.tlens_pretrained,
        tlens_model_path=config.tlens_model_path,
        eps=config.eps,
        dtype=dtype,
        device=device,
    )
    seq_model.eval()
    seq_model.to(device=torch.device(device), dtype=dtype)
    seq_model.fold_bias()
    hooked_model = HookedModel(seq_model)

    # Importantly, use the same dataset as was used for training
    dataset = load_dataset(
        dataset_config=config.dataset,
        return_set="train",
        tlens_model_path=config.tlens_model_path,
    )

    train_loader = create_data_loader(dataset, shuffle=True, batch_size=config.batch_size)

    # Test model accuracy before graph building, ta be sure
    accuracy = eval_model_accuracy(hooked_model, train_loader, dtype=dtype, device=device)
    logger.info("Model accuracy on dataset: %.2f%%", accuracy * 100)

    # Don't build the graph for the section of the model before the first node layer
    graph_module_names = [f"sections.{sec}" for sec in seq_model.sections if sec != "pre"]

    # Only need gram matrix for logits if we're rotating the final node layer
    collect_output_gram = config.logits_node_layer and config.rotate_final_node_layer

    gram_matrices = collect_gram_matrices(
        hooked_model=hooked_model,
        module_names=graph_module_names,
        data_loader=train_loader,
        dtype=dtype,
        device=device,
        collect_output_gram=collect_output_gram,
        hook_names=config.node_layers,
    )

    Cs, Us = calculate_interaction_rotations(
        gram_matrices=gram_matrices,
        module_names=graph_module_names,
        hooked_model=hooked_model,
        data_loader=train_loader,
        dtype=dtype,
        device=device,
        n_intervals=config.n_intervals,
        logits_node_layer=config.logits_node_layer,
        truncation_threshold=config.truncation_threshold,
        rotate_final_node_layer=config.rotate_final_node_layer,
        hook_names=config.node_layers,
    )

    if not config.calculate_edges:
        logger.info("Skipping edge calculation.")
        E_hats = {}
    else:
        logger.info("Calculating edges.")
        E_hats = collect_interaction_edges(
            Cs=Cs,
            hooked_model=hooked_model,
            n_intervals=config.n_intervals,
            module_names=graph_module_names,
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
        "edges": [(node_layer, E_hats[node_layer].cpu()) for node_layer in E_hats],
        "config": json.loads(config.model_dump_json()),
        "model_config_dict": tlens_cfg_dict,
    }

    # Save the results (which include torch tensors) to file
    torch.save(results, out_interaction_graph_file)
    logger.info("Saved results to %s", out_interaction_graph_file)


if __name__ == "__main__":
    fire.Fire(main)
