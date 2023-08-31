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
"""
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Optional

import fire
import torch
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from rib.data import ModularArithmeticDataset
from rib.data_accumulator import collect_gram_matrices, collect_interaction_edges
from rib.hook_manager import HookedModel
from rib.interaction_algos import calculate_interaction_rotations
from rib.log import logger
from rib.models import SequentialTransformer, SequentialTransformerConfig
from rib.models.sequential_transformer.converter import convert_tlens_weights
from rib.tlens_mapper import model_fold_bias
from rib.types import TORCH_DTYPES
from rib.utils import load_config, overwrite_output, set_seed


class Config(BaseModel):
    exp_name: str = Field(..., description="The name of the experiment")
    seed: int = Field(..., description="The random seed value for reproducibility")
    tlens_pretrained: Optional[Literal["gpt2"]] = Field(
        None, description="Pretrained transformer lens model"
    )
    tlens_model_path: Optional[Path] = Field(
        None, description="Path to saved transformer lens model"
    )
    node_layers: list[str] = Field(
        ..., description="Names of the node layers to build the graph with"
    )
    dataset: Literal["modular_arithmetic"] = Field(
        ...,
        description="The dataset to use to build the graph. Currently only supports modular arithmetic",
    )
    batch_size: int = Field(..., description="The batch size to use when building the graph")
    truncation_threshold: float = Field(
        ...,
        description="Remove eigenvectors with eigenvalues below this threshold.",
    )
    rotate_output: bool = Field(
        ...,
        description="Whether to rotate the output layer to its eigenbasis.",
    )
    dtype: str = Field(..., description="The dtype to use when building the graph")

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


def load_sequential_transformer(config: Config) -> tuple[SequentialTransformer, dict]:
    """Load a SequentialTransformer model from a pretrained transformerlens model.

    Requires config to contain a pretrained model name or a path to a transformerlens model.

    First loads a HookedTransformer model, then uses its config to create an instance of
    SequentialTransformerConfig, which is then used to create a SequentialTransformer.

    Args:
        config (Config): The config, containing either `tlens_pretrained` or `tlens_model_path`.

    Returns:
        - SequentialTransformer: The SequentialTransformer model.
        - dict: The config used in the transformerlens model.
    """

    if config.tlens_pretrained is not None:
        tlens_model = HookedTransformer.from_pretrained(config.tlens_pretrained)
        # Create a SequentialTransformerConfig from the HookedTransformerConfig
        tlens_cfg_dict = tlens_model.cfg.to_dict()
    elif config.tlens_model_path is not None:
        with open(config.tlens_model_path.parent / "config.yaml", "r") as f:
            # The config specified in the YAML file used to train the tlens model
            provided_tlens_cfg_dict = yaml.safe_load(f)["model"]
        tlens_model = HookedTransformer(provided_tlens_cfg_dict)
        # The entire tlens config (including default values)
        tlens_cfg_dict = tlens_model.cfg.to_dict()

    seq_cfg = SequentialTransformerConfig(**tlens_cfg_dict)
    # Set the dtype to the one specified in the config for this script (as opposed to the one used
    # to train the tlens model)
    seq_cfg.dtype = TORCH_DTYPES[config.dtype]
    seq_model = SequentialTransformer(seq_cfg, config.node_layers)

    # Load the transformer-lens weights into the sequential transformer model
    state_dict = convert_tlens_weights(list(seq_model.state_dict().keys()), tlens_model)
    seq_model.load_state_dict(state_dict)

    return seq_model, tlens_cfg_dict


def create_data_loader(config: Config, train: bool = False) -> DataLoader:
    """Create a DataLoader for the dataset specified in `config.dataset`.

    Args:
        config (Config): The config, containing the dataset name.

    Returns:
        DataLoader: The DataLoader to use for building the graph.
    """
    if config.dataset == "modular_arithmetic":
        # Get the dataset config from our training config
        assert config.tlens_model_path is not None, "tlens_model_path must be specified"
        with open(config.tlens_model_path.parent / "config.yaml", "r") as f:
            # The config specified in the YAML file used to train the tlens model
            train_config = yaml.safe_load(f)["train"]
        test_data = ModularArithmeticDataset(
            train_config["modulus"], train_config["frac_train"], seed=config.seed, train=train
        )
        # Note that the batch size for training typically gives 1 batch per epoch. We use a smaller
        # batch size here, mostly for verifying that our iterative code works.
        test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)
    else:
        raise NotImplementedError(f"Dataset {config.dataset} not implemented")
    return test_loader


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_model, tlens_cfg_dict = load_sequential_transformer(config)
    seq_model.eval()

    seq_model.to(device=torch.device(device), dtype=TORCH_DTYPES[config.dtype])
    hooked_model = HookedModel(seq_model)

    data_loader = create_data_loader(config)

    # Don't build the graph for the section of the model before the first node layer
    graph_module_names = [f"sections.{sec}" for sec in seq_model.sections if sec != "pre"]

    gram_matrices = collect_gram_matrices(
        hooked_model=hooked_model,
        module_names=graph_module_names,
        data_loader=data_loader,
        device=device,
        collect_output_gram=True,
        hook_names=config.node_layers,
    )

    Cs = calculate_interaction_rotations(
        gram_matrices=gram_matrices,
        module_names=graph_module_names,
        hooked_model=hooked_model,
        data_loader=data_loader,
        device=device,
        truncation_threshold=config.truncation_threshold,
        rotate_output=config.rotate_output,
        hook_names=config.node_layers,
    )

    E_hats = collect_interaction_edges(
        Cs=Cs,
        hooked_model=hooked_model,
        module_names=graph_module_names,
        data_loader=data_loader,
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

    results = {
        "exp_name": config.exp_name,
        "gram_matrices": {k: v.cpu() for k, v in gram_matrices.items()},
        "interaction_rotations": interaction_rotations,
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
