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
from pathlib import Path
from typing import Any, Literal, Optional

import fire
import yaml
from pydantic import BaseModel, Field, model_validator
from transformer_lens import HookedTransformer

from rib.models import SequentialTransformer, SequentialTransformerConfig
from rib.utils import load_config, set_seed


class Config(BaseModel):
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

    @model_validator(mode="after")
    def verify_model_info(self) -> "Config":
        if sum(1 for val in [self.tlens_pretrained, self.tlens_model_path] if val is not None) != 1:
            raise ValueError(
                "Exactly one of [tlens_pretrained, tlens_model_path] must be specified"
            )
        return self


def map_tlens_to_seq(seq_model: SequentialTransformer, tlens_model: HookedTransformer) -> None:
    """Map the weights from a transformer lens model to a sequential transformer model."""
    raise NotImplementedError("Haven't yet implemented loading a saved model")


def main(config_path_str: str) -> Optional[dict[str, Any]]:
    """Build the interaction graph and store it on disk."""
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)
    set_seed(config.seed)

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
    seq_model = SequentialTransformer(seq_cfg, config.node_layers)

    map_tlens_to_seq(seq_model, tlens_model)
    return None


if __name__ == "__main__":
    fire.Fire(main)
