"""Converts a transformer lens model to a sequential transformer model."""
from pathlib import Path
from typing import Literal, Optional

import fire
import torch
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from transformer_lens import HookedTransformer, HookedTransformerConfig

from rib.models import SequentialTransformer, SequentialTransformerConfig
from rib.models.utils import map_state_dict
from rib.types import TORCH_DTYPES
from rib.utils import load_config, set_seed


class Config(BaseModel):
    seed: int = Field(..., description="The random seed value for reproducibility")
    tlens_pretrained: Optional[str] = Field(None, description="Pretrained transformer lens model")
    tlens_model_path: Optional[str] = Field(
        None, description="Path to saved transformer lens model"
    )
    node_layers: list[str] = Field(
        ..., description="Names of the node layers to build the graph with"
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


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)
    set_seed(config.seed)

    if config.tlens_pretrained is not None:
        tlens_model = HookedTransformer.from_pretrained(config.tlens_pretrained)
        tlens_cfg_dict = tlens_model.cfg.to_dict()
    elif config.tlens_model_path is not None:
        with open(
            "empty",
            "r",
        ) as f:
            # The config specified in the YAML file used to train the tlens model
            provided_tlens_cfg_dict = yaml.safe_load(f)["model"]

        tlens_model = HookedTransformer(provided_tlens_cfg_dict)
        tlens_cfg_dict = tlens_model.cfg.to_dict()
        model_state_dict = torch.load(config.tlens_model_path, map_location="cpu")
        tlens_model.load_state_dict(model_state_dict, strict=False)

    seq_cfg = SequentialTransformerConfig(**tlens_cfg_dict)

    seq_cfg.dtype = TORCH_DTYPES[config.dtype]
    seq_model = SequentialTransformer(seq_cfg, config.node_layers)

    mapped_state_dict = map_state_dict(tlens_model.state_dict(), seq_model.state_dict())

    # Test model produces same output  # TODO not implemented
    input_ids = torch.randint(0, seq_model.cfg.d_vocab, size=(2, seq_model.cfg.n_ctx))
    output = seq_model(input_ids)

    # Restructure model # TODO not implemented
    seq_model.toggle_state_dict_renaming(False)  # Disable renaming state_dict
    seq_model.structure_graph()


if __name__ == "__main__":
    fire.Fire(main)
