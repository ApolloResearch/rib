from pathlib import Path
from typing import Literal, Optional

import fire
import torch
from pydantic import BaseModel, Field, model_validator
from transformer_lens import HookedTransformer, HookedTransformerConfig

from rib.models import SequentialTransformer, SequentialTransformerConfig
from rib.utils import load_config, set_seed

TLENS_PRETRAINED = Literal["gpt2"]


class Config(BaseModel):
    seed: int = Field(..., description="The random seed value for reproducibility")
    tlens_pretrained: TLENS_PRETRAINED = Field(
        None, description="Pretrained transformer lens model"
    )
    tlens_model_path: Optional[str] = Field(
        None, description="Path to saved transformer lens model"
    )
    model: Optional[SequentialTransformerConfig] = Field(
        None, description="Variable names match HookedTransformerConfig names"
    )
    node_layers: list[str] = Field(
        ..., description="Names of the node layers to build the graph with"
    )

    @model_validator(mode="after")
    def verify_model_info(self) -> None:
        if (
            sum(
                1
                for val in [self.tlens_pretrained, self.tlens_model_path, self.model]
                if val is not None
            )
            != 1
        ):
            raise ValueError(
                "Exactly one of [tlens_pretrained, tlens_model_path, model] must be specified"
            )
        return self


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)
    set_seed(config.seed)

    if config.tlens_pretrained is not None:
        tlens_model = HookedTransformer.from_pretrained(config.tlens_pretrained)
        seq_cfg = SequentialTransformerConfig(**tlens_model.cfg.to_dict())
    elif config.model is not None:
        tlens_cfg = HookedTransformerConfig.from_dict(config.model.model_dump())
        tlens_model = HookedTransformer(tlens_cfg)
    elif config.tlens_model_path is not None:
        raise NotImplementedError("Haven't yet implemented loading a saved model")

    print(seq_cfg)
    seq_model = SequentialTransformer(seq_cfg, config.node_layers)
    input_ids = torch.randint(0, seq_model.cfg.d_vocab, size=(2, seq_model.cfg.n_ctx))
    output = seq_model(input_ids)
    print(seq_model)


if __name__ == "__main__":
    fire.Fire(main)
