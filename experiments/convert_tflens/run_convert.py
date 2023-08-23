from typing import Any, Optional, Type, Union

import fire
from pydantic import BaseModel

from rib.models.sequential_transformer import SequentialTransformerConfig


class Config(BaseModel):
    seed: int
    tlens_pretrained: Optional[str]  # Pretrained transformer lens model
    model: SequentialTransformerConfig  # Variable names match HookedTransformerConfig names


def main():
    pass


if __name__ == "__main__":
    fire.Fire(main)
