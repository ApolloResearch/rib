from typing import Literal

import torch

TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}

DATASET_TYPES = Literal["modular_arithmetic", "wikitext"]
