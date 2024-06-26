from pathlib import Path
from typing import Annotated, Any, Literal

import torch
from pydantic import BeforeValidator, PlainSerializer

from rib.utils import to_root_path

StrDtype = Literal["float32", "float64", "bfloat16"]

TORCH_DTYPES: dict[StrDtype, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}


def convert_str_to_torch_dtype(v: Any) -> torch.dtype:
    """Convert dtype from str to a supported torch dtype."""
    if v in TORCH_DTYPES:
        return TORCH_DTYPES[v]
    elif v in TORCH_DTYPES.values():
        return v
    else:
        raise ValueError(f"Invalid dtype: {v}")


def serialize_torch_dtype_to_str(v: torch.dtype) -> str:
    """Convert dtype from torch dtype to str."""
    for k, v2 in TORCH_DTYPES.items():
        if v == v2:
            return k
    raise ValueError(f"Invalid dtype found during serialization: {v}")


TorchDtype = Annotated[
    torch.dtype,
    BeforeValidator(convert_str_to_torch_dtype),
    PlainSerializer(serialize_torch_dtype_to_str),
]

# This is a type for pydantic configs that will convert all relative paths
# to be relative to the ROOT_DIR of rib.
RootPath = Annotated[Path, BeforeValidator(to_root_path), PlainSerializer(lambda x: str(x))]

IntegrationMethod = Literal["trapezoidal", "gauss-legendre", "gradient"]
