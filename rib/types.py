from pathlib import Path
from typing import Annotated, Literal, Union

import torch
from pydantic import BeforeValidator

TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}

DTYPE_STR = Literal["float32", "float64", "bfloat16"]


def to_root_path(path: Union[str, Path]):
    if Path(path).is_absolute():
        return Path(path)
    else:
        ROOT_DIR = Path(__file__).parent.parent
        return Path(ROOT_DIR / path)


RootPath = Annotated[Path, BeforeValidator(to_root_path)]

# %%
