from pathlib import Path
from typing import Annotated, Any, Literal, Optional, TypedDict, Union

import torch
from pydantic import BeforeValidator

from rib.utils import REPO_ROOT

StrDtype = Literal["float32", "float64", "bfloat16"]

TORCH_DTYPES: dict[StrDtype, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}


def to_root_path(path: Union[str, Path]):
    return Path(path) if Path(path).is_absolute() else Path(REPO_ROOT / path)


# This is a type for pydantic configs that will convert all relative paths
# to be relative to the ROOT_DIR of rib.
RootPath = Annotated[Path, BeforeValidator(to_root_path)]


# We specify the results of a rib build.
# Split into two classes to mark most fields as required but not the timing fields.
class _RibBuildResultsTotal(TypedDict, total=True):
    exp_name: str
    gram_matrices: dict[str, torch.Tensor]
    interaction_rotations: list[dict[str, Any]]  # serialized Interaction Rotation object
    eigenvectors: list[dict[str, Any]]  # serialized Eigenvector object
    edges: list[tuple[str, torch.Tensor]]
    config: dict[str, Any]
    model_config_dict: dict[str, Any]


class RibBuildResults(_RibBuildResultsTotal, total=False):
    calc_C_time: Optional[str]
    calc_edges_time: Optional[str]
    dist_info: dict[str, Any]
