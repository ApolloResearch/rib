import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Union

import fire
import torch
import yaml
from pydantic import BaseModel, field_validator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rib.data_accumulator import collect_gram_matrices, collect_interaction_edges
from rib.hook_manager import HookedModel
from rib.interaction_algos import calculate_interaction_rotations
from rib.log import logger
from rib.models import MLP
from rib.types import TORCH_DTYPES
from rib.utils import REPO_ROOT, load_config, overwrite_output, set_seed
from experiments.build_interaction_graph import (
    Config,
    load_mlp,
    load_mnist_dataloader,
)


def main(config_path_str: str) -> Optional[dict[str, Any]]:
    """Test for ReLU interactions (separate to main RIB algorithm)."""
