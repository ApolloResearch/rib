from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from rib.log import logger

ACTIVATION_MAP = {
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
}


def save_model(config_dict: dict[str, Any], save_dir: Path, model: nn.Module, epoch: int) -> None:
    # If the save_dir doesn't exist, create it and save the config
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
        logger.info("Saving config to %s", save_dir)
        with open(save_dir / "config.yaml", "w") as f:
            yaml.dump(config_dict, f)
    logger.info("Saving model to %s", save_dir)
    torch.save(model.state_dict(), save_dir / f"model_epoch_{epoch + 1}.pt")


def get_model_attr(model: torch.nn.Module, attr_path: str) -> torch.nn.Module:
    """Retrieve a nested attribute of a PyTorch module by a string of attribute names.

    Navigates through the model's structure following the provided attribute path.
    Each attribute name in the path is separated by a period ('.').

    Args:
        model (torch.nn.Module): The PyTorch model.
        attr_path (str): A string representing the path to the attribute.

    Returns:
        torch.nn.Module: The attribute (which may be a module or other object) at the specified path in the model.

    Example:
        >>> mlp = MLP(...)
        >>> linear_layer = get_model_attr(mlp, "layers.linear_1")
        >>> print(linear_layer)
        Linear(in_features=100, out_features=10, bias=False)
    """
    attr_names = attr_path.split(".")
    attr = model
    for name in attr_names:
        attr = getattr(attr, name)
    return attr
