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

    Each attribute name in the path is separated by a period ('.').

    Since models often have lists of modules, the attribute path can also include an index.

    Args:
        model (torch.nn.Module): The PyTorch model.
        attr_path (str): A string representing the path to the attribute.

    Returns:
        torch.nn.Module: The attribute of the model.

    Example:
        >>> mlp = MLP([5], input_size=2, output_size=3)
        >>> mlp
        MLP(
            (layers): ModuleList(
                (0): Layer(
                    (linear): Linear(in_features=2, out_features=5, bias=True)
                    (activation): ReLU()
                )
                (1): Layer(
                    (linear): Linear(in_features=5, out_features=3, bias=True)
                )
            )
        )
        - get_model_attr(model, "layers") -> ModuleList(...)
        - get_model_attr(model, "layers.0") -> Layer(...)
        - get_model_attr(model, "layers.0.activation") -> ReLU()
        - get_model_attr(model, "layers.1.linear") -> LinearFoldedBias(...)
    """
    attr_names = attr_path.split(".")
    attr = model

    for name in attr_names:
        try:
            if isinstance(attr, torch.nn.ModuleList) and name.isdigit():
                attr = attr[int(name)]
            else:
                attr = getattr(attr, name)
        except AttributeError:
            logger.error(f"Attribute '{name}' not found in the path '{attr_path}'.")
            raise
    return attr
