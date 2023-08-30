import warnings
from pathlib import Path
from typing import Any, TypeVar

import torch
import yaml
from torch import nn

from rib.log import logger

T = TypeVar("T")


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


def create_list_partitions(in_list: list[T], sub_list: list[T]) -> list[list[T]]:
    """Create partitions of a list based on a sub-list of matching values

    Args:
        in_list: The list to partition.
        sub_list: The sub-list of values to partition by.

    Returns:
        A list of lists, where each sub-list is a partition of the input list.

    Example:
        >>> all_layers = ['embed', 'pos_embed', 'add_embed', 'ln1.0', 'attn.0', 'add_resid1.0']
        >>> node_layers = ['ln1.0', 'add_resid1.0']
        >>> create_list_partitions(all_layers, node_layers)
        [['embed', 'pos_embed', 'add_embed'], ['ln1.0', 'attn.0'], ['add_resid1.0']]
    """
    indices: list[int] = []
    for entry in sub_list:
        assert entry in in_list, f"Entry '{entry}' not found in the input list."
        indices.append(in_list.index(entry))

    partitions: list[list[T]] = []
    for i, j in zip([0] + indices, indices + [None]):
        sub_list = in_list[i:j]
        if sub_list:
            partitions.append(sub_list)
    return partitions


def map_state_dict(tlens_state_dict: dict, seq_state_dict: dict) -> dict:
    """Maps the state dict from a transformer lens model to a sequential transformer model.

    Args:
        tlens_state_dict: The state dict from the transformer lens model
        seq_state_dict: The state dict from the sequential transformer model

    Returns:
        The mapped state dict
    """
    assert set(tlens_state_dict.keys()) == set(
        seq_state_dict.keys()
    ), "State dict keys do not match"

    for name, param in tlens_state_dict.items():
        if name in seq_state_dict:
            seq_state_dict[name].copy_(param)
            assert torch.allclose(
                tlens_state_dict[name], seq_state_dict[name]
            ), f"Parameter {name} not copied correctly"
        else:
            warnings.warn(f"Parameter {name} not found in sequential transformer model")

    return seq_state_dict
