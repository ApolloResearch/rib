from typing import Union

import torch

ACTIVATION = Union[torch.nn.ReLU, torch.nn.GELU, torch.nn.Tanh, torch.nn.Sigmoid]
ACTIVATION_MAP = {
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
}
