"""Contains hook functions whose only purpose is to change the model's activations on forward pass,
not store data."""
from functools import partial
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from rib.linalg import (
    edge_norm,
    integrated_gradient_trapezoidal_jacobian,
    integrated_gradient_trapezoidal_norm,
)
from rib.models.utils import get_model_attr


def relu_swap_forward_hook_fn(
    module: torch.nn.Module,
    inputs: Union[
        tuple[Float[Tensor, "batch d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden1"],
              Float[Tensor, "batch pos d_hidden2"]],
    ],
    output: Union[
        Float[Tensor, "batch d_hidden"],
        Float[Tensor, "batch pos d_hidden"],
        tuple[Float[Tensor, "batch pos d_hidden1"],
              Float[Tensor, "batch pos d_hidden2"]],
    ],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    replacement_idx_list: list[tuple],
) -> None:
    """Hooks activation layers."""
    assert isinstance(data_key, list) or isinstance(data_key, str), "data_key must be a str or list of strings."
    # Code below would be `in_acts = torch.cat(inputs,d dim=-1)` if not detaching
    # Inputs always tuple
    # For this function, which always hooks an activation layer, inputs ARE the preactivations
    inputs = torch.cat([x for x in inputs], dim=-1)

    outputs = output if isinstance(output, tuple) else (output,)
    outputs = torch.cat([x for x in outputs], dim=-1) # Concat over hidden dimension
    operator: Float[Tensor, "batch d_hidden_out"] = torch.div(outputs, inputs)
    if operator.dim() == 2:
        batch_size, d_hidden = operator.shape # [256, 101]
    elif operator.dim() == 3:
        batch_size, token_len, d_hidden_concat = operator.shape

    edited_operator = operator.clone()
    edited_operator[:, torch.arange(d_hidden)] = operator[:, replacement_idx_list]

    edited_output = edited_operator * inputs

    return edited_output
