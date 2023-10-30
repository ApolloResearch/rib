"""Contains hook functions whose only purpose is to change the model's activations on forward pass,
not store data."""
from functools import partial
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from jaxtyping import Float, Int
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
    replacement_idxs: Int[Tensor, "d_hidden"],
) -> Union[
        tuple[Float[Tensor, "batch d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden1"],
              Float[Tensor, "batch pos d_hidden2"]],
    ]:
    """Swap each ReLU with another ReLU. Whether the swap is based on clustering or otherwise
    depends on the indices you pass in.

    Hook activation layers.
    """
    assert isinstance(data_key, list) or isinstance(data_key, str), "data_key must be a str or list of strings."

    output_is_tuple: bool = True if isinstance(output, tuple) else False
    outputs = output if output_is_tuple else (output,)
    out_hidden_dims = [x.shape[-1] for x in outputs]

    inputs = torch.cat([x for x in inputs], dim=-1)
    outputs = torch.cat([x for x in outputs], dim=-1) # Concat over hidden dimension

    operator: Float[Tensor, "batch d_hidden_out"] = torch.div(outputs, inputs)
    if operator.dim() == 2:
        batch_size, d_hidden = operator.shape # [256, 101]
    elif operator.dim() == 3:
        batch_size, token_len, d_hidden = operator.shape

    edited_operator = operator.clone()
    # `replacement_idxs` is shorter than `operator` by however large residual stream dim is
    # Trivially extend for residual stream by avoiding replacing these indices
    extended_replacement_idxs = torch.arange(d_hidden)
    extended_replacement_idxs[:len(replacement_idxs)] = replacement_idxs
    operator[..., torch.arange(d_hidden)] = operator[..., extended_replacement_idxs]

    edited_output = operator * inputs

    # Split back into tuple form if the ouput should have been tuple
    if output_is_tuple:
        edited_output = tuple(torch.split(edited_output, out_hidden_dims, dim=-1))

    return edited_output

