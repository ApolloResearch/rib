from typing import List, Literal, Optional

import torch
import torch.nn as nn
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from rib.models import MLP, MLPConfig
from rib.types import TorchDtype


class ModularMLPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    config_type: Literal["ModularMLP"] = "ModularMLP"
    n_hidden_layers: int = Field(
        4,
        description="The number of hidden layers [input, hidden, ..., hidden, output]",
    )
    width: int = Field(
        4,
        description="The width of each layer.",
    )
    first_block_width: Optional[int] = Field(
        None,
        description="Width of the first block. If None, defaults to width // 2.",
        validate_default=True,
    )
    weight_variances: List[float] = Field(
        [1.0, 1.0],
        description="Variance of the two blocks of the block diagonal matrix.",
    )
    weight_equal_columns: bool = Field(
        False,
        description="Whether to make the columns of each block equal.",
    )
    bias: float = Field(
        0.0,
        description="Value of the bias term to add to the linear transformations. Default is 0.0.",
    )
    activation_fn: str = Field(
        "relu",
        description="The activation function to use for all but the last layer. Default is "
        '"relu".',
    )
    dtype: TorchDtype = Field(torch.float32, description="The dtype to initialize the model with.")
    seed: Optional[int] = Field(
        0,
        description="Seed for generating the weights. If None, no seed is set.",
    )


def generate_block_diagonal_weights(
    dtype: TorchDtype,
    total_width: int,
    first_block_width: Optional[int],
    block_variances: List[float],
    equal_columns: bool,
    seed: Optional[int] = 0,
) -> Float[Tensor, "width width"]:
    """Generate a random block diagonal matrix.

    Args:
        dtype: The dtype of the weights
        total_width: The width of the matrix
        first_block_width: The width of the first block. If None, defaults to total_width // 2
        block_variances: The variances of the two blocks
        equal_columns: Whether to make the columns of each block equal

    Returns:
        A random block diagonal matrix
    """
    if first_block_width is None:
        first_block_width = total_width // 2

    assert total_width > first_block_width, "First block width must be smaller than total width"
    assert len(block_variances) == 2, "Only two blocks supported"

    second_block_width = total_width - first_block_width

    if seed is not None:
        torch.manual_seed(seed)

    if equal_columns:
        # Duplicate the same columns in each block
        first_block = (
            block_variances[0]
            * torch.randn(1, first_block_width, dtype=dtype).repeat(first_block_width, 1).T
        )
        second_block = (
            block_variances[1]
            * torch.randn(1, second_block_width, dtype=dtype).repeat(second_block_width, 1).T
        )
    else:
        # Normal random weights
        first_block = block_variances[0] * torch.randn(
            first_block_width, first_block_width, dtype=dtype
        )
        second_block = block_variances[1] * torch.randn(
            second_block_width, second_block_width, dtype=dtype
        )

    return torch.block_diag(first_block, second_block)


def create_modular_mlp(modular_mlp_config: ModularMLPConfig) -> MLP:
    """Generate a block diagonal MLP.

    Args:
        modular_mlp_config: Config class for the block diagonal MLP
        seed: Seed for generating the weights

    Returns:
        A block diagonal MLP
    """
    mlp_config = MLPConfig(
        hidden_sizes=[modular_mlp_config.width] * modular_mlp_config.n_hidden_layers,
        input_size=modular_mlp_config.width,
        output_size=modular_mlp_config.width,
        activation_fn=modular_mlp_config.activation_fn,
        dtype=modular_mlp_config.dtype,
        fold_bias=False,  # Don't fold bias here, it should be done after initialisation
        bias=True,
    )
    mlp = MLP(mlp_config)

    # Hardcode weights and biases
    assert len(mlp.layers) == modular_mlp_config.n_hidden_layers + 1
    for layer in mlp.layers:
        layer.W = nn.Parameter(
            generate_block_diagonal_weights(
                dtype=modular_mlp_config.dtype,
                total_width=modular_mlp_config.width,
                first_block_width=modular_mlp_config.first_block_width,
                block_variances=modular_mlp_config.weight_variances,
                equal_columns=modular_mlp_config.weight_equal_columns,
                seed=modular_mlp_config.seed,
            )
        )
        layer.b = nn.Parameter(
            torch.full(
                (modular_mlp_config.width,),
                fill_value=modular_mlp_config.bias,
                dtype=modular_mlp_config.dtype,
            )
        )

    return mlp
