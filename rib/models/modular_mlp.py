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
    mixing_layers: List[int] = Field(
        [],
        description="List of layer indices with off-diagonal weights. Empty by default. Starts at zero.",
    )
    weight_variances: List[float] = Field(
        [1.0, 1.0, 1.0, 1.0],
        description="Variances of the two on-diagonal blocks and two off diagonal the block diagonal matrix. Ordered [first_diagonal, second_diagonal, first_off_diagonal, second_off_diagonal]",
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


def generate_block_diagonal_weights(
    dtype: TorchDtype,
    total_width: int,
    first_block_width: Optional[int],
    block_variances: List[float],
    equal_columns: bool,
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
    assert (
        len(block_variances) == 4
    ), "Only matrices with two on-diagonal and two off-diagonal blocks supported"

    second_block_width = total_width - first_block_width

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


def generate_block_weights(
    dtype: TorchDtype,
    total_width: int,
    first_diagonal_block_width: Optional[int],
    block_variances: List[float],
    equal_columns: bool,
) -> Float[Tensor, "width width"]:
    """Generate a random matrix consisting of two diagonal and two off-diagonal blocks with
    specified variances for each block.

    Args:
        dtype: The dtype of the weights
        total_width: The width of the matrix
        first_diagonal_block_width: The width of the first block. If None, defaults to total_width // 2
        block_variances: The variances of the four blocks
        equal_columns: Whether to make the columns of all block equal

    Returns:
        A random block matrix
    """
    if first_diagonal_block_width is None:
        first_diagonal_block_width = total_width // 2

    assert (
        total_width > first_diagonal_block_width
    ), "First block width must be smaller than total width"
    assert len(block_variances) == 4, "Only two diagonal and two off-diagonal blocks supported"

    second_diagonal_block_width = total_width - first_diagonal_block_width

    if equal_columns:
        # Duplicate the same columns in each block
        first_diagonal_block = (
            block_variances[0]
            * torch.randn(1, first_diagonal_block_width, dtype=dtype)
            .repeat(first_diagonal_block_width, 1)
            .T
        )
        second_diagonal_block = (
            block_variances[1]
            * torch.randn(1, second_diagonal_block_width, dtype=dtype)
            .repeat(second_diagonal_block_width, 1)
            .T
        )
        first_off_diagonal_block = (
            block_variances[2]
            * torch.randn(1, first_diagonal_block_width, dtype=dtype)
            .repeat(second_diagonal_block_width, 1)
            .T
        )
        second_off_diagonal_block = (
            block_variances[3]
            * torch.randn(1, second_diagonal_block_width, dtype=dtype)
            .repeat(first_diagonal_block_width, 1)
            .T
        )
    else:
        # Normal random weights
        first_diagonal_block = block_variances[0] * torch.randn(
            first_diagonal_block_width, first_diagonal_block_width, dtype=dtype
        )
        second_diagonal_block = block_variances[1] * torch.randn(
            second_diagonal_block_width, second_diagonal_block_width, dtype=dtype
        )
        first_off_diagonal_block = block_variances[2] * torch.randn(
            first_diagonal_block_width, second_diagonal_block_width, dtype=dtype
        )
        second_off_diagonal_block = block_variances[3] * torch.randn(
            second_diagonal_block_width, first_diagonal_block_width, dtype=dtype
        )

    # Horizontally concatenate the first row of blocks
    top_row = torch.cat([first_diagonal_block, first_off_diagonal_block], dim=1)

    # Horizontally concatenate the second row of blocks
    bottom_row = torch.cat([second_off_diagonal_block, second_diagonal_block], dim=1)

    # Vertically concatenate the top and bottom rows to form the final matrix
    final_matrix = torch.cat([top_row, bottom_row], dim=0)
    assert final_matrix.shape == (
        total_width,
        total_width,
    ), "final_matrix shape must be (total_width, total_width)"
    return final_matrix


def create_modular_mlp(modular_mlp_config: ModularMLPConfig, seed: Optional[int] = None) -> MLP:
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

    if seed is not None:
        torch.manual_seed(seed)

    # Hardcode weights and biases
    assert (
        len(mlp.layers) == modular_mlp_config.n_hidden_layers + 1
    ), "Total number of network layers must match n_hidden_layers + 1"
    for l in range(len(mlp.layers)):
        layer = mlp.layers[l]
        if l in modular_mlp_config.mixing_layers:
            layer.W = nn.Parameter(
                generate_block_weights(
                    dtype=modular_mlp_config.dtype,
                    total_width=modular_mlp_config.width,
                    first_diagonal_block_width=modular_mlp_config.first_block_width,
                    block_variances=modular_mlp_config.weight_variances,
                    equal_columns=modular_mlp_config.weight_equal_columns,
                )
            )
        else:
            layer.W = nn.Parameter(
                generate_block_diagonal_weights(
                    dtype=modular_mlp_config.dtype,
                    total_width=modular_mlp_config.width,
                    first_block_width=modular_mlp_config.first_block_width,
                    block_variances=modular_mlp_config.weight_variances,
                    equal_columns=modular_mlp_config.weight_equal_columns,
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
