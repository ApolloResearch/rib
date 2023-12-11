from typing import List, Optional

import torch
import torch.nn as nn
from jaxtyping import Float
from pydantic import BaseModel, Field
from torch import Tensor

from rib.models import MLP, MLPConfig
from rib.types import TORCH_DTYPES, StrDtype


class ModularMLPConfig(BaseModel):
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

    def __init__(self, **data):
        super().__init__(**data)
        if self.first_block_width is None:
            self.first_block_width = self.width // 2


class ModularMLP(MLP):
    def __init__(
        self,
        mlp_config: ModularMLPConfig,
        dtype: StrDtype = "float64",
        seed: Optional[int] = None,
    ):
        """Generate a block diagonal MLP

        Args:
            mlp_config: Config class for the block diagonal MLP
            dtype: Data type of the weights
            seed: Seed for generating the weights
        """
        self.cfg = mlp_config
        self.mlp_config = MLPConfig(
            hidden_sizes=[self.cfg.width] * self.cfg.n_hidden_layers,
            input_size=self.cfg.width,
            output_size=self.cfg.width,
            activation_fn=self.cfg.activation_fn,
            dtype=dtype,
            fold_bias=False,  # Don't fold bias here, we'll do it later
            bias=True,
        )
        super(ModularMLP, self).__init__(config=self.mlp_config)

        # Hardcode weights and biases
        assert len(self.layers) == self.cfg.n_hidden_layers + 1
        for layer in self.layers:
            layer.W = nn.Parameter(self.generate_weights(dtype=TORCH_DTYPES[dtype], seed=seed))
            layer.b = nn.Parameter(
                self.cfg.bias * torch.ones(self.cfg.width, dtype=TORCH_DTYPES[dtype])
            )

        self.fold_bias()

    def generate_weights(
        self, dtype=torch.float32, seed: Optional[int] = None
    ) -> Float[Tensor, "width width"]:
        """Generate a random block diagonal matrix

        Args:
            dtype: data type of the matrix.
            seed: random seed

        Returns:
            A random block diagonal matrix
        """
        total_width = self.cfg.width
        first_block_width = self.cfg.first_block_width
        block_variances = self.cfg.weight_variances
        equal_columns = self.cfg.weight_equal_columns

        first_block_width = first_block_width or total_width // 2
        assert total_width > first_block_width, "First block width must be smaller than total width"
        assert len(block_variances) == 2, "Only two blocks supported"

        if seed is not None:
            torch.manual_seed(seed)

        block_matrix = torch.zeros((total_width, total_width), dtype=dtype)

        if equal_columns:
            # Duplicate the same columns in each block
            block_matrix[:first_block_width, :first_block_width] = (
                block_variances[0]
                * torch.randn(1, first_block_width, dtype=dtype).repeat(first_block_width, 1).T
            )
            second_block_width = total_width - first_block_width
            block_matrix[first_block_width:, first_block_width:] = (
                block_variances[1]
                * torch.randn(1, second_block_width, dtype=dtype).repeat(second_block_width, 1).T
            )
        else:
            # Normal random weights
            block_matrix[:first_block_width, :first_block_width] = block_variances[0] * torch.randn(
                first_block_width, first_block_width, dtype=dtype
            )
            second_block_width = total_width - first_block_width
            block_matrix[first_block_width:, first_block_width:] = block_variances[1] * torch.randn(
                second_block_width, second_block_width, dtype=dtype
            )

        return block_matrix
