from typing import List, Optional

import torch
import torch.nn as nn
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from torch import Tensor

from rib.models import MLP, MLPConfig
from rib.types import TorchDtype


class ModularMLPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
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

    @field_validator("first_block_width", mode="after")
    @classmethod
    def set_first_block_width(cls, v: Optional[int], info: ValidationInfo) -> int:
        if v is None:
            return info.data["width"] // 2
        return v

    @property
    def mlp_config(self) -> MLPConfig:
        return MLPConfig(
            hidden_sizes=[self.width] * self.n_hidden_layers,
            input_size=self.width,
            output_size=self.width,
            activation_fn=self.activation_fn,
            dtype=self.dtype,
            fold_bias=False,  # Don't fold bias here, it should be done after initialisation
            bias=True,
        )


class ModularMLP(MLP):
    def __init__(
        self,
        mlp_config: ModularMLPConfig,
        seed: Optional[int] = None,
    ):
        """Generate a block diagonal MLP

        Args:
            mlp_config: Config class for the block diagonal MLP
            seed: Seed for generating the weights
        """
        self.cfg = mlp_config
        super(ModularMLP, self).__init__(config=self.cfg.mlp_config)

        # Hardcode weights and biases
        assert len(self.layers) == self.cfg.n_hidden_layers + 1
        for layer in self.layers:
            layer.W = nn.Parameter(self.generate_weights(seed=seed))
            layer.b = nn.Parameter(
                torch.full((self.cfg.width,), fill_value=self.cfg.bias, dtype=self.cfg.dtype)
            )

    def generate_weights(self, seed: Optional[int] = None) -> Float[Tensor, "width width"]:
        """Generate a random block diagonal matrix

        Args:
            seed: random seed

        Returns:
            A random block diagonal matrix
        """
        dtype = self.cfg.dtype
        total_width = self.cfg.width
        first_block_width = self.cfg.first_block_width or total_width // 2
        block_variances = self.cfg.weight_variances
        equal_columns = self.cfg.weight_equal_columns

        assert total_width > first_block_width, "First block width must be smaller than total width"
        assert len(block_variances) == 2, "Only two blocks supported"

        if seed is not None:
            torch.manual_seed(seed)

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
