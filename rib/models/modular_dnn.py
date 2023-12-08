from typing import Callable, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import Dataset

from rib.models import MLP, MLPConfig
from rib.types import TORCH_DTYPES, StrDtype


def generate_weights(
    total_width: int,
    first_block_width: Optional[int],
    block_variances: List[float],
    equal_columns: bool,
    dtype=torch.float32,
    seed: Optional[int] = None,
):
    """Generate a random block diagonal matrix

    Args:
        total_width: width of the matrix
        first_block_width: width of the first block
        block_variances: variances of the two blocks
        equal_columns: whether to duplicate the same columns in each block. This makes the outputs
            of the matrix extremely correlated withing each block.
        dtype: data type of the matrix

    Returns:
        A: a random block diagonal matrix
    """
    first_block_width = first_block_width or total_width // 2
    assert total_width > first_block_width, "First block width must be smaller than total width"
    assert len(block_variances) == 2, "Only two blocks supported"

    if seed is not None:
        torch.manual_seed(seed)

    A = torch.zeros((total_width, total_width), dtype=dtype)

    if equal_columns:
        # Duplicate the same columns in each block
        A[:first_block_width, :first_block_width] = (
            block_variances[0]
            * torch.randn(1, first_block_width, dtype=dtype).repeat(first_block_width, 1).T
        )
        second_block_width = total_width - first_block_width
        A[first_block_width:, first_block_width:] = (
            block_variances[1]
            * torch.randn(1, second_block_width, dtype=dtype).repeat(second_block_width, 1).T
        )
    else:
        # Normal random weights
        A[:first_block_width, :first_block_width] = block_variances[0] * torch.randn(
            first_block_width, first_block_width, dtype=dtype
        )
        second_block_width = total_width - first_block_width
        A[first_block_width:, first_block_width:] = block_variances[1] * torch.randn(
            second_block_width, second_block_width, dtype=dtype
        )

    assert A.dtype == dtype, "dtype mismatch happened somewhere"

    return A


class ModularDNNConfig(BaseModel):
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


class ModularDNN(MLP):
    def __init__(
        self,
        dnn_config: ModularDNNConfig,
        dtype: StrDtype = "float64",
        seed: Optional[int] = None,
    ):
        """Generate a block diagonal DNN

        Args:
            hidden_layers: number of layers. Total number of layers is layers + 2 (input and output).
                node_layers will be layer.0, [hidden layers i.e. layer.1 to layer.hidden_layers], output
            width: width of each layer
            first_block_width: width of the first block
            dtype: data type
            bias: bias of each layer
            activation_fn: activation function
            block_variances: variances of the two blocks
            weight_assignment_function: function to assign weights to each layer
        """
        self.cfg = dnn_config
        # Initialize MLP
        self.mlp_config = MLPConfig(
            hidden_sizes=[self.cfg.width] * self.cfg.n_hidden_layers,
            input_size=self.cfg.width,
            output_size=self.cfg.width,
            activation_fn=self.cfg.activation_fn,
            dtype=dtype,
            # Don't fold bias yet, fold later
            fold_bias=False,
            bias=True,
        )
        super(ModularDNN, self).__init__(config=self.mlp_config)

        # Hardcode weights and biases
        assert len(self.layers) == self.cfg.n_hidden_layers + 1
        for layer in self.layers:
            layer.W = nn.Parameter(
                generate_weights(
                    self.cfg.width,
                    self.cfg.first_block_width,
                    self.cfg.weight_variances,
                    self.cfg.weight_equal_columns,
                    dtype=TORCH_DTYPES[dtype],
                    seed=seed,
                )
            )
            layer.b = nn.Parameter(
                self.cfg.bias * torch.ones(self.cfg.width, dtype=TORCH_DTYPES[dtype])
            )

        # Now fold biases
        self.fold_bias()


def generate_data(
    size: int,
    length: int,
    first_block_length: Optional[int],
    data_variances: List[float],
    data_perfect_correlation: bool,
    dtype: torch.dtype,
    seed: Optional[int] = None,
):
    first_block_length = first_block_length or length // 2
    second_block_length = length - first_block_length
    data = torch.empty((size, length), dtype=dtype)

    if seed is not None:
        torch.manual_seed(seed)

    if not data_perfect_correlation:
        data[:, 0:first_block_length] = data_variances[0] * torch.randn(
            size, first_block_length, dtype=dtype
        )
        data[:, first_block_length:] = data_variances[1] * torch.randn(
            size, second_block_length, dtype=dtype
        )
    else:
        data[:, 0:first_block_length] = data_variances[0] * torch.randn(
            size, 1, dtype=dtype
        ).repeat(1, first_block_length)
        data[:, first_block_length:] = data_variances[1] * torch.randn(size, 1, dtype=dtype).repeat(
            1, second_block_length
        )

    assert data.dtype == dtype, "dtype mismatch happened somewhere"

    return data


class BlockVectorDatasetConfig(BaseModel):
    size: int = Field(
        1000,
        description="Number of samples in the dataset.",
    )
    length: int = Field(
        4,
        description="Length of each vector.",
    )
    first_block_length: Optional[int] = Field(
        None,
        description="Length of the first block. If None, defaults to length // 2.",
    )
    data_variances: List[float] = Field(
        [1.0, 1.0],
        description="Variance of the two blocks of the vectors.",
    )
    data_perfect_correlation: bool = Field(
        False,
        description="Whether to make the data within each block perfectly correlated.",
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.first_block_length is None:
            self.first_block_length = self.length // 2


class BlockVectorDataset(Dataset):
    def __init__(
        self,
        data_config: BlockVectorDatasetConfig,
        dtype: StrDtype = "float64",
        seed: Optional[int] = None,
    ):
        """Generate a dataset of vectors with two blocks of variance"""
        self.cfg = data_config
        self.data = generate_data(
            size=self.cfg.size,
            length=self.cfg.length,
            first_block_length=self.cfg.first_block_length,
            data_variances=self.cfg.data_variances,
            data_perfect_correlation=self.cfg.data_perfect_correlation,
            dtype=TORCH_DTYPES[dtype],
            seed=seed,
        )
        # Not needed, just here for Dataset class
        self.labels = torch.nan * torch.ones(self.cfg.size)

    def __len__(self):
        return self.cfg.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
