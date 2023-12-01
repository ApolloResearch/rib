import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import fire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pydantic import BaseModel, field_validator
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from rib.data_accumulator import collect_gram_matrices, collect_interaction_edges
from rib.hook_manager import HookedModel
from rib.interaction_algos import InteractionRotation, calculate_interaction_rotations
from rib.log import logger
from rib.models import MLP, MLPConfig, modular_dnn
from rib.plotting import plot_interaction_graph
from rib.types import TORCH_DTYPES
from rib.utils import REPO_ROOT, check_outfile_overwrite, load_config, set_seed


def random_block_diagonal_matrix(
    total_width, first_block_width, block_variances, dtype=torch.float32
):
    """Generate a random block diagonal matrix

    Args:
        total_width: width of the matrix
        first_block_width: width of the first block
        block_variances: variances of the two blocks
        dtype: data type of the matrix

    Returns:
        A: a random block diagonal matrix
    """
    assert total_width > first_block_width
    A = torch.zeros((total_width, total_width), dtype=dtype)
    A[:first_block_width, :first_block_width] = block_variances[0] * torch.randn(
        first_block_width, first_block_width, dtype=dtype
    )
    second_block_width = total_width - first_block_width
    A[first_block_width:, first_block_width:] = block_variances[1] * torch.randn(
        second_block_width, second_block_width, dtype=dtype
    )
    return A


def random_block_diagonal_matrix_equal_columns(
    total_width, first_block_width, block_variances, dtype=torch.float32
):
    """Generate a random block diagonal matrix with equal columns

    This makes the outputs of the matrix extremely correlated withing each block.

    Args:
        total_width: width of the matrix
        first_block_width: width of the first block
        block_variances: variances of the two blocks
        dtype: data type of the matrix

    Returns:
        A: a random block diagonal matrix
    """
    assert total_width > first_block_width
    A = torch.zeros((total_width, total_width), dtype=dtype)
    A[:first_block_width, :first_block_width] = (
        block_variances[0]
        * torch.randn(1, first_block_width, dtype=dtype).repeat(first_block_width, 1).T
    )
    second_block_width = total_width - first_block_width
    A[first_block_width:, first_block_width:] = (
        block_variances[1]
        * torch.randn(1, second_block_width, dtype=dtype).repeat(second_block_width, 1).T
    )
    return A


class BlockDiagonalDNN(MLP):
    def __init__(
        self,
        layers: int = 3,
        width: int = 4,
        first_block_width: Optional[int] = None,
        dtype: Literal["float32", "float64"] = "float32",
        bias: float = 0.0,
        activation_fn: str = "relu",
        block_variances: List[float] = [1.0, 1.0],
        weight_assignment_function: Callable = random_block_diagonal_matrix,
    ):
        # Initialize MLP
        self.mlp_config = MLPConfig(
            hidden_sizes=[width] * layers,
            input_size=width,
            output_size=width,
            bias=True,
            dtype=dtype,
            # Don't fold bias yet, fold later
            fold_bias=False,
            activation_fn=activation_fn,
        )
        super(BlockDiagonalDNN, self).__init__(config=self.mlp_config)

        # Hardcode weights and biases
        self.width = width
        self.first_block_width = first_block_width if first_block_width is not None else width // 2
        self.block_variances = block_variances
        self.bias = bias
        assert len(self.layers) == layers + 1
        for layer in self.layers:
            layer.W = nn.Parameter(
                weight_assignment_function(
                    self.width, self.first_block_width, self.block_variances
                ).to(self.dtype)
            )
            layer.b = nn.Parameter(bias * torch.ones(self.width, dtype=self.dtype))

        # Now fold biases
        self.fold_bias()


class BlockVectorDataset(Dataset):
    def __init__(
        self,
        length: int = 4,
        first_block_length: Optional[int] = None,
        variances: List[float] = [1.0, 1.0],
        perfect_correlation: bool = False,
        size: int = 1000,
        dtype: Literal["float32", "float64"] = "float32",
    ):
        """Generate a dataset of vectors with two blocks of variance

        Args:
            length: length of each vector
            first_block_length: length of the first block
            variances: variances of the two blocks
            size: number of samples in the dataset
            dtype: data type
        """
        self.length = length
        self.first_block_length = (
            first_block_length if first_block_length is not None else length // 2
        )
        self.variances = variances
        self.size = size
        self.dtype = TORCH_DTYPES[dtype]
        # Generate data
        self.data = torch.empty((self.size, self.length), dtype=self.dtype)
        second_block_length = self.length - self.first_block_length
        if not perfect_correlation:
            self.data[:, 0 : self.first_block_length] = self.variances[0] * torch.randn(
                self.size, self.first_block_length, dtype=self.dtype
            )
            self.data[:, self.first_block_length :] = self.variances[1] * torch.randn(
                self.size, second_block_length, dtype=self.dtype
            )
        else:
            self.data[:, 0 : self.first_block_length] = self.variances[0] * torch.randn(
                size, 1, dtype=self.dtype
            ).repeat(1, self.first_block_length)
            self.data[:, self.first_block_length :] = self.variances[1] * torch.randn(
                size, 1, dtype=self.dtype
            ).repeat(1, second_block_length)
        # Not needed, just here for Dataset
        self.labels = torch.nan * torch.ones(self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
