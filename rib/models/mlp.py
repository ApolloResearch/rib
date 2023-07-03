"""
Defines a generic MLP to be used for rib.
"""
from typing import List, Optional, Union

import torch
from torch import nn

from rib.models.utils import ACTIVATION, ACTIVATION_MAP


class MLP(nn.Module):
    def __init__(
        self,
        hidden_sizes: Optional[List[int]],
        input_size: int,
        output_size: int,
        activation_fn: str = "relu",
    ):
        super().__init__()

        self.layers = self.make_layers(input_size, hidden_sizes, output_size, activation_fn)

    @staticmethod
    def make_layers(
        input_size: int,
        hidden_sizes: Optional[List[int]],
        output_size: int,
        activation_fn: str = "relu",
    ) -> nn.Sequential:
        """Create layers for MLP.

        The total number of layers is len(hidden_sizes) + 1.
        An activations layer is added after each Linear layer except the last one.

        Args:
            input_size: The size of the input.
            hidden_sizes: A list of hidden layer sizes. If None, no hidden layers are added.
            output_size: The size of the output.

        Returns:
            A nn.Sequential containing the Linear and activation layers.
        """
        if hidden_sizes is None:
            hidden_sizes = []

        sizes = [input_size] + hidden_sizes + [output_size]

        activation_module = ACTIVATION_MAP[activation_fn]

        layers: List[Union[nn.Linear, ACTIVATION]] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            # Don't add ReLU to the last layer
            if i < len(sizes) - 2:
                layers.append(activation_module())
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
