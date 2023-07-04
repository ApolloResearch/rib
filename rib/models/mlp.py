"""
Defines a generic MLP to be used for rib.
"""
from collections import OrderedDict
from typing import Optional, Type, Union

import torch
from torch import nn
from torch.nn import functional as F

from rib.models.utils import ACTIVATION_MAP


class LinearFoldedBias(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None
    ) -> None:
        """A Linear layer with an extra feature of ones to act as the bias.

        Ignore the bias parameter, it is always set to False.
        """
        super().__init__(in_features + 1, out_features, bias=False, device=device, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Add an extra feature of ones to act as the bias
        bias = torch.ones(input.size(0), 1, device=input.device)
        input = torch.cat([input, bias], dim=1)
        return F.linear(input, self.weight, self.bias)  # self.bias is None


class MLP(nn.Module):
    def __init__(
        self,
        hidden_sizes: Optional[list[int]],
        input_size: int,
        output_size: int,
        activation_fn: str = "relu",
        bias: bool = True,
        fold_bias: bool = True,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = []

        linear_module = LinearFoldedBias if bias and fold_bias else nn.Linear

        self.layers = self.make_layers(
            linear_module, input_size, hidden_sizes, output_size, activation_fn, bias
        )

    @staticmethod
    def make_layers(
        linear_module: Union[Type[nn.Linear], Type[LinearFoldedBias]],
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        activation_fn: str = "relu",
        bias: bool = True,
    ) -> nn.Sequential:
        """Create layers for MLP.

        The total number of layers is len(hidden_sizes) + 1.
        An activations layer is added after each Linear layer except the last one.

        Args:
            linear_module: The module to use for the Linear layers.
            input_size: The size of the input.
            hidden_sizes: A list of hidden layer sizes. If None, no hidden layers are added.
            output_size: The size of the output.
            activation_fn: The activation function to use.
            bias: Whether to add a bias to the Linear layers.

        Returns:
            A nn.Sequential containing the Linear and activation layers.
        """
        sizes = [input_size] + hidden_sizes + [output_size]

        activation_module = ACTIVATION_MAP[activation_fn]

        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(len(sizes) - 1):
            layers[f"linear_{i}"] = linear_module(sizes[i], sizes[i + 1], bias=bias)
            # Don't add activation function to the last layer
            if i < len(sizes) - 2:
                layers[f"act_{i}"] = activation_module()
        return nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
