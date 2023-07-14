"""
Defines a generic MLP to be used for rib.
"""
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
        """Add a feature of ones to the input and then perform a linear transformation.

        Input may be batched or unbatched, and may or may not include a seq_len dimension.
        We handle all cases by ensuring that the bias tensor has the same size as the input
        except for the last dimension which should have size 1.
        """
        rank = len(input.shape)
        bias_size = [input.shape[i] if i < rank - 1 else 1 for i in range(rank)]
        bias = torch.ones(bias_size, device=input.device, dtype=input.dtype)
        biased_input = torch.cat([input, bias], dim=rank - 1)
        return F.linear(biased_input, self.weight, self.bias)  # self.bias is None


class Layer(nn.Module):
    """
    Neural network layer consisting of a linear layer followed by an optional activation function.

    Args:
        in_features: The size of each input.
        out_features: The size of each output.
        linear_module: A type defining whether to use a folded bias layer or a regular linear layer.
        activation_fn: The activation function to use. Default is "relu".
        bias: Whether to add a bias term to the linear transformation. Default is True.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        linear_module: Union[Type[nn.Linear], Type[LinearFoldedBias]],
        activation_fn: Optional[str] = "relu",
        bias: bool = True,
    ):
        super().__init__()
        self.linear = linear_module(in_features, out_features, bias=bias)
        if activation_fn is not None:
            self.activation = ACTIVATION_MAP[activation_fn]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if hasattr(self, "activation"):
            x = self.activation(x)
        return x


class MLP(nn.Module):
    """
    This class defines an MLP with a variable number of hidden layers.

    Args:
        hidden_sizes: A list of integers specifying the sizes of the hidden layers.
        input_size: The size of each input sample.
        output_size: The size of each output sample.
        activation_fn: The activation function to use for all but the last layer. Default is "relu".
        bias: Whether to add a bias term to the linear transformations. Default is True.
        fold_bias: Whether to use the LinearFoldedBias class for linear layers. If true (and if
            bias is True), the biases are folded into the weight matrices and the forward pass
            is modified to add a vector of 1s to the input. Default is True.
    """

    def __init__(
        self,
        hidden_sizes: Optional[list[int]],
        input_size: int,
        output_size: int,
        activation_fn: str = "relu",
        bias: bool = True,
        fold_bias: bool = False,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = []

        linear_module = LinearFoldedBias if bias and fold_bias else nn.Linear

        # Size of each layer (including input and output)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            # No activation for final layer
            layer_act = activation_fn if i < len(sizes) - 2 else None
            self.layers.append(
                Layer(
                    in_features=sizes[i],
                    out_features=sizes[i + 1],
                    linear_module=linear_module,
                    activation_fn=layer_act,
                    bias=bias,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x
