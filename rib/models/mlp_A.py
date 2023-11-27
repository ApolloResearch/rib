"""
Defines a generic MLP to be used for rib.
"""
from typing import Any, Optional, Type, Union

import torch
from torch import nn
from torch.nn import functional as F

from rib.models.utils import ACTIVATION_MAP


class LinearFoldedBias(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        concat_bias_to_output: bool = True,
    ) -> None:
        """A Linear layer with an extra input feature to act as a folded bias.

        Ignore the bias parameter, it is always set to False.
        """
        self.concat_bias_to_output = concat_bias_to_output
        super().__init__(in_features + 1, out_features, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a standard linear transformation and concatenate a vector of 1s to the output.

        If self.concat_bias_to_output is True, we concatenate a vector of 1s to the output.
        """
        x = F.linear(x, self.weight, self.bias)
        if self.concat_bias_to_output:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        return x


class Layer(nn.Module):
    """
    Neural network layer consisting of a linear layer followed by an optional activation function.

    Args:
        in_features: The size of each input.
        out_features: The size of each output.
        linear_module: A type defining whether to use a folded bias layer or a regular linear layer.
        activation_fn: The activation function to use. Default is "relu".
        bias: Whether to add a bias term to the linear transformation. Default is True.
        concat_bias_to_output: Whether to concatenate a vector of 1s to the output. Default is False.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        linear_module: Union[Type[nn.Linear], Type[LinearFoldedBias]],
        activation_fn: Optional[str] = "relu",
        bias: bool = True,
        concat_bias_to_output: bool = False,
    ):
        super().__init__()
        kwargs: dict[str, Any] = {"bias": bias}
        if linear_module is LinearFoldedBias:
            kwargs["concat_bias_to_output"] = concat_bias_to_output

        self.linear = linear_module(in_features, out_features, **kwargs)
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
            is modified to add a vector of 1s to the output of each layer, except the last layer.
            Default is True.

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

        self.fold_bias = fold_bias
        if hidden_sizes is None:
            hidden_sizes = []

        linear_module = LinearFoldedBias if bias and fold_bias else nn.Linear

        # Size of each layer (including input and output)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            final_layer = i == len(sizes) - 2
            # No activation for final layer
            layer_act = activation_fn if not final_layer else None
            concat_bias_to_output = not final_layer and fold_bias
            self.layers.append(
                Layer(
                    in_features=sizes[i],
                    out_features=sizes[i + 1],
                    linear_module=linear_module,
                    activation_fn=layer_act,
                    bias=bias,
                    concat_bias_to_output=concat_bias_to_output,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP on the input.

        We first flatten the input to a single feature dimension.

        If self.fold_bias is True, each layer will need to ensure that a vector of 1s is
        concatenated to each layer's input. This is done automatically in the output of each
        LinearFoldedBias layer except the final layer. Therefore, we just need to add a vector
        of 1s to the input of the first layer.
        """
        # Flatten input
        x = x.view(x.size(0), -1)
        if self.fold_bias:
            # Concatenate a vector of 1s to the input
            x = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)], dim=-1)

        for layer in self.layers:
            x = layer(x)
        return x
