"""
Defines a generic MLP to be used for rib.
"""
from typing import Iterable, Optional, Tuple
from warnings import warn

import torch
from fancy_einsum import einsum
from jaxtyping import Float
from torch import Tensor, nn

from rib.models.utils import ACTIVATION_MAP, fold_mlp_in


class MLPLayer(nn.Module):
    """
    Neural network layer consisting of a linear layer followed by an optional activation function.

    Note: after biases are folded in, the layer expects an input of size `[..., in_features + 1]`,
    and outputs a tensor of size [..., out_features + 1].

    Args:
        in_features: The size of each input.
        out_features: The size of each output.
        linear_module: A type defining whether to use a folded bias layer or a regular linear layer.
        activation_fn: The activation function to use. Default is "relu".
        use_bias: Whether to add a bias term to the linear transformation. Default is True.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_fn: Optional[str] = "relu",
        use_bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn: Optional[str] = activation_fn

        def make_param(size: Tuple[int, ...]):
            # consistent init with nn.Linear
            bound = 1 / (self.in_features**0.5)
            p = nn.Parameter(torch.empty(size=size, dtype=dtype))
            nn.init.uniform_(p, a=-bound, b=bound)
            return p

        self.W = make_param((self.in_features, self.out_features))
        self.b = make_param((self.out_features,)) if use_bias else None
        if activation_fn is not None:
            self.activation = ACTIVATION_MAP[activation_fn]()

        self.has_folded_bias = False

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        x = einsum("... d_model, d_model d_mlp -> ... d_mlp", x, self.W)
        if self.b is not None:
            x += self.b
        if self.activation_fn is not None:
            x = self.activation(x)
        return x

    def fold_bias(self) -> None:
        if self.b is None:
            warn("trying to fold a bias on an MLP layer that has none")
            # create a 0 bias to fold in
            self.b = torch.zeros(self.out_features, dtype=self.W.dtype, device=self.W.device)
        assert not self.has_folded_bias
        fold_mlp_in(self.activation_fn, self.W, self.b)
        self.b = None  # fold_mlp_in sets to 0, but None is clearer
        self.has_folded_bias = True


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
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = []

        # Size of each layer (including input and output)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.layers: Iterable[MLPLayer] = nn.ModuleList()
        for i in range(len(sizes) - 1):
            final_layer = i == len(sizes) - 2
            # No activation for final layer
            layer_act = activation_fn if not final_layer else None
            self.layers.append(
                MLPLayer(
                    in_features=sizes[i],
                    out_features=sizes[i + 1],
                    activation_fn=layer_act,
                    use_bias=bias,
                    dtype=dtype,
                )
            )

        self.has_folded_bias = False
        if fold_bias:
            self.fold_bias()

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
        if self.has_folded_bias:  # concat 1 to every batch element
            x = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)], dim=-1)

        for layer in self.layers:
            x = layer(x)

        if self.has_folded_bias:
            # output has an extra 1 at the end
            x = x[:, :-1]
        return x

    def fold_bias(self):
        for layer in self.layers:
            layer.fold_bias()

        self.has_folded_bias = True
