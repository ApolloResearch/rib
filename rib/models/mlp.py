"""
Defines a generic MLP to be used for rib.
"""
from typing import Literal, Optional, Tuple

import torch
from fancy_einsum import einsum
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from rib.models.components import SequentialComponent
from rib.models.utils import ACTIVATION_MAP, fold_mlp_in
from rib.types import TorchDtype


class MLPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    config_type: Literal["MLP"] = "MLP"
    hidden_sizes: Optional[list[int]] = Field(
        None,
        description="A list of integers specifying the sizes of the hidden layers. If None, "
        "there are no hidden layers.",
    )
    input_size: int = Field(..., description="The size of each input sample (after flattening)")
    output_size: int = Field(..., description="The size of each output sample")
    activation_fn: str = Field(
        "relu",
        description="The activation function to use for all but the last layer. Default is "
        '"relu".',
    )
    bias: bool = Field(
        True,
        description="Whether to add a bias term to the linear transformations. Default is True.",
    )
    fold_bias: bool = Field(
        False,
        description="Whether to fold the bias in after initialization. If done, model is no "
        "longer valid to train! Doesn't change the input / output behavior or input / output "
        "gradients, but will append a 1 to intermediate activations between layers.",
    )
    dtype: TorchDtype = Field(torch.float32, description="The dtype to initialize the model with.")


class MLP(nn.Module):
    """
    This class defines an MLP with a variable number of hidden layers.

    Args:
        hidden_sizes: A list of integers specifying the sizes of the hidden layers.
        input_size: The size of each input sample (after flattening)
        output_size: The size of each output sample.
        activation_fn: The activation function to use for all but the last layer. Default is "relu".
        bias: Whether to add a bias term to the linear transformations. Default is True.
        fold_bias: Whether to fold the bias in after initialization. If done, model is no longer
            valid to train! Doesn't change the input / output behavior or input / output gradients,
            but will append a 1 to intermediate activations between layers.
    """

    def __init__(self, cfg: MLPConfig):
        super().__init__()

        self.cfg = cfg
        self.hidden_sizes = cfg.hidden_sizes if cfg.hidden_sizes is not None else []
        self.input_size = cfg.input_size
        self.output_size = cfg.output_size

        # Size of each layer (including input and output)
        sizes = [self.input_size] + self.hidden_sizes + [self.output_size]

        self.layers: nn.ModuleList = nn.ModuleList()
        for i in range(len(sizes) - 1):
            final_layer = i == len(sizes) - 2
            # No activation for final layer
            layer_act = cfg.activation_fn if not final_layer else None
            self.layers.append(
                MLPLayer(
                    in_features=sizes[i],
                    out_features=sizes[i + 1],
                    activation_fn=layer_act,
                    use_bias=cfg.bias,
                    dtype=cfg.dtype,
                )
            )

        self.has_folded_bias = False
        if cfg.fold_bias:
            self.fold_bias()

    def forward(self, x: Float[Tensor, "batch ..."]) -> Float[Tensor, "batch outdim"]:
        """Run the MLP on the input.

        We first flatten the input to a single feature dimension (preserving batch dimension)

        If we have folded biases in, we add an extra 1 to each intermediate output between layers.
        This is handled within each layer, but we need to concat it to the input. The last layer
        excludes this automatically.
        """
        # Flatten input
        x = x.view(x.size(0), -1)
        if self.has_folded_bias:  # concat 1 to every batch element
            x = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)], dim=-1)

        for layer in self.layers:
            x = layer(x)
        return x

    def fold_bias(self):
        for layer in self.layers:
            last_layer = layer is self.layers[-1]
            layer.fold_bias(include_ones_in_out=not last_layer)

        self.has_folded_bias = True


class MLPLayer(SequentialComponent):
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
        super().__init__(in_dims=(in_features,))

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

    @property
    def out_dims(self) -> tuple[int]:
        return (self.W.shape[-1],)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        x = einsum("... d_in, d_in d_out -> ... d_out", x, self.W)
        if self.b is not None:
            x += self.b
        if self.activation_fn is not None:
            x = self.activation(x)
        return x

    def fold_bias(self, include_ones_in_out=True) -> None:
        assert not self.has_folded_bias, "bias already folded"
        assert self.b is not None, "trying to fold bias on a layer that has no bias"
        fold_mlp_in(self.activation_fn, self.W, self.b)
        if not include_ones_in_out:  # for last layer
            self.W.data = self.W.data[..., :-1]
        self.b = None  # fold_mlp_in sets to 0, but None is clearer
        self.has_folded_bias = True
