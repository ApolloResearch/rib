from typing import Optional, Tuple

import pytest
from torch import nn

from rib.models import MLP
from rib.models.mlp import Layer, LinearFoldedBias
from rib.models.utils import ACTIVATION_MAP


@pytest.mark.parametrize(
    "hidden_sizes, activation_fn, fold_bias, bias, expected_layer_sizes",
    [
        # 2 hidden layers with ReLU, fold_bias True, bias False
        ([4, 3], "relu", True, False, [(3, 4), (4, 3), (3, 4)]),
        # no hidden layers with ReLU, fold_bias False, bias True
        ([], "relu", False, True, [(3, 4)]),
        # 1 hidden layer with Tanh, fold_bias True, bias True
        ([4], "tanh", True, True, [(4, 4), (5, 4)]),
        # 2 hidden layers with Sigmoid, fold_bias False, bias False
        ([4, 3], "sigmoid", False, False, [(3, 4), (4, 3), (3, 4)]),
        # 1 hidden layer with default ReLU, fold_bias True, bias True
        ([4], None, True, True, [(4, 4), (5, 4)]),
        # 1 hidden layer with Tanh, fold_bias False, bias False
        ([4], "tanh", False, False, [(3, 4), (4, 4)]),
    ],
)
def test_mlp_layers(
    hidden_sizes: list[int],
    activation_fn: Optional[str],
    fold_bias: bool,
    bias: bool,
    expected_layer_sizes: list[Tuple[int, int]],
) -> None:
    """Test the MLP constructor for fixed input and output sizes.

    Verifies the created layers' types, sizes and bias. Also checks whether the
    layers are instances of LinearFoldedBias when fold_bias is True, and nn.Linear when it's False.

    Args:
        hidden_sizes: A list of hidden layer sizes. If None, no hidden layers are added.
        activation_fn: The activation function to use.
        fold_bias: If True and bias is True, biases will be folded into the input features.
        bias: Whether to add a bias to the Linear layers.
        expected_layer_sizes: A list of tuples where each tuple is a pair of in_features and
            out_features of a layer.
    """
    input_size = 3
    output_size = 4
    if activation_fn is None:
        model = MLP(
            hidden_sizes,
            input_size,
            output_size,
            bias=bias,
            fold_bias=fold_bias,
        )
    else:
        model = MLP(
            hidden_sizes,
            input_size,
            output_size,
            activation_fn,
            bias,
            fold_bias,
        )

    assert isinstance(model, MLP)

    activation_fn = activation_fn or "relu"

    for i, layer in enumerate(model.layers):
        assert isinstance(layer, Layer)

        if fold_bias and bias:
            assert isinstance(layer.linear, LinearFoldedBias)
        else:
            assert isinstance(layer.linear, nn.Linear)

        # Check the in/out feature sizes of Linear layers
        assert layer.linear.in_features == expected_layer_sizes[i][0]
        assert layer.linear.out_features == expected_layer_sizes[i][1]
        # Check bias is None when fold_bias is True, and not None otherwise
        assert (
            layer.linear.bias is None
            if fold_bias or bias is False
            else layer.linear.bias is not None
        )

        if i < len(model.layers) - 1:
            # Activation layers at indices before the last layer
            assert isinstance(layer.activation, ACTIVATION_MAP[activation_fn.lower()])
        else:
            # No activation function for the last layer
            assert not hasattr(layer, "activation")
