from typing import List, Optional, Tuple

import pytest
from torch import nn

from rib.models import MLP
from rib.models.mlp import LinearFoldedBias
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
def test_make_layers(
    hidden_sizes: List[int],
    activation_fn: Optional[str],
    fold_bias: bool,
    bias: bool,
    expected_layer_sizes: List[Tuple[int, int]],
):
    """Test the make_layers method of MLP class for fixed input and output sizes.

    Verifies the returned layers' types, sizes and bias. Also checks whether the
    layers are instances of LinearFoldedBias when fold_bias is True, and nn.Linear when it's False.

    Args:
        hidden_sizes: A list of hidden layer sizes. If None, no hidden layers are added.
        activation_fn: The activation function to use.
        fold_bias: If True and bias is True, biases will be folded into the input features.
        bias: Whether to add a bias to the Linear layers.
        expected_layer_sizes: A list of tuples where each tuple is a pair of in_features and out_features of a layer.
    """
    input_size = 3
    output_size = 4

    if activation_fn is not None:
        layers = MLP.make_layers(
            LinearFoldedBias if fold_bias and bias else nn.Linear,
            input_size,
            hidden_sizes,
            output_size,
            activation_fn,
            bias=bias,
        )
    else:
        # Test the default activation_fn (relu)
        layers = MLP.make_layers(
            LinearFoldedBias if fold_bias and bias else nn.Linear,
            input_size,
            hidden_sizes,
            output_size,
            bias=bias,
        )

    assert isinstance(layers, nn.Sequential)
    # Account for activation layers
    assert len(layers) == len(expected_layer_sizes) * 2 - 1

    activation_fn = activation_fn or "relu"

    # Linear layers are at even indices (0, 2, 4, ...)
    linear_layer_indices = range(0, len(layers), 2)
    for i, layer in enumerate(layers):
        if i in linear_layer_indices:
            if fold_bias and bias:
                assert isinstance(layer, LinearFoldedBias)
            else:
                assert isinstance(layer, nn.Linear)

            # Check the in/out feature sizes of Linear layers
            assert layer.in_features == expected_layer_sizes[i // 2][0]
            assert layer.out_features == expected_layer_sizes[i // 2][1]
            # Check bias is None when fold_bias is True, and not None otherwise
            assert layer.bias is None if fold_bias or bias is False else layer.bias is not None
        else:
            # Activation layers at other indices
            assert isinstance(layer, ACTIVATION_MAP[activation_fn.lower()])
