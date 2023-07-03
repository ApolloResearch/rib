from typing import List, Optional, Tuple

import pytest
from torch import nn

from rib.models import MLP
from rib.models.utils import ACTIVATION_MAP


@pytest.mark.parametrize(
    "input_size, hidden_sizes, output_size, activation_fn, expected_layer_sizes",
    [
        # 2 hidden layers with ReLU
        (784, [128, 64], 10, "relu", [(784, 128), (128, 64), (64, 10)]),
        # no hidden layers with ReLU
        (784, [], 10, "relu", [(784, 10)]),
        # 1 hidden layer with Tanh
        (784, [128], 10, "tanh", [(784, 128), (128, 10)]),
        # 3 hidden layers with Sigmoid
        (784, [128, 64, 32], 10, "sigmoid", [(784, 128), (128, 64), (64, 32), (32, 10)]),
        # 2 hidden layers with default ReLU
        (784, [128, 64], 10, None, [(784, 128), (128, 64), (64, 10)]),
    ],
)
def test_make_layers(
    input_size: int,
    hidden_sizes: List[int],
    output_size: int,
    activation_fn: Optional[str],
    expected_layer_sizes: List[Tuple[int, int]],
):
    """Test the _make_layers method of MLP class.

    Verifies the returned layers' types and sizes.
    """
    if activation_fn is not None:
        layers = MLP.make_layers(input_size, hidden_sizes, output_size, activation_fn)
    else:
        # Test the default activation_fn (relu)
        layers = MLP.make_layers(input_size, hidden_sizes, output_size)

    assert isinstance(layers, nn.Sequential)
    # Multiply by 2 for activation layers and subtract 1 as there's no activation after last Linear layer
    assert len(layers) == len(expected_layer_sizes) * 2 - 1

    activation_fn = activation_fn or "relu"
    # Check types and sizes of layers
    # Indices of Linear layers (0, 2, 4, ...)
    linear_layer_indices = range(0, len(layers), 2)
    for i, layer in enumerate(layers):
        if i in linear_layer_indices:
            assert isinstance(layer, nn.Linear)
            assert layer.in_features == expected_layer_sizes[i // 2][0]
            assert layer.out_features == expected_layer_sizes[i // 2][1]
        else:
            # Activation layers at other indices
            assert isinstance(layer, ACTIVATION_MAP[activation_fn.lower()])
