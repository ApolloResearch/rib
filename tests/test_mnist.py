import pytest
from torch import nn

from rib.mnist import MLP


@pytest.mark.parametrize(
    "input_size, hidden_sizes, output_size, expected_num_layers",
    [
        (784, [128, 64], 10, 5),  # 3 linear layers, 2 ReLU layers
        (784, [], 10, 1),  # only 1 linear layer
        (784, [128], 10, 3),  # 2 linear layers, 1 ReLU layer
        (784, [128, 64, 32], 10, 7),  # 3 4 linear layers, 3 ReLU layers
    ],
)
def test_make_layers(input_size, hidden_sizes, output_size, expected_num_layers):
    layers = MLP.make_layers(input_size, hidden_sizes, output_size)
    assert isinstance(layers, nn.Sequential)
    assert len(layers) == expected_num_layers

    # Check types of layers
    for i, layer in enumerate(layers):
        if i % 2 == 0:  # Linear layers at even indices
            assert isinstance(layer, nn.Linear)
        else:  # ReLU layers at odd indices (except for the last Linear layer)
            assert isinstance(layer, nn.ReLU)
