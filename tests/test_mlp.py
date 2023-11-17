from typing import Optional, Tuple

import pytest
from torch import nn

from rib.models import MLP, MLPLayer
from rib.models.utils import ACTIVATION_MAP


@pytest.mark.parametrize(
    "hidden_sizes, activation_fn, fold_bias, bias",
    [
        # 2 hidden layers with ReLU, fold_bias True, bias False
        ([4, 3], "relu", True, False),
        # no hidden layers with ReLU, fold_bias False, bias True
        ([], "relu", False, True),
        # 1 hidden layer with Tanh, fold_bias True, bias True
        ([4], "gelu", True, True),
        # 2 hidden layers with Sigmoid, fold_bias False, bias False
        ([4, 3], "sigmoid", False, False),
        # 1 hidden layer with default ReLU, fold_bias True, bias True
        ([4], None, True, True),
        # 1 hidden layer with Tanh, fold_bias False, bias False
        ([4], "tanh", False, False),
    ],
)
def test_mlp_layers(
    hidden_sizes: list[int],
    activation_fn: Optional[str],
    fold_bias: bool,
    bias: bool,
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
    expected_layer_sizes = list(zip([input_size] + hidden_sizes, hidden_sizes + [output_size]))
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
            bias=bias,
            fold_bias=fold_bias,
        )

    assert isinstance(model, MLP)

    activation_fn = activation_fn or "relu"

    for i, layer in enumerate(model.layers):
        assert isinstance(layer, MLPLayer)

        assert layer.has_folded_bias == fold_bias
        assert (layer.b is None) == (not bias or fold_bias)

        # Check the in/out feature sizes of Linear layers
        assert layer.in_features == expected_layer_sizes[i][0]
        assert layer.out_features == expected_layer_sizes[i][1]

        if i < len(model.layers) - 1:
            # Activation layers at indices before the last layer
            assert isinstance(layer.activation, ACTIVATION_MAP[activation_fn.lower()])
        else:
            # No activation function for the last layer
            assert not hasattr(layer, "activation")

    # input = torch.rand((3, in_size))
