from contextlib import nullcontext

import pytest
import torch

from rib.models import MLP, MLPConfig, MLPLayer
from rib.models.utils import ACTIVATION_MAP
from rib.utils import set_seed

set_seed(0)


@pytest.mark.parametrize("hidden_sizes", [[], [4, 3]])
@pytest.mark.parametrize("activation_fn", ["relu", "gelu", "sigmoid"])
@pytest.mark.parametrize("bias, fold_bias", [(False, False), (True, False), (True, True)])
def test_mlp_layers(
    hidden_sizes: list[int],
    activation_fn: str,
    fold_bias: bool,
    bias: bool,
) -> None:
    """Test the MLP constructor for fixed input and output sizes.

    Verifies the created layers' shapes and bias.
    Also tests that when folding the bias in the forward and backward passes remain the same.

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

    # some activation functions can't be folded
    exception_expected = (
        (activation_fn in ["tanh", "sigmoid"]) and fold_bias and (len(hidden_sizes) > 0)
    )
    with pytest.raises(ValueError) if exception_expected else nullcontext():
        mlp_config = MLPConfig(
            hidden_sizes=hidden_sizes,
            input_size=input_size,
            output_size=output_size,
            activation_fn=activation_fn,
            bias=bias,
            fold_bias=fold_bias,
        )
        model = MLP(mlp_config)
    if exception_expected:
        return None

    assert isinstance(model, MLP)
    assert model.has_folded_bias == fold_bias
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

    batch_size = 10
    rand_input = torch.rand((batch_size, input_size), requires_grad=True)
    output = model(rand_input)
    in_grad = torch.autograd.grad(output.sum(), rand_input)[0]
    assert output.shape == (batch_size, output_size)

    if bias and not fold_bias and (activation_fn not in ["sigmoid", "tanh"]):
        model.fold_bias()
        assert model.has_folded_bias
        folded_output = model(rand_input)
        folded_in_grad = torch.autograd.grad(folded_output.sum(), rand_input)[0]
        assert torch.allclose(output, folded_output)
        assert torch.allclose(in_grad, folded_in_grad)

    if not bias and not fold_bias:
        with pytest.raises(AssertionError):
            model.fold_bias()
