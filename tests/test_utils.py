from collections import OrderedDict

from torch import nn

from rib.models.utils import get_model_attr


def test_get_model_attr():
    """Test the get_model_attr function to retrieve model layers.

    Creates a custom Sequential model with nested sub-layers, then uses get_model_attr to retrieve
    a specific layer.
    """

    # Define a nested Sequential model
    nested_layers = nn.Sequential(
        OrderedDict([("linear_0", nn.Linear(in_features=3, out_features=5)), ("relu_0", nn.ReLU())])
    )

    # Define the parent Sequential model and add the nested_layers to it
    model = nn.Sequential(OrderedDict([("layers", nested_layers)]))

    # Test the function
    nested_layers = get_model_attr(model, "layers")
    assert isinstance(nested_layers, nn.Sequential)

    linear_0 = get_model_attr(model, "layers.linear_0")
    assert isinstance(linear_0, nn.Linear)
    assert linear_0.in_features == 3
    assert linear_0.out_features == 5
