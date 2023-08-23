from typing import Optional
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformer_lens.utils import gelu_new

from rib.models import MLP, Layer
from rib.models.utils import get_model_attr
from rib.utils import calc_ablation_schedule, eval_model_accuracy, find_root


def test_get_model_attr() -> None:
    """Test the get_model_attr function to retrieve model layers.

    Creates a custom MLP with nested sub-layers and module list, then uses get_model_attr to retrieve a
    specific layer.
    """

    # Define the parent MLP model
    model = MLP(hidden_sizes=[5], input_size=2, output_size=3, fold_bias=False)

    # Test the function
    layers = get_model_attr(model, "layers")
    assert isinstance(layers, nn.ModuleList)

    layer_0 = get_model_attr(model, "layers.0")
    assert isinstance(layer_0, Layer)

    layer_0_linear = get_model_attr(model, "layers.0.linear")
    assert isinstance(layer_0_linear, torch.nn.Linear)
    assert layer_0_linear.in_features == 2
    assert layer_0_linear.out_features == 5

    if hasattr(layer_0, "activation"):
        layer_0_activation = get_model_attr(model, "layers.0.activation")
        assert isinstance(
            layer_0_activation, nn.Module
        )  # replace nn.Module with specific activation function if known

    layer_1_linear = get_model_attr(model, "layers.1.linear")
    assert isinstance(layer_1_linear, torch.nn.Linear)
    assert layer_1_linear.in_features == 5
    assert layer_1_linear.out_features == 3


def test_eval_model_accuracy() -> None:
    """Test the eval_model_accuracy function.

    Mocks a hooked model, hooks, dataloader and device, then checks if the function calculates
    accuracy correctly.
    """
    torch.manual_seed(0)
    # Create mock objects
    hooked_model = Mock()
    hooks = [Mock() for _ in range(3)]  # assume there are 3 hooks
    device = "cpu"

    # Create a simple DataLoader with hardcoded tensors
    data = torch.randn(3, 2)
    labels = torch.tensor([0, 1, 0])  # Assume binary classification
    dataloader: DataLoader = DataLoader(list(zip(data, labels)), batch_size=1)

    # Define generator function for outputting one label at a time (since our batch_size=1)
    def model_output_generator(output_list):
        for output in output_list:
            yield torch.tensor(output).unsqueeze(0)

    # Test case 1: All predictions are correct
    hooked_model.side_effect = model_output_generator([[1.2, -0.8], [-1.5, 2.1], [3.0, -1.5]])
    assert eval_model_accuracy(hooked_model, dataloader, hooks, device) == 1.0

    # Test case 2: Only one prediction is correct
    hooked_model.side_effect = model_output_generator([[-0.7, 1.2], [-1.5, 2.1], [3.0, -1.5]])
    assert eval_model_accuracy(hooked_model, dataloader, hooks, device) == 2 / 3

    # Test case 3: No predictions are correct
    hooked_model.side_effect = model_output_generator([[-0.7, 1.2], [1.5, -2.1], [-3.0, 1.5]])
    assert eval_model_accuracy(hooked_model, dataloader, hooks, device) == 0.0


@pytest.mark.parametrize(
    "ablate_every_vec_cutoff, n_eigenvecs, expected",
    [
        (None, 12, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
        (0, 12, [12, 11, 9, 5, 0]),
        (1, 12, [12, 11, 10, 8, 4, 0]),
        (3, 12, [12, 11, 10, 9, 8, 6, 2, 0]),
        (3, 24, [24, 23, 22, 21, 20, 18, 14, 6, 0]),
    ],
)
def test_calc_ablation_schedule(
    ablate_every_vec_cutoff: Optional[int],
    n_eigenvecs: int,
    expected: list[int],
):
    schedule = calc_ablation_schedule(ablate_every_vec_cutoff, n_eigenvecs)
    assert schedule == expected


class TestFindRoot:
    def test_quadratic(self):
        """Test find_root function for a quadratic function."""
        func = lambda x: x**2 - 4
        root = find_root(func, xmin=0, xmax=3)
        assert root == pytest.approx(2.0, abs=1e-6)

    def test_bad_bracketing(self):
        """Test find_root with xmin and xmax that do not bracket the root."""
        func = lambda x: x**2 - 4
        with pytest.raises(AssertionError):
            find_root(func, xmin=0, xmax=2)

    def test_convergence_error(self):
        """Test find_root with a function that is slow to converge."""
        func = lambda x: torch.exp(-x) - 0.5
        with pytest.raises(ValueError):
            find_root(func, xmin=torch.tensor(0.0), xmax=torch.tensor(1.0), max_iter=10)

    def test_gelu_new(self):
        """Test the gelu_new function."""
        func = lambda x: gelu_new(x) - 1.0
        root = find_root(func, xmin=torch.tensor(-1.0), xmax=torch.tensor(4.0))
        assert root == pytest.approx(1.1446, abs=1e-4)
