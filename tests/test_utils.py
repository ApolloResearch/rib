from typing import Optional
from unittest.mock import Mock

import pytest
import torch
from pydantic import BaseModel, ConfigDict, ValidationError
from torch import nn
from torch.utils.data import DataLoader

from rib.ablations import (
    BisectSchedule,
    BisectScheduleConfig,
    ExponentialSchedule,
    ExponentialScheduleConfig,
    LinearSchedule,
    LinearScheduleConfig,
)
from rib.models import MLP, MLPConfig, MLPLayer
from rib.models.utils import gelu_new, get_model_attr
from rib.utils import eval_model_accuracy, find_root, replace_pydantic_model


def test_get_model_attr() -> None:
    """Test the get_model_attr function to retrieve model layers.

    Creates a custom MLP with nested sub-layers and module list, then uses get_model_attr to retrieve a
    specific layer.
    """

    # Define the parent MLP model
    mlp_config = MLPConfig(hidden_sizes=[5], input_size=2, output_size=3)
    model = MLP(mlp_config)

    # Test the function
    layers = get_model_attr(model, "layers")
    assert isinstance(layers, nn.ModuleList)

    layer_0 = get_model_attr(model, "layers.0")
    assert isinstance(layer_0, MLPLayer)

    layer_0_W = get_model_attr(model, "layers.0.W")
    assert isinstance(layer_0_W, torch.nn.Parameter)
    assert layer_0.in_features == 2
    assert layer_0.out_features == 5

    if hasattr(layer_0, "activation"):
        layer_0_activation = get_model_attr(model, "layers.0.activation")
        assert isinstance(
            layer_0_activation, nn.Module
        )  # replace nn.Module with specific activation function if known

    layer_1 = get_model_attr(model, "layers.1")
    assert layer_1.in_features == 5
    assert layer_1.out_features == 3


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
    dtype = torch.float32

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
    assert eval_model_accuracy(hooked_model, dataloader, hooks, dtype, device) == 1.0

    # Test case 2: Only one prediction is correct
    hooked_model.side_effect = model_output_generator([[-0.7, 1.2], [-1.5, 2.1], [3.0, -1.5]])
    assert eval_model_accuracy(hooked_model, dataloader, hooks, dtype, device) == 2 / 3

    # Test case 3: No predictions are correct
    hooked_model.side_effect = model_output_generator([[-0.7, 1.2], [1.5, -2.1], [-3.0, 1.5]])
    assert eval_model_accuracy(hooked_model, dataloader, hooks, dtype, device) == 0.0


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
def test_calc_exponential_ablation_schedule(
    ablate_every_vec_cutoff: Optional[int],
    n_eigenvecs: int,
    expected: list[int],
):
    schedule_config = ExponentialScheduleConfig(
        schedule_type="exponential", ablate_every_vec_cutoff=ablate_every_vec_cutoff
    )
    schedule = ExponentialSchedule(schedule_config, n_eigenvecs)
    assert schedule._ablation_schedule == expected


@pytest.mark.parametrize(
    "n_points, n_eigenvecs, expected",
    [
        (3, 12, [12, 6, 0]),
        (11, 10, [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
        (11, 20, [20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0]),
    ],
)
def test_calc_linear_ablation_schedule(
    n_points: int,
    n_eigenvecs: int,
    expected: list[int],
):
    schedule_config = LinearScheduleConfig(schedule_type="linear", n_points=n_points)
    schedule = LinearSchedule(schedule_config, n_eigenvecs)
    assert schedule._ablation_schedule == expected


@pytest.mark.parametrize(
    "loss_fn, loss_target, n_eigenvecs, lower, upper",
    [
        (lambda x: -10.0 + x, -8.5, 10, 1, 2),
        (lambda x: -10.0 + x, -5.1, 10, 4, 5),
        (lambda x: -3.14 + x**2, -3, 10, 0, 1),
    ],
)
def test_bisect_schedule(loss_fn, loss_target, n_eigenvecs, upper, lower):
    """The schedule should have schedule._upper_bound == smallest n_vec_ablated with bad loss
    (loss > loss_target), and schedule._lower_bound == largest n_vec_ablated with good loss
    (loss <= loss_target).

    The function assumes that ablating every vector returns a bad loss, and ablating no vectors
    returns a good loss.
    """
    schedule_config = BisectScheduleConfig(
        schedule_type="bisect",
        score_target=loss_target,
    )
    schedule = BisectSchedule(schedule_config, n_eigenvecs, "ce_loss")
    for n_vec_ablated in schedule:
        print(f"{n_vec_ablated=}")
        loss = loss_fn(n_vec_ablated)
        schedule.update_bounds(loss)
    assert schedule._upper_bound == upper
    assert schedule._lower_bound == lower


@pytest.mark.parametrize(
    "acc_fn, accuracy_target, n_eigenvecs, lower, upper",
    [
        (lambda x: (100 - x) / 100, 0.95, 100, 5, 6),
        (lambda x: (3 - x), 1.5, 3, 1, 2),
        (lambda x: (10 - x) / 10, 0.49, 10, 5, 6),
        (lambda x: (10 - x) / 10, 0.5, 10, 5, 6),
    ],
)
def test_bisect_schedule_accuracy(acc_fn, accuracy_target, n_eigenvecs, upper, lower):
    """The schedule should have schedule._upper_bound == smallest n_vec_ablated with bad loss
    (loss > loss_target), and schedule._lower_bound == largest n_vec_ablated with good loss
    (loss <= loss_target).

    The function assumes that ablating every vector returns a bad loss, and ablating no vectors
    returns a good loss.
    """
    schedule_config = BisectScheduleConfig(
        schedule_type="bisect",
        score_target=accuracy_target,
    )
    schedule = BisectSchedule(schedule_config, n_eigenvecs, "accuracy")
    for n_vec_ablated in schedule:
        print(f"{n_vec_ablated=}")
        score = acc_fn(n_vec_ablated)
        schedule.update_bounds(score)
    assert schedule._upper_bound == upper
    assert schedule._lower_bound == lower


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


# For TestUpdatePydanticModel (must be defined outside for linting reasons)
class SimpleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    x: int
    y: str


class NestedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    value: SimpleModel


class DeeplyNestedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    inner: NestedModel


class TestReplacePydanticModel:
    """Test the replace_pydantic_model function.

    Note that we only care about configs defined with extra="forbid" and frozen=True.
    """

    def test_replace_simple_fields(self):
        model = SimpleModel(x=1, y="hello")
        replaced_model = replace_pydantic_model(model, {"x": 2})
        assert replaced_model.x == 2
        assert replaced_model.y == "hello"

    def test_nonexistent_fields(self):
        model = SimpleModel(x=1, y="hello")
        with pytest.raises(ValidationError):
            replace_pydantic_model(model, {"z": 2})

    def test_replace_nested_model(self):
        model = NestedModel(value=SimpleModel(x=1, y="hello"))
        replaced_model = replace_pydantic_model(model, {"value": {"x": 2}})
        assert replaced_model.value.x == 2
        assert replaced_model.value.y == "hello"

    def test_deeply_nested_model_replace(self):
        model = DeeplyNestedModel(inner=NestedModel(value=SimpleModel(x=1, y="hello")))
        replaced_model = replace_pydantic_model(model, {"inner": {"value": {"x": 2}}})
        assert replaced_model.inner.value.x == 2
        assert replaced_model.inner.value.y == "hello"

    def test_replace_with_invalid_data_type(self):
        model = SimpleModel(x=1, y="hello")
        with pytest.raises(ValidationError):
            replace_pydantic_model(model, {"x": "help"})

    def test_replace_with_multiple_dicts(self):
        model = SimpleModel(x=1, y="hello")
        replaced_model = replace_pydantic_model(model, {"x": 2, "y": "world"}, {"x": 3})
        assert replaced_model.x == 3
        assert replaced_model.y == "world"
