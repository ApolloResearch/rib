from unittest.mock import Mock

import pytest
import torch

from rib.hook_manager import Hook, HookedModel
from rib.hook_registry import HOOK_REGISTRY


@pytest.fixture
def test_model():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, x):
            return self.linear(x)

    return TestModel()


def test_hooked_model_add_and_remove_hooks(test_model):
    """Test adding and removing hooks in the HookedModel"""
    torch.manual_seed(0)
    data = torch.randn((1, 3))
    hooked_model = HookedModel(test_model)
    mock_fn = Mock(return_value=None)

    with pytest.raises(ValueError):
        # Should raise ValueError if hook function is not registered
        hooks = [
            Hook(name="test_forward", data_key="gram", fn_name="mock_hook", module_name="linear"),
        ]

    # Register mock hook function
    HOOK_REGISTRY.update({"mock_hook": (mock_fn, "forward")})
    hooks = [
        Hook(name="test_forward", data_key="gram", fn_name="mock_hook", module_name="linear"),
    ]

    # Register mock hook and run forward pass
    result = hooked_model(data, hooks=hooks)

    # Ensure hook function was called
    mock_fn.assert_called_once()

    # Ensure all hooks are removed after forward pass
    assert not hooked_model.hook_handles

    # Ensure output matches expected output from base model (gram hook does not modify output)
    assert torch.allclose(result, test_model(data))


def test_gram_forward_hook_fn_accumulates_over_forward_passes(test_model):
    """Test the gram_forward_hook_fn accumulates the gram matrix over multiple forward passes"""
    torch.manual_seed(0)
    data = torch.randn((1, 3))
    hooked_model = HookedModel(test_model)

    # Register hook function
    hooks = [
        Hook(
            name="test_forward",
            data_key="gram",
            fn_name="gram_forward_hook_fn",
            module_name="linear",
        ),
    ]

    # Run forward passes with the HookedModel
    hooked_model(data, hooks=hooks)
    data_2 = torch.randn((1, 3))
    hooked_model(data_2, hooks=hooks)

    # Compute the expected Gram matrix
    output_1 = test_model.linear(data)
    output_2 = test_model.linear(data_2)
    expected_gram = output_1.T @ output_1 + output_2.T @ output_2

    # Compare hooked_data with the expected gram matrix
    assert torch.allclose(hooked_model.hooked_data["test_forward"]["gram"], expected_gram)
