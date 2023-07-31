import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from rib.data_accumulator import collect_gram_matrices
from rib.hook_manager import HookConfig, HookedModel


def test_collect_gram_matrices():
    """Test collect gram matrices function"""

    torch.manual_seed(0)

    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, x):
            return self.linear(x)

    # Setup the model, data_loader, and HookConfig
    hooked_mlp = HookedModel(TestModel())
    data = torch.randn((30, 3))
    targets = torch.ones(30)
    data_loader = DataLoader(TensorDataset(data, targets), batch_size=13)

    # Test with no hook_configs should raise assertion
    with pytest.raises(AssertionError):
        collect_gram_matrices(hooked_mlp, [], data_loader, "cpu")

    hook_configs = [
        HookConfig(
            hook_name="test_forward", module_name="linear", hook_type="forward", layer_size=3
        ),
        HookConfig(
            hook_name="test_pre_forward",
            module_name="linear",
            hook_type="pre_forward",
            layer_size=3,
        ),
    ]
    device = "cpu"

    # Run collect_gram_matrices
    collect_gram_matrices(hooked_mlp, hook_configs, data_loader, device)

    for config in hook_configs:
        # Ensure hooked_data contains gram matrices for each hook_config
        assert config.hook_name in hooked_mlp.hooked_data
        assert "gram" in hooked_mlp.hooked_data[config.hook_name]

        # Ensure gram matrices are of correct size
        assert hooked_mlp.hooked_data[config.hook_name]["gram"].size() == torch.Size(
            [config.layer_size, config.layer_size]
        )
