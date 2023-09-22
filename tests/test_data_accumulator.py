import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from rib.data_accumulator import collect_gram_matrices
from rib.hook_manager import HookedModel


def test_collect_gram_matrices():
    """Test collect gram matrices function"""

    torch.manual_seed(0)
    dtype = torch.float32
    device = "cpu"

    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear(x))

    # Setup the model, data_loader, and HookConfig
    hooked_mlp = HookedModel(TestModel())
    data = torch.randn((30, 3))
    targets = torch.ones(30)
    data_loader = DataLoader(TensorDataset(data, targets), batch_size=13)

    # Test with no module_names should raise assertion
    with pytest.raises(AssertionError):
        collect_gram_matrices(hooked_mlp, [], data_loader, device=device, dtype=dtype)

    module_names = ["linear", "relu"]

    # Run collect_gram_matrices
    gram_matrices = collect_gram_matrices(
        hooked_mlp, module_names, data_loader, device=device, dtype=dtype
    )

    # Ensure hooked_mlp.hooked_data got removed after collect_gram_matrices
    assert not hooked_mlp.hooked_data

    for module_name in module_names:
        # Ensure gram matrices are of correct size
        assert gram_matrices[module_name].size() == torch.Size([3, 3])
