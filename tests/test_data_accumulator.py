import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from rib.data_accumulator import collect_dataset_means, collect_gram_matrices
from rib.hook_manager import HookedModel
from rib.models.mlp import MLP, MLPConfig
from rib.utils import set_seed


def test_collect_gram_matrices():
    """Test collect gram matrices function"""

    set_seed(0)
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


def test_collect_dataset_means():
    """Test collect dataset means function"""

    set_seed(0)
    dtype = torch.float32
    device = "cpu"

    config = MLPConfig(
        input_size=3,
        output_size=1,
        hidden_sizes=[5],
        activation_fn="relu",
    )
    mlp = MLP(config)
    b0 = mlp.layers[0].b[:]
    mlp.fold_bias()

    # Setup the model, data_loader, and HookConfig
    hooked_mlp = HookedModel(mlp)
    data = torch.zeros(10, 3)
    targets = torch.ones(10)
    zero_data_loader = DataLoader(TensorDataset(data, targets), batch_size=7)

    # Test with no module_names should raise assertion
    with pytest.raises(AssertionError):
        collect_dataset_means(hooked_mlp, [], zero_data_loader, device=device, dtype=dtype)

    module_names = ["layers.0", "layers.1"]
    means, index_pos = collect_dataset_means(
        hooked_mlp, module_names, zero_data_loader, device=device, dtype=dtype
    )

    # Ensure hooked_mlp.hooked_data got removed after collect_gram_matrices
    assert not hooked_mlp.hooked_data

    # The acts before layer 0 are the inputs with bias folded in. The inputs are all zeros here.
    assert torch.allclose(means["layers.0"], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    # The acts before layer 1 will be `relu(Wx + b)`. This is relu(b) since x is all zeros.
    mean_l1 = torch.concatenate([torch.clip(b0, min=0), torch.tensor([1.0])])
    assert torch.allclose(means["layers.1"], mean_l1)
    assert index_pos["layers.0"] == torch.tensor([mlp.input_size])
    assert index_pos["layers.1"] == torch.tensor([mlp.hidden_sizes[0]])
