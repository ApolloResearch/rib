from inspect import signature
from unittest.mock import Mock

import pytest
import torch

from rib.hook_fns import gram_forward_hook_fn, gram_pre_forward_hook_fn
from rib.hook_manager import Hook, HookedModel
from rib.loader import load_sequential_transformer


@pytest.fixture
def model():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, x):
            return self.linear(x)

    return TestModel()


def test_register_hook_with_bad_name():
    """Check that a hook fn without 'forward' or 'pre_forward' in the name raises an error"""
    mock_fn = Mock(return_value=None)
    # A forward hook should be a function with 'forward' in the name and have a signature of
    # (module, inputs, output, ...)
    mock_fn.__signature__ = signature(gram_forward_hook_fn)

    with pytest.raises(AssertionError):
        # Check that we get an assertion error without 'forward' in the name
        mock_fn.__name__ = "mock_fn"
        Hook(
            name="test_forward",
            data_key="gram",
            fn=mock_fn,
            module_name="linear",
            fn_kwargs={"dataset_size": 1},
        )
        # Check that we get an assertion error with 'pre_forward' in the name
        mock_fn.__name__ = "mock_fn_pre_forward"
        Hook(
            name="test_forward",
            data_key="gram",
            fn=mock_fn,
            module_name="linear",
            fn_kwargs={"dataset_size": 1},
        )
        # Check that we get an assertion error with 'forward' in the name but the signature is wrong
        mock_fn.__name__ = "mock_fn_forward"
        mock_fn.__signature__ = signature(gram_pre_forward_hook_fn)
        Hook(
            name="test_forward",
            data_key="gram",
            fn=mock_fn,
            module_name="linear",
            fn_kwargs={"dataset_size": 1},
        )


def test_hooked_model_add_and_remove_hooks(model):
    """Test adding and removing hooks in the HookedModel"""
    torch.manual_seed(0)
    data = torch.randn((1, 3))
    hooked_model = HookedModel(model)
    mock_fn = Mock(return_value=None)
    # Ensure mock_fn has the same signature as gram_forward_hook_fn so that the hook_type is set correctly
    mock_fn.__signature__ = signature(gram_forward_hook_fn)
    mock_fn.__name__ = "mock_fn_forward"

    hooks = [
        Hook(
            name="test_forward",
            data_key="gram",
            fn=mock_fn,
            module_name="linear",
            fn_kwargs={"dataset_size": 3},
        ),
    ]

    # Register mock hook and run forward pass
    result = hooked_model(data, hooks=hooks)

    # Ensure hook function was called
    mock_fn.assert_called_once()

    # Ensure all hooks are removed after forward pass
    assert not hooked_model.forward_hook_handles

    # Ensure output matches expected output from base model (gram hook does not modify output)
    assert torch.allclose(result, model(data))


def test_gram_forward_hook_fn_accumulates_over_forward_passes(model):
    """Test the gram_forward_hook_fn accumulates the gram matrix over multiple forward passes"""
    torch.manual_seed(0)
    data = torch.randn((1, 3))
    hooked_model = HookedModel(model)

    # Register hook function
    hooks = [
        Hook(
            name="test_forward",
            data_key="gram",
            fn=gram_forward_hook_fn,
            module_name="linear",
            fn_kwargs={"dataset_size": 4},
        ),
    ]

    # Run forward passes with the HookedModel
    hooked_model(data, hooks=hooks)
    data_2 = torch.randn((1, 3))
    hooked_model(data_2, hooks=hooks)

    # Compute the expected Gram matrix (and scale by 1 / dataset_size)
    output_1 = model.linear(data)
    output_2 = model.linear(data_2)
    expected_gram = (output_1.T @ output_1 + output_2.T @ output_2) / 4

    # Compare hooked_data with the expected gram matrix
    assert torch.allclose(hooked_model.hooked_data["test_forward"]["gram"], expected_gram)


@pytest.mark.slow()
def test_pythia_attn_causal_mask():
    """Test that the causal mask is applied to the attention scores in the Pythia model

    This test mainly exists to make sure the attention scores hook works sensibly.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    module_id = "attn_out.0"
    batch_size = 2
    n_ctx = 2048

    seq_model = load_sequential_transformer(
        node_layers=[module_id],
        last_pos_module_type=None,
        tlens_pretrained="pythia-14m",
        tlens_model_path=None,
        fold_bias=True,
        dtype=dtype,
        device=device,
    )
    seq_model.eval()
    hooked_model = HookedModel(seq_model)

    # Create a fake data sample and run the model
    input_ids = torch.randint(0, seq_model.cfg.d_vocab, (batch_size, n_ctx), device=device)
    with torch.inference_mode():
        _, cache = hooked_model.run_with_cache(input_ids)

    # Check that the attention scores are masked correctly
    for k in cache.keys():
        if "attention_scores" in k:
            attn_scores = cache[k]["acts"][0]
            mask = torch.triu(torch.ones_like(attn_scores), diagonal=1)
            assert torch.allclose(
                attn_scores[mask == 1], torch.tensor(-float("inf"), dtype=torch.float64)
            ), f"Attention scores for {k} are not masked correctly"
