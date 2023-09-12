"""Tests for the tlens_mapper module."""

from dataclasses import asdict

import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from rib.hook_manager import HookedModel
from rib.models import SequentialTransformer, SequentialTransformerConfig
from rib.models.utils import get_model_attr
from rib.tlens_mapper import model_fold_bias
from rib.utils import set_seed


@torch.inference_mode()
def _folded_bias_comparison(
    model_raw: SequentialTransformer, model_folded: SequentialTransformer, atol=1e-6
) -> None:
    """Compare the outputs of raw model and one with biases folded into its weights.

    Args:
        model_raw: The raw model.
        model_folded: The model with biases folded into its weights.
        atol: The absolute tolerance for the comparison.
    """
    input_ids = torch.randint(0, model_raw.cfg.d_vocab, size=(1, model_raw.cfg.n_ctx))
    outputA, cacheA = HookedModel(model_raw).run_with_cache(input_ids)
    outputB, cacheB = HookedModel(model_folded).run_with_cache(input_ids)

    for k in cacheA.keys():
        # Tuple of outputs for each module in the layer
        outputsA, outputsB = cacheA[k]["acts"], cacheB[k]["acts"]
        for vA, vB in zip(outputsA, outputsB):
            if vB.shape[-1] == vA.shape[-1] + 1:
                # Consider the final dimension (it should be our constant function)
                vB = vB[..., :-1]

            assert vA.shape == vB.shape, f"shape mismatch for {k}: {vA.shape} vs {vB.shape}"

            assert torch.allclose(vA, vB, atol=atol), f"WARNING: mismatched values for {k}"
    for outA, outB in zip(outputA, outputB):
        assert torch.allclose(outA, outB, atol=atol), "WARNING: mismatched output values"


def test_modular_arithmetic_folded_bias() -> None:
    """Test that the folded bias trick works for a model used for modular arithmetic.

    Floating point errors heavily accumulate here with float32 or less, so we use float64.
    """
    set_seed(42)
    cfg = {
        "n_layers": 2,
        "d_model": 129,
        "d_head": 32,
        "n_heads": 4,
        "d_mlp": 512,
        "d_vocab": 114,  # modulus + 1
        "n_ctx": 3,
        "act_fn": "gelu_new",
        "normalization_type": "LNPre",
        "dtype": torch.float64,
    }
    # Need atols to be larger for lower precision dtypes (1e3 for bfloat16, 1e-2 for float32)
    atol = 1e-8

    tlens_cfg = HookedTransformerConfig.from_dict(cfg)
    cfg = SequentialTransformerConfig(**asdict(tlens_cfg))

    node_layers = ["attn.0", "mlp_act.0"]

    model_raw = SequentialTransformer(cfg, node_layers)
    model_raw.eval()
    # Manually set all bias vectors to random values (to avoid the default of 0)

    seq_tf_keys = list(model_raw.state_dict().keys())
    # Initialise all params to random values
    for key in seq_tf_keys:
        module = get_model_attr(model_raw, key)
        if module.dtype != torch.bool:  # Ignore the preset boolean mask tensor
            # Use kaiming normal initialisation for weights
            torch.nn.init.normal_(module)

    model_folded = SequentialTransformer(cfg, node_layers)
    model_folded.load_state_dict(model_raw.state_dict())
    model_folded.fold_bias()
    model_folded.eval()

    _folded_bias_comparison(model_raw, model_folded, atol=atol)


@pytest.mark.slow()
def test_gpt2_folded_bias() -> None:
    """Test that the folded bias trick works for GPT2.

    Uses float64 to avoid floating point errors.
    """
    set_seed(42)
    model_raw = HookedTransformer.from_pretrained("gpt2")
    model_folded = HookedTransformer(cfg=model_raw.cfg, tokenizer=model_raw.tokenizer)
    model_folded.load_state_dict(model_raw.state_dict())
    model_fold_bias(model_folded)

    model_raw.eval()
    model_folded.eval()
    model_raw.to(torch.float64)
    model_folded.to(torch.float64)
    # Need to use atol=1e-3 if you want to test float32.
    _folded_bias_comparison(model_raw, model_folded, atol=1e-6)
