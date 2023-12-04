"""Test that folding in the bias works for Sequential Transformer."""

from dataclasses import asdict
from typing import Literal

import einops
import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from rib.hook_manager import HookedModel
from rib.models import SequentialTransformer, SequentialTransformerConfig
from rib.models.sequential_transformer.converter import convert_tlens_weights
from rib.models.utils import get_model_attr
from rib.utils import set_seed


@torch.inference_mode()
def _folded_bias_comparison(
    model_raw: SequentialTransformer,
    model_folded: SequentialTransformer,
    atol=1e-8,
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
                # Remove the final dimension (it should be our constant function)
                vB = vB[..., :-1]
            elif vB.shape[-1] == vA.shape[-1] + model_folded.cfg.n_heads:
                # We add a constant dimension per attention head to v and then concatenate heads
                # This is a bit more complicated to cut out
                vB = einops.rearrange(
                    vB,
                    "... pos (head_index d_head_v) -> ... pos head_index d_head_v",
                    head_index=model_folded.cfg.n_heads,
                )
                assert torch.equal(vB[..., :, -1], torch.ones_like(vB[..., :, -1]))
                vB = vB[..., :-1]
                vB = einops.rearrange(
                    vB, "... pos head_index d_head_v -> ... pos (head_index d_head_v)"
                )

            assert vA.shape == vB.shape, f"shape mismatch for {k}: {vA.shape} vs {vB.shape}"

            assert not torch.isnan(vA).any(), f"NaNs in {k}"
            assert not torch.isnan(vB).any(), f"NaNs in {k}"

            assert torch.allclose(vA, vB, atol=atol), f"WARNING: mismatched values for {k}"

    for outA, outB in zip(outputA, outputB):
        assert torch.allclose(outA, outB, atol=atol), "WARNING: mismatched output values"


def test_modular_arithmetic_folded_bias() -> None:
    """Test that the folded bias trick works for a model used for modular arithmetic."""
    set_seed(42)
    dtype = torch.float32
    # Works with atol=0 for float64 and atol=1e-3 for float32
    atol = 1e-3
    cfg = {
        "n_layers": 2,  # If going up to 50 layers, need atol=1e-1
        "d_model": 129,
        "d_head": 32,
        "n_heads": 4,
        "d_mlp": 512,
        "d_vocab": 114,  # modulus + 1
        "n_ctx": 3,
        "act_fn": "gelu_new",  # Even though we use relu, this will test our root_one conversion
        "normalization_type": "LNPre",
        "dtype": dtype,
    }

    tlens_cfg = HookedTransformerConfig.from_dict(cfg)
    cfg = SequentialTransformerConfig(**asdict(tlens_cfg))

    node_layers = ["attn_in.0", "mlp_act.0"]

    model_raw = SequentialTransformer(cfg, node_layers)
    model_raw.eval()
    # Manually set all params to random values (but not the buffers which include the mask tensor)
    seq_param_names = [named_param[0] for named_param in model_raw.named_parameters()]

    # Initialise all params to random values
    for key in seq_param_names:
        module = get_model_attr(model_raw, key)
        # Use kaiming normal initialisation for weights
        torch.nn.init.normal_(module)

    model_folded = SequentialTransformer(cfg, node_layers)
    model_folded.load_state_dict(model_raw.state_dict())
    model_folded.fold_bias()
    model_folded.eval()

    _folded_bias_comparison(model_raw, model_folded, atol=atol)


def pretrained_lm_folded_bias_comparison(
    hf_model_str: str,
    node_layers: list[str],
    positional_embedding_type: Literal["standard", "rotary"],
    atol: float = 1e-11,
    dtype: torch.dtype = torch.float64,
) -> None:
    """Test that the folded bias trick works for a pretrained language model.

    Uses float64 to avoid floating point errors.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tlens_model = HookedTransformer.from_pretrained(hf_model_str, device="cpu", torch_dtype=dtype)
    cfg = SequentialTransformerConfig(**asdict(tlens_model.cfg))
    assert cfg.dtype == dtype
    model_raw = SequentialTransformer(cfg, node_layers)
    # Load the transformer-lens weights into the sequential transformer model
    state_dict = convert_tlens_weights(
        seq_model=model_raw,
        tlens_model=tlens_model,
        positional_embedding_type=positional_embedding_type,
    )
    model_raw.load_state_dict(state_dict)
    model_raw.to(device=device)
    model_raw.eval()
    model_folded = SequentialTransformer(cfg, node_layers)
    model_folded.load_state_dict(state_dict)
    model_folded.fold_bias()
    model_folded.to(device=device)
    model_folded.eval()

    _folded_bias_comparison(model_raw, model_folded, atol=atol)


@pytest.mark.slow()
@pytest.mark.parametrize("model_str", ["gpt2", "tiny-stories-1M"])
def test_gpt_folded_bias(model_str) -> None:
    """Test that the folded bias trick works for GPT2 and tiny-stories."""
    set_seed(42)
    dtype = torch.float32
    # float64 can do 1e-10, float32 can do 1e-3. We use float32 in this test because it's faster and
    # to avoid OOM errors in the Github runner.
    atol = 1e-3
    node_layers = ["attn_in.0", "mlp_act.0"]
    pretrained_lm_folded_bias_comparison(
        hf_model_str=model_str,
        node_layers=node_layers,
        positional_embedding_type="standard",
        atol=atol,
        dtype=dtype,
    )


@pytest.mark.slow()
def test_pythia_folded_bias() -> None:
    """Test that the folded bias trick works for Pythia."""
    set_seed(42)
    dtype = torch.float64
    # float64 can do atol=1e-11, float32 can do atol=1e2.
    atol = 1e-11
    node_layers = ["mlp_in.1", "add_resid2.3"]
    pretrained_lm_folded_bias_comparison(
        hf_model_str="pythia-14m",
        node_layers=node_layers,
        positional_embedding_type="rotary",
        atol=atol,
        dtype=dtype,
    )
