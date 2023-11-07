"""Test that folding in the bias works for Sequential Transformer."""

from dataclasses import asdict
from typing import Literal

import einops
import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from rib.hook_manager import HookedModel
from rib.models import SequentialTransformer, SequentialTransformerConfig
from rib.models.sequential_transformer.components import AttentionIn
from rib.models.sequential_transformer.converter import convert_tlens_weights
from rib.models.utils import get_model_attr
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
            if vB.shape[-1] == vA.shape[-1] + model_folded.cfg.n_heads:
                # We add a constant dimension per attention head to v and then concatenate heads
                # This is a bit more complicated to cut out
                vB = einops.rearrange(
                    vB,
                    "... pos (head_index d_head_v) -> ... pos head_index d_head_v",
                    head_index=model_folded.cfg.n_heads,
                )
                assert torch.allclose(vB[..., :, -1], torch.tensor(1, dtype=vB.dtype))
                vB = vB[..., :, :-1]
                vB = einops.rearrange(
                    vB, "... pos head_index d_head_v -> ... pos (head_index d_head_v)"
                )

            assert vA.shape == vB.shape, f"shape mismatch for {k}: {vA.shape} vs {vB.shape}"

            # Check if this is a Pythia attention module:
            stages = k.split(".")
            if (
                isinstance(
                    getattr(getattr(model_raw, stages[0]), stages[1])[int(stages[2])], AttentionIn
                )
                and model_raw.cfg.rotary_dim is not None
            ):
                assert torch.allclose(vA, vB, atol=1e-3), f"WARNING: mismatched values for {k}"
            else:
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

    node_layers = ["attn_in.0", "mlp_act.0"]

    model_raw = SequentialTransformer(cfg, node_layers)
    model_raw.eval()
    # Manually set all bias vectors to random values (to avoid the default of 0)

    seq_param_names = list(model_raw.state_dict().keys())
    # Initialise all params to random values
    for key in seq_param_names:
        module = get_model_attr(model_raw, key)
        if module.dtype != torch.bool:  # Ignore the preset boolean mask tensor
            # Use kaiming normal initialisation for weights
            torch.nn.init.normal_(module)

    model_folded = SequentialTransformer(cfg, node_layers)
    model_folded.load_state_dict(model_raw.state_dict())
    model_folded.fold_bias()
    model_folded.eval()

    _folded_bias_comparison(model_raw, model_folded, atol=atol)


test_modular_arithmetic_folded_bias()


def pretrained_lm_folded_bias_comparison(
    hf_model_str: str,
    node_layers: list[str],
    positional_embedding_type: Literal["standard", "rotary"],
    atol: float = 1e-6,
) -> None:
    """Test that the folded bias trick works for a pretrained language model.

    Uses float64 to avoid floating point errors.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tlens_model = HookedTransformer.from_pretrained(hf_model_str)
    cfg = SequentialTransformerConfig(**asdict(tlens_model.cfg))
    model_raw = SequentialTransformer(cfg, node_layers)
    # Load the transformer-lens weights into the sequential transformer model
    state_dict = convert_tlens_weights(
        seq_param_names=list(model_raw.state_dict().keys()),
        tlens_model=tlens_model,
        positional_embedding_type=positional_embedding_type,
    )
    model_raw.load_state_dict(state_dict)
    model_raw.to(torch.float64)
    model_raw.to(device=device)
    model_raw.eval()
    model_folded = SequentialTransformer(cfg, node_layers)
    model_folded.load_state_dict(state_dict)
    model_folded.fold_bias()
    model_folded.to(torch.float64)
    model_folded.to(device=device)
    model_folded.eval()

    _folded_bias_comparison(model_raw, model_folded, atol=atol)


@pytest.mark.slow()
def test_gpt2_folded_bias() -> None:
    """Test that the folded bias trick works for GPT2."""
    set_seed(42)
    node_layers = ["attn_in.0", "mlp_act.0"]
    pretrained_lm_folded_bias_comparison(
        hf_model_str="gpt2",
        node_layers=node_layers,
        positional_embedding_type="standard",
        atol=1e-5,
    )


@pytest.mark.slow()
def test_pythia_folded_bias() -> None:
    """Test that the folded bias trick works for Pythia."""
    set_seed(42)
    node_layers = ["mlp_in.1", "add_resid2.3"]
    pretrained_lm_folded_bias_comparison(
        hf_model_str="pythia-14m",
        node_layers=node_layers,
        positional_embedding_type="rotary",
        atol=1e-5,
    )
