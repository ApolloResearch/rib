"""Test that the tlens model and a corresponding SequentialTransformer model are the same."""

from dataclasses import asdict

import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from rib.hook_manager import HookedModel
from rib.models import SequentialTransformer, SequentialTransformerConfig
from rib.models.sequential_transformer.converter import convert_tlens_weights
from rib.utils import set_seed


def test_modular_arithmetic_conversion() -> None:
    """Test that a transformer with the same architecture as used in modular arithmetic maps from
    tlens to sequential transformer correctly.

    The sequential transformer with node layer node_layers = ["mlp_act.0", "unembed"] has the
    following architecture:
    HookedModel(
        (model): SequentialTransformer(
            (sections): ModuleDict(
            (pre): MultiSequential(
                (0): Embed()
                (1): PosEmbed()
                (2): Add()
                (3): LayerNormPre()
                (4): Attention()
                (5): Add()
                (6): LayerNormPre()
                (7): MLPIn()
            )
            (section_0): MultiSequential(
                (0): MLPAct()
                (1): MLPOut()
                (2): Add()
                (3): LayerNormPre()
                (4): Attention()
                (5): Add()
                (6): LayerNormPre()
                (7): MLPIn()
                (8): MLPAct()
                (9): MLPOut()
                (10): Add()
                (11): LayerNormPre()
            )
            (section_1): MultiSequential(
                (0): Unembed()
            )
            )
        )
    )

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
    tlens_model = HookedTransformer(tlens_cfg)
    tlens_model.to(torch.float64)
    tlens_model.to("cpu")
    tlens_model.eval()

    cfg = SequentialTransformerConfig(**asdict(tlens_cfg))

    node_layers = ["mlp_act.0", "unembed"]
    seq_model_raw = SequentialTransformer(cfg, node_layers).to(torch.float64)
    seq_model_raw.eval()
    # Load the transformer-lens weights into the sequential transformer model
    state_dict = convert_tlens_weights(list(seq_model_raw.state_dict().keys()), tlens_model)
    seq_model_raw.load_state_dict(state_dict)
    seq_model = HookedModel(seq_model_raw)

    input_ids = torch.randint(0, tlens_model.cfg.d_vocab, size=(1, tlens_model.cfg.n_ctx))

    # Mapping from some tlens cache keys to SequentialTransformer cache keys (and their tuple index)
    mappings = {
        "blocks.0.hook_resid_pre": {
            "seq_key": "sections.pre.2",
            "tuple_idx": 0,
        },
        "blocks.0.hook_resid_mid": {
            "seq_key": "sections.pre.5",
            "tuple_idx": 0,
        },
        "blocks.1.hook_mlp_out": {
            "seq_key": "sections.section_0.9",
            "tuple_idx": 1,
        },
    }
    outputA, cacheA = tlens_model.run_with_cache(input_ids)
    # Only store the activations we care about (storing all activations may cause CI failure)
    cacheA = [cacheA[key] for key in mappings]

    outputB, cacheB = seq_model.run_with_cache(input_ids)
    cacheB = [cacheB[v["seq_key"]]["acts"][v["tuple_idx"]] for v in mappings.values()]

    assert torch.allclose(outputA, outputB[0], atol=atol), "Outputs are not equal"
    for i, (tlens_act, seq_act) in enumerate(zip(cacheA, cacheB)):
        assert torch.allclose(
            tlens_act, seq_act, atol=atol
        ), f"Activations are not equal for mapping index {i}"


@pytest.mark.slow()
def test_gpt2_conversion():
    """Test that gpt2 in tlens and SequentialTransformer give the same outputs and internal
    activations.

    The SequentialTransformer with node layer node_layers = ["ln2.1", "unembed"] has the
    following architecture:
    HookedModel(
        (model): SequentialTransformer(
            (sections): ModuleDict(
            (pre): MultiSequential(
                (0): Embed()
                (1): PosEmbed()
                (2): Add()
                (3): LayerNormPre()
                (4): Attention()
                (5): Add()
                (6): LayerNormPre()
                (7): MLPIn()
                (8): MLPAct()
                (9): MLPOut()
                (10): Add()
                (11): LayerNormPre()
                (12): Attention()
                (13): Add()
                (14): LayerNormPre()
            )
            (section_0): MultiSequential(
                (0): MLPIn()
                (1): MLPAct()
                ...
                (85): LayerNormPre()
            )
            (section_1): MultiSequential(
                (0): Unembed()
            )
            )
        )
    )

    Floating point errors heavily accumulate here with float32 or less, so we use float64.
    """
    set_seed(42)
    # Need atols to be larger for lower precision dtypes (1e3 for bfloat16, 1e-2 for float32)
    atol = 1e-8

    tlens_model = HookedTransformer.from_pretrained("gpt2")
    tlens_model.to(torch.float64)
    tlens_model.to("cpu")
    tlens_model.eval()

    seq_cfg = SequentialTransformerConfig(**asdict(tlens_model.cfg))
    node_layers = ["ln2.1", "unembed"]
    seq_model_raw = SequentialTransformer(seq_cfg, node_layers).to(torch.float64)
    seq_model_raw.eval()
    # Load the transformer-lens weights into the sequential transformer model
    state_dict = convert_tlens_weights(list(seq_model_raw.state_dict().keys()), tlens_model)
    seq_model_raw.load_state_dict(state_dict)
    seq_model = HookedModel(seq_model_raw)

    input_ids = torch.randint(0, tlens_model.cfg.d_vocab, size=(1, tlens_model.cfg.n_ctx))
    # Mapping from some tlens cache keys to SequentialTransformer cache keys (and their tuple index)
    mappings = {
        "blocks.0.hook_resid_pre": {
            "seq_key": "sections.pre.2",
            "tuple_idx": 0,
        },
        "blocks.3.hook_resid_mid": {
            "seq_key": "sections.section_0.15",
            "tuple_idx": 0,
        },
        "blocks.8.hook_resid_post": {
            "seq_key": "sections.section_0.60",
            "tuple_idx": 0,
        },
    }
    outputA, cacheA = tlens_model.run_with_cache(input_ids)
    # Only store the activations we care about (storing all activations may cause CI failure)
    cacheA = [cacheA[key] for key in mappings]

    outputB, cacheB = seq_model.run_with_cache(input_ids)
    cacheB = [cacheB[v["seq_key"]]["acts"][v["tuple_idx"]] for v in mappings.values()]

    assert torch.allclose(outputA, outputB[0], atol=atol), "Outputs are not equal"
    for i, (tlens_act, seq_act) in enumerate(zip(cacheA, cacheB)):
        assert torch.allclose(
            tlens_act, seq_act, atol=atol
        ), f"Activations are not equal for mapping index {i}"
