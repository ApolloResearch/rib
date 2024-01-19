"""Test that the tlens model and a corresponding SequentialTransformer model are the same."""

from dataclasses import asdict

import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from rib.hook_fns import acts_forward_hook_fn
from rib.hook_manager import Hook, HookedModel
from rib.models import SequentialTransformer, SequentialTransformerConfig
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
                (4): AttentionIn()
                (5): AttentionOut()
                (6): Add()
                (7): LayerNormPre()
                (8): MLPIn()
            )
            (section_0): MultiSequential(
                (0): MLPAct()
                (1): MLPOut()
                (2): Add()
                (3): LayerNormPre()
                (4): AttentionIn()
                (5): AttentionOut()
                (6): Add()
                (7): LayerNormPre()
                (8): MLPIn()
                (9): MLPAct()
                (10): MLPOut()
                (11): Add()
                (12): LayerNormPre()
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
    dtype = torch.float32
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
        "dtype": dtype,
        "device": "cpu",
    }

    tlens_cfg = HookedTransformerConfig.from_dict(cfg)
    tlens_model = HookedTransformer(tlens_cfg)
    tlens_model.eval()

    cfg = SequentialTransformerConfig(**asdict(tlens_cfg))

    node_layers = ["mlp_act.0", "unembed"]
    seq_model_raw = SequentialTransformer(cfg, node_layers)
    assert seq_model_raw.cfg.dtype == dtype
    seq_model_raw.eval()
    seq_model_raw.load_tlens_weights(
        tlens_model, positional_embedding_type=cfg.positional_embedding_type
    )

    seq_model = HookedModel(seq_model_raw)

    input_ids = torch.randint(0, tlens_model.cfg.d_vocab, size=(1, tlens_model.cfg.n_ctx))

    # Mapping from some tlens cache keys to SequentialTransformer cache keys (and their tuple index)
    mappings = {
        "blocks.0.hook_resid_pre": {
            "section_id": "sections.pre.2",
            "tuple_idx": 0,
        },
        "blocks.0.hook_resid_mid": {
            "section_id": "sections.pre.6",
            "tuple_idx": 0,
        },
        "blocks.1.hook_mlp_out": {
            "section_id": "sections.section_0.10",
            "tuple_idx": 0,
        },
    }
    outputA, cacheA = tlens_model.run_with_cache(input_ids)
    # Only store the activations we care about (storing all activations may cause CI failure)
    cacheA = [cacheA[key] for key in mappings]

    outputB, cacheB = seq_model.run_with_cache(input_ids)
    cacheB = [cacheB[v["section_id"]]["acts"][v["tuple_idx"]] for v in mappings.values()]

    assert torch.equal(outputA, outputB[0]), "Outputs are not equal"
    for i, (tlens_act, seq_act) in enumerate(zip(cacheA, cacheB)):
        assert torch.equal(tlens_act, seq_act), f"Activations are not equal for mapping index {i}"


def pretrained_lm_comparison(hf_model_str: str, mappings: dict[str, dict[str, str]]) -> None:
    """Test that a pretrained lm in tlens and SequentialTransformer give the same outputs and
    internal activations.

    Args:
        hf_model_str: The model to test.
        mappings: A mapping from some tlens cache keys to SequentialTransformer cache keys.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    tlens_model = HookedTransformer.from_pretrained(hf_model_str, device=device, torch_dtype=dtype)
    tlens_model.eval()

    seq_cfg = SequentialTransformerConfig(**asdict(tlens_model.cfg))
    node_layers = ["ln2.1", "unembed"]
    seq_model_raw = SequentialTransformer(seq_cfg, node_layers).to(device=device)
    assert seq_model_raw.cfg.dtype == dtype
    seq_model_raw.eval()

    seq_model_raw.load_tlens_weights(
        tlens_model, positional_embedding_type=seq_cfg.positional_embedding_type
    )
    seq_model = HookedModel(seq_model_raw)

    input_ids = torch.randint(0, tlens_model.cfg.d_vocab, size=(1, tlens_model.cfg.n_ctx))

    # Collect activations from the tlens model
    tlens_cache = []

    def tlens_store_activations_hook(output, hook):
        tlens_cache.append(output.detach().cpu())

    tlens_hooks = [(name, tlens_store_activations_hook) for name in mappings]
    tlens_output = tlens_model.run_with_hooks(input_ids, fwd_hooks=tlens_hooks)

    # Collect activations from the sequential transformer model
    seq_hooks: list[Hook] = []
    for section_id in [v["section_id"] for v in mappings.values()]:
        seq_hooks.append(
            Hook(
                name=section_id,
                data_key="acts",
                fn=acts_forward_hook_fn,
                module_name=section_id,
            )
        )
    seq_output = seq_model(input_ids, hooks=seq_hooks)[0]
    seq_cache = [
        seq_model.hooked_data[v["section_id"]]["acts"][v["tuple_idx"]] for v in mappings.values()
    ]
    assert torch.equal(tlens_output, seq_output), "Outputs are not equal"
    for i, (tlens_act, seq_act) in enumerate(zip(tlens_cache, seq_cache)):
        assert torch.equal(tlens_act, seq_act), f"Activations are not equal for mapping index {i}"


@pytest.mark.slow()
@pytest.mark.parametrize("model_str", ["gpt2", "tiny-stories-1M"])
def test_gpt_conversion(model_str):
    """Test that gpt2 and tiny-stories have the same ouputs and inernal actiavtions in tlens as SequentialTransformer. The architecture of gpt2 and tiny-stories is close enough for the
    mapping to be the same.

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
                    (4): AttentionIn()
                    (5): AttentionOut()
                    (6): Add()
                    (7): LayerNormPre()
                    (8): MLPIn()
                    (9): MLPAct()
                    (10): MLPOut()
                    (11): Add()
                    (12): LayerNormPre()
                    (13): AttentionIn()
                    (14): AttentionOut()
                    (15): Add()
                )
                (section_0): MultiSequential(
                    (0): LayerNormPre()
                    (1): MLPIn()
                    (2): MLPAct()
                    (3): MLPOut()
                    (4): Add()
                    ...
                    (95): LayerNormPre()
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
    # Mapping from some tlens cache keys to SequentialTransformer cache keys
    # The tuple_idx is the index of the module's output to use
    mappings = {
        "blocks.0.hook_resid_pre": {
            "section_id": "sections.pre.2",
            "tuple_idx": 0,
        },
        "blocks.3.hook_resid_mid": {
            "section_id": "sections.section_0.17",
            "tuple_idx": 0,
        },
        "blocks.6.hook_resid_post": {
            "section_id": "sections.section_0.49",
            "tuple_idx": 0,
        },
    }
    pretrained_lm_comparison(model_str, mappings)


@pytest.mark.slow()
def test_pythia_conversion():
    """Test that pythia-14m in tlens and SequentialTransformer give the same outputs and internal
    activations.

    The SequentialTransformer with node layer node_layers = ["ln2.1", "unembed"] has the
    following architecture:
    HookedModel(
        (model): SequentialTransformer(
            (sections): ModuleDict(
                (pre): MultiSequential(
                    (0): Embed()
                    (1): LayerNormPre()
                    (2): AttentionIn()
                    (3): AttentionOut()
                    (4): Add()
                    (5): DualLayerNormPre()
                    (6): MLPIn()
                    (7): MLPAct()
                    (8): MLPOut()
                    (9): Add()
                    (10): LayerNormPre()
                    (11): AttentionIn()
                    (12): AttentionOut()
                    (13): Add()
                )
                (section_0): MultiSequential(
                    (0): DualLayerNormPre()
                    (1): MLPIn()
                    (2): MLPAct()
                    (3): MLPOut()
                    (4): Add()
                    (5): LayerNormPre()
                    ...
                    (41): LayerNormPre()
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
    # Mapping from some tlens cache keys to SequentialTransformer cache keys
    # The tuple_idx is the index of the module's output to use
    mappings = {
        "blocks.0.hook_resid_pre": {
            "section_id": "sections.pre.0",
            "tuple_idx": 0,
        },
        "blocks.3.hook_mlp_out": {
            "section_id": "sections.section_0.21",
            "tuple_idx": 0,
        },
        "blocks.4.hook_resid_post": {
            "section_id": "sections.section_0.31",
            "tuple_idx": 0,
        },
    }
    pretrained_lm_comparison("pythia-14m", mappings)
