from typing import TypeVar

import pytest
import torch

from rib.hook_manager import HookedModel
from rib.loader import load_sequential_transformer
from rib.models import SequentialTransformer, SequentialTransformerConfig
from rib.models.sequential_transformer.transformer import MultiSequential
from rib.models.utils import create_list_partitions
from rib.utils import set_seed

T = TypeVar("T")


class TestCreatePartitions:
    @pytest.mark.parametrize(
        "in_list, sub_list, expected_output",
        [
            (
                ["embed", "pos_embed", "add_embed", "ln1.0", "attn.0", "add_resid1.0"],
                ["ln1.0", "add_resid1.0"],
                [["embed", "pos_embed", "add_embed"], ["ln1.0", "attn.0"], ["add_resid1.0"]],
            ),
            # Elements in sub_list are at the start and end of in_list
            (["a", "b", "c", "d"], ["a", "d"], [["a", "b", "c"], ["d"]]),
            # All elements in in_list are in sub_list
            ([0, 1, 2], [0, 1, 2], [[0], [1], [2]]),
        ],
    )
    def test_create_list_partitions(
        self, in_list: list[T], sub_list: list[T], expected_output: list[list[T]]
    ):
        result = create_list_partitions(in_list, sub_list)
        assert result == expected_output

    @pytest.mark.parametrize(
        "in_list, sub_list",
        [(["a", "b", "c"], ["a", "x"])],
    )
    def test_create_list_partitions_invalid(self, in_list: list[T], sub_list: list[T]):
        with pytest.raises(AssertionError):
            create_list_partitions(in_list, sub_list)


class TestCreateSectionIdToModuleIdMapping:
    @staticmethod
    def compare_mappings(
        seq_model: SequentialTransformer, expected_mappings: list[tuple[str, str]]
    ):
        for section_id, module_id in expected_mappings:
            assert seq_model.section_id_to_module_id[section_id] == module_id, (
                f"section_id_to_module_id[{section_id}]={seq_model.section_id_to_module_id[section_id]} "
                f"!= {module_id}"
            )
            assert seq_model.module_id_to_section_id[module_id] == section_id, (
                f"module_id_to_section_id[{module_id}]={seq_model.module_id_to_section_id[module_id]} "
                f"!= {section_id}"
            )

    def test_create_section_id_to_module_id_mapping_two_layer(self):
        node_layers = ["mlp_act.0", "unembed"]

        cfg = {
            "n_layers": 2,
            "d_model": 6,
            "d_head": 2,
            "n_heads": 3,
            "d_mlp": 12,
            "d_vocab": 3,  # modulus + 1
            "n_ctx": 3,
            "act_fn": "gelu_new",
            "normalization_type": "LNPre",
            "eps": 1e-5,
            "dtype": torch.bfloat16,
            "use_attn_scale": True,
            "use_split_qkv_input": False,
            "positional_embedding_type": "standard",
            "rotary_dim": None,
            "parallel_attn_mlp": False,
            "original_architecture": None,
            "use_local_attn": False,
        }
        cfg = SequentialTransformerConfig(**cfg)
        seq_model = SequentialTransformer(cfg, node_layers)

        # Obtained by manually inspecting the model and counting the layers
        expected_mappings = [
            ("sections.pre.2", "add_embed"),
            ("sections.pre.5", "attn_out.0"),
            ("sections.section_0.9", "mlp_act.1"),
        ]
        TestCreateSectionIdToModuleIdMapping.compare_mappings(seq_model, expected_mappings)

    def test_create_section_id_to_module_id_mapping_gpt2(self):
        # Copying the GPT2 config that can be obtained by running the slow code below
        # from transformer_lens.loading_from_pretrained import get_pretrained_model_config
        # from dataclasses import asdict
        # tlens_cfg = get_pretrained_model_config("gpt2")
        # print(SequentialTransformerConfig(**asdict(tlens_cfg)).dict())

        cfg_dict = {
            "n_layers": 12,
            "d_model": 768,
            "d_head": 64,
            "n_heads": 12,
            "d_mlp": 3072,
            "d_vocab": 50257,
            "n_ctx": 1024,
            "act_fn": "gelu_new",
            "normalization_type": "LNPre",  # GPT2 actually uses LN but tlens converts it to LNPre
            "eps": 1e-05,
            "dtype": torch.float32,
            "use_attn_scale": True,
            "use_split_qkv_input": False,
            "positional_embedding_type": "standard",
            "rotary_dim": None,
            "parallel_attn_mlp": False,
            "original_architecture": "GPT2LMHeadModel",
            "use_local_attn": False,
        }

        cfg = SequentialTransformerConfig(**cfg_dict)
        node_layers = ["add_embed", "ln2.4", "mlp_act.7"]
        seq_model = SequentialTransformer(cfg, node_layers)

        # Obtained by manually inspecting the model and counting the layers
        expected_mappings = [
            ("sections.pre.1", "pos_embed"),
            ("sections.section_0.0", "add_embed"),
            ("sections.section_0.17", "mlp_out.1"),
            ("sections.section_2.9", "mlp_act.8"),
            ("sections.section_2.29", "add_resid2.10"),
        ]
        TestCreateSectionIdToModuleIdMapping.compare_mappings(seq_model, expected_mappings)

    def test_create_section_id_to_module_id_mapping_pythia(self):
        # Copying the pythia-14m config that can be obtained by running the slow code below
        # from transformer_lens.loading_from_pretrained import get_pretrained_model_config
        # from dataclasses import asdict
        # tlens_cfg = get_pretrained_model_config("pythia-14m")
        # print(SequentialTransformerConfig(**asdict(tlens_cfg)).dict())

        cfg_dict = {
            "n_layers": 6,
            "d_model": 128,
            "d_head": 32,
            "n_heads": 4,
            "d_mlp": 512,
            "d_vocab": 50304,
            "n_ctx": 2048,
            "act_fn": "gelu",
            "normalization_type": "LNPre",  # pythia actually uses LN but tlens converts it to LNPre
            "eps": 1e-05,
            "dtype": torch.float32,
            "use_attn_scale": True,
            "use_split_qkv_input": False,
            "positional_embedding_type": "rotary",
            "rotary_dim": 8,
            "parallel_attn_mlp": True,
            "original_architecture": "GPTNeoXForCausalLM",
            "use_local_attn": False,
        }

        cfg = SequentialTransformerConfig(**cfg_dict)
        node_layers = ["ln2.2", "mlp_out.4"]
        seq_model = SequentialTransformer(cfg, node_layers)

        # Obtained by manually inspecting the model and counting the layers
        expected_mappings = [
            ("sections.pre.1", "ln1.0"),
            ("sections.section_0.0", "ln2.2"),
            ("sections.section_0.17", "add_resid1.4"),
            ("sections.section_1.9", "mlp_out.5"),
            ("sections.section_1.11", "ln_final"),
        ]

        TestCreateSectionIdToModuleIdMapping.compare_mappings(seq_model, expected_mappings)

    def test_create_section_id_to_module_id_mapping_tinystories(self):
        # Copying the tiny-stories-1m config that can be obtained by running the slow code below
        # from transformer_lens.loading_from_pretrained import get_pretrained_model_config
        # from dataclasses import asdict
        # tlens_cfg = get_pretrained_model_config("tiny-stories-1m")
        # print(SequentialTransformerConfig(**asdict(tlens_cfg)).dict())

        cfg_dict = {
            "n_layers": 8,
            "d_model": 64,
            "d_head": 4,
            "n_heads": 16,
            "d_mlp": 256,
            "d_vocab": 50257,
            "n_ctx": 2048,
            "act_fn": "gelu_new",
            "normalization_type": "LNPre",  # tinystories actually uses LN but tlens converts it to LNPre
            "eps": 1e-05,
            "dtype": torch.float32,
            "use_attn_scale": False,
            "use_split_qkv_input": False,
            "use_local_attn": True,
            "positional_embedding_type": "standard",
            "rotary_dim": None,
            "parallel_attn_mlp": False,
            "original_architecture": "GPTNeoForCausalLM",
        }

        cfg = SequentialTransformerConfig(**cfg_dict)
        node_layers = ["ln1.3", "mlp_out.6"]
        seq_model = SequentialTransformer(cfg, node_layers)

        # TODO
        expected_mappings = [
            ("sections.pre.3", "ln1.0"),
            ("sections.section_0.1", "attn_in.3"),
            ("sections.section_0.2", "attn_out.3"),
        ]

        TestCreateSectionIdToModuleIdMapping.compare_mappings(seq_model, expected_mappings)


@pytest.mark.parametrize(
    "node_layers, module_ids",
    [
        (
            ["add_resid1.0", "ln1.1"],
            ["embed", "add_resid1.0", "mlp.0", "ln1.1"],
        ),
        (
            ["add_resid1.0", "ln1.1", "mlp.3"],
            ["add_resid1.0", "ln1.1", "ln1.2", "mlp.3"],
        ),
        (
            ["add_resid1.0", "ln1.1", "mlp.3", "output"],  # output is optional
            ["add_resid1.0", "ln1.1", "ln1.2", "mlp.3"],
        ),
    ],
)
def test_validate_node_layers_valid(node_layers: list[str], module_ids: list[str]):
    # The below should not raise an error
    SequentialTransformer.validate_node_layers(node_layers, module_ids)


@pytest.mark.parametrize(
    "node_layers, module_ids",
    [
        (
            ["add_resid1.0", "ln1.1", "mlp.0"],
            ["embed", "add_resid1.0", "mlp.0", "ln1.1"],  # mlp.0 appears before ln1.1
        ),
        (
            ["add_resid1.0", "ln1.1", "mlp.3"],
            ["add_resid1.0", "ln1.1", "ln1.2", "mlp.2"],  # mlp.3 does not exist
        ),
        (
            ["add_resid1.0", "ln1.1", "output", "mlp.3"],  # output in wrong position
            ["add_resid1.0", "ln1.1", "ln1.2", "mlp.3"],
        ),
    ],
)
def test_validate_node_layers_invalid(node_layers: list[str], module_ids: list[str]):
    with pytest.raises(ValueError):
        SequentialTransformer.validate_node_layers(node_layers, module_ids)


@pytest.mark.slow()
def test_n_ctx_pythia():
    """Test that varying n_ctx produces the same attention scores and outputs for pythia-14m.

    Note that this test doesn't test any particular feature that we've implemented, but rather
    that pythia in SequentialTransformer will work regardless of the value of n_ctx.

    Process is a follows:
    1. Initialize a pythia-14m model.
    2. Create a fake data sample of length short_n_ctx.
    3. Add random tokens to the fake data sample so that it is length long_n_ctx=2048.
    4. Apply a pre-forward hook to attn_out.0 to calculate and save the attention scores.
    5. Assert that the attention scores are the same for the first short_n_ctx positions.
    6. Assert that the model outputs are the same for the first short_n_ctx positions.
    """
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    atol = 0  # Works with atol=0 for float64 and atol=1e-6 for float32
    module_id = "attn_out.0"
    batch_size = 2
    short_n_ctx = 20
    long_n_ctx = 2048

    seq_model, _ = load_sequential_transformer(
        node_layers=[module_id],
        last_pos_module_type=None,
        tlens_pretrained="pythia-14m",
        tlens_model_path=None,
        fold_bias=True,
        dtype=dtype,
        device=device,
    )
    seq_model.eval()

    # We only care about the first module in section_0, so remove the rest
    seq_model.sections.section_0 = MultiSequential(seq_model.sections.section_0[0])

    hooked_model = HookedModel(seq_model)

    # Create a fake data sample of length short_n_ctx
    input_ids_short = torch.randint(
        0, seq_model.cfg.d_vocab, (batch_size, short_n_ctx), device=device
    )
    # Add extra tokens to the fake data sample so that it is length n_ctx=2048
    input_ids_long = torch.cat(
        [
            input_ids_short,
            torch.randint(
                0, seq_model.cfg.d_vocab, (batch_size, long_n_ctx - short_n_ctx), device=device
            ),
        ],
        dim=1,
    )

    with torch.inference_mode():
        # Ran short_ctx example through model
        out_short, cache_short = hooked_model.run_with_cache(input_ids_short)

    attn_scores_short = (
        cache_short["sections.section_0.0.attention_scores"]["acts"][0].detach().clone()
    )

    with torch.inference_mode():
        # Ran long_ctx example through model
        out_long, cache_long = hooked_model.run_with_cache(input_ids_long)

    # Collect the attention scores for the first short_n_ctx positions
    attn_scores_long = (
        cache_long["sections.section_0.0.attention_scores"]["acts"][0][
            ..., :short_n_ctx, :short_n_ctx
        ]
        .detach()
        .clone()
    )

    assert torch.allclose(attn_scores_short, attn_scores_long, atol=atol), (
        f"Attention scores for short_n_ctx={short_n_ctx} and long_n_ctx={long_n_ctx} are not equal."
        f"Max difference: {torch.max(torch.abs(attn_scores_short - attn_scores_long))}"
    )

    # Check that the outputs for the first short_n_ctx positions are the same
    assert torch.allclose(out_short[0], out_long[0][:, :short_n_ctx, :], atol=atol), (
        f"Model output for short_n_ctx={short_n_ctx} and long_n_ctx={long_n_ctx} are not equal."
        f"Max difference: {torch.max(torch.abs(out_short[0] - out_long[0][..., :short_n_ctx, :]))}"
    )
