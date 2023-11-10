from typing import TypeVar

import pytest
import torch

from rib.models import SequentialTransformer, SequentialTransformerConfig
from rib.models.utils import create_list_partitions

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
        }
        cfg = SequentialTransformerConfig(**cfg)
        seq_model = SequentialTransformer(cfg, node_layers)
        # mappings: list[tuple[str, str]] = seq_model.create_section_id_to_module_id_mapping()

        # Obtained by manually inspecting the model and counting the layers
        expected_mappings = [
            ("sections.pre.2", "add_embed"),
            ("sections.pre.5", "add_resid1.0"),
            ("sections.section_0.9", "mlp_out.1"),
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
            "normalization_type": "LNPre",  # GPT2 actually uses LN but we currently don't support it
            "eps": 1e-05,
            "dtype": torch.float32,
            "use_attn_scale": True,
            "use_split_qkv_input": False,
            "positional_embedding_type": "standard",
            "rotary_dim": None,
            "parallel_attn_mlp": False,
            "original_architecture": "GPT2LMHeadModel",
        }

        cfg = SequentialTransformerConfig(**cfg_dict)
        node_layers = ["add_embed", "ln2.4", "mlp_act.7"]
        seq_model = SequentialTransformer(cfg, node_layers)

        # Obtained by manually inspecting the model and counting the layers
        expected_mappings = [
            ("sections.pre.1", "pos_embed"),
            ("sections.section_0.0", "add_embed"),
            ("sections.section_0.17", "ln1.2"),
            ("sections.section_2.9", "mlp_out.8"),
            ("sections.section_2.29", "add_resid1.11"),
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
            "normalization_type": "LNPre",  # pythia actually uses LN but we currently don't support it
            "eps": 1e-05,
            "dtype": torch.float32,
            "use_attn_scale": True,
            "use_split_qkv_input": False,
            "positional_embedding_type": "rotary",
            "rotary_dim": 8,
            "parallel_attn_mlp": True,
            "original_architecture": "GPTNeoXForCausalLM",
        }

        cfg = SequentialTransformerConfig(**cfg_dict)
        node_layers = ["ln2.2", "mlp_out.4"]
        seq_model = SequentialTransformer(cfg, node_layers)

        # Obtained by manually inspecting the model and counting the layers
        expected_mappings = [
            ("sections.pre.1", "ln1.0"),
            ("sections.section_0.0", "ln2.2"),
            ("sections.section_0.17", "mlp_in.4"),
            ("sections.section_1.9", "add_resid2.5"),
            ("sections.section_1.11", "unembed"),
        ]

        TestCreateSectionIdToModuleIdMapping.compare_mappings(seq_model, expected_mappings)


@staticmethod
def validate_node_layers(node_layers: list[str], module_ids: list[str]) -> None:
    """Check that all the node_layers are valid module_ids, and that they appear in order."""
    module_id_idx = 0
    for node_layer in node_layers:
        try:
            node_layer_idx = module_ids.index(node_layer)
        except ValueError:
            raise ValueError(f"Invalid node_layer {node_layer}")
        if node_layer_idx < module_id_idx:
            raise ValueError(
                f"Node layers must be in order. {node_layer} appears before {module_ids[module_id_idx]}"
            )
        module_id_idx = node_layer_idx


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
    ],
)
def test_validate_node_layers_valid(node_layers: list[str], module_ids: list[str]):
    # The below should not raise an error
    # SequentialTransformer.validate_node_layers(node_layers, module_ids)
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
    ],
)
def test_validate_node_layers_invalid(node_layers: list[str], module_ids: list[str]):
    with pytest.raises(ValueError):
        SequentialTransformer.validate_node_layers(node_layers, module_ids)
