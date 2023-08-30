"""
Defines a Transformer based on transformer lens but with a module hierarchy that allows for easier
building of a RIB graph.
"""
from collections import OrderedDict
from typing import Type

from jaxtyping import Int
from torch import Tensor, nn

from rib.models.sequential_transformer.components import (
    SEQUENTIAL_COMPONENT_REGISTRY,
    SeqLayerNormPre_Folded,
)
from rib.models.sequential_transformer.config import SequentialTransformerConfig
from rib.models.utils import create_list_partitions


class MultiSequential(nn.Sequential):
    """Sequential module where containing modules that may have multiple inputs and outputs."""

    def forward(self, *inputs):
        for module in self._modules.values():
            inputs = inputs if isinstance(inputs, tuple) else (inputs,)
            inputs = module(*inputs)
        return inputs


class SequentialTransformer(nn.Module):
    """Transformer whose modules are organised into a hierarchy based on the desired RIB graph.

    If the first node_layer is not "embed", we will have a pre-section of modules from embed to
    the first node_layer. This pre-section will not be part of the graph but needs to be run
    with all forward passes in order to feed the correct data to susbequent sections.

    Args:
        cfg: The SequentialTransformer config
        node_layers: The names of the node layers used to partition the transformer. There will be
            `node_layers - 1` sections in the graph, one between each node layer.

    For example:
    >>> cfg = ... # config for gpt2
    >>> node_layers = ["attn.0", "mlp_act.0"]
    >>> model = SequentialTransformer(cfg, node_layers)
    >>> print(model)
        SequentialTransformer(
            (sections): ModuleDict(
                (pre): MultiSequential(
                    (0): Embed()
                    (1): PosEmbed()
                    (2): Add()
                    (3): SeqLayerNormPre_Folded()
                )
                (section_0): MultiSequential(
                    (0): Attention()
                    (1): Add()
                    (2): SeqLayerNormPre_Folded()
                    (3): MLPIn()
                    (4): MLPAct()
                )
            )
        )
    """

    embed_module_names: list[str] = ["embed", "pos_embed", "add_embed"]
    block_module_names: list[str] = [
        "ln1",
        "attn",
        "add_resid1",
        "ln2",
        "mlp_in",
        "mlp_act",
        "mlp_out",
        "add_resid2",
    ]
    unembed_module_name: str = "unembed"

    def __init__(self, cfg: SequentialTransformerConfig, node_layers: list[str]):
        super().__init__()
        self.cfg = cfg
        self.node_layers = node_layers
        assert len(node_layers) > 1, "Must have at least 2 node layers"

        self.module_name_sections = None
        self.sections = None
        self.rename_state_dict_keys = True

        if self.cfg.normalization_type == "LNPre":
            self.ln_class = SeqLayerNormPre_Folded
        elif self.cfg.normalization_type is None:
            self.ln_class = nn.Identity
        else:
            raise ValueError(f"Normalization type {self.cfg.normalization_type} not supported")

        self.flattened_layers = self.initialize_flat_model()

    def toggle_state_dict_renaming(self, flag: bool):
        self.rename_state_dict_keys = flag

    def state_dict(self, destination=None, prefix="", keep_vars=False) -> OrderedDict[str, Tensor]:
        sd = super().state_dict(destination, prefix, keep_vars)

        if not self.rename_state_dict_keys:
            return sd

        new_sd = OrderedDict()
        for k, v in sd.items():
            # Strip the undesired prefix and replace the underscore
            new_k = k
            if new_k.startswith("flattened_layers."):
                new_k = new_k[len("flattened_layers.") :]
            new_k = new_k.replace("blocks_", "blocks.")
            new_k = new_k.replace("mlp_in", "mlp")
            new_k = new_k.replace("mlp_out", "mlp")
            new_sd[new_k] = v

        return new_sd

    def initialize_flat_model(self) -> nn.ModuleDict[str, nn.Module]:
        main_dict = OrderedDict()

        for module_name in SequentialTransformer.embed_module_names:
            module_class = SEQUENTIAL_COMPONENT_REGISTRY[module_name]
            module = module_class(self.cfg)
            main_dict[module_name] = module

        for block_num in range(self.cfg.n_layers):
            block_dict = OrderedDict()
            for module_name in SequentialTransformer.block_module_names:
                module_class = (
                    self.ln_class
                    if module_name in ["ln1", "ln2"]
                    else SEQUENTIAL_COMPONENT_REGISTRY[module_name]
                )
                module = module_class(self.cfg)
                block_dict[module_name] = module
            block_module_dict = nn.ModuleDict(block_dict)
            main_dict[f"blocks_{block_num}"] = block_module_dict

        unembed_module_class = SEQUENTIAL_COMPONENT_REGISTRY[
            SequentialTransformer.unembed_module_name
        ]
        unembed_module = unembed_module_class(self.cfg)
        main_dict[SequentialTransformer.unembed_module_name] = unembed_module

        return nn.ModuleDict(main_dict)

    # TODO refactor given initialize_flat_model
    def structure_graph(self):
        self.module_name_sections = self.create_module_name_sections(
            self.cfg.n_layers, self.node_layers
        )

        has_pre_section = self.node_layers[0] != "embed"
        # Initialize the modules, creating a ModuleList of Sequential modules for each graph section
        sections: dict[str, MultiSequential] = {}
        # If has_pre_section, we need to start at -1 because the first section will be the
        # pre-section which should not be part of the graph
        for i, module_names in enumerate(self.module_name_sections, -1 if has_pre_section else 0):
            section_name = "pre" if self.node_layers[0] != "embed" and i == -1 else f"section_{i}"

            module_section: list[nn.Module] = []
            for module_name in module_names:
                module_type = module_name.split(".")[0]
                module_class: Type[nn.Module]
                if module_type in ["ln1", "ln2"]:
                    module_class = self.ln_class
                else:
                    module_class = SEQUENTIAL_COMPONENT_REGISTRY[module_type]
                module = module_class(self.cfg)
                module_section.append(module)
            sections[section_name] = MultiSequential(*module_section)
        self.sections: nn.ModuleDict = nn.ModuleDict(sections)

    @staticmethod
    # TODO refactor given initialize_flat_model
    def create_module_name_sections(n_blocks: int, node_layers: list[str]) -> list[list[str]]:
        """Create ordered groups of module names.

        Each group will be a section of a RIB graph, with the exception that, if the first
        node_layer is not "embed", the first group will be all of the modules from embed to the
        first node layer (which will not make up part of the graph).

        We first create a flat list of all module names. We then iterate through this list,
        appending each module name to a group until we reach the next node layer, and repeat.

        If the first node_layer is not "embed", we will have a pre-section of modules from embed to
        the first node_layer. This pre-section will not be part of the graph but needs to be run
        with all forward passes regardless.

        Args:
            n_blocks: The number of layers/blocks in the model.
            node_layers: The names of the node layers to build the graph with.

        Returns:
            A list of lists of module names, where each list is a graph section.
        """
        all_layers = SequentialTransformer.embed_module_names.copy()
        for i in range(n_blocks):
            all_layers.extend(
                [f"{module_name}.{i}" for module_name in SequentialTransformer.block_module_names]
            )
        all_layers.append(SequentialTransformer.unembed_module_name)

        module_name_sections = create_list_partitions(all_layers, node_layers)
        return module_name_sections

    def forward(self, input_ids: Int[Tensor, "batch n_ctx"]) -> tuple[Tensor]:
        """Forward pass through the model.

        Args:
            input_ids: The input token IDs.

        Returns:
            A tuple of tensors, the number of which depends on how many outputs the final module in
            the graph gives.
        """
        xs = input_ids
        for module_section in self.sections.values():
            inputs = xs if isinstance(xs, tuple) else (xs,)
            xs = module_section(*inputs)
        out = xs if isinstance(xs, tuple) else (xs,)
        return out
