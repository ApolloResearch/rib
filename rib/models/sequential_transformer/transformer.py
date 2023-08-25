"""
Defines a Transformer based on transformer lens but with a module hierarchy that allows for easier
building of a RIB graph.
"""

from typing import Type

from jaxtyping import Int
from torch import Tensor, nn

from rib.models.sequential_transformer.components import (
    SEQUENTIAL_COMPONENT_REGISTRY,
    SeqLayerNormPre_Folded,
)
from rib.models.sequential_transformer.config import SequentialTransformerConfig


class MultiSequential(nn.Sequential):
    """Sequential module where containing modules that may have multiple inputs and outputs."""

    def forward(self, *inputs):
        for module in self._modules.values():
            inputs = inputs if isinstance(inputs, tuple) else (inputs,)
            inputs = module(*inputs)
        return inputs


class SequentialTransformer(nn.Module):
    """Transformer whose modules are organised into a hierarchy based on the desired RIB graph.

    Args:
        cfg: The SequentialTransformer config
        node_layers: The names of the node layers used to partition the transformer. There will be
            `node_layers - 1` sections in the graph, one between each node layer.

    For example:
    >>> cfg = ... # config for gpt2
    >>> node_layers = ["embed", "attn.0", "mlp_act.0"]
    >>> model = SequentialTransformer(cfg, node_layers)
    >>> print(model)
        SequentialTransformer(
            (module_sections): ModuleList(
                (0): MultiSequential(
                    (0): Embed()
                    (1): PosEmbed()
                    (2): Add()
                    (3): SeqLayerNormPre_Folded()
                )
                (1): MultiSequential(
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

        assert len(node_layers) > 1, "Must have at least 2 node layers"
        self.module_name_sections = self.create_module_name_sections(cfg.n_layers, node_layers)

        if cfg.normalization_type == "LNPre":
            ln_class = SeqLayerNormPre_Folded
        elif cfg.normalization_type is None:
            ln_class = nn.Identity
        else:
            raise ValueError(f"Normalization type {cfg.normalization_type} not supported")

        # Initialize the modules, creating a ModuleList of Sequential modules for each graph section
        module_sections: list[MultiSequential] = []
        for module_names in self.module_name_sections:
            module_section: list[nn.Module] = []
            for module_name in module_names:
                module_type = module_name.split(".")[0]
                module_class: Type[nn.Module]
                if module_type in ["ln1", "ln2"]:
                    module_class = ln_class
                else:
                    module_class = SEQUENTIAL_COMPONENT_REGISTRY[module_type]
                module = module_class(cfg)
                module_section.append(module)
            module_sections.append(MultiSequential(*module_section))
        self.module_sections: nn.ModuleList = nn.ModuleList(module_sections)

    @staticmethod
    def create_module_name_sections(n_blocks: int, node_layers: list[str]) -> list[list[str]]:
        """Create ordered groups of module names.

        Each group will be a section of a RIB graph, with the exception that, if the first
        node_layer is not "embed", the first group will be all of the modules from embed to the
        first node layer (which will not make up part of the graph).

        We first create a flat list of all module names. We then iterate through this list,
        appending each module name to a group until we reach the next node layer, and repeat.

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

        module_name_sections: list[list[str]] = []
        current_module_name_section: list[str] = []
        node_idx = 0
        for module_name in all_layers:
            if module_name == node_layers[node_idx]:
                if current_module_name_section:
                    # Need to start a new module group, store the current one if non-empty
                    module_name_sections.append(current_module_name_section)
                    current_module_name_section = []
                if node_idx == len(node_layers) - 1:
                    # We've reached the end of the node layers, so we're done
                    break
                node_idx += 1
            current_module_name_section.append(module_name)

        return module_name_sections

    def forward(self, input_ids: Int[Tensor, "batch n_ctx"]) -> tuple[Tensor]:
        """Forward pass through the model.

        Args:
            input_ids: The input token IDs.

        Returns:
            A tuple of tensors, the number of which depends on how many outputs the final module in
            the graph gives.
        """
        xs = (input_ids,)
        for module_section in self.module_sections:
            xs = module_section(*xs)
        return xs
