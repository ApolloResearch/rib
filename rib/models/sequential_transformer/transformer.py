"""
Defines a Transformer based on the transformer lens but with a flattened, sequential structure.
"""

from torch import nn

from rib.models.sequential_transformer.components import (
    SEQUENTIAL_COMPONENT_REGISTRY,
    SeqLayerNormPre_Folded,
)
from rib.models.sequential_transformer.config import SequentialTransformerConfig


class MultiSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class SequentialTransformer(nn.Module):
    def __init__(self, cfg: SequentialTransformerConfig, node_layers: list[str]):
        self.module_group_names = self.create_module_groups(cfg.n_layers, node_layers)

        if cfg.normalization_type == "LNPre":
            ln_class = SeqLayerNormPre_Folded
        elif cfg.normalization_type is None:
            ln_class = nn.Identity
        else:
            raise ValueError(f"Normalization type {cfg.normalization_type} not supported")

        # Initialize the modules, creating a ModuleList of Sequential modules
        self.modules: nn.ModuleList = []
        for module_names in self.module_group_names:
            module_group: list[nn.Module] = []
            for module_name in module_names:
                if module_name.split(".")[0] in ["ln1", "ln2"]:
                    module_class = ln_class
                else:
                    module_class = SEQUENTIAL_COMPONENT_REGISTRY[module_name]
                module = module_class(cfg)
                module_group.append(module)
            self.modules.append(nn.Sequential(*module_group))

        print(self.modules)

    @staticmethod
    def create_module_groups(n_layers: int, node_layers: list[str]) -> list[list[str]]:
        """Create ordered groups of modules, where each group is a graph section.

        We first create a flat list of all module names. We then iterate through this list,
        appending each module name to a group until we reach the next node layer, and repeat.

        Args:
            n_layers: The number of layers in the model.
            node_layers: The names of the node layers to build the graph with.

        Returns:
            A list of lists of module names, where each list is a graph section.
        """
        block_module_names = [
            "ln1",
            "attn",
            "add_resid1",
            "ln2",
            "mlp_in",
            "mlp_act",
            "mlp_out",
            "add_resid2",
        ]
        all_layers = ["embed", "pos_embed"]
        for i in range(n_layers):
            all_layers.extend([f"{module_name}.{i}" for module_name in block_module_names])
        all_layers.extend(["unembed"])

        module_groups: list[list[str]] = []
        current_module_group: list[str] = []
        node_idx = 0
        for module_name in all_layers:
            if module_name == node_layers[node_idx]:
                # Need to start a new module group, store the current one
                module_groups.append(current_module_group)
                current_module_group = []
                if node_idx == len(node_layers) - 1:
                    # We've reached the end of the node layers, so we're done
                    break
                node_idx += 1
            current_module_group.append(module_name)

        return module_groups

    def forward(self, x):
        pass
