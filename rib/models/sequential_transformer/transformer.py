"""
Defines a Transformer based on transformer lens but with a module hierarchy that allows for easier
building of a RIB graph.
"""

from functools import partial
from typing import Callable, Type

from jaxtyping import Int
from torch import Tensor, nn

from rib.models.sequential_transformer.components import (
    SEQUENTIAL_COMPONENT_REGISTRY,
    IdentitySplit,
    LayerNormPre,
    LayerNormPreFolded,
)
from rib.models.sequential_transformer.config import SequentialTransformerConfig
from rib.models.utils import (
    add_dim_attn_O,
    concat_ones,
    concat_zeros,
    create_list_partitions,
    fold_attn_QKV,
    fold_mlp_in,
    fold_mlp_out,
    fold_unembed,
    get_model_attr,
)


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

    A SequentialTransformer contains a fold_bias method which modifies the weights of the model
    to fold in the bias parameters. If called, beware that the dimensions of the weight matrices
    will change.

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
    ln_final_name: str = "ln_final"
    unembed_module_name: str = "unembed"

    def __init__(
        self, cfg: SequentialTransformerConfig, node_layers: list[str], last_pos_only: bool = False
    ):
        super().__init__()
        self.cfg = cfg
        self.last_pos_only = last_pos_only

        assert len(node_layers) > 1, "Must have at least 2 node layers"
        self.module_name_sections = self.create_module_name_sections(cfg.n_layers, node_layers)

        assert cfg.normalization_type in [None, "LNPre"], (
            f"Normalization type {cfg.normalization_type} not supported. "
            "Only LayerNormPre and None are currently supported."
        )

        has_pre_section = node_layers[0] != "embed"
        # Initialize the modules, creating a ModuleList of Sequential modules for each graph section
        sections: dict[str, MultiSequential] = {}
        # If has_pre_section, we need to start at -1 because the first section will be the
        # pre-section which should not be part of the graph
        for i, module_names in enumerate(self.module_name_sections, -1 if has_pre_section else 0):
            section_name = "pre" if node_layers[0] != "embed" and i == -1 else f"section_{i}"

            module_section: list[nn.Module] = []
            for module_name in module_names:
                module_type = module_name.split(".")[0]
                module_class: Type[nn.Module]
                kwargs = {}
                if module_type in ["ln1", "ln2", "ln_final"]:
                    if cfg.normalization_type == "LNPre":
                        module_class = LayerNormPre
                        # ln1 and ln2 need to output both the residual and normed residual
                        kwargs["return_residual"] = module_type in ["ln1", "ln2"]
                    else:
                        module_class = nn.Identity if module_type == "ln_final" else IdentitySplit
                elif module_type == "unembed":
                    module_class = SEQUENTIAL_COMPONENT_REGISTRY[module_type]
                    kwargs = {"last_pos_only": last_pos_only}
                else:
                    module_class = SEQUENTIAL_COMPONENT_REGISTRY[module_type]
                    kwargs = {}
                module = module_class(cfg, **kwargs)
                module_section.append(module)
            sections[section_name] = MultiSequential(*module_section)
        self.sections: nn.ModuleDict = nn.ModuleDict(sections)

        self.has_folded_bias = False

    @staticmethod
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
        all_layers.append(SequentialTransformer.ln_final_name)
        all_layers.append(SequentialTransformer.unembed_module_name)

        module_name_sections = create_list_partitions(all_layers, node_layers)
        return module_name_sections

    def fold_bias(self) -> None:
        """Fold the bias parameters into the weight parameters.

        Also converts any instances of LayerNormPre into LayerNormPreFolded to ensure that the
        layer norm does not consider the extra feature of ones when calculating the mean and std.

        We define a mapping from each weight-bias pair to a function defining how to fold the bias.
        """
        if self.has_folded_bias:
            raise ValueError("Model already has folded bias")

        fold_fns: dict[str, Callable] = {
            "W_E": concat_zeros,
            "W_pos": concat_ones,
            "W_Q": fold_attn_QKV,
            "W_K": fold_attn_QKV,
            "W_V": fold_attn_QKV,
            "W_O": add_dim_attn_O,
            "W_in": partial(fold_mlp_in, self.cfg.act_fn),
            "W_out": fold_mlp_out,
            "W_U": fold_unembed,
        }

        # Get the parameter keys of the model
        seq_tf_keys = list(self.state_dict().keys())
        for seq_tf_key in seq_tf_keys:
            sections, section_name, module_idx, param_name = seq_tf_key.split(".")
            if param_name[:2] == "W_":
                if param_name not in fold_fns:
                    # No folding needed for this weight parameter
                    continue
                # It's a weight parameter. Get it's corresponding bias if it has one
                bias_key = f"{sections}.{section_name}.{module_idx}.b_{param_name[2:]}"
                bias_param = get_model_attr(self, bias_key) if bias_key in seq_tf_keys else None
                weight_param = get_model_attr(self, seq_tf_key)
                args = (weight_param,) if bias_param is None else (weight_param, bias_param)
                fold_fn = fold_fns[param_name]
                fold_fn(*args)

        # We also need to convert all instances of LayerNormPre into LayerNormPreFolded
        lnpre_folded_cfg = self.cfg.model_copy(
            update={"d_model:": self.cfg.d_model + 1, "d_mlp": self.cfg.d_mlp + 1}
        )
        for section_name, section in self.sections.items():
            for module_idx, module in enumerate(section):  # type: ignore
                if isinstance(module, LayerNormPre):
                    modified_module = LayerNormPreFolded(
                        lnpre_folded_cfg, return_residual=module.return_residual
                    )
                    self.sections[section_name][module_idx] = modified_module

        self.has_folded_bias = True

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
