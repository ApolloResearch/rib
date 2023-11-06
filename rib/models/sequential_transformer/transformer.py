"""
Defines a Transformer based on transformer lens but with a module hierarchy that allows for easier
building of a RIB graph.
"""

from functools import partial
from typing import Callable, Literal, Optional, Type

from jaxtyping import Int
from torch import Tensor, nn

from rib.models.sequential_transformer.components import (
    SEQUENTIAL_COMPONENT_REGISTRY,
    DualLayerNormPre,
    DualLayerNormPreFolded,
    IdentitySplit,
    LayerNormPre,
    LayerNormPreFolded,
)
from rib.models.sequential_transformer.config import SequentialTransformerConfig
from rib.models.utils import (
    concat_ones,
    concat_zeros,
    create_list_partitions,
    fold_attn_O,
    fold_attn_QK,
    fold_attn_V,
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
    """Transformer written as a sequence of nn.Modules.

    The modules are organised into sections based on the provided `node_layers` list. Each section
    is a MultiSequential object where the outputs of each module in the section is fed as inputs to
    the next, and the outputs of the final module in the section are the fed as inputs to the first
    module in the next section.

    The order of module names in the transformer must be as follows:
    - embed
    - pos_embed (optional)
    - add_embed (optional, only if pos_embed is present)
    - ln1 (may be an identity module if cfg.normalization_type is None)
    - attn
    - add_resid1
    - ln2 (may be an identity module if cfg.normalization_type is None)
    - mlp_in
    - mlp_act
    - mlp_out
    - add_resid2
    - ln_final (may be an identity module if cfg.normalization_type is None)
    - unembed

    This same module structure is used for both sequential (GPT2) and parallel (Pythia) attention
    models, with the difference being handled by the module classes and arguments that get used for
    each module name above.

    The `node_layers` specify the points in which to partition the model into sections.

    If the first node_layer is not "embed", we will have a pre-section of modules from embed to
    the first node_layer. This pre-section will not be part of the RIB graph but needs to be run
    with all forward passes in order to feed the correct data to susbequent sections.

    The node_layers list may end in an `output` layer, meaning that the outputs of the model will
    be the first basis in our RIB graph. We ignore this `output` layer when partitioning the
    model into sections.

    A SequentialTransformer contains a fold_bias method which modifies the weights of the model
    to fold in the bias parameters. If called, beware that the dimensions of the weight matrices
    will change. After running the `fold_bias` method, it will no longer be valid to train the
    model. This is because fold_bias adds vectors of zeros and ones to the weight matrices that must
    be fixed for the model to be valid.

    Args:
        cfg (SequentialTransformerConfig): The SequentialTransformer config.
        node_layers (list[str]): The names of the node layers used to partition the transformer.
        last_pos_module_type (Optional[Literal["add_resid1", "unembed"]]): The name of the module
            in which to only output the last position index. This is used for modular addition.

    For example:
    >>> cfg = ... # config for gpt2
    >>> node_layers = ["attn.0", "mlp_out.0"]
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

    def __init__(
        self,
        cfg: SequentialTransformerConfig,
        node_layers: list[str],
        last_pos_module_type: Optional[Literal["add_resid1", "unembed"]] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.node_layers = node_layers
        self.last_pos_module_type = last_pos_module_type
        self.has_folded_bias = False

        assert len(node_layers) > 0, "Must have at least 1 node layer"
        self.module_name_sections = self.create_module_name_sections(
            cfg.n_layers, node_layers, positional_embedding_type=cfg.positional_embedding_type
        )

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
                if module_type == last_pos_module_type:
                    # Used for modular addition where we only care about the last position index
                    kwargs["last_pos_only"] = True
                if module_type == "embed" and cfg.positional_embedding_type == "rotary":
                    kwargs["return_tokens"] = False
                if module_type == "add_resid1" and cfg.parallel_attn_mlp:
                    kwargs["return_residual"] = True
                if module_type in ["ln1", "ln2", "ln_final"]:
                    if cfg.normalization_type == "LNPre":
                        if cfg.parallel_attn_mlp and module_type == "ln2":
                            module_class = DualLayerNormPre
                        else:
                            module_class = LayerNormPre
                            # ln1 and ln2 need to output both the residual and normed residual
                            kwargs["return_residual"] = module_type in ["ln1", "ln2"]
                    else:
                        module_class = nn.Identity if module_type == "ln_final" else IdentitySplit
                else:
                    module_class = SEQUENTIAL_COMPONENT_REGISTRY[module_type]
                module = module_class(cfg, **kwargs)
                module_section.append(module)
            sections[section_name] = MultiSequential(*module_section)
        self.sections: nn.ModuleDict = nn.ModuleDict(sections)

    @staticmethod
    def create_module_name_sections(
        n_blocks: int,
        node_layers: list[str],
        positional_embedding_type: Literal["rotary", "standard"],
    ) -> list[list[str]]:
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
            positional_embedding_type: The type of positional embedding to use.

        Returns:
            A list of lists of module names, where each list is a graph section.
        """
        embed_module_names: list[str] = ["embed"]
        if positional_embedding_type == "standard":
            embed_module_names.extend(["pos_embed", "add_embed"])

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

        all_layers = embed_module_names.copy()
        for i in range(n_blocks):
            all_layers.extend([f"{module_name}.{i}" for module_name in block_module_names])
        all_layers.append(ln_final_name)
        all_layers.append(unembed_module_name)

        # We ignore the optional `output` layer when partitioning the model into sections
        partition_modules = [layer for layer in node_layers if layer != "output"]
        module_name_sections = create_list_partitions(all_layers, partition_modules)
        return module_name_sections

    def fold_bias(self) -> None:
        """Fold the bias parameters into the weight parameters.

        Also converts any instances of LayerNormPre into LayerNormPreFolded to ensure that the
        layer norm does not consider the extra feature of ones when calculating the mean and std.

        We define a mapping from each weight-bias pair to a function defining how to fold the bias.

        Note that, after running this method, it will no longer be valid to train the model!
        """
        if self.has_folded_bias:
            raise ValueError("Model already has folded bias")

        # If using rotary, we don't have a later positional embedding to concat 1s to, so we must
        # concat the 1s to the token embeddings
        token_embed_fold_fn = (
            concat_ones if self.cfg.positional_embedding_type == "rotary" else concat_zeros
        )
        fold_fns: dict[str, Callable] = {
            "W_E": token_embed_fold_fn,
            "W_pos": concat_ones,
            "W_Q": fold_attn_QK,
            "W_K": fold_attn_QK,
            "W_V": fold_attn_V,
            "W_O": fold_attn_O,
            "W_in": partial(fold_mlp_in, self.cfg.act_fn),
            "W_out": fold_mlp_out,
            "W_U": fold_unembed,
        }

        # Get the parameter keys of the model
        seq_param_names = list(self.state_dict().keys())
        for seq_param_name in seq_param_names:
            sections, section_name, module_idx, param_name = seq_param_name.split(".")
            if param_name[:2] == "W_":
                if param_name not in fold_fns:
                    # No folding needed for this weight parameter
                    continue
                # It's a weight parameter. Get it's corresponding bias if it has one
                bias_key = f"{sections}.{section_name}.{module_idx}.b_{param_name[2:]}"
                bias_param = get_model_attr(self, bias_key) if bias_key in seq_param_names else None
                weight_param = get_model_attr(self, seq_param_name)
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
                    self.sections[section_name][module_idx] = LayerNormPreFolded(
                        lnpre_folded_cfg, return_residual=module.return_residual
                    )
                elif isinstance(module, DualLayerNormPre):
                    self.sections[section_name][module_idx] = DualLayerNormPreFolded(
                        lnpre_folded_cfg
                    )

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
