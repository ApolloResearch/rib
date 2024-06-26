"""
Defines a Transformer made from a sequence of modules, allowing for calculation of a RIB graph.
"""

import warnings
from functools import partial
from typing import Callable, Literal, Optional, Type, Union

import torch
from jaxtyping import Int
from pydantic import BaseModel, ConfigDict, field_validator
from torch import Tensor, nn
from transformer_lens import HookedTransformer

from rib.models.components import (
    SEQUENTIAL_COMPONENT_REGISTRY,
    DualLayerNormIn,
    DualLayerNormOut,
    IdentitySplit,
    LayerNormIn,
    LayerNormOut,
    MultiSequential,
)
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
from rib.types import TORCH_DTYPES, StrDtype


class SequentialTransformerConfig(BaseModel):
    """Config for the sequential transformer model.

    The fields must be a subset of those in transformer-lens' HookedTransformerConfig (with exactly
    the same names).

    Args:
        n_layers: The number of layers in the model.
        d_model: The dimensionality of the model (i.e. the size of the residual stream).
        d_head: The dimensionality of the attention heads.
        n_heads: The number of attention heads.
        d_mlp: The dimensionality of the MLP (typically 4 * d_model).
        d_vocab: The size of the vocabulary.
        n_ctx: The context size (often denoted `seq` or `pos`).
        act_fn: The activation function to use in the MLP.
        normalization_type: The type of normalization to use in the model.
        eps: The epsilon value used to prevent numerical instability in normalization.
        dtype: The dtype to use for the model.
        use_attn_scale: Whether to scale the attention scores by sqrt(d_head).
        use_split_qkv_input: Whether to split the input into separate q, k, and v inputs (less
            memory efficient but easier for analysis).
        positional_embedding_type: The type of positional embedding to use ("rotary" for pythia,
            "standard" for gpt2).
        parallel_attn_mlp: Whether to parallelize the attention and MLP computations (as done in
            pythia).
        original_architecture: The family of the model, used to help load weights from HuggingFace
            or initialized to "custom" if not passed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    config_type: Literal["SequentialTransformer"] = "SequentialTransformer"

    n_layers: int
    d_model: int
    d_head: int
    n_heads: int
    d_mlp: int
    d_vocab: int
    n_ctx: int
    act_fn: str
    normalization_type: Optional[str]
    eps: float
    dtype: torch.dtype
    use_attn_scale: bool
    use_split_qkv_input: bool
    use_local_attn: bool
    positional_embedding_type: Literal["rotary", "standard"]
    rotary_dim: Optional[int]
    parallel_attn_mlp: bool
    original_architecture: Optional[str]

    @field_validator("dtype")
    @classmethod
    def set_dtype(cls, v: Union[StrDtype, torch.dtype]) -> torch.dtype:
        """Verify torch dtype or convert str to torch.dtype."""
        if isinstance(v, torch.dtype):
            if v not in TORCH_DTYPES.values():
                raise ValueError(f"Unsupported dtype {v}")
            return v
        elif isinstance(v, str):
            if v not in TORCH_DTYPES:
                raise ValueError(f"Unsupported dtype {v}")
            return TORCH_DTYPES[v]


class SequentialTransformer(nn.Module):
    """Transformer written as a sequence of nn.Modules.

    The modules are organised into sections based on the provided `node_layers` list. Each section
    is a MultiSequential object where the outputs of each module in the section is fed as inputs to
    the next module in the section, and the outputs of the final module in the section are the fed
    as inputs to the first module in the next section.

    The order of modules in the transformer are as follows:
    - embed
    - pos_embed (optional, not included if cfg.positional_embedding_type == "rotary")
    - add_embed (optional, only if pos_embed is present)
    - ln1 (may be an identity module if cfg.normalization_type is None)
    - ln1_out (may be an identity module if cfg.normalization_type is None)
    - attn_in
    - attn_out
    - add_resid1
    - ln2 (may be an identity module if cfg.normalization_type is None)
    - ln2_out (may be an identity module if cfg.normalization_type is None)
    - mlp_in
    - mlp_act
    - mlp_out
    - add_resid2
    - ln_final (may be an identity module if cfg.normalization_type is None)
    - ln_final_out (may be an identity module if cfg.normalization_type is None)
    - unembed

    Note that ln1 and ln2 just calculate the variance of the input, and ln1_out and ln2_out
    calculate the layer norm given the variance.

    We use the term `module_id` to refer to the naming convention `module_name[.layer_idx]`. E.g.
    "mlp_in.0" refers to the MLP in module in the 0th transformer layer (zero-indexed), and "ln2.3"
    refers to the layer norm module before the MLP in the 3rd transformer layer, and "embed" refers
    to the token embedding module.

    This same module structure is used for both sequential (GPT2) and parallel (Pythia) attention
    models, with the difference being handled by the module classes and arguments that are
    associated with each module name. See the diagram in `docs/SequentialTransformer.drawio.png`
    for a visual representation of the module structure.

    The `node_layers` argument is a list of module_ids specifying the points in which to partition
    the model into sections.

    The first section will be labelled "pre", and includes all modules up until (BUT NOT INCLUDING)
    the first node_layer. The next section will be labelled "section_0" and includes the modules
    from the first node_layer up until (BUT NOT INCLUDING) the second node_layer. And so on.

    The "pre" section will not be part of the RIB graph but needs to be run with all
    forward passes in order to feed the correct data to susbequent sections.

    A `section_id` is a string of the form `sections.section_name.section_idx`, where `section_name`
    is the name of the section (e.g. "pre", "section_0", "section_1", etc.), and `section_idx` is
    the index of the module in the section (zero-indexed). To access a module in a
    SequentialTransformer, `rib.models.utils.get_model_attr` can be used with the section_id as the
    attribute name. Alternatively, you can run pass `sections.section_name` to `get_model_attr`
    which will return the MultiSequential object (which is a subclass of nn.Module) that contains
    all the modules in that section. For example, to get the first module in the "section_0"
    section, use: `get_model_attr(model, "sections.section_0.0")`. To get the section itself, use:

    To convert between a section_id and a module_id, use the `section_id_to_module_id` and
    `module_id_to_section_id` attributes of the SequentialTransformer.

    Every `sections.section_name` is be a MultiSequential object (i.e. a module in itself).

    The node_layers list may end in an `output` layer, meaning that the outputs of the model will
    be the first basis in our RIB graph. We ignore this `output` layer when partitioning the
    model into sections. Since `output` is a valid node layer, we say that it is also a `module_id`.

    A SequentialTransformer contains a fold_bias method which modifies the weights of the model
    to fold in the bias parameters. If called, beware that the dimensions of the weight matrices
    will change. After running the `fold_bias` method, it will no longer be valid to train the
    model. This is because fold_bias adds vectors of zeros and ones to the weight matrices that must
    be fixed for the model to be valid.

    Args:
        cfg (SequentialTransformerConfig): The SequentialTransformer config.
        node_layers (list[str]): The module_ids indicating where to partition the transformer.
        last_pos_module_type (Optional[Literal["add_resid1", "unembed"]]): The name of the module
            in which to only output the last position index. This is used for modular addition.

    For example:
    >>> cfg = ... # config for pythia-14m
    >>> # Including "output" won't affect the section structure but will change the RIB computation
    >>> node_layers = ["mlp_out.0", "ln2.3", "mlp_out.3", "output"]
    >>> model = SequentialTransformer(cfg, node_layers)
    >>> print(model)
    SequentialTransformer(
        (sections): ModuleDict(
            (pre): MultiSequential(
                (0): Embed()
                (1): LayerNormIn()
                (2): LayerNormOut()
                (3): AttentionIn()
                (4): AttentionOut()
                (5): Add()
                (6): DualLayerNormIn()
                (7): DualLayerNormOut()
                (8): MLPIn()
                (9): MLPAct()
            )
            (section_0): MultiSequential(
                (0): MLPOut()
                (1): Add()
                (2): LayerNormIn()
                (3): LayerNormOut()
                ...
                (27): AttentionOut()
                (28): Add()
            )
            (section_1): MultiSequential(
                (0): DualLayerNormIn()
                (1): DualLayerNormOut()
                (2): MLPIn()
                (3): MLPAct()
            )
            (section_2): MultiSequential(
                (0): MLPOut()
                (1): Add()
                (2): LayerNormIn()
                ...
                (222): LayerNormIn()
                (223): LayerNormOut()
                (224): Unembed()
            )
    )
    """

    LAYER_MODULE_NAMES: list[str] = [
        "ln1",
        "ln1_out",
        "attn_in",
        "attn_out",
        "add_resid1",
        "ln2",
        "ln2_out",
        "mlp_in",
        "mlp_act",
        "mlp_out",
        "add_resid2",
    ]
    LN_FINAL_NAME: str = "ln_final"
    LN_FINAL_OUT_NAME: str = "ln_final_out"
    UNEMBED_MODULE_NAME: str = "unembed"

    def __init__(
        self,
        cfg: SequentialTransformerConfig,
        node_layers: list[str],
        last_pos_module_type: Optional[Literal["add_resid1", "unembed"]] = None,
    ):
        super().__init__()

        assert cfg.normalization_type in [None, "LNPre"], (
            f"Normalization type {cfg.normalization_type} not supported. "
            "Only LayerNorm and None are currently supported."
        )

        self.cfg = cfg
        self.node_layers = node_layers
        self.last_pos_module_type = last_pos_module_type
        self.has_folded_bias = False

        module_ids: list[str] = self.get_module_ids()

        SequentialTransformer.validate_node_layers(node_layers, module_ids)

        self.sections: nn.ModuleDict = self.create_sections(module_ids)

        id_mappings: list[tuple[str, str]] = self.create_section_id_to_module_id_mapping(module_ids)
        self.section_id_to_module_id = dict(id_mappings)
        self.module_id_to_section_id = {v: k for k, v in self.section_id_to_module_id.items()}

    @staticmethod
    def validate_node_layers(node_layers: list[str], module_ids: list[str]) -> None:
        """Check that all the node_layers are valid.

        We check that:
        1. There is at least 1 node layer.
        2. The first node layer is not an embedding layer.
        3. There are no duplicate node_layers.
        4. All are valid module_ids.
        5. They appear in order.
        6. There are no duplicates.
        """
        assert len(node_layers) > 0, "Must have at least 1 node layer"
        assert node_layers[0] not in ["embed", "pos_embed"], (
            f"The first node layer must be a node layer in the transformer, not an embedding layer "
            f"that takes in token IDs. Got {node_layers[0]}"
        )
        assert len(node_layers) == len(set(node_layers)), "Duplicate node layers provided"

        node_layers_no_output = node_layers[:-1] if node_layers[-1] == "output" else node_layers

        module_id_idx = 0
        for node_layer in node_layers_no_output:
            try:
                node_layer_idx = module_ids.index(node_layer)
            except ValueError:
                raise AssertionError(f"Provided node_layer: {node_layer} is not a valid module_id")
            assert node_layer_idx >= module_id_idx, (
                f"Node layers must be in order. {node_layer} appears before "
                f"{module_ids[module_id_idx]}"
            )
            module_id_idx = node_layer_idx

    def create_sections(
        self,
        module_ids: list[str],
    ) -> nn.ModuleDict:
        """Create ordered sections of module_ids.

        Each section spans the start-end of an edge in a RIB graph.

        Note that the first section, labelled "pre", will not be part of the graph but needs to be
        run with all forward passes regardless.

        Args:
            module_ids: The names (and layer indices) of all modules of the model, in order.

        Returns:
            A ModuleDict where each key is the name of the section and each value is a
            MultiSequential object containing the modules in the section.
        """

        # We ignore the optional `output` layer when partitioning the model into sections
        node_layers_no_output = [layer for layer in self.node_layers if layer != "output"]
        paritioned_module_ids = create_list_partitions(module_ids, node_layers_no_output)

        sections: dict[str, MultiSequential] = {}
        # We need to start at -1 because the first section is the pre-section which should not be
        # part of the graph
        for i, module_names in enumerate(paritioned_module_ids, -1):
            section_name = "pre" if i == -1 else f"section_{i}"

            module_section: list[nn.Module] = []
            for module_name in module_names:
                module_type = module_name.split(".")[0]
                module_class: Type[nn.Module]
                kwargs = {}
                if module_type == self.last_pos_module_type:
                    # Used for modular addition where we only care about the last position index
                    kwargs["last_pos_only"] = True
                if module_type == "embed" and self.cfg.positional_embedding_type == "rotary":
                    kwargs["return_tokens"] = False
                if module_type == "add_resid1" and self.cfg.parallel_attn_mlp:
                    kwargs["return_residual"] = True
                if module_type == "attn_out" and self.cfg.use_local_attn:
                    assert self.cfg.original_architecture == "GPTNeoForCausalLM"
                    layer_idx = module_name.split(".")[-1]
                    kwargs["use_local_attn"] = (int(layer_idx) % 2) != 0  # odd layers use local

                if module_type in ["ln1", "ln2", "ln_final"]:
                    if self.cfg.normalization_type == "LNPre":
                        if self.cfg.parallel_attn_mlp and module_type == "ln2":
                            module_class = DualLayerNormIn
                        else:
                            module_class = LayerNormIn
                    else:
                        module_class = nn.Identity
                elif module_type in ["ln1_out", "ln2_out", "ln_final_out"]:
                    if self.cfg.normalization_type == "LNPre":
                        if self.cfg.parallel_attn_mlp and module_type == "ln2_out":
                            module_class = DualLayerNormOut
                        else:
                            module_class = LayerNormOut
                            # ln1_out and ln2_out need to output both the resid and normed resid
                            kwargs["return_residual"] = module_type in ["ln1_out", "ln2_out"]
                    else:
                        # Since ln1 and ln2 are identities, we need to split here (except for
                        # ln_final_out which is always identity)
                        module_class = (
                            nn.Identity if module_type == "ln_final_out" else IdentitySplit
                        )
                else:
                    module_class = SEQUENTIAL_COMPONENT_REGISTRY[module_type]
                module = module_class(self.cfg, **kwargs)
                module_section.append(module)
            sections[section_name] = MultiSequential(*module_section)

        return nn.ModuleDict(sections)

    def fold_bias(self) -> None:
        """Fold the bias parameters into the weight parameters.

        Also ensures that the final dimension is not calculated in layernorm modules.

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
            # E.g. sections.section_0.0.W_E or sections.section_1.14.attention_scores.mask
            # Number of "." in the param_name can vary, thus split at most 3 times.
            sections, section_name, module_idx, param_name = seq_param_name.split(".", 3)
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

        # We want to ensure that layernorm doesn't include the final dimension in its calcs
        ln_classes = (LayerNormOut, DualLayerNormOut, LayerNormIn, DualLayerNormIn)
        ln_modules = [module for module in self.modules() if isinstance(module, ln_classes)]
        for module in ln_modules:
            module.exclude_final_dim(True)

        self.has_folded_bias = True

    def create_section_id_to_module_id_mapping(
        self, module_ids: list[str]
    ) -> list[tuple[str, str]]:
        """Create a list of tuples mapping `sections.section_name.section_idx` (section_ids) to
        `module_name[.layer_idx]` (module_ids).

        Args:
            module_ids: The names of the modules (and their layer index) in the model.

        Returns:
            A list of tuples where each tuple is a (section_id, module_id) pair.
        """
        section_ids: list[str] = [
            f"sections.{section_name}.{section_idx}"
            for section_name, section in self.sections.items()
            for section_idx, _ in enumerate(section)
        ]
        mapping: list[tuple[str, str]] = list(zip(section_ids, module_ids))
        return mapping

    def get_module_ids(self) -> list[str]:
        """Create a flat list of all module_ids in the model.

        Recall that module_ids take the form `module_name[.layer_idx]`. For example:
        `embed`, `ln1.0`, `mlp_in.2`.

        Returns:
            A list of module_ids.
        """
        module_ids: list[str] = []

        # Embed module_ids
        module_ids.extend(
            ["embed"]
            if self.cfg.positional_embedding_type == "rotary"
            else ["embed", "pos_embed", "add_embed"]
        )

        for i in range(self.cfg.n_layers):
            module_ids.extend(
                [f"{module_name}.{i}" for module_name in SequentialTransformer.LAYER_MODULE_NAMES]
            )
        module_ids.append(SequentialTransformer.LN_FINAL_NAME)
        module_ids.append(SequentialTransformer.LN_FINAL_OUT_NAME)
        module_ids.append(SequentialTransformer.UNEMBED_MODULE_NAME)
        return module_ids

    def load_tlens_weights(
        self,
        tlens_model: HookedTransformer,
        positional_embedding_type: Literal["standard", "rotary"],
    ) -> None:
        """Load the weights from a transformer lens model into the sequential transformer model.

        Args:
            tlens_model: The transformer lens model.
            positional_embedding_type: The type of positional embedding to use ("rotary" for pythia,
                "standard" for gpt2).
        """
        seq_param_names = list(self.state_dict().keys())
        named_buffers = dict(self.named_buffers())

        attn_names: list[str] = [
            "W_Q",
            "b_Q",
            "W_K",
            "b_K",
            "W_V",
            "b_V",
            "W_O",
            "b_O",
            "IGNORE",
            "mask",
        ]
        if positional_embedding_type == "rotary":
            attn_names += ["rotary_sin", "rotary_cos"]

        mlp_names: list[str] = ["W_in", "b_in", "W_out", "b_out"]

        if positional_embedding_type == "standard":
            embed_names = ["W_E", "W_pos"]
            expected_embedding_names = ["sections.pre.0.W_E", "sections.pre.1.W_pos"]
        elif positional_embedding_type == "rotary":
            embed_names = ["W_E"]
            expected_embedding_names = ["sections.pre.0.W_E"]
        assert all(
            [param_name in seq_param_names for param_name in expected_embedding_names]
        ), "The embedding layers must be in the 'pre' section of the model"

        expected_param_names = attn_names + mlp_names + embed_names + ["W_U", "b_U"]

        assert set([key.split(".")[-1] for key in seq_param_names]) == set(
            expected_param_names
        ), f"seq_param_names has params not in {seq_param_names}"

        # The current block number in the tlens model
        block_num: int = 0
        # The names of all params in the current block
        tlens_block_names = set(attn_names + mlp_names)
        state_dict: dict[str, torch.Tensor] = {}
        for seq_param_name in seq_param_names:
            # Check if tlens_block_names is empty and if so, increment block_num and reset
            if len(tlens_block_names) == 0:
                block_num += 1
                tlens_block_names = set(attn_names + mlp_names)

            param_name = seq_param_name.split(".")[-1]

            if param_name == "W_E":
                state_dict[seq_param_name] = tlens_model.embed.W_E
            elif param_name == "W_pos":
                state_dict[seq_param_name] = tlens_model.pos_embed.W_pos
                if tlens_model.cfg.model_name.startswith("TinyStories"):
                    # We special case n_ctx in tinystories due to this bug:
                    # https://github.com/neelnanda-io/TransformerLens/issues/492
                    # this gives the pos embed matrix a different shape
                    state_dict[seq_param_name] = state_dict[seq_param_name][: self.cfg.n_ctx]
            elif param_name == "W_U":
                state_dict[seq_param_name] = tlens_model.unembed.W_U
            elif param_name == "b_U":
                state_dict[seq_param_name] = tlens_model.unembed.b_U
            else:
                tlens_block_names.remove(param_name)
                if param_name in attn_names:
                    tlens_param_val = getattr(tlens_model.blocks[block_num].attn, param_name)
                elif param_name in mlp_names:
                    tlens_param_val = getattr(tlens_model.blocks[block_num].mlp, param_name)
                else:
                    raise ValueError(
                        f"Param name not an embed, unembed, attn or mlp param: {param_name}"
                    )

                if tlens_model.cfg.model_name.startswith("TinyStories") and seq_param_name.endswith(
                    "attention_scores.mask"
                ):
                    # We special case n_ctx in tinystories due to this bug:
                    # https://github.com/neelnanda-io/TransformerLens/issues/492
                    # this gives the attn mask a different shape
                    n_ctx = self.cfg.n_ctx
                    tlens_param_val = tlens_param_val[:n_ctx, :n_ctx]

                buffer_val = named_buffers.get(seq_param_name)
                if buffer_val is not None:
                    assert buffer_val.dtype == tlens_param_val.dtype, (
                        f"Buffer {seq_param_name} has dtype {buffer_val.dtype} but tlens_param_val "
                        f"has dtype {tlens_param_val.dtype}. It is not a good idea to map "
                        f"parameters of different dtypes."
                    )
                    if not torch.allclose(buffer_val, tlens_param_val.to(buffer_val.dtype)):
                        if seq_param_name.endswith("IGNORE"):
                            warnings.warn(
                                f"Mismatch ignored in parameter {seq_param_name} ({buffer_val} vs {tlens_param_val})"
                            )
                        else:
                            raise ValueError(
                                f"Buffer {seq_param_name} does not match between seq_model "
                                "and tlens_model"
                            )
                state_dict[seq_param_name] = tlens_param_val

        self.load_state_dict(state_dict)

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
