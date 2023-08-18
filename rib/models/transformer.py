from transformer_lens import HookedTransformer, HookedTransformerConfig


class TransformerLensHooked:
    """
    This class defines a Transformer as a wrapper around HookedTransformer from transformer_lens.

    Args:
        n_layers: The number of layers in the Transformer.
        d_model: The size of the residual stream.
        d_head: The size of the attention heads. Default is d_model / n_heads.
        n_heads: The number of attention heads.
        d_mlp: The size of the hidden layer in the MLP.
        d_vocab: The size of the vocabulary embedding.
        n_ctx: The number of tokens in the context window.
        act_fn: The activation function to use in the MLP. Default is "relu".
        normalization_type: The type of normalization to use. Default is None.
    """

    def __init__(
        self,
        d_mlp: int = 512,
        n_layers: int = 1,
        d_model: int = 128,
        n_heads: int = 4,
        d_vocab: int = 113,
        n_ctx: int = 3,
        act_fn: str = "relu",
        normalization_type: str = "LN",
    ):
        self.hooked_transformer_config = HookedTransformerConfig(
            n_layers=n_layers,
            d_model=d_model,
            d_head=d_model // n_heads,
            n_heads=n_heads,
            d_mlp=d_mlp,
            d_vocab=d_vocab,
            n_ctx=n_ctx,
            act_fn=act_fn,
            normalization_type=normalization_type,
        )
        self.hooked_transformer = HookedTransformer(self.hooked_transformer_config)
