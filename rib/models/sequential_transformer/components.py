"""Defines components to be used in a sequential transformer architecture."""
from typing import Callable, Optional, Union, cast

import einops
import numpy as np
import torch
from fancy_einsum import einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.nn import functional as F
from transformer_lens.utils import gelu_new

from rib.models import SequentialTransformerConfig


class Embed(nn.Module):
    def __init__(self, cfg: SequentialTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.W_E: Float[Tensor, "d_vocab d_model"] = nn.Parameter(
            torch.empty(self.cfg.d_vocab, self.cfg.d_model, dtype=cfg.dtype)
        )

    def forward(
        self, tokens: Int[Tensor, "batch pos"]
    ) -> tuple[Int[Tensor, "d_vocab d_model"], Float[Tensor, "batch pos d_model"]]:
        """Calculate token embeddings of the input tokens.

        Args:
            tokens (Int[Tensor, "batch pos"]): The input tokens.

        Returns:
            - The input tokens
            - The token embeddings
        """
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        return tokens, self.W_E[tokens, :]


class PosEmbed(nn.Module):
    def __init__(self, cfg: SequentialTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty(self.cfg.n_ctx, self.cfg.d_model, dtype=cfg.dtype))

    def forward(
        self, tokens: Int[Tensor, "batch pos"], x: Float[Tensor, "batch pos d_model"]
    ) -> tuple[Int[Tensor, "batch pos d_model"], Float[Tensor, "batch pos d_model"]]:
        """Add positional embeddings to the input.

        Args:
            tokens (Int[Tensor, "batch pos"]): The input tokens.
            x (Float[Tensor, "batch pos d_model"]): Tokens after embedding.

        Returns:
            - Positional embeddings
            - Token embeddings
        """

        tokens_length = tokens.size(-1)
        pos_embed = self.W_pos[:tokens_length, :]  # [pos, d_model]
        broadcast_pos_embed = einops.repeat(
            pos_embed, "pos d_model -> batch pos d_model", batch=tokens.size(0)
        )  # [batch, pos, d_model]
        return broadcast_pos_embed.clone(), x


class Unembed(nn.Module):
    def __init__(self, cfg: SequentialTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.W_U: Float[Tensor, "d_model d_vocab"] = nn.Parameter(
            torch.empty(self.cfg.d_model, self.cfg.d_vocab, dtype=cfg.dtype)
        )
        self.b_U: Float[Tensor, "d_vocab"] = nn.Parameter(
            torch.zeros(self.cfg.d_vocab, dtype=cfg.dtype)
        )

    def forward(
        self, residual: Float[Tensor, "batch pos d_model"]
    ) -> Float[Tensor, "batch pos d_vocab"]:
        return (
            einsum(
                "batch pos d_model, d_model vocab -> batch pos vocab",
                residual,
                self.W_U,
            )
            + self.b_U
        )


class Add(nn.Module):
    def __init__(self, cfg: SequentialTransformerConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self, x: Float[Tensor, "#dims"], y: Float[Tensor, "#dims"]
    ) -> Float[Tensor, "#dims"]:
        return x + y


class Attention(nn.Module):
    def __init__(
        self,
        cfg: SequentialTransformerConfig,
        layer_id: Optional[int] = None,
    ):
        """Attention Block - params have shape [head_index, d_model, d_head] (or [head_index, d_head, d_model] for W_O) and multiply on the right. attn_scores refers to query key dot product immediately before attention softmax

        Convention: All attention pattern-style matrices have shape [batch, head_index, query_pos, key_pos]

        Args:
            cfg (SequentialTransformerConfig): Config
            layer_id (int, optional): The index of the current layer. Used by the Mistal models (labelled here as stanford-gpt2) to scale down attention scores pre softmax for numerical stability reasons by 1/(layer_id+1). Defaults to None.
        """
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(
            torch.empty(self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head, dtype=cfg.dtype)
        )
        self.W_K = nn.Parameter(
            torch.empty(self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head, dtype=cfg.dtype)
        )
        self.W_V = nn.Parameter(
            torch.empty(self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head, dtype=cfg.dtype)
        )
        self.W_O = nn.Parameter(
            torch.empty(self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model, dtype=cfg.dtype)
        )
        self.b_Q = nn.Parameter(torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=cfg.dtype))
        self.b_K = nn.Parameter(torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=cfg.dtype))
        self.b_V = nn.Parameter(torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=cfg.dtype))
        self.b_O = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=cfg.dtype))

        # Create a max_ctx x max_ctx mask, with True iff that query position
        # can attend to that key position (query is first axis, key is second axis)
        causal_mask: Bool[Tensor, "pos pos"] = torch.tril(
            torch.ones((self.cfg.n_ctx, self.cfg.n_ctx)).bool()
        )
        self.register_buffer("mask", causal_mask)

        self.register_buffer("IGNORE", torch.tensor(-1e5))

        self.layer_id = layer_id

        # attn_scale is a constant that we divide the attention scores by pre-softmax.
        # I'm not entirely sure why it matters, but it's probably a mix of softmax not being
        # scale invariant and numerical stability?
        if self.cfg.use_attn_scale:
            self.attn_scale = np.sqrt(self.cfg.d_head)
        else:
            self.attn_scale = 1.0

    def forward(
        self,
        residual: Float[Tensor, "batch pos d_model"],
        x: Float[Tensor, "batch pos d_model"],
    ) -> tuple[Float[Tensor, "batch pos d_model"], Float[Tensor, "batch pos d_model"]]:
        """Forward through the entire attention block.

        TODO: Split into multiple modules so we can create graphs at each layer.
        """

        def add_head_dimension(tensor):
            return einops.repeat(
                tensor,
                "batch pos d_model -> batch pos n_heads d_model",
                n_heads=self.cfg.n_heads,
            ).clone()

        if self.cfg.use_split_qkv_input:
            query_input: Float[Tensor, "batch pos head_index d_model"] = add_head_dimension(x)
            key_input: Float[Tensor, "batch pos head_index d_model"] = add_head_dimension(x)
            value_input: Float[Tensor, "batch pos head_index d_model"] = add_head_dimension(x)
        else:
            query_input, key_input, value_input = residual, residual, residual

        if self.cfg.use_split_qkv_input:
            qkv_einops_string = "batch pos head_index d_model"
        else:
            qkv_einops_string = "batch pos d_model"

        q = (
            einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> batch pos head_index d_head",
                query_input,
                self.W_Q,
            )
            + self.b_Q
        )
        # [batch, pos, head_index, d_head]
        k = (
            einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> batch pos head_index d_head",
                key_input,
                self.W_K,
            )
            + self.b_K
        )
        # [batch, pos, head_index, d_head]
        v = (
            einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> batch pos head_index d_head",
                value_input,
                self.W_V,
            )
            + self.b_V
        )
        # [batch, pos, head_index, d_head]

        if self.cfg.dtype not in [torch.float32, torch.float64]:
            # If using 16 bits, increase the precision to avoid numerical instabilities
            q = q.to(torch.float32)
            k = k.to(torch.float32)

        attn_scores = (
            einsum(
                "batch query_pos head_index d_head, \
                        batch key_pos head_index d_head \
                        -> batch head_index query_pos key_pos",
                q,
                k,
            )
            / self.attn_scale
        )  # [batch, head_index, query_pos, key_pos]

        # Only supports causal attention (not bidirectional)
        # If causal attention, we mask it to only attend backwards. If bidirectional, we don't mask.
        attn_scores = self.apply_causal_mask(attn_scores)  # [batch, head_index, query_pos, key_pos]

        pattern = F.softmax(attn_scores, dim=-1)  # [batch, head_index, query_pos, key_pos]
        pattern = pattern.to(self.cfg.dtype)
        z = einsum(
            "batch key_pos head_index d_head, \
                batch head_index query_pos key_pos -> \
                batch query_pos head_index d_head",
            v,
            pattern,
        )  # [batch, pos, head_index, d_head]

        out = (
            (
                einsum(
                    "batch pos head_index d_head, \
                            head_index d_head d_model -> \
                            batch pos d_model",
                    z,
                    self.W_O,
                )
            )
            + self.b_O
        )  # [batch, pos, d_model]

        return residual, out

    def apply_causal_mask(
        self,
        attn_scores: Float[Tensor, "batch head_index pos pos_plus_past_kv_pos_offset"],
    ):
        # The key context length is the number of positions in the past - this includes all positions in the cache
        # If not caching, query_ctx_length == key_ctx_length
        key_ctx_length = attn_scores.size(-1)

        mask: Bool[Tensor, "pos pos"] = cast(Tensor, self.mask)
        return torch.where(
            mask[:key_ctx_length],
            attn_scores,
            cast(Tensor, self.IGNORE),
        )


class MLPIn(nn.Module):
    def __init__(self, cfg: SequentialTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_mlp, dtype=cfg.dtype))
        self.W_out = nn.Parameter(torch.empty(self.cfg.d_mlp, self.cfg.d_model, dtype=cfg.dtype))

    def forward(
        self,
        residual: Float[Tensor, "batch pos d_model"],
        x: Float[Tensor, "batch pos d_model"],
    ) -> tuple[Float[Tensor, "batch_size pos d_model"], Float[Tensor, "batch pos d_model"]]:
        pre_act = einsum("batch pos d_model, d_model d_mlp -> batch pos d_mlp", x, self.W_in)
        # [batch, pos, d_mlp]
        return residual, pre_act


class MLPAct(nn.Module):
    def __init__(self, cfg: SequentialTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_mlp, dtype=cfg.dtype))
        self.W_out = nn.Parameter(torch.empty(self.cfg.d_mlp, self.cfg.d_model, dtype=cfg.dtype))

        self.act_fn: Callable[[Float[Tensor, "... d_model"]], Tensor]
        if self.cfg.act_fn == "relu":
            self.act_fn = F.relu
        elif self.cfg.act_fn == "gelu":
            self.act_fn = F.gelu
        elif self.cfg.act_fn == "gelu_new":
            self.act_fn = gelu_new
        else:
            raise ValueError(f"Invalid activation function name: {self.cfg.act_fn}")

    def forward(
        self,
        residual: Float[Tensor, "batch pos d_model"],
        pre_act: Float[Tensor, "batch pos d_model"],
    ) -> tuple[Float[Tensor, "batch_size pos d_model"], Float[Tensor, "batch pos d_model"]]:
        # Technically, all these einsums could be done with a single matmul, but this is more readable.
        post_act = self.act_fn(pre_act)  # [batch, pos, d_mlp]
        return residual, post_act


class MLPOut(nn.Module):
    def __init__(self, cfg: SequentialTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_mlp, dtype=cfg.dtype))
        self.W_out = nn.Parameter(torch.empty(self.cfg.d_mlp, self.cfg.d_model, dtype=cfg.dtype))

    def forward(
        self,
        residual: Float[Tensor, "batch pos d_model"],
        post_act: Float[Tensor, "batch pos d_model"],
    ) -> tuple[Float[Tensor, "batch pos d_model"], Float[Tensor, "batch pos d_model"]]:
        out = einsum(
            "batch pos d_mlp, d_mlp d_model -> batch pos d_model",
            post_act,
            self.W_out,
        )
        return residual, out


class SeqLayerNormPre_Folded(torch.nn.Module):
    """Sequential version of rib.models.tlens_components.LayerNormPre_Folded.

    LayerNormPre - the 'center and normalise' part of LayerNorm. Length is
    normally d_model, but is d_mlp for softmax. Not needed as a parameter. This
    should only be used in inference mode after folding in LayerNorm weights

    Folded: the last hidden dimension is the constant function 1, and does not participate in the layernorm

    """

    def __init__(self, cfg: SequentialTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.eps = self.cfg.eps

    def forward(
        self,
        residual: Float[Tensor, "batch pos d_model"],
    ) -> tuple[Float[Tensor, "batch pos d_model"], Float[Tensor, "batch pos d_model"]]:
        x0 = residual[..., :-1]  # [batch, pos, length-1]

        x0 = x0 - x0.mean(-1, keepdim=True)  # [batch, pos, length-1]
        scale: Float[Tensor, "batch pos 1"] = (x0.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        x0 = x0 / scale  # [batch, pos, length-1]
        x = torch.cat([x0, residual[..., -1:]], dim=-1)  # [batch, pos, length]
        return residual, x


# Map from module names in SequentialTransformer to the corresponding component modules
SEQUENTIAL_COMPONENT_REGISTRY = {
    "embed": Embed,
    "pos_embed": PosEmbed,
    "add_embed": Add,
    "attn": Attention,
    "add_resid1": Add,
    "mlp_in": MLPIn,
    "mlp_act": MLPAct,
    "mlp_out": MLPOut,
    "add_resid2": Add,
    "unembed": Unembed,
}