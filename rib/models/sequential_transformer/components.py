"""Defines components to be used in a sequential transformer architecture."""
from typing import Callable, Optional, Union, cast

import einops
import numpy as np
import torch
from fancy_einsum import einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from rib.models import SequentialTransformerConfig
from rib.models.utils import gelu_new, layer_norm


class Embed(nn.Module):
    def __init__(self, cfg: SequentialTransformerConfig, return_tokens: bool = True):
        super().__init__()
        self.cfg = cfg
        self.W_E: Float[Tensor, "d_vocab d_model"] = nn.Parameter(
            torch.empty(self.cfg.d_vocab, self.cfg.d_model, dtype=cfg.dtype)
        )
        self.return_tokens = return_tokens

    def forward(
        self, tokens: Int[Tensor, "..."]
    ) -> Union[
        tuple[Int[Tensor, "d_vocab d_model"], Float[Tensor, "... d_model"]],
        Float[Tensor, "... d_model"],
    ]:
        """Calculate token embeddings of the input tokens.

        For models with rotary embeddings, we don't need to return the raw tokens as the positional
        embeddings are handled in the attention layer and not added to the raw residual stream.

        Args:
            tokens: The input tokens, typically (batch, pos)

        Returns:
            - The input tokens (if return_tokens is True)
            - The token embeddings
        """
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        token_embeddings: Float[Tensor, "... d_model"] = self.W_E[tokens, :]
        if self.return_tokens:
            return tokens, token_embeddings
        else:
            return token_embeddings


class PosEmbed(nn.Module):
    def __init__(self, cfg: SequentialTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty(self.cfg.n_ctx, self.cfg.d_model, dtype=cfg.dtype))

    def forward(
        self, tokens: Int[Tensor, "... pos"], token_embed: Float[Tensor, "... pos d_model"]
    ) -> tuple[Int[Tensor, "... pos d_model"], Float[Tensor, "... pos d_model"]]:
        """Add positional embeddings to the input.

        Args:
            tokens (Int[Tensor, "... pos"]): The input tokens.
            token_embed (Float[Tensor, "... pos d_model"]): Tokens after embedding.

        Returns:
            - Positional embeddings
            - Token embeddings
        """

        n_tokens = tokens.size(-1)
        pos_embed = self.W_pos[:n_tokens, :]  # [pos, d_model]
        # If there is a batch dimension, we need to broadcast the positional embeddings
        if tokens.dim() > 1:
            pos_embed = einops.repeat(
                pos_embed, "pos d_model -> batch pos d_model", batch=tokens.size(0)
            )  # [..., pos, d_model]
        return pos_embed, token_embed


class Unembed(nn.Module):
    def __init__(self, cfg: SequentialTransformerConfig, last_pos_only: bool = False):
        super().__init__()
        self.cfg = cfg
        self.last_pos_only = last_pos_only
        self.W_U: Float[Tensor, "d_model d_vocab"] = nn.Parameter(
            torch.empty(self.cfg.d_model, self.cfg.d_vocab, dtype=cfg.dtype)
        )
        self.b_U: Float[Tensor, "d_vocab"] = nn.Parameter(
            torch.zeros(self.cfg.d_vocab, dtype=cfg.dtype)
        )

    def forward(
        self, residual: Float[Tensor, "... pos d_model"]
    ) -> Union[Float[Tensor, "... pos d_vocab"], Float[Tensor, "... 1 d_vocab"]]:
        if self.last_pos_only:
            if residual.dim() == 3:
                residual = residual[:, -1:, :]
            elif residual.dim() == 2:
                # No batch dimension (e.g. due to vmap)
                residual = residual[-1:, :]
            else:
                raise ValueError(f"residual should have dim 2 or 3, but has dim {residual.dim()}")
        return (
            einsum(
                "... pos_trunc d_model, d_model vocab -> ... pos_trunc vocab",
                residual,
                self.W_U,
            )
            + self.b_U
        )


class Add(nn.Module):
    """Add the residual stream to the output stream of the previous module.

    If last_pos_only is True, only the last position index is returned.

    If return_residual is True, the residual stream is returned as well as the output stream. This
    is needed for add_resid1 when we have parallel attention and mlp streams (as in pythia).
    """

    def __init__(
        self,
        cfg: SequentialTransformerConfig,
        last_pos_only: bool = False,
        return_residual: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.last_pos_only = last_pos_only
        self.return_residual = return_residual

    def forward(
        self, residual: Float[Tensor, "#dims"], y: Float[Tensor, "#dims"]
    ) -> Union[tuple[Float[Tensor, "#dims"], Float[Tensor, "#dims"]], Float[Tensor, "#dims"]]:
        summed = residual + y
        if self.last_pos_only:
            if summed.dim() == 3:
                summed = summed[:, -1:, :]
                if self.return_residual:
                    residual = residual[:, -1:, :]
            elif summed.dim() == 2:
                # No batch dimension (e.g. due to vmap)
                summed = summed[-1:, :]
                if self.return_residual:
                    residual = residual[-1:, :]
            else:
                raise ValueError(f"summed should have dim 2 or 3, but has dim {summed.dim()}")
        if self.return_residual:
            return residual, summed
        else:
            return summed


class AttentionIn(nn.Module):
    def __init__(
        self,
        cfg: SequentialTransformerConfig,
    ):
        """Attention Block - params have shape [head_index, d_model, d_head] (or [head_index, d_head, d_model] for W_O) and multiply on the right. attn_scores refers to query key dot product immediately before attention softmax

        Convention: All attention pattern-style matrices have shape [..., head_index, query_pos,
        key_pos]

        Supports rotary attention.

        This code was taken mostly verbatim from:
        https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/components.py

        Args:
            cfg (SequentialTransformerConfig): Config
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
        self.b_Q = nn.Parameter(torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=cfg.dtype))
        self.b_K = nn.Parameter(torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=cfg.dtype))
        self.b_V = nn.Parameter(torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=cfg.dtype))

        if self.cfg.positional_embedding_type == "rotary":
            assert (
                self.cfg.rotary_dim is not None
            ), "rotary_dim must be specified for rotary attention"
            # Applies a rotation to each two-element chunk of keys and queries pre dot producting to bake in relative position. See HookedTransformerConfig for details
            sin, cos = self.calculate_sin_cos_rotary(
                self.cfg.rotary_dim, self.cfg.n_ctx, dtype=self.cfg.dtype
            )
            self.register_buffer("rotary_sin", sin)
            self.register_buffer("rotary_cos", cos)

    def forward(
        self,
        residual: Float[Tensor, "... pos d_model"],
        x: Float[Tensor, "... pos d_model"],
    ) -> tuple[
        Float[Tensor, "... pos d_model"],
        Float[Tensor, "... pos n_head_times_d_head"],
        Float[Tensor, "... pos n_head_times_d_head"],
        Float[Tensor, "... pos n_head_times_d_head"],
    ]:
        """Forward through the entire attention block.

        Args:
            residual (Float[Tensor, "... pos d_model]): The "pure" residual stream
            x (Float[Tensor, "... pos d_model]): The normed residual stream (the input to the attention block)
        """

        def add_head_dimension(tensor):
            return einops.repeat(
                tensor,
                "... pos d_model -> ... pos n_heads d_model",
                n_heads=self.cfg.n_heads,
            ).clone()

        if self.cfg.use_split_qkv_input:
            query_input: Float[Tensor, "... pos head_index d_model"] = add_head_dimension(x)
            key_input: Float[Tensor, "... pos head_index d_model"] = add_head_dimension(x)
            value_input: Float[Tensor, "... pos head_index d_model"] = add_head_dimension(x)
        else:
            query_input, key_input, value_input = x, x, x

        if self.cfg.use_split_qkv_input:
            qkv_einops_string = "... pos head_index d_model"
        else:
            qkv_einops_string = "... pos d_model"

        q = (
            einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> ... pos head_index d_head",
                query_input,
                self.W_Q,
            )
            + self.b_Q
        )

        k = (
            einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> ... pos head_index d_head",
                key_input,
                self.W_K,
            )
            + self.b_K
        )

        # Note the d_head_v instead of d_head because when we fold in the bias of the v matrix
        # we get d_head_v = d_head+1
        v = (
            einsum(
                f"{qkv_einops_string}, head_index d_model d_head_v \
                -> ... pos head_index d_head_v",
                value_input,
                self.W_V,
            )
            + self.b_V
        )

        if self.cfg.positional_embedding_type == "rotary":
            q, k = self.rotary_rotate_qk(q, k)

        # Concatenate the last two dimension to keep the shapes the rest of the code is expecting
        q = einops.rearrange(q, "... pos head_index d_head -> ... pos (head_index d_head)")
        k = einops.rearrange(k, "... pos head_index d_head -> ... pos (head_index d_head)")
        v = einops.rearrange(v, "... pos head_index d_head_v -> ... pos (head_index d_head_v)")

        return residual, q, k, v

    def calculate_sin_cos_rotary(
        self,
        rotary_dim: int,
        n_ctx: int,
        base: int = 10000,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Float[torch.Tensor, "n_ctx rotary_dim"], Float[torch.Tensor, "n_ctx rotary_dim"]]:
        """
        Calculate the sine and cosine waves to use in a rotary embedding. See https://blog.eleuther.ai/rotary-embeddings/ for details

        Note: For some inexplicable reason, in GPT-J each ADJACENT pair of elements in k and q are rotated, in GPT-NeoX the pair of elements at k and k+n//2 are rotated (ie folding the full length in half, and then looking at pairs accordingly). I have absolutely no clue why, it should be completely equivalent.
        To resolve this, I've coded it to default to the GPT-J mode, but to explicitly check whether it's GPT-NeoX and then do the GPT-NeoX thing if it is.
        """
        high_precision = torch.float32 if dtype != torch.float64 else torch.float64
        pos = torch.arange(n_ctx, dtype=high_precision)
        dim = torch.arange(rotary_dim // 2, dtype=high_precision)

        # A set of frequencies evenly spaced in log space
        freq = base ** (dim / (rotary_dim / 2))
        if self.cfg.original_architecture in ["GPTNeoXForCausalLM", "LlamaForCausalLM"]:
            freq = einops.repeat(freq, "d -> (2 d)")
        else:
            freq = einops.repeat(freq, "d -> (d 2)")
        # Create a n_ctx x rotary_dim tensor, where each column is an arithmetic sequence of angles in that frequency
        angles = pos[:, None] / freq[None, :]
        return torch.sin(angles).to(dtype), torch.cos(angles).to(dtype)

    def rotary_rotate_qk(
        self,
        q: Float[torch.Tensor, "batch q_pos head_index d_head"],
        k: Float[torch.Tensor, "batch k_pos head_index d_head"],
    ) -> tuple[
        Float[torch.Tensor, "batch q_pos head_index d_head"],
        Float[torch.Tensor, "batch k_pos head_index d_head"],
    ]:
        # We first apply standard q and k calculation
        q = self.apply_rotary(q)
        k = self.apply_rotary(k)
        return q, k

    def rotate_every_two(
        self, x: Float[torch.Tensor, "... rotary_dim"]
    ) -> Float[torch.Tensor, "... rotary_dim"]:
        """
        Rotary helper function, splits x into blocks of size 2 along the final axis and maps [x0, x1] to [-x1, x0]

        The final axis of x must have even length.

        GPT-NeoX and GPT-J do rotary subtly differently, see calculate_sin_cos_rotary for details.
        """
        rot_x = x.clone()
        if self.cfg.original_architecture in ["GPTNeoXForCausalLM", "LlamaForCausalLM"]:
            n = x.size(-1) // 2
            rot_x[..., :n] = -x[..., n:]
            rot_x[..., n:] = x[..., :n]
        else:
            rot_x[..., ::2] = -x[..., 1::2]
            rot_x[..., 1::2] = x[..., ::2]

        return rot_x

    def apply_rotary(
        self,
        x: Float[torch.Tensor, "... head_index d_head"],
        past_kv_pos_offset=0,
    ) -> Float[torch.Tensor, "... head_index d_head"]:
        """Apply rotary embeddings to the input x.

        Note that x may or may not have a batch dimension (e.g. no batch dimension if using vmap).
        """
        # Only apply rotary to first rotary_dim dimensions (eg, if rotary_dim=64 and d_head=256,
        # only apply to first 1/4 of dimensions)
        x_pos = x.size(1) if x.dim() == 4 else x.size(0)
        x_rot = x[..., : self.cfg.rotary_dim]
        x_pass = x[..., self.cfg.rotary_dim :]
        x_flip = self.rotate_every_two(x_rot)
        rotary_cos = cast(Tensor, self.rotary_cos)
        rotary_sin = cast(Tensor, self.rotary_sin)
        x_rotated = (
            x_rot * rotary_cos[past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :]
            + x_flip * rotary_sin[past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :]
        )
        return torch.cat([x_rotated, x_pass], dim=-1)


class AttentionOut(nn.Module):
    def __init__(
        self,
        cfg: SequentialTransformerConfig,
    ):
        """Attention Block - params have shape [head_index, d_model, d_head] (or [head_index, d_head, d_model] for W_O) and multiply on the right. attn_scores refers to query key dot product immediately before attention softmax

        Convention: All attention pattern-style matrices have shape [..., head_index, query_pos,
        key_pos]

        Supports rotary attention.

        This code was taken mostly verbatim from:
        https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/components.py

        Args:
            cfg (SequentialTransformerConfig): Config
        """
        super().__init__()
        self.cfg = cfg
        self.W_O = nn.Parameter(
            torch.empty(self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model, dtype=cfg.dtype)
        )
        self.b_O = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=cfg.dtype))

        # Create a max_ctx x max_ctx mask, with True iff that query position
        # can attend to that key position (query is first axis, key is second axis)
        causal_mask: Bool[Tensor, "pos pos"] = torch.tril(
            torch.ones((self.cfg.n_ctx, self.cfg.n_ctx)).bool()
        )
        self.register_buffer("mask", causal_mask)

        self.register_buffer("IGNORE", torch.tensor(-1e5))

        # attn_scale is a constant that we divide the attention scores by pre-softmax.
        # I'm not entirely sure why it matters, but it's probably a mix of softmax not being
        # scale invariant and numerical stability?
        if self.cfg.use_attn_scale:
            self.attn_scale = np.sqrt(self.cfg.d_head)
        else:
            self.attn_scale = 1.0

    def forward(
        self,
        residual: Float[Tensor, "... pos d_model"],
        q: Float[Tensor, "... pos n_head_times_d_head"],
        k: Float[Tensor, "... pos n_head_times_d_head"],
        v: Float[Tensor, "... pos n_head_times_d_head"],
    ) -> tuple[Float[Tensor, "... pos d_model"], Float[Tensor, "... pos d_model"]]:
        """Forward through the entire attention block.

        Args:
            residual (Float[Tensor, "... pos d_model]): The "pure" residual stream
            q (Float[Tensor, "... pos n_head_times_d_head]): The query tensor
            k (Float[Tensor, "... pos n_head_times_d_head]): The key tensor
            v (Float[Tensor, "... pos n_head_times_d_head]): The value tensor
        """

        # Separate the last dimension into head_index and d_head (undo the operation from AttentionIn)
        q = einops.rearrange(
            q,
            "... pos (head_index d_head) -> ... pos head_index d_head",
            head_index=self.cfg.n_heads,
        )
        k = einops.rearrange(
            k,
            "... pos (head_index d_head) -> ... pos head_index d_head",
            head_index=self.cfg.n_heads,
        )
        v = einops.rearrange(
            v,
            "... pos (head_index d_head_v) -> ... pos head_index d_head_v",
            head_index=self.cfg.n_heads,
        )

        in_dtype = v.dtype

        if in_dtype not in [torch.float32, torch.float64]:
            # If using 16 bits, increase the precision to avoid numerical instabilities
            q = q.to(torch.float32)
            k = k.to(torch.float32)
        attn_scores = (
            einsum(
                "... query_pos head_index d_head, \
                        ... key_pos head_index d_head \
                        -> ... head_index query_pos key_pos",
                q,
                k,
            )
            / self.attn_scale
        )  # [..., head_index, query_pos, key_pos]

        # Only supports causal attention (not bidirectional)
        # If causal attention, we mask it to only attend backwards. If bidirectional, we don't mask.
        attn_scores = self.apply_causal_mask(attn_scores)  # [..., head_index, query_pos, key_pos]

        pattern = F.softmax(attn_scores, dim=-1)  # [..., head_index, query_pos, key_pos]
        pattern = pattern.to(in_dtype)
        z = einsum(
            "... key_pos head_index d_head, \
                ... head_index query_pos key_pos -> \
                ... query_pos head_index d_head",
            v,
            pattern,
        )  # [..., pos, head_index, d_head]

        out = (
            (
                einsum(
                    "... pos head_index d_head, \
                            head_index d_head d_model -> \
                            ... pos d_model",
                    z,
                    self.W_O,
                )
            )
            + self.b_O
        )  # [..., pos, d_model]

        return residual, out

    def apply_causal_mask(
        self,
        attn_scores: Float[Tensor, "... head_index pos pos_plus_past_kv_pos_offset"],
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
        self.b_in = nn.Parameter(torch.zeros(self.cfg.d_mlp, dtype=cfg.dtype))

    def forward(
        self,
        residual: Float[Tensor, "... d_model"],
        x: Float[Tensor, "... d_model"],
    ) -> tuple[Float[Tensor, "... d_model"], Float[Tensor, "... d_model"]]:
        pre_act = einsum("... d_model, d_model d_mlp -> ... d_mlp", x, self.W_in) + self.b_in
        return residual, pre_act


class MLPAct(nn.Module):
    def __init__(self, cfg: SequentialTransformerConfig):
        super().__init__()
        self.cfg = cfg

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
        residual: Float[Tensor, "... d_model"],
        pre_act: Float[Tensor, "... d_model"],
    ) -> tuple[Float[Tensor, "... d_model"], Float[Tensor, "... d_model"]]:
        # Technically, all these einsums could be done with a single matmul, but this is more readable.
        post_act = self.act_fn(pre_act)  # [..., d_mlp]
        return residual, post_act


class MLPOut(nn.Module):
    def __init__(self, cfg: SequentialTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.W_out = nn.Parameter(torch.empty(self.cfg.d_mlp, self.cfg.d_model, dtype=cfg.dtype))
        self.b_out = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=cfg.dtype))

    def forward(
        self,
        residual: Float[Tensor, "... d_model"],
        post_act: Float[Tensor, "... d_model"],
    ) -> tuple[Float[Tensor, "... d_model"], Float[Tensor, "... d_model"]]:
        out = (
            einsum(
                "... d_mlp, d_mlp d_model -> ... d_model",
                post_act,
                self.W_out,
            )
            + self.b_out
        )
        return residual, out


class LayerNormPre(torch.nn.Module):
    """Sequential version of transformer-lens' LayerNormPre.

    A standard LayerNorm without the element-wise affine parameters.
    """

    def __init__(self, cfg: SequentialTransformerConfig, return_residual: bool = False):
        super().__init__()
        self.cfg = cfg
        self.return_residual = return_residual

    def forward(
        self,
        residual: Float[Tensor, "... d_model"],
    ) -> Union[
        Float[Tensor, "... d_model"],
        tuple[Float[Tensor, "... d_model"], Float[Tensor, "... d_model"]],
    ]:
        out = layer_norm(residual.clone(), self.cfg.eps)
        if self.return_residual:
            return residual, out
        else:
            return out


class DualLayerNormPre(torch.nn.Module):
    """A version of LayerNormPre that handles two inputs.

    Simply passes through the second input as the new residual. This is used for models like pythia
    that use parallel attention and mlp blocks.
    """

    def __init__(self, cfg: SequentialTransformerConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        residual: Float[Tensor, "... d_model"],
        attn_resid: Float[Tensor, "... d_model"],
    ) -> tuple[Float[Tensor, "... d_model"], Float[Tensor, "... d_model"]]:
        """Forward through the module.

        Args:
            residual: The raw residual stream.
            attn_resid: The residual stream after adding the attention block.

        Returns:
            - The residual stream after adding the attention block. This will be considered the
                "new" residual stream in the subsequent (MLP) layers.
            - The application of layer norm to the raw residual stream.
        """
        out = layer_norm(residual.clone(), self.cfg.eps)
        return attn_resid, out


class LayerNormPreFolded(torch.nn.Module):
    """A version of LayerNormPre where we assume the input has a constant final dimension."""

    def __init__(self, cfg: SequentialTransformerConfig, return_residual: bool = False):
        super().__init__()
        self.cfg = cfg
        self.return_residual = return_residual

    def forward(
        self,
        residual: Float[Tensor, "... d_model"],
    ) -> Union[
        Float[Tensor, "... d_model"],
        tuple[Float[Tensor, "... d_model"], Float[Tensor, "... d_model"]],
    ]:
        x0 = residual[..., :-1].clone()  # [..., length-1]

        x0_out = layer_norm(x0, self.cfg.eps)
        out = torch.cat([x0_out, residual[..., -1:]], dim=-1)  # [..., length]
        if self.return_residual:
            return residual, out
        else:
            return out


class DualLayerNormPreFolded(torch.nn.Module):
    """A version of LayerNormPreFolded that handles two inputs."""

    def __init__(self, cfg: SequentialTransformerConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        residual: Float[Tensor, "... d_model"],
        attn_resid: Float[Tensor, "... d_model"],
    ) -> tuple[Float[Tensor, "... d_model"], Float[Tensor, "... d_model"]]:
        """Forward through the module.

        Args:
            residual: The raw residual stream.
            attn_resid: The residual stream after adding the attention block.

        Returns:
            - The residual stream after adding the attention block. This will be considered the
                "new" residual stream in the subsequent (MLP) layers.
            - The application of layer norm to the raw residual stream.
        """
        x0 = residual[..., :-1].clone()  # [..., length-1]

        x0_out = layer_norm(x0, self.cfg.eps)
        out = torch.cat([x0_out, residual[..., -1:]], dim=-1)
        return attn_resid, out


class IdentitySplit(torch.nn.Module):
    """Identity that splits the input into two outputs."""

    def __init__(self, cfg: SequentialTransformerConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        residual: Float[Tensor, "... d_model"],
    ) -> tuple[Float[Tensor, "... d_model"], Float[Tensor, "... d_model"]]:
        return residual, residual.clone()


# Map from module names in SequentialTransformer to the corresponding component modules
SEQUENTIAL_COMPONENT_REGISTRY = {
    "embed": Embed,
    "pos_embed": PosEmbed,
    "add_embed": Add,
    "attn_in": AttentionIn,
    "attn_out": AttentionOut,
    "add_resid1": Add,
    "mlp_in": MLPIn,
    "mlp_act": MLPAct,
    "mlp_out": MLPOut,
    "add_resid2": Add,
    "unembed": Unembed,
}