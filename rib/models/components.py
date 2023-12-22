"""Defines components to be used in a sequential transformer architecture."""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional, Type, Union, cast

import einops
import numpy as np
import torch
from fancy_einsum import einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from rib.models import SequentialTransformerConfig

from rib.models.utils import gelu_new, layer_norm


class SequentialComponent(ABC, nn.Module):
    """Abstract base class for all components of a sequential model."""

    def __init__(self, *_args, in_dims: Optional[tuple[int, ...]], **_kwargs):
        super().__init__()
        self._in_dims = in_dims

    @property
    def in_dims(self) -> Optional[tuple[int, ...]]:
        return self._in_dims

    @property
    @abstractmethod
    def out_dims(self) -> Optional[tuple[int, ...]]:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class MultiSequential(nn.Sequential):
    """Sequential module where containing modules that may have multiple inputs and outputs."""

    def __init__(self, *args: SequentialComponent, in_dims: Optional[tuple[int, ...]]):
        super().__init__(*args)
        self._in_dims = in_dims

    @property
    def in_dims(self) -> Optional[tuple[int, ...]]:
        return self._in_dims

    @property
    def out_dims(self) -> Optional[tuple[int, ...]]:
        final_component = self[-1]
        assert isinstance(final_component, SequentialComponent)
        return final_component.out_dims

    def forward(self, *inputs):
        for module in self._modules.values():
            inputs = inputs if isinstance(inputs, tuple) else (inputs,)
            inputs = module(*inputs)
        return inputs


class Embed(SequentialComponent):
    def __init__(
        self, cfg: "SequentialTransformerConfig", in_dims: None, return_tokens: bool = True
    ):
        super().__init__(in_dims=None)
        self.W_E: Float[Tensor, "d_vocab d_model"] = nn.Parameter(
            torch.empty(cfg.d_vocab, cfg.d_model, dtype=cfg.dtype)
        )
        self.return_tokens = return_tokens

    @property
    def out_dims(self) -> Optional[tuple[int]]:
        # If we're returning tokens, there must be a pos embed layer after this, so we don't need
        # to return the output dimension
        return None if self.return_tokens else (self.W_E.shape[-1],)

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
            - The input tokens (if self.return_tokens is True)
            - The token embeddings
        """
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        token_embeddings: Float[Tensor, "... d_model"] = self.W_E[tokens, :]
        if self.return_tokens:
            return tokens, token_embeddings
        else:
            return token_embeddings


class PosEmbed(SequentialComponent):
    def __init__(self, cfg: "SequentialTransformerConfig", in_dims: None):
        super().__init__(in_dims=None)
        self.W_pos = nn.Parameter(torch.empty(cfg.n_ctx, cfg.d_model, dtype=cfg.dtype))

    @property
    def out_dims(self) -> tuple[int, int]:
        return (self.W_pos.shape[-1], self.W_pos.shape[-1])

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


class Unembed(SequentialComponent):
    def __init__(
        self, cfg: "SequentialTransformerConfig", in_dims: tuple[int], last_pos_only: bool = False
    ):
        super().__init__(in_dims=in_dims)
        self.last_pos_only = last_pos_only
        self.W_U: Float[Tensor, "d_model d_vocab"] = nn.Parameter(
            torch.empty(cfg.d_model, cfg.d_vocab, dtype=cfg.dtype)
        )
        self.b_U: Float[Tensor, "d_vocab"] = nn.Parameter(torch.zeros(cfg.d_vocab, dtype=cfg.dtype))

    @property
    def out_dims(self) -> tuple[int]:
        return (self.W_U.shape[-1],)

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


class Add(SequentialComponent):
    """Add the residual stream to the output stream of the previous module.

    If last_pos_only is True, only the last position index is returned.

    If return_residual is True, the residual stream is returned as well as the output stream. This
    is needed for add_resid1 when we have parallel attention and mlp streams (as in pythia).
    """

    def __init__(
        self,
        cfg: "SequentialTransformerConfig",
        in_dims: tuple[int, int],
        last_pos_only: bool = False,
        return_residual: bool = False,
    ):
        super().__init__(in_dims=in_dims)
        self._in_dims = in_dims
        self.last_pos_only = last_pos_only
        self.return_residual = return_residual

    @property
    def out_dims(self) -> Union[tuple[int], tuple[int, int]]:
        assert self.in_dims is not None and len(self.in_dims) == 2, "Add must have two inputs"
        summed = sum(self.in_dims)
        return (self.in_dims[0], summed) if self.return_residual else (summed,)

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


class AttentionIn(SequentialComponent):
    rotary_sin: Float[Tensor, "n_ctx rotary_dim"]
    rotary_cos: Float[Tensor, "n_ctx rotary_dim"]

    def __init__(self, cfg: "SequentialTransformerConfig", in_dims: tuple[int, int]):
        """Attention Block part 1 (computes q, k, v) - params have shape [head_index, d_model, d_head] (or [head_index, d_head, d_model] for W_O) and multiply on the right. attn_scores refers to query key dot product immediately before attention softmax

        Convention: All attention pattern-style matrices have shape [..., head_index, query_pos,
        key_pos]

        Supports rotary attention.

        This code was taken mostly verbatim from:
        https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/components.py

        """
        super().__init__(in_dims=in_dims)
        self.cfg = cfg

        self.W_Q = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head, dtype=cfg.dtype))
        self.W_K = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head, dtype=cfg.dtype))
        self.W_V = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head, dtype=cfg.dtype))
        self.b_Q = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype))
        self.b_K = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype))
        self.b_V = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype))

        if cfg.positional_embedding_type == "rotary":
            assert cfg.rotary_dim is not None, "rotary_dim must be specified for rotary attention"
            # Applies a rotation to each two-element chunk of keys and queries pre dot producting to bake in relative position. See HookedTransformerConfig for details
            sin, cos = self.calculate_sin_cos_rotary(cfg.rotary_dim, cfg.n_ctx, dtype=cfg.dtype)
            self.register_buffer("rotary_sin", sin)
            self.register_buffer("rotary_cos", cos)

    @property
    def out_dims(self) -> tuple[int, int, int, int]:
        return (self.W_Q.shape[1], self.W_Q.shape[-1], self.W_K.shape[-1], self.W_V.shape[-1])

    def forward(
        self, residual: Float[Tensor, "... pos d_model"], x: Float[Tensor, "... pos d_model"]
    ) -> tuple[
        Float[Tensor, "... pos d_model"],
        Float[Tensor, "... pos n_head_times_d_head"],
        Float[Tensor, "... pos n_head_times_d_head"],
        Float[Tensor, "... pos n_head_times_d_head"],
    ]:
        """Forward through the entire attention block.

        Args:
            residual (Float[Tensor, "... pos d_model]): The "pure" residual stream
            x (Float[Tensor, "... pos d_model]): The normed residual stream (the input to the
                attention block)

        Returns:
            - The residual stream
            - The query tensor
            - The key tensor
            - The value tensor
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
        self, rotary_dim: int, n_ctx: int, base: int = 10000, dtype: torch.dtype = torch.float32
    ) -> tuple[Float[Tensor, "n_ctx rotary_dim"], Float[Tensor, "n_ctx rotary_dim"]]:
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
        q: Float[Tensor, "batch q_pos head_index d_head"],
        k: Float[Tensor, "batch k_pos head_index d_head"],
    ) -> tuple[
        Float[Tensor, "batch q_pos head_index d_head"],
        Float[Tensor, "batch k_pos head_index d_head"],
    ]:
        # We first apply standard q and k calculation
        q = self.apply_rotary(q)
        k = self.apply_rotary(k)
        return q, k

    def rotate_every_two(
        self, x: Float[Tensor, "... rotary_dim"]
    ) -> Float[Tensor, "... rotary_dim"]:
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
        self, x: Float[Tensor, "... head_index d_head"], past_kv_pos_offset=0
    ) -> Float[Tensor, "... head_index d_head"]:
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


class AttentionScores(nn.Module):
    mask: Bool[Tensor, "pos pos"]
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: "SequentialTransformerConfig", use_local_attn: bool) -> None:
        """Compute the attention scores.

        This module exists purely so that we can hook the attention
        scores (and pattern). Note that unlike the other modules, does not return a tuple but just a
        single tensor.
        """
        super().__init__()
        # Create a max_ctx x max_ctx mask, with True iff that query position
        # can attend to that key position (query is first axis, key is second axis)
        causal_mask: Bool[Tensor, "pos pos"] = torch.tril(torch.ones((cfg.n_ctx, cfg.n_ctx)).bool())
        if use_local_attn:
            assert cfg.original_architecture == "GPTNeoForCausalLM"
            # Only attend to the previous window_size positions
            # so mask true iff query - window_size < key <= query
            # we fix window_size to 256 (the default for GPTNeo) instead of putting into config
            window_size = 256
            causal_mask = torch.triu(causal_mask, 1 - window_size)
        self.register_buffer("mask", causal_mask)

        self.register_buffer("IGNORE", torch.tensor(-torch.inf))

        # attn_scale is a constant that we divide the attention scores by pre-softmax.
        # I'm not entirely sure why it matters, but it's probably a mix of softmax not being
        # scale invariant and numerical stability?
        if cfg.use_attn_scale:
            self.attn_scale = np.sqrt(cfg.d_head)
        else:
            self.attn_scale = 1.0

    def forward(
        self,
        q: Float[Tensor, "... query_pos head_index d_head"],
        k: Float[Tensor, "... key_pos head_index d_head"],
    ) -> Float[Tensor, "... head_index query_pos key_pos"]:
        """Calculate the attention scores.

        Args:
            q: The query tensor.
            k: The key tensor.
            attn_scale: The scale to divide the attention scores by.

        Returns:
            The attention scores.
        """
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
        return attn_scores

    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "... head_index pos pos_plus_past_kv_pos_offset"]
    ):
        # The key context length is the number of positions in the past - this includes all positions in the cache
        # If not caching, query_ctx_length == key_ctx_length
        key_ctx_length = attn_scores.size(-1)
        query_ctx_length = attn_scores.size(-2)

        mask: Bool[Tensor, "pos pos"] = cast(Tensor, self.mask)
        return torch.where(
            mask[:query_ctx_length, :key_ctx_length],
            attn_scores,
            cast(Tensor, self.IGNORE),
        )


class AttentionOut(SequentialComponent):
    def __init__(
        self,
        cfg: "SequentialTransformerConfig",
        in_dims: tuple[int, int, int, int],
        use_local_attn: bool = False,
    ):
        """Attention Block part 2 (takes q, k, v; computes out) - params have shape [head_index,
        d_model, d_head] (or [head_index, d_head, d_model] for W_O) and multiply on the right.
        attn_scores refers to query key dot product immediately before attention softmax

        Convention: All attention pattern-style matrices have shape [..., head_index, query_pos,
        key_pos]

        Supports rotary attention.

        This code was taken mostly verbatim from:
        https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/components.py
        """
        super().__init__(in_dims=in_dims)
        self.n_heads = cfg.n_heads
        self.attention_scores = AttentionScores(cfg, use_local_attn=use_local_attn)
        self.W_O = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head, cfg.d_model, dtype=cfg.dtype))
        self.b_O = nn.Parameter(torch.zeros(cfg.d_model, dtype=cfg.dtype))

    @property
    def out_dims(self) -> tuple[int, int]:
        return (self.W_O.shape[-1], self.W_O.shape[-1])

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

        Returns:
            - The residual stream
            - The attention output
        """

        # Separate the last dimension into head_index and d_head (undo the operation from AttentionIn)
        q = einops.rearrange(
            q,
            "... pos (head_index d_head) -> ... pos head_index d_head",
            head_index=self.n_heads,
        )
        k = einops.rearrange(
            k,
            "... pos (head_index d_head) -> ... pos head_index d_head",
            head_index=self.n_heads,
        )
        v = einops.rearrange(
            v,
            "... pos (head_index d_head_v) -> ... pos head_index d_head_v",
            head_index=self.n_heads,
        )

        in_dtype = v.dtype

        if in_dtype not in [torch.float32, torch.float64]:
            # If using 16 bits, increase the precision to avoid numerical instabilities
            q = q.to(torch.float32)
            k = k.to(torch.float32)
        attn_scores = self.attention_scores(q, k)  # [..., head_index, query_pos, key_pos]

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


class MLPIn(SequentialComponent):
    def __init__(self, cfg: "SequentialTransformerConfig", in_dims: tuple[int, int]):
        super().__init__(in_dims=in_dims)
        self.W_in = nn.Parameter(torch.empty(cfg.d_model, cfg.d_mlp, dtype=cfg.dtype))
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp, dtype=cfg.dtype))

    @property
    def out_dims(self) -> tuple[int, int]:
        return (self.W_in.shape[0], self.W_in.shape[-1])

    def forward(
        self, residual: Float[Tensor, "... d_model"], x: Float[Tensor, "... d_model"]
    ) -> tuple[Float[Tensor, "... d_model"], Float[Tensor, "... d_mlp"]]:
        pre_act = einsum("... d_model, d_model d_mlp -> ... d_mlp", x, self.W_in) + self.b_in
        return residual, pre_act


class MLPAct(SequentialComponent):
    def __init__(self, cfg: "SequentialTransformerConfig", in_dims: tuple[int, int]):
        super().__init__(in_dims=in_dims)

        self.act_fn: Callable[[Float[Tensor, "... d_model"]], Tensor]
        if cfg.act_fn == "relu":
            self.act_fn = F.relu
        elif cfg.act_fn == "gelu":
            self.act_fn = F.gelu
        elif cfg.act_fn == "gelu_new":
            self.act_fn = gelu_new
        else:
            raise ValueError(f"Invalid activation function name: {cfg.act_fn}")

    @property
    def out_dims(self) -> tuple[int, int]:
        return cast(tuple[int, int], self.in_dims)

    def forward(
        self, residual: Float[Tensor, "... d_model"], pre_act: Float[Tensor, "... d_mlp"]
    ) -> tuple[Float[Tensor, "... d_model"], Float[Tensor, "... d_mlp"]]:
        # Technically, all these einsums could be done with a single matmul, but this is more readable.
        post_act = self.act_fn(pre_act)  # [..., d_mlp]
        return residual, post_act


class MLPOut(SequentialComponent):
    def __init__(self, cfg: "SequentialTransformerConfig", in_dims: tuple[int, int]):
        super().__init__(in_dims=in_dims)
        self.W_out = nn.Parameter(torch.empty(cfg.d_mlp, cfg.d_model, dtype=cfg.dtype))
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model, dtype=cfg.dtype))

    @property
    def out_dims(self) -> tuple[int, int]:
        return (self.W_out.shape[-1], self.W_out.shape[-1])

    def forward(
        self, residual: Float[Tensor, "... d_model"], post_act: Float[Tensor, "... d_mlp"]
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


class LayerNormPre(SequentialComponent):
    """Sequential version of transformer-lens' LayerNormPre.

    A standard LayerNorm without the element-wise affine parameters.
    """

    def __init__(
        self, cfg: "SequentialTransformerConfig", in_dims: tuple[int], return_residual: bool = False
    ):
        super().__init__(in_dims=in_dims)
        self.eps = cfg.eps
        self.return_residual = return_residual

    @property
    def out_dims(self) -> Union[tuple[int], tuple[int, int]]:
        in_dims = cast(tuple[int], self.in_dims)
        return in_dims if not self.return_residual else (in_dims[0], in_dims[0])

    def forward(
        self, residual: Float[Tensor, "... d_model"]
    ) -> Union[
        Float[Tensor, "... d_model"],
        tuple[Float[Tensor, "... d_model"], Float[Tensor, "... d_model"]],
    ]:
        out = layer_norm(residual.clone(), self.eps)
        if self.return_residual:
            return residual, out
        else:
            return out


class DualLayerNormPre(SequentialComponent):
    """A version of LayerNormPre that handles two inputs.

    Simply passes through the second input as the new residual. This is used for models like pythia
    that use parallel attention and mlp blocks.
    """

    def __init__(self, cfg: "SequentialTransformerConfig", in_dims: tuple[int, int]):
        super().__init__(in_dims=in_dims)
        self.eps = cfg.eps

    @property
    def out_dims(self) -> tuple[int, int]:
        assert (
            self.in_dims is not None and len(self.in_dims) == 2
        ), "DualLayerNormPre must have two inputs"
        # Note that the dims switch in this layer
        return (self.in_dims[1], self.in_dims[0])

    def forward(
        self, residual: Float[Tensor, "... d_model"], attn_resid: Float[Tensor, "... d_model"]
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
        out = layer_norm(residual.clone(), self.eps)
        return attn_resid, out


class LayerNormPreFolded(SequentialComponent):
    """A version of LayerNormPre where we assume the input has a constant final dimension."""

    def __init__(
        self, cfg: "SequentialTransformerConfig", in_dims: tuple[int], return_residual: bool = False
    ):
        super().__init__(in_dims=in_dims)
        self.eps = cfg.eps
        self.return_residual = return_residual

    @property
    def out_dims(self) -> Union[tuple[int], tuple[int, int]]:
        in_dims = cast(tuple[int], self.in_dims)
        return in_dims if not self.return_residual else (in_dims[0], in_dims[0])

    def forward(
        self, residual: Float[Tensor, "... d_model"]
    ) -> Union[
        Float[Tensor, "... d_model"],
        tuple[Float[Tensor, "... d_model"], Float[Tensor, "... d_model"]],
    ]:
        x0 = residual[..., :-1].clone()  # [..., length-1]

        x0_out = layer_norm(x0, self.eps)
        out = torch.cat([x0_out, residual[..., -1:]], dim=-1)  # [..., length]
        if self.return_residual:
            return residual, out
        else:
            return out


class DualLayerNormPreFolded(SequentialComponent):
    """A version of LayerNormPreFolded that handles two inputs."""

    def __init__(self, cfg: "SequentialTransformerConfig", in_dims: tuple[int, int]):
        super().__init__(in_dims=in_dims)
        self.eps = cfg.eps

    @property
    def out_dims(self) -> tuple[int, int]:
        in_dims = cast(tuple[int, int], self.in_dims)
        # Note that the dims switch in this layer
        return (in_dims[1], in_dims[0])

    def forward(
        self, residual: Float[Tensor, "... d_model"], attn_resid: Float[Tensor, "... d_model"]
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

        x0_out = layer_norm(x0, self.eps)
        out = torch.cat([x0_out, residual[..., -1:]], dim=-1)
        return attn_resid, out


class IdentitySplit(SequentialComponent):
    """Identity that splits the input into two outputs."""

    def __init__(self, cfg: "SequentialTransformerConfig", in_dims: tuple[int]):
        super().__init__(in_dims=in_dims)

    @property
    def out_dims(self) -> tuple[int, int]:
        in_dims = cast(tuple[int], self.in_dims)
        return (in_dims[0], in_dims[0])

    def forward(
        self, residual: Float[Tensor, "... d_model"]
    ) -> tuple[Float[Tensor, "... d_model"], Float[Tensor, "... d_model"]]:
        return residual, residual.clone()


class Identity(SequentialComponent):
    """Identity module."""

    def __init__(self, cfg: "SequentialTransformerConfig", in_dims: tuple[int]):
        super().__init__(in_dims=in_dims)

    @property
    def out_dims(self) -> tuple[int]:
        return cast(tuple[int], self.in_dims)

    def forward(self, residual: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return residual


# Map from module names in SequentialTransformer to the corresponding component modules
SEQUENTIAL_COMPONENT_REGISTRY: dict[str, Type[SequentialComponent]] = {
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
