"""This module contains alternative components for transformer-lens models."""
from typing import Union

import torch
from jaxtyping import Float
from transformer_lens import HookedTransformerConfig
from transformer_lens.hook_points import HookPoint


class LayerNormPre_Folded(torch.nn.Module):
    """Drop-in replacement for transformer_lens.components.LayerNormPre"""

    def __init__(self, cfg: Union[dict, HookedTransformerConfig]):
        """LayerNormPre - the 'center and normalise' part of LayerNorm. Length is
        normally d_model, but is d_mlp for softmax. Not needed as a parameter. This
        should only be used in inference mode after folding in LayerNorm weights

        Folded: the last hidden dimension is the constant function 1, and does not participate in the layernorm
        """
        super().__init__()
        if isinstance(cfg, dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.eps = self.cfg.eps

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos]
        # Hook Normalized captures LN output - here it's a vector with std 1 and mean 0
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(
        self,
        x: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos head_index d_model"],
        ],
    ) -> Union[
        Float[torch.Tensor, "batch pos d_model"],
        Float[torch.Tensor, "batch pos head_index d_model"],
    ]:
        x0 = x[..., :-1]  # [batch, pos, length-1]

        x0 = x0 - x0.mean(-1, keepdim=True)  # [batch, pos, length-1]
        scale: Float[torch.Tensor, "batch pos 1"] = self.hook_scale(
            (x0.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )
        x0 = x0 / scale  # [batch, pos, length-1]
        x = torch.cat([x0, x[..., -1:]], dim=-1)  # [batch, pos, length]
        return self.hook_normalized(x)
