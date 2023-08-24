from typing import Union

import torch
from pydantic import BaseModel, ConfigDict, field_validator

from rib.types import TORCH_DTYPES


class SequentialTransformerConfig(BaseModel):
    """Config for the sequential transformer model.

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
            memory efficient but easier for analysis)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_layers: int
    d_model: int
    d_head: int
    n_heads: int
    d_mlp: int
    d_vocab: int
    n_ctx: int
    act_fn: str
    normalization_type: str
    eps: float
    dtype: torch.dtype
    use_attn_scale: bool
    use_split_qkv_input: bool

    @field_validator("dtype")
    @classmethod
    def set_dtype(cls, v: Union[str, torch.dtype]) -> torch.dtype:
        """Verify and if needed convert the dtype to a torch.dtype."""
        if isinstance(v, torch.dtype):
            if v not in TORCH_DTYPES.values():
                raise ValueError(f"Unsupported dtype {v}")
            return v
        elif isinstance(v, str):
            if v not in TORCH_DTYPES:
                raise ValueError(f"Unsupported dtype {v}")
            return TORCH_DTYPES[v]
