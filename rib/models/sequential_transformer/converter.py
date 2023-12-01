from typing import Literal

import torch
from transformer_lens import HookedTransformer

from rib.models import SequentialTransformer


def convert_tlens_weights(
    seq_model: SequentialTransformer,
    tlens_model: HookedTransformer,
    positional_embedding_type: Literal["standard", "rotary"],
) -> dict[str, torch.Tensor]:
    """Converts the weights from a transformer lens model to a sequential transformer state dict.

    Note that this algorithm assumes that the seq_param_names are ordered by section number, which
    will be the case if pulled from SequentialTransformer.state_dict().
    """
    seq_param_names = list(seq_model.state_dict().keys())
    named_buffers = dict(seq_model.named_buffers())

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

            buffer_val = named_buffers.get(seq_param_name)
            if buffer_val is not None:
                assert buffer_val.dtype == tlens_param_val.dtype, (
                    f"Buffer {seq_param_name} has dtype {buffer_val.dtype} but tlens_param_val "
                    f"has dtype {tlens_param_val.dtype}. It is not a good idea to map parameters "
                    f"of different dtypes."
                )
                if not torch.allclose(buffer_val, tlens_param_val.to(buffer_val.dtype)):
                    raise ValueError(
                        f"Buffer {seq_param_name} does not match between seq_model and tlens_model"
                    )
            state_dict[seq_param_name] = tlens_param_val

    return state_dict
