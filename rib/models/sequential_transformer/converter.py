import torch
from transformer_lens import HookedTransformer


def convert_tlens_weights(
    seq_tf_keys: list[str], tlens_model: HookedTransformer
) -> dict[str, torch.Tensor]:
    """Converts the weights from a transformer lens model to a sequential transformer state dict.

    Note that this algorithm assumes that the seq_tf_keys are ordered by section number, which
    will be the case if pulled from SequentialTransformer.state_dict().
    """

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
    mlp_names: list[str] = ["W_in", "b_in", "W_out", "b_out"]

    state_dict: dict[str, torch.Tensor] = {}
    assert "sections.pre.0.W_E" in seq_tf_keys and "sections.pre.1.W_pos" in seq_tf_keys, (
        "We currently only support the token and positional embeddings in the `pre` section."
        "This will occur if the first element of node_layers is not in embed_module_names"
    )
    "We currently only support the token embedding"

    # The current block number in the tlens model
    block_num: int = 0
    # The names of all params in the current block
    tlens_block_names = set(attn_names + mlp_names)
    for seq_tf_key in seq_tf_keys:
        # Check if tlens_block_names is empty and if so, increment block_num and reset
        if len(tlens_block_names) == 0:
            block_num += 1
            tlens_block_names = set(attn_names + mlp_names)

        param_name = seq_tf_key.split(".")[-1]
        if param_name == "W_E":
            state_dict[seq_tf_key] = tlens_model.embed.W_E
        elif param_name == "W_pos":
            state_dict[seq_tf_key] = tlens_model.pos_embed.W_pos
        elif param_name == "W_U":
            state_dict[seq_tf_key] = tlens_model.unembed.W_U
        elif param_name == "b_U":
            state_dict[seq_tf_key] = tlens_model.unembed.b_U
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
            state_dict[seq_tf_key] = tlens_param_val

    return state_dict
