"""This module contains functionality for changing the structure of transformer-lens models.

Credit to Jorn Stohler, whose code on folding in biases was basically copied to here.
"""
from dataclasses import asdict

import torch
from transformer_lens import HookedTransformer
from transformer_lens.components import MLP, Attention, LayerNormPre
from transformer_lens.utils import gelu_new

from rib.models.tlens_components import LayerNormPre_Folded
from rib.utils import find_root


def embedding_fold_bias(model: HookedTransformer) -> None:
    """Modifies the embedding layer to output an extra feature of ones.

    Since future layers will have their biases folded into their weights, we need our embedding
    layer to output an extra feature of ones so that the bias is preserved.

    To do this, we have the positional embedding layer output a feature of ones, and the token
    embedding layer output a feature of zeros. When these are added together, the result is a
    feature of ones.

    Args:
        model (HookedTransformer): A transformerlens model.

    Returns:
        None: The model is modified in-place.
    """
    assert hasattr(model, "embed"), "Model does not have a token embedding module"
    assert hasattr(model, "pos_embed"), "Model does not have a positional embedding module"
    device = model.embed.W_E.device
    dtype = model.embed.W_E.dtype
    model.embed.W_E.data = torch.cat(
        [
            model.embed.W_E.data,
            torch.zeros(model.cfg.d_vocab, 1, device=device, dtype=dtype),
        ],
        dim=1,
    )  # [d_vocab, d_model_new]

    model.pos_embed.W_pos.data = torch.cat(
        [
            model.pos_embed.W_pos.data,
            torch.ones(model.cfg.n_ctx, 1, device=device, dtype=dtype),
        ],
        dim=1,
    )  # [n_ctx, d_model_new]


def attention_fold_bias(attn: Attention) -> None:
    """Modifies the attention layer to fold in biases into its weights.

    For the output matrix (W_O), we concatenate a feature of zeros to the end of the matrix. This is
    because this output is added together with the residual stream, which already has a feature of
    ones concatenated to the end of it. Adding both of these together will result in a feature of
    ones, which is what is needed to preserve the bias in the next layer.

    TODO: Fold in b_O into W_O. To do this, we need z to be (batch, q_pos, head_idx, d_head_new),
    and thus we need v to be (batch, j_pos, head_idx, d_head_new), and thus we need W_V to be
    (head_idx, d_model_new, d_head_new). This is not a trivial operation, we would have to modify
    also modify W_K and W_Q to create an attention pattern that has an extra dimension so that
    when QK is multiplied by V we get a z of the correct shape.

    Args:
        attn (Attention): A transformerlens attention layer.

    Returns:
        None: The attention layer is modified in-place.
    """

    attn.W_Q.data = torch.cat(
        [attn.W_Q.data, attn.b_Q.data[:, None]], dim=1
    )  # [head_idx, d_model_new, d_head_new]
    attn.b_Q.data = torch.zeros_like(attn.b_Q.data)

    attn.W_K.data = torch.cat(
        [attn.W_K.data, attn.b_K.data[:, None]], dim=1
    )  # [head_idx, d_model_new, d_head]
    attn.b_K.data = torch.zeros_like(attn.b_K.data)

    attn.W_V.data = torch.cat(
        [attn.W_V.data, attn.b_V.data[:, None]], dim=1
    )  # [head_idx, d_model_new, d_head]
    attn.b_V.data = torch.zeros_like(attn.b_V.data)

    # Temporary hack that avoids folding in b_O
    n_heads, d_head, _ = attn.W_O.shape
    attn.W_O.data = torch.cat(
        [
            attn.W_O.data,
            torch.zeros(n_heads, d_head, 1, dtype=attn.W_O.dtype, device=attn.W_O.device),
        ],
        dim=2,
    )  # [head_idx, d_head, d_model_new]
    attn.b_O.data = torch.cat(
        [
            attn.b_O.data,
            torch.zeros(1, device=attn.b_O.device, dtype=attn.b_O.dtype),
        ],
        dim=0,
    )  # [d_model_new]


def mlp_fold_bias(mlp: MLP) -> None:
    """Modifies the MLP layer to fold in biases into its weights.

    For the input matrix W_in of shape (d_model, d_mlp), we concat the bias vector to the row
    dimension and then add an extra column with all zeros and a single value at the end equal to
    the value that get transformed to 1 by the activation function (denoted root_1). I.e.
    W_in_folded will be of shape (d_model + 1, d_mlp + 1).
    If the activation function has a root_1 of 0 (such as ReLU), the value added to the end will be
    1. This will result in act_fn(x @ W_in_folded) giving the same result as act_fn(x @ W_in + b_in).

    For the output matrix W_out of shape (d_mlp, d_model), we concat the bias vector to the row
    dimension and then add an extra column of all zeros. I.e. W_out_folded will be of shape
    (d_mlp + 1, d_model + 1). Since the MLP block is added to the residual stream, which already
    has a feature of ones concatenated to the end of it, adding both of these together will result
    in a feature of ones, which is what is needed to preserve the bias in the next layer.

    Args:
        mlp (MLP): A transformerlens MLP layer.

    Returns:
        None: The MLP layer is modified in-place.
    """

    if mlp.act_fn == torch.nn.functional.relu:
        # relu(1) = 1
        root_one = 1.0
    elif mlp.act_fn == gelu_new:
        # Find the value of x such that act_fn(x) = 1
        root_one = find_root(
            lambda x: gelu_new(x) - 1.0,
            xmin=torch.tensor(-1.0),
            xmax=torch.tensor(4.0),
            tol=1e-11,
            max_iter=1000,
        )
    else:
        raise ValueError(f"Unsupported activation function {mlp.act_fn}.")

    mlp.W_in.data = torch.cat(
        [
            torch.cat([mlp.W_in.data, mlp.b_in.data[None, :]], dim=0),
            torch.cat(
                [
                    torch.zeros(mlp.W_in.shape[0], 1, device=mlp.W_in.device, dtype=mlp.W_in.dtype),
                    torch.ones(1, 1, device=mlp.W_in.device, dtype=mlp.W_in.dtype) * root_one,
                ],
                dim=0,
            ),
        ],
        dim=1,
    )  # [d_model_new, d_mlp_new]
    mlp.b_in.data = torch.zeros(mlp.b_in.shape[0] + 1, device=mlp.b_in.device, dtype=mlp.b_in.dtype)

    mlp.W_out.data = torch.cat(
        [
            torch.cat([mlp.W_out.data, mlp.b_out.data[None, :]], dim=0),
            torch.zeros(mlp.W_out.shape[0] + 1, 1, device=mlp.W_out.device, dtype=mlp.W_out.dtype),
        ],
        dim=1,
    )  # [d_mlp_new, d_model_new]
    mlp.b_out.data = torch.zeros(
        mlp.b_out.shape[0] + 1, device=mlp.b_out.device, dtype=mlp.b_out.dtype
    )  # [d_model_new]


def unembed_fold_bias(model: HookedTransformer) -> None:
    """Modifies the unembedding layer to fold in biases into its weights."""

    assert hasattr(model, "unembed"), "Model does not have an unembedding module"
    device = model.unembed.W_U.device
    dtype = model.unembed.W_U.dtype

    model.unembed.W_U.data = torch.cat(
        [model.unembed.W_U.data, model.unembed.b_U.data[None, :]], dim=0
    )
    model.unembed.b_U.data = torch.zeros(1, model.cfg.d_vocab, device=device, dtype=dtype)

    # transformer_lens.HookedRootModule has some internal stuff that needs to be reinitialized
    model.setup()


def model_fold_bias(model: HookedTransformer) -> None:
    """Fold the bias parameters into the weight parameters of the transformer-lens model.

    - Converts any instances of LayerNormPre into LayerNormPre_Folded.
    - Folds the biases in embedding, attention, mlp, unmbedding layers into their weight parameters.

    Args:
        model (HookedTransformer): A transformerlens model.

    Returns:
        None: The model is modified in-place to have its bias parameters folded into its weights
    """

    # Used to modify the cfg of the LayerNormPre modules
    lnpre_folded_cfg = {
        **asdict(model.cfg),
        "d_model": model.cfg.d_model + 1,
        "d_mlp": model.cfg.d_mlp + 1,
    }

    embedding_fold_bias(model)

    for block_idx in range(model.cfg.n_layers):
        # Convert layer norm modules if they are of type LayerNormPre
        for ln in ["ln1", "ln2"]:
            ln_module = getattr(model.blocks[block_idx], ln)
            if isinstance(ln_module, LayerNormPre):
                setattr(model.blocks[block_idx], ln, LayerNormPre_Folded(lnpre_folded_cfg))
            else:
                assert isinstance(
                    ln_module, torch.nn.Identity
                ), f"Unexpected layer norm module {ln_module}"

        attention_fold_bias(model.blocks[block_idx].attn)

        assert isinstance(
            model.blocks[block_idx].mlp, MLP
        ), f"Fold bias only supports a regular MLP, got {type(model.blocks[block_idx].mlp)}"
        mlp_fold_bias(model.blocks[block_idx].mlp)

    if hasattr(model, "ln_final") and isinstance(model.ln_final, LayerNormPre):
        model.ln_final = LayerNormPre_Folded(lnpre_folded_cfg)

    unembed_fold_bias(model)

    # transformer_lens.HookedRootModule has some internal stuff that needs to be reinitialized
    model.setup()
