import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import fire
import torch
import torch.optim as optim
import transformer_lens
import wandb
from jaxtyping import Float, Int
from pydantic import BaseModel
from torch import nn
from torch.func import jacrev, vmap
from torch.nn.modules.container import ModuleList
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.components import (
    MLP,
    Attention,
    Embed,
    HookPoint,
    LayerNorm,
    PosEmbed,
    TransformerBlock,
    Unembed,
)

from rib.data import ModularArithmeticDataset
from rib.log import logger
from rib.models.utils import save_model
from rib.utils import load_config


class ModelConfig(BaseModel):
    n_layers: int
    d_model: int
    d_head: int
    n_heads: int
    d_mlp: int
    d_vocab: int
    n_ctx: int
    act_fn: str
    normalization_type: str


class TrainConfig(BaseModel):
    modulus: int
    frac_train: float
    fn_name: str
    learning_rate: float
    batch_size: int  # Set to max(batch_size, <number of samples in dataset>)
    epochs: int
    save_dir: Optional[Path]
    save_every_n_epochs: Optional[int]


class InteractionConfig(BaseModel):
    layer_types: Optional[set[str]]
    jacobian_pairs: Optional[list[dict[str, str]]]
    layers_of_interest: Optional[dict]


class Config(BaseModel):
    seed: int
    model: ModelConfig
    train: TrainConfig
    interaction: InteractionConfig


def do_everything(
    config: Config,
    model: HookedTransformer,
    train_loader: DataLoader,
    layer_types_classes: dict,
    device: str,
):
    seq2 = extract_layers_of_interest(
        model, layer_types_classes, config.interaction.layers_of_interest
    )
    if config.interaction.jacobian_pairs is not None:
        for pair in config.interaction.jacobian_pairs:
            jacobian = compute_jacobian_between_layers(
                model, seq2, pair["start"], pair["end"], train_loader
            )


def extract_layers_of_interest(model, layer_types: dict, layers_of_interest: dict):
    if layers_of_interest is None:
        layers_of_interest = {}
    if layer_types is None:
        layer_types = (Embed, PosEmbed, Attention, MLP, LayerNorm, Unembed)

    for name, child in model.named_children():
        if any(isinstance(child, layer_class) for layer_class in layer_types):
            layers_of_interest[name] = child
        elif isinstance(child, HookPoint):
            continue
        elif isinstance(child, ModuleList) or isinstance(child, TransformerBlock):
            extract_layers_of_interest(child, layer_types, layers_of_interest)

    return layers_of_interest


def compute_jacobian_between_layers(
    model, layers_of_interest, start_layer_name, end_layer_name, train_loader
):
    # Store the start value
    start_val = None

    # Define a hook to capture the start value
    def start_hook(module, input, output):
        nonlocal start_val
        start_val = output

    # Register the hook with the start layer
    hook_handle = layers_of_interest[start_layer_name].register_forward_hook(start_hook)

    # Perform a forward pass to capture the start value
    x = train_loader.dataset[0][0]
    model(x)

    hook_handle.remove()

    # Define a function for the Jacobian computation
    def fn(input_val):
        found_start = False
        for name, layer in model.named_children():
            if name == start_layer_name:
                found_start = True
                continue  # Skip applying the start layer again
            if found_start and not isinstance(layer, HookPoint):
                input_val = layer(input_val)
                if name == end_layer_name:
                    break
        return input_val

    # Compute the Jacobian of this function
    jacobian = jacrev(fn)(start_val)

    return jacobian


def compute_jacobian_between_layers2(
    model, layers_of_interest, start_layer_name, end_layer_name, train_loader
):
    # Store the start value
    start_val = None

    # Define a hook to capture the start value
    def start_hook(module, input, output):
        nonlocal start_val
        start_val = output

    # Register the hook with the start layer
    hook_handle = layers_of_interest[start_layer_name].register_forward_hook(start_hook)

    # Perform a forward pass to capture the start value
    x = train_loader.dataset[0][0]
    model(x)

    hook_handle.remove()

    # Define a function for the Jacobian computation
    def process_layers(input_val, layers_iterator, start_layer_name, end_layer_name):
        found_start = False
        for name, layer in layers_iterator:
            if isinstance(layer, (torch.nn.ModuleList, TransformerBlock)):
                input_val = process_layers(
                    input_val, layer.named_children(), start_layer_name, end_layer_name
                )
            else:
                if name == start_layer_name:
                    found_start = True
                    continue  # Skip applying the start layer again
                if found_start:
                    input_val = layer(input_val)
                    if name == end_layer_name:
                        break
        return input_val

    def fn(input_val):
        return process_layers(input_val, model.named_children(), start_layer_name, end_layer_name)

    # Compute the Jacobian of this function
    jacobian = jacrev(fn)(start_val)

    return jacobian


# def jorn_seq(model: HookedTransformer) -> nn.Sequential:
#     # convert a HookedTransformer to a nn.Sequential
#     # this makes calculating jacobians way more easy
#     layers = []
#
#     def layer_onehot(tokens: Int[torch.Tensor, "batch_size seq_len"]):
#         # one hot encode
#         return torch.nn.functional.one_hot(tokens, num_classes=model.cfg.d_vocab).to(torch.float32)
#
#     def layer_embed(tokens: Int[torch.Tensor, "batch_size seq_len d_vocab"]):
#         # for one-hot encoded tokens
#         batch_size, seq_len, d_vocab = tokens.shape
#
#         embed = tokens @ model.embed.W_E
#
#         pos_embed = model.pos_embed.W_pos[:seq_len, :]  # [seq_len, d_model]
#         pos_embed = pos_embed[None, :, :].repeat(batch_size, 1, 1)  # [batch, seq_len, d_model]
#
#         return embed + pos_embed
#
#     # attention
#     def layer_attn(iblock: int):
#         def fn(resid_pre: Float[torch.Tensor, "batch_size seq_len d_model"]):
#             assert not model.cfg.use_split_qkv_input
#             assert not model.cfg.parallel_attn_mlp
#
#             qkv = model.blocks[iblock].ln1(resid_pre)
#             attn_out = model.blocks[iblock].attn(qkv, qkv, qkv)
#
#             resid_mid = resid_pre + attn_out
#
#             return resid_mid
#
#         return fn
#
#     # mlp
#     def layer_mlp(iblock: int):
#         def fn(resid_mid: Float[torch.Tensor, "batch_size seq_len d_model"]):
#             assert not model.cfg.attn_only and not model.cfg.parallel_attn_mlp
#             normalized_resid_mid = model.blocks[iblock].ln2(resid_mid)
#             mlp_out = model.blocks[iblock].mlp(normalized_resid_mid)
#             resid_post = resid_mid + mlp_out
#             return resid_post
#
#         return fn
#
#     # ln_final and unembed
#     def layer_unembed(resid_post: Float[torch.Tensor, "batch_size seq_len d_model"]):
#         if model.cfg.normalization_type is not None:
#             residual = model.ln_final(resid_post)
#             logits = model.unembed(residual)
#         else:
#             logits = model.unembed(resid_post)
#         return logits
#
#     def layer_restrict_last(resid_post: Float[torch.Tensor, "batch_size seq_len d_vocab_out"]):
#         # multiply with mask
#         mask = torch.zeros_like(resid_post, requires_grad=False)
#         mask[:, -1, :] = 1
#         return resid_post * mask
#
#     layers.append(("one_hot", layer_onehot))
#     layers.append(("embed", layer_embed))
#     for iblock in range(model.cfg.n_layers):
#         layers.append((f"attn_{iblock}", layer_attn(iblock)))
#         layers.append((f"mlp_{iblock}", layer_mlp(iblock)))
#     layers.append(("unembed", layer_unembed))
#     layers.append(("restrict_last", layer_restrict_last))
#
#     return layers


def main(config_path_str: str):
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)

    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer_classes = {
        "Embed": Embed,
        "PosEmbed": PosEmbed,
        "Attention": Attention,
        "MLP": MLP,
        "Unembed": Unembed,
    }
    layers_of_interest_classes = [
        layer_classes[name] for name in config.interaction.layer_types if not None
    ]

    # Load the Modular Arithmetic train dataset
    train_data = ModularArithmeticDataset(
        config.train.modulus,
        config.train.frac_train,
        device=device,
        seed=config.seed,
        train=True,
    )
    train_loader = DataLoader(train_data, batch_size=config.train.batch_size, shuffle=False)

    # Initialize the Transformer model
    transformer_lens_config = HookedTransformerConfig(**config.model.model_dump())
    model = HookedTransformer(transformer_lens_config)
    model = model.to(device)

    sequence = do_everything(
        config=config,
        model=model,
        train_loader=train_loader,
        layer_types_classes=layers_of_interest_classes,
        device=device,
    )


if __name__ == "__main__":
    fire.Fire(main)
