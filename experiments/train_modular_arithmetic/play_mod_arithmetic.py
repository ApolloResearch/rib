# %%
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import einops
import fire
import numpy as np
import torch
import torch.optim as optim
import wandb
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig

from rib.ablations import (
    ExponentialScheduleConfig,
    LinearScheduleConfig,
    load_basis_matrices,
    run_ablations,
)
from rib.data import ModularArithmeticDatasetConfig
from rib.hook_manager import HookedModel
from rib.loader import create_data_loader, load_dataset, load_sequential_transformer
from rib.log import logger
from rib.models.utils import save_model
from rib.types import TORCH_DTYPES
from rib.utils import load_config, set_seed

torch.set_grad_enabled(False)


class ModelConfig(BaseModel):
    n_layers: int
    d_model: int
    d_head: int
    n_heads: int
    d_mlp: int
    d_vocab: int
    n_ctx: int
    act_fn: str
    normalization_type: Optional[str]


class TrainConfig(BaseModel):
    learning_rate: float
    batch_size: int  # Set to max(batch_size, <number of samples in dataset>)
    epochs: int
    eval_every_n_epochs: int
    save_dir: Optional[Path] = None
    save_every_n_epochs: Optional[int] = None


class WandbConfig(BaseModel):
    project: str
    entity: Optional[str]


class Config(BaseModel):
    seed: int
    model: ModelConfig
    train: TrainConfig
    dataset: ModularArithmeticDatasetConfig
    wandb: Optional[WandbConfig]


# %%

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = TORCH_DTYPES["float32"]

seq_model, tlens_cfg_dict = load_sequential_transformer(
    node_layers=["ln1.0", "ln2.0", "mlp_out.0", "unembed"],
    last_pos_module_type="add_resid1",  # module type in which to only output the last position index
    tlens_pretrained=None,
    tlens_model_path=Path(
        "/mnt/ssd-apollo/checkpoints/rib/modular_arthimetic/lr-0.001_bs-10000_norm-None_2023-09-27_18-19-33/model_epoch_60000.pt"
    ),
    eps=1e-5,
    dtype=dtype,
    device=device,
)

seq_model.eval()
seq_model.to(device=torch.device(device), dtype=dtype)
seq_model.fold_bias()
hooked_model = HookedModel(seq_model)

print(hooked_model)

# %%

# Interp plans
# Run model with hooks and collect all activations, do a PCA and stuff
# Analyze attentiohn patterns
# Collect RIB-activations rather than standard activations
#     Do resample-ablation tests on these, what happens if I replace some?
#     Do logit-attribution of these, and confirm things match the way they should
#     Maximum activating dataset examples for these

# %%
config_path = Path("mod_arithmetic_config.yaml")
config = load_config(config_path, config_model=Config)

datasets = load_dataset(dataset_config=config.dataset, return_set=config.dataset.return_set)
train_loader, test_loader = create_data_loader(datasets, shuffle=True, batch_size=11, seed=0)

print(train_loader.dataset[0:10])  #

# %%

logits = hooked_model(torch.tensor([3, 3, 113]))[0]

logits.argmax(dim=-1)

# %%

logits, cache = hooked_model.run_with_cache(torch.tensor([3, 3, 113]))

for key, value in cache.items():
    for index, stream in enumerate(value["acts"]):
        print(key, index, cache[key]["acts"][index].shape)


# %%

# Plot activations of MLP Act section 1.2
# Store all in tensor
resid_acts = torch.empty([113, 113, 129])
mlp_acts = torch.empty([113, 113, 513])
# Iterate all inputs
p = 113
for x in tqdm(range(p)):
    for y in range(p):
        logits, cache = hooked_model.run_with_cache(torch.tensor([x, y, p]))
        # Get activations
        acts = cache["sections.section_1.2"]["acts"]
        resid_acts[x, y] = acts[0][0]  # stream, batch
        mlp_acts[x, y] = acts[1][0]

# %%

# Now collect RIB features
interaction_graph_path = "/mnt/ssd-apollo/dan/RIB/rib/experiments/lm_rib_build/out/modular_arithmetic_interaction_graph.pt"
interaction_graph_info = torch.load(interaction_graph_path)
logits_node_layer = interaction_graph_info["config"]["node_layers"]
node_layers = interaction_graph_info["config"]["node_layers"]
ablation_node_layers = node_layers if logits_node_layer else node_layers[:-1]
ablation_type = "rib"

basis_matrices = load_basis_matrices(
    interaction_graph_info=interaction_graph_info,
    ablation_node_layers=ablation_node_layers,
    ablation_type=ablation_type,
    dtype=dtype,
    device=device,
)

node_index = np.where("mlp_out.0" == np.array(node_layers))[0][0]
target = "sections.section_1.2"

# Turns out it's 556 cause basis matrices are 556 rather than 642 (I assume trunc'd zeros)
rib_acts = torch.empty([113, 113, basis_matrices[node_index][0].shape[1]])
for x in tqdm(range(p)):
    for y in range(p):
        logits, cache = hooked_model.run_with_cache(torch.tensor([x, y, p]))
        # Get activations
        acts = cache[target]["acts"]
        acts = torch.cat([acts[0][0], acts[1][0]], dim=-1)
        transformed_acts = einops.einsum(acts, basis_matrices[node_index][0], "act, act rib -> rib")
        rib_acts[x, y] = transformed_acts

# def load_basis_matrices(
#     interaction_graph_info: dict,
#     ablation_node_layers: list[str],
#     ablation_type: Literal["rib", "orthogonal"],
#     dtype: torch.dtype,
#     device: str,
# ) -> list[tuple[BasisVecs, BasisVecsPinv]]:

#
# for x in tqdm(range(p)):
#     for y in range(p):
#   - ln1.0
#   - ln2.0
#   - mlp_out.0
#   - unembed

# %%

import matplotlib.pyplot as plt

nrows = 3
ncols = 2

fig, axes = plt.subplots(nrows, ncols, constrained_layout=True, figsize=(6, 8))
for i, axs in enumerate(axes):
    for j, ax in enumerate(axs):
        ax.set_title(f"({i * ncols + j})")
        im = ax.imshow(mlp_acts[:, :, i * ncols + j].numpy())
        fig.colorbar(im, ax=ax)

fig, axes = plt.subplots(nrows, ncols, constrained_layout=True, figsize=(6, 8))
for i, axs in enumerate(axes):
    for j, ax in enumerate(axs):
        ax.set_title(f"({i * ncols + j})")
        im = ax.imshow(resid_acts[:, :, i * ncols + j].numpy())
        fig.colorbar(im, ax=ax)

# %%

fig, axes = plt.subplots(nrows, ncols, constrained_layout=True, figsize=(8, 10))
for i, axs in enumerate(axes):
    for j, ax in enumerate(axs):
        ax.set_title(f"({i * ncols + j})")
        im = ax.imshow(rib_acts[:, :, i * ncols + j].numpy())
        # Color
        fig.colorbar(im, ax=ax)

# plt.imshow(rib_acts[:, :, 0].numpy())
# %%
