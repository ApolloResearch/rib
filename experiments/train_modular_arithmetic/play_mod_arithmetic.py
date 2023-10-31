# %%
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import einops
import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import wandb
from pydantic import BaseModel
from sklearn.decomposition import PCA
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

logits = hooked_model(torch.tensor([3, 5, 113]))[0]
assert logits.argmax(dim=-1).item() == 8

logits, cache = hooked_model.run_with_cache(torch.tensor([3, 8, 113]))

print(hooked_model)

for key, value in cache.items():
    for index, stream in enumerate(value["acts"]):
        print(key, index, cache[key]["acts"][index].shape)

# %%
config_path = Path("mod_arithmetic_config.yaml")
config = load_config(config_path, config_model=Config)

datasets = load_dataset(dataset_config=config.dataset, return_set=config.dataset.return_set)
train_loader, test_loader = create_data_loader(datasets, shuffle=True, batch_size=11, seed=0)

print(train_loader.dataset[0:10])

# %%

# Interp plans
# Run model with hooks and collect all activations, do a PCA and stuff
# Analyze attention patterns
# Collect RIB-activations rather than standard activations
#     Do resample-ablation tests on these, what happens if I replace some?
#     Do logit-attribution of these, and confirm things match the way they should
#     Maximum activating dataset examples for these

# %%


def get_normal_activation_sizes(hooked_model=hooked_model, section="sections.section_1.2", p=113):
    _, cache = hooked_model.run_with_cache(torch.tensor([0, 0, p]))
    # Get activations
    acts = cache[section]["acts"]

    sizes = []
    for i, act in enumerate(acts):
        sizes.append(act[0].shape[0])
    return sizes


def get_normal_activations(
    hooked_model=hooked_model, section="sections.section_1.2", sizes=None, p=113
):
    return_acts = []

    if sizes is None:
        sizes = get_normal_activation_sizes(hooked_model=hooked_model, section=section, p=p)

    for size in sizes:
        return_acts.append(torch.empty([p, p, size]))

    for x in tqdm(range(p)):
        for y in range(p):
            batch_index = 0
            _, cache = hooked_model.run_with_cache(torch.tensor([x, y, p]))
            # Get activations
            acts = cache[section]["acts"]
            assert len(acts) == len(sizes), f"{len(acts)} != {len(sizes)}"
            for i, act in enumerate(acts):
                assert act[batch_index].shape[0] == sizes[i], f"{act.shape[0]} != {sizes[i]}"
                return_acts[i][x, y] = act[batch_index]
    return tuple(return_acts)


# %%
sec_0_2_resid_acts = get_normal_activations(section="sections.section_0.2")[0]

sec_1_1_resid_acts, sec_1_1_mlp_pre_acts = get_normal_activations(
    section="sections.section_1.1", sizes=(129, 513)
)
sec_1_2_resid_acts, sec_1_2_mlp_post_acts = get_normal_activations(
    section="sections.section_1.2", sizes=(129, 513)
)
assert torch.allclose(sec_1_1_resid_acts, sec_1_2_resid_acts)
assert torch.allclose(torch.relu(sec_1_1_mlp_pre_acts), sec_1_2_mlp_post_acts)

sec_2_1_resid_acts = get_normal_activations(section="sections.section_2.1", sizes=(129,))[0]
# %%


def plot_activations(acts, title="Default title", nrows=3, ncols=2, figsize=(8, 10)):
    fig, axes = plt.subplots(nrows, ncols, constrained_layout=True, figsize=figsize)
    fig.suptitle(title)
    for i, axs in enumerate(axes):
        for j, ax in enumerate(axs):
            ax.set_title(f"({i * ncols + j})")
            im = ax.imshow(acts[:, :, i * ncols + j].numpy())
            # Color
            fig.colorbar(im, ax=ax)


# plot_activations(sec_1_1_resid_acts, title="Residual activations section 1.1")
# plot_activations(sec_1_1_mlp_pre_acts, title="MLP pre-activations section 1.1")
# plot_activations(sec_1_2_resid_acts, title="Residual activations section 1.2")
# plot_activations(sec_1_2_mlp_post_acts, title="MLP post-activations section 1.2")
# plot_activations(sec_2_1_resid_acts, title="Residual activations section 2.1")
plot_activations(sec_0_2_resid_acts, title="Residual activations section 0.2 (extended embedding)")


# %%


def collect_RIB_acts(
    hooked_model=hooked_model,
    section="sections.section_0.2",
    p=113,
    interaction_graph_path="/mnt/ssd-apollo/dan/RIB/rib/experiments/lm_rib_build/out/modular_arithmetic_interaction_graph.pt",
    ablation_type="rib",
):
    # Note that the mapping feels shifted because hooks give us outputs but rib acts are inputs
    node_layer_dict = {
        "sections.section_pre.2": "ln1.0",
        "sections.section_0.2": "ln2.0",
        "sections.section_1.2": "mlp_out.0",
        "sections.section_2.2": "unembed",
    }

    interaction_graph_info = torch.load(interaction_graph_path)
    logits_node_layer = interaction_graph_info["config"]["node_layers"]
    node_layers = interaction_graph_info["config"]["node_layers"]
    ablation_node_layers = node_layers if logits_node_layer else node_layers[:-1]

    basis_matrices = load_basis_matrices(
        interaction_graph_info=interaction_graph_info,
        ablation_node_layers=ablation_node_layers,
        ablation_type=ablation_type,
        dtype=dtype,
        device=device,
    )

    node_layer = node_layer_dict[section]
    node_index = np.where(node_layer == np.array(node_layers))[0][0]
    assert (
        len(basis_matrices[node_index]) == 2
    ), f"This should be C and C_inv but contains {len(basis_matrices[node_index])} elements"
    basis_matrices_index = 0  # get C rather than C_inv
    # basis_matrices[node_index][0].shape = (act_dim_concat, rib_dim) with rib_dim <= act_dim
    print("Rotation matrix shape", basis_matrices[node_index][basis_matrices_index].shape)
    rib_return_acts = torch.empty([p, p, basis_matrices[node_index][basis_matrices_index].shape[1]])
    for x in tqdm(range(p)):
        for y in range(p):
            batch_index = 0
            _, cache = hooked_model.run_with_cache(torch.tensor([x, y, p]))
            acts = cache[section]["acts"]
            # print("Shapes", [act.shape for act in acts])
            acts = torch.cat([acts[i][batch_index] for i in range(len(acts))], dim=-1)
            transformed_acts = einops.einsum(
                acts, basis_matrices[node_index][basis_matrices_index], "act, act rib -> rib"
            )
            rib_return_acts[x, y] = transformed_acts
    return rib_return_acts


rib_acts_extended_embedding = collect_RIB_acts(section="sections.section_0.2")
rib_acts_mlp_post_act = collect_RIB_acts(section="sections.section_1.2")
rib_acts_pre_unembed = collect_RIB_acts(section="sections.section_2.2")
# Last two should be basically identical because only differ due to linear transform? Yep look visually the same!

# %%

plot_activations(
    rib_acts_extended_embedding,
    title="RIB activations section 0.2 (extended embedding)",
    nrows=10,
    figsize=(8, 30),
)
# plot_activations(rib_acts_mlp_post_act, title="RIB activations section 1.2 (MLP post-act)")
# plot_activations(rib_acts_pre_unembed, title="RIB activations section 2.2 (pre-unembed)")
# %%


def plot_fft_activations(
    acts, title="Default title", nrows=3, ncols=2, figsize=(8, 10), fftshift=True
):
    fig, axes = plt.subplots(nrows, ncols, constrained_layout=True, figsize=figsize)
    freqs = torch.fft.fftfreq(acts.shape[0])
    if fftshift:
        freqs = torch.fft.fftshift(freqs).numpy()
        acts = torch.fft.fftshift(acts, dim=[0, 1])
        title += "(fftshift'ed)"
        extent = [freqs[0], freqs[-1], freqs[0], freqs[-1]]
    else:
        extent = None
    title += "\n" + "".join([f"{freq:.2f} " for freq in freqs[::10]])
    fig.suptitle(title)
    for i, axs in enumerate(axes):
        for j, ax in enumerate(axs):
            ax.set_title(f"({i * ncols + j})")
            vminmax = acts[:, :, i * ncols + j].abs().max() / 10
            im = ax.imshow(
                acts[:, :, i * ncols + j].numpy(),
                cmap="RdBu",
                vmin=-vminmax,
                vmax=vminmax,
                aspect="equal",
                extent=extent,
                origin="lower",
            )
            # Color
            fig.colorbar(im, ax=ax)


# %%

# Fourier transform
rib_acts_extended_embedding_fft = torch.fft.fft2(rib_acts_extended_embedding, dim=(0, 1))

# %%

plot_fft_activations(
    rib_acts_extended_embedding_fft.real,
    title="RIB activations section 0.2 (extended embedding), real",
    nrows=3,
    figsize=(8, 10),
)

plot_fft_activations(
    rib_acts_extended_embedding_fft.imag,
    title="RIB activations section 0.2 (extended embedding), imag",
    nrows=3,
    figsize=(8, 10),
)
# %%

sec_0_2_resid_acts_fft = torch.fft.fft2(sec_0_2_resid_acts, dim=(0, 1))

plot_fft_activations(
    sec_0_2_resid_acts_fft.real,
    title="Normal activations section 0.2 (extended embedding), real",
    nrows=3,
    figsize=(8, 10),
)

# %%


def pca_activations(acts):
    acts_pca = acts.reshape(-1, acts.shape[-1])
    pca = PCA(n_components=acts_pca.shape[-1])
    acts_transformed = pca.fit_transform(acts_pca)
    acts_transformed = acts_transformed.reshape(acts.shape[0], acts.shape[1], -1)
    print("Explained variance", pca.explained_variance_ratio_)
    acts_transformed = torch.tensor(acts_transformed)
    return acts_transformed


sec_0_2_resid_acts_transformed = pca_activations(sec_0_2_resid_acts)
sec_0_2_resid_acts_transformed_fft = torch.fft.fft2(sec_0_2_resid_acts_transformed, dim=(0, 1))

plot_fft_activations(
    sec_0_2_resid_acts_transformed_fft.real,
    title="PCA activations section 0.2 (extended embedding), real",
    nrows=3,
    figsize=(8, 10),
)
# %%
