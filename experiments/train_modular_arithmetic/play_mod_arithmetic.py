# %%
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import einops
import fire
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import torch
import torch.optim as optim
import wandb
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
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


def get_normal_activation_sizes(
    hooked_model=hooked_model, section="sections.section_1.2", p=113, pos_index=-1
):
    _, cache = hooked_model.run_with_cache(torch.tensor([0, 0, p]))
    # Get activations
    acts = cache[section]["acts"]
    print(type(acts))
    print(acts[0].shape)

    sizes = []
    for i, act in enumerate(acts):
        sizes.append(act[pos_index].shape[0])
    return sizes


def get_normal_activations(
    hooked_model=hooked_model, section="sections.section_1.2", sizes=None, p=113, pos_index=-1
):
    return_acts = []

    if sizes is None:
        sizes = get_normal_activation_sizes(hooked_model=hooked_model, section=section, p=p)
        print("Determined sizes", sizes)

    for size in sizes:
        return_acts.append(torch.empty([p, p, size]))

    for x in tqdm(range(p)):
        for y in range(p):
            batch_index = 0
            _, cache = hooked_model.run_with_cache(torch.tensor([[x, y, p]]))
            # Get activations
            acts = cache[section]["acts"]
            assert len(acts) == len(sizes), f"{len(acts)} != {len(sizes)}"
            for i, act in enumerate(acts):
                assert (
                    act[batch_index][pos_index].shape[0] == sizes[i]
                ), f"{act.shape[0]} != {sizes[i]}"
                return_acts[i][x, y] = act[batch_index][pos_index]
    return tuple(return_acts)


# %%$
def get_attn_scores(
    hooked_model=hooked_model,
    section="sections.section_0.1.attention_scores",
    p=113,
):
    batch_index = 0
    head = slice(None)
    key = slice(None)
    query = slice(None)

    _, cache = hooked_model.run_with_cache(torch.tensor([[0, 0, p]]))
    acts = cache[section]["acts"]
    assert len(acts) == 1, f"acts should be tuple size 1 for attn scores"
    batch_size, n_heads, n_query, n_key = acts[0].shape
    # confirmed these, acts[0][47,47,3,1] has a -1e5 at last position
    assert n_query == n_key, f"n_ctx should be equal to n_ctx2"
    assert batch_size == 1, f"batch_size should be 1"
    assert n_heads == 4, f"n_heads should be 4"
    assert n_query == 3, f"n_query should be 3"

    return_acts = torch.empty([p, p, n_heads, n_query, n_key])

    for x in tqdm(range(p)):
        for y in range(p):
            _, cache = hooked_model.run_with_cache(torch.tensor([[x, y, p]]))
            # Get activations
            act = cache[section]["acts"][0]
            return_acts[x, y] = act[batch_index, head, key, query]
    return return_acts


sec_attention_scores = get_attn_scores(section="sections.section_0.1.attention_scores")
sec_attention_pattern = torch.softmax(sec_attention_scores, dim=-1)[:, :, :, -1]
# import slice


# %%
sec_0_2_resid_acts = get_normal_activations(section="sections.section_0.2")[0]

sec_1_1_resid_acts, sec_1_1_mlp_pre_acts = get_normal_activations(
    section="sections.section_1.1", sizes=(129, 513)
)  # type: ignore
sec_1_2_resid_acts, sec_1_2_mlp_post_acts = get_normal_activations(
    section="sections.section_1.2", sizes=(129, 513)
)
assert torch.allclose(sec_1_1_resid_acts, sec_1_2_resid_acts)
assert torch.allclose(torch.relu(sec_1_1_mlp_pre_acts), sec_1_2_mlp_post_acts)

sec_2_1_resid_acts = get_normal_activations(section="sections.section_2.1", sizes=(129,))[0]

# %%


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


plot_activations(
    sec_attention_pattern[:, :, :, 0],
    title="Attention pattern on pos x",
    nrows=2,
    ncols=2,
    figsize=(8, 10),
)


# plot_activations(sec_1_1_resid_acts, title="Residual activations section 1.1")
# plot_activations(sec_1_1_mlp_pre_acts, title="MLP pre-activations section 1.1")
# plot_activations(sec_1_2_resid_acts, title="Residual activations section 1.2")
# plot_activations(sec_1_2_mlp_post_acts, title="MLP post-activations section 1.2")
# plot_activations(sec_2_1_resid_acts, title="Residual activations section 2.1")
# plot_activations(sec_0_2_resid_acts, title="Residual activations section 0.2 (extended embedding)")


# %%


def collect_RIB_acts(
    hooked_model=hooked_model,
    section="sections.section_0.2",
    p=113,
    interaction_graph_path="/mnt/ssd-apollo/dan/RIB/rib/experiments/lm_rib_build/out/modular_arithmetic_interaction_graph.pt",
    ablation_type="rib",
    pos_index=-1,
):
    # Note that the mapping feels shifted because hooks give us outputs but rib acts are inputs
    node_layer_dict = {
        "sections.pre.2": "ln1.0",
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
            _, cache = hooked_model.run_with_cache(torch.tensor([[x, y, p]]))
            acts = cache[section]["acts"]
            if x == 0 and y == 0:
                print("In shapes (pre-cat)", [act.shape for act in acts])
            acts = torch.cat([acts[i][batch_index][pos_index] for i in range(len(acts))], dim=-1)

            transformed_acts = einops.einsum(
                acts, basis_matrices[node_index][basis_matrices_index].cpu(), "act, act rib -> rib"
            )
            rib_return_acts[x, y] = transformed_acts
            if x == 0 and y == 0:
                print("In shapes (cat tuples, batch 0)", acts.shape)
                print("Out shapes", transformed_acts.shape)
    return rib_return_acts


rib_acts_embedding_x = collect_RIB_acts(section="sections.pre.2", pos_index=0)
rib_acts_embedding_y = collect_RIB_acts(section="sections.pre.2", pos_index=1)
rib_acts_embedding_z = collect_RIB_acts(section="sections.pre.2", pos_index=-1)
rib_acts_extended_embedding = collect_RIB_acts(section="sections.section_0.2")
rib_acts_mlp_post_act = collect_RIB_acts(section="sections.section_1.2")
rib_acts_pre_unembed = collect_RIB_acts(section="sections.section_2.2")
# Last two should be basically identical because only differ due to linear transform? Yep look visually the same!

# %%

plot_activations(
    rib_acts_embedding_z,
    title="RIB activations section 0.2 (extended embedding)",
    nrows=10,
    figsize=(8, 30),
)
# plot_activations(rib_acts_mlp_post_act, title="RIB activations section 1.2 (MLP post-act)")
# plot_activations(rib_acts_pre_unembed, title="RIB activations section 2.2 (pre-unembed)")
# %%

# Fourier transform
rib_acts_extended_embedding_fft = torch.fft.fft2(rib_acts_extended_embedding, dim=(0, 1))
rib_acts_mlp_post_act_fft = torch.fft.fft2(rib_acts_mlp_post_act, dim=(0, 1))
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

    assert ncols == 2, "Only works for 2 columns with Re/Im"

    fig.suptitle(title)
    for col, title_info in enumerate(["Magnitude", "Phase"]):
        if col == 0:
            acts_component = acts.abs()
        else:
            acts_component = acts.angle()
        for row, ax in enumerate(axes[:, col]):
            ax.set_title(f"({row}) {title_info}")
            vminmax = acts_component[:, :, row].abs().max().item()
            if col == 0:
                # SymLogNorm
                norm = SymLogNorm(linthresh=vminmax / 10, linscale=1, vmin=-vminmax, vmax=vminmax)
            else:
                norm = Normalize(vmin=-vminmax, vmax=vminmax)
            im = ax.imshow(
                acts_component[:, :, row].numpy(),
                cmap="RdBu",
                norm=norm,
                aspect="equal",
                extent=extent,
                origin="lower",
            )
            # Color
            fig.colorbar(im, ax=ax)
    # Save to filename based on title without bad characters
    filename = title.replace(" ", "_").replace("\n", "_").replace(".", "_")
    filename = filename.replace("(", "").replace(")", "").replace(",", "").replace("'", "")
    plt.savefig(f"fft_activations_{filename}.png")


# %%

plot_fft_activations(
    rib_acts_extended_embedding_fft,
    title="RIB activations section 0.2 (extended embedding), combined",
    nrows=20,
    figsize=(8, 60),
)


plot_fft_activations(
    rib_acts_extended_embedding_fft,
    title="RIB activations section 0.2 (extended embedding), combined",
    nrows=20,
    figsize=(8, 60),
)

# %%

sec_0_2_resid_acts_fft = torch.fft.fft2(sec_0_2_resid_acts, dim=(0, 1))
rib_acts_mlp_post_act_fft = torch.fft.fft2(rib_acts_mlp_post_act, dim=(0, 1))

plot_fft_activations(
    rib_acts_mlp_post_act_fft,
    title="RIB activations section 1.2 (MLP post-act), combined",
    nrows=20,
    figsize=(8, 60),
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


sec_1_2_mlp_post_acts_pca = pca_activations(sec_1_2_mlp_post_acts)
sec_1_2_mlp_post_acts_pca_fft = torch.fft.fft2(sec_1_2_mlp_post_acts_pca, dim=(0, 1))

# %%


def svd_activations(acts):
    acts_flat = acts.reshape(-1, acts.shape[-1])
    U, S, Vt = np.linalg.svd(acts_flat, full_matrices=False)
    acts_transformed = np.dot(acts_flat, Vt.T)
    acts_transformed = acts_transformed.reshape(acts.shape[0], acts.shape[1], -1)
    explained_variance_ratio = S**2 / (S**2).sum()
    print("Explained variance", explained_variance_ratio)
    acts_transformed = torch.tensor(acts_transformed)
    return acts_transformed


sec_attn_pattern_svd = svd_activations(sec_attention_pattern[:, :, :, 0])
plot_activations(sec_attn_pattern_svd, nrows=2)
sec_attn_pattern_svd_fft = torch.fft.fft2(sec_attn_pattern_svd, dim=(0, 1))
plot_fft_activations(sec_attn_pattern_svd_fft, nrows=4)

# You can call this function with your activations 'acts' and the number of components 'n_components'.
# If you don't specify 'n_components', it will use the minimum of the number of samples and features.
# %%

sec_0_2_resid_acts_transformed = pca_activations(sec_0_2_resid_acts)
sec_0_2_resid_acts_transformed_fft = torch.fft.fft2(sec_0_2_resid_acts_transformed, dim=(0, 1))

plot_fft_activations(
    sec_0_2_resid_acts_transformed_fft,
    title="PCA activations section 0.2 (extended embedding), real",
    nrows=20,
    figsize=(8, 60),
)


# %%

plot_fft_activations(
    sec_0_2_resid_acts_transformed_fft.imag,
    title="PCA activations section 0.2 (extended embedding), imag",
    nrows=20,
    figsize=(8, 60),
)
# %%

sec_attention_pattern_fft = torch.fft.fft2(sec_attention_pattern[:, :, :, 0], dim=(0, 1))


plot_fft_activations(
    sec_attention_pattern_fft,
    title="FFT",
    nrows=4,
    figsize=(8, 10),
)
# %%

sec_attention_pattern_pca = pca_activations(sec_attention_pattern[:, :, :, 0])
sec_attention_pattern_pca_fft = torch.fft.fft2(sec_attention_pattern_pca, dim=(0, 1))

plot_fft_activations(
    sec_attention_pattern_pca_fft,
    title="Attn pattern PCA'ed",
    nrows=4,
    figsize=(8, 10),
)

# %%
# plotly

# Imshow sec_attention_pattern_pca_fft[:,:,0].abs().numpy()
px.imshow(torch.fft.fftshift(sec_attention_pattern_pca_fft[:, :, 0]).abs().numpy())
px.imshow(torch.fft.fftshift(sec_attention_pattern_pca_fft[:, :, 1]).abs().numpy())
freqs = torch.fft.fftfreq(sec_attention_pattern_pca_fft.shape[0])
freqs = torch.fft.fftshift(freqs).numpy()
print("Index 0:", freqs[26], freqs[56], freqs[86])
# 56 with both, both with 56
print("Index 0:", freqs[3], freqs[26], freqs[86], freqs[109], "less bright")

print("Index 1:", freqs[20], freqs[56], freqs[92])
print(torch.fft.fftshift(sec_attention_pattern_pca_fft)[56, 26].abs())
# %%
# 26 56
# 1
# 20 56, 92 56, 20 56, 56, 20, 56 92


# %%
def print_acts_and_phases(ffted_acts, index, p=113, lower=300):
    for x in range(p):
        for y in range(p):
            if ffted_acts[x, y, index].abs() > lower and ffted_acts[x, y, 0].abs() < 1e10:
                freqs = torch.fft.fftfreq(ffted_acts.shape[0])
                val = ffted_acts[x, y, index].abs().item()
                phase = ffted_acts[x, y, index].angle().item()
                print(
                    f"({freqs[x]:.3f}, {freqs[y]:.3f})",
                    f"Value {val:.1f}",
                    f"Phase {phase/np.pi*180:.1f} deg",
                    f"Phrase as real & imag: e^(i phi) = {np.cos(phase):.3f} + i {np.sin(phase):.3f}",
                )


print_acts_and_phases(sec_attention_pattern_pca_fft, 0)

# %%

plot_fft_activations(sec_1_2_mlp_post_acts_pca_fft, nrows=8, figsize=(8, 50))

# %%

plot_fft_activations(rib_acts_mlp_post_act_fft, nrows=8, figsize=(8, 50))

# %%
print_acts_and_phases(rib_acts_mlp_post_act_fft, 0, lower=200000)

# %%
print_acts_and_phases(rib_acts_mlp_post_act_fft, 0, lower=200000)

# %%


plot_activations(
    rib_acts_mlp_post_act,
    title="rib_acts_mlp_post_act",
    nrows=2,
    ncols=2,
    figsize=(8, 10),
)

# %%
