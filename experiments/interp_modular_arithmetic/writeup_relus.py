# %%
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from activations import Activations
from plotting import annotated_fft_line_plot, plot_activations, plot_fft_activations
from tqdm import tqdm
from transformations import fft2, pca_activations, svd_activations

from rib.models.utils import get_model_attr

torch.set_grad_enabled(False)

# %%

activations = Activations()

# %%

# Extract all normal activations

attention_scores = activations.get_section_activations(
    section="sections.section_0.1.attention_scores",
)[0]

sec_pre_2_resid_embed_acts = activations.get_section_activations(section="sections.pre.2")[0]
(
    sec_pre_2_resid_embed_acts_x,
    sec_pre_2_resid_embed_acts_y,
    sec_pre_2_resid_embed_acts_z,
) = einops.rearrange(sec_pre_2_resid_embed_acts, "x y p h -> p x y h")

sec_0_1_emed_parallel, sec_0_1_attn_out = activations.get_section_activations(
    section="sections.section_0.1",
)
assert torch.allclose(sec_0_1_emed_parallel[:, :, 2, :], sec_pre_2_resid_embed_acts_z)
sec_0_1_attn_out_z = sec_0_1_attn_out[:, :, 2, :]

sec_0_2_resid_post_attn_acts = activations.get_section_activations(section="sections.section_0.2")[
    0
]

sec_1_1_resid_pre_mlp_acts, sec_1_1_mlp_pre_acts = activations.get_section_activations(
    section="sections.section_1.1",
)
# Note: Resid contains the same numbers from 0.2 to 1.2
assert torch.allclose(sec_1_1_resid_pre_mlp_acts, sec_0_2_resid_post_attn_acts)

sec_1_2_resid_parallel_acts, sec_1_2_mlp_post_acts = activations.get_section_activations(
    section="sections.section_1.2",
)

assert torch.allclose(sec_1_2_resid_parallel_acts, sec_1_1_resid_pre_mlp_acts)
# Note: Resid contains the same numbers from 0.2 to 1.2


sec_2_1_resid_post_mlp_acts = activations.get_section_activations(section="sections.section_2.1")[0]


# %%

# RIB acts

rib_acts_embedding = activations.get_rib_activations(section="sections.pre.2")
rib_acts_extended_embedding = activations.get_rib_activations(section="sections.section_0.2")[
    :, :, 0, :
]
# Note: mlp_post and pre_umembed are basically the same
rib_acts_mlp_post = activations.get_rib_activations(section="sections.section_1.2")[:, :, 0, :]
rib_acts_pre_unembed = activations.get_rib_activations(section="sections.section_2.2")[:, :, 0, :]
# Embeddings have token dimension, split up.
rib_acts_embedding_x, rib_acts_embedding_y, rib_acts_embedding_z = einops.rearrange(
    rib_acts_embedding, "x y p h -> p x y h"
)

# %%
# Get RIB graph

graph_path = "/mnt/ssd-apollo/stefan/rib/modular_arithmetic_interaction_graph.pt"
edges = dict(torch.load(graph_path)["edges"])
for key in edges:
    print("Edge going trhough module", key, edges[key].shape)

# Examples:
# Biggest input to ln2.0#9 seems to be ln1.0#7
# Biggest input to mlp_out.0#2 seems to be ln2.0#5

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
fig.suptitle("Edges graph (read from file)")

ax1.plot(dict(edges)["ln1.0"][9, :])
ax1.set_title("Edges through ln1.0")
ax1.set_xlabel("ln1.0-input node #")
ax1.set_ylabel("edge strength into ln2.0-input #9")
ax1.set_xlim(0, 10)

ax2.plot(dict(edges)["mlp_out.0"][2, :])
ax2.set_title("Edges through mlp_out.0")
ax2.set_xlabel("mlp_out.0-input node #")
ax2.set_ylabel("edge strength into mlp_out.0-input #2")
ax2.set_xlim(0, 10)


# %%

# Test if edges make sense -- patching

# Get required matrices
model = activations.hooked_model.model
W_in = get_model_attr(model, "sections.section_1.1").W_in.to(torch.float64).to("cpu")
W_out = get_model_attr(model, "sections.section_2.0").W_out.to(torch.float64).to("cpu")

print([a[0].shape for a in activations.rib_basis_matrices])


class ScaledMLP:
    def __init__(self, W_in, dtype=torch.float64):
        _, self.Cinv_ell = activations.rib_basis_matrices[1]  # extended embedding ln2.0-input "l"
        self.Cinv_ell = self.Cinv_ell.to("cpu").to(dtype)
        self.C_ellp1, _ = activations.rib_basis_matrices[2]  # post-ReLU mlp_out-input "l + 1"
        self.C_ellp1 = self.C_ellp1.to("cpu").to(dtype)

        self.in_acts = rib_acts_extended_embedding.to(dtype)
        self.parallel_acts = sec_1_2_resid_parallel_acts
        # self.parallel_acts = torch.unsqueeze(self.parallel_acts, 0)
        self.W_hat = einops.einsum(W_in, self.Cinv_ell, "embed mlp, rib embed -> rib mlp")
        self.d_mlp = self.W_hat.shape[-1]
        self.d_concat = self.C_ellp1.shape[0]

    def forward(self, scaling, mlp_filter=None):
        """Forward pass through the RIB-MLP, i.e. though C^l^{-1} W_in C^{l+1}

        Args:
            scaling (torch.Tensor): Scaling of the RIB dimensions
            mlp_filter (torch.Tensor, optional): Select specific neurons only. Defaults to all.

        """

        scaled_in_acts = einops.einsum(self.in_acts, scaling, "x y rib, rib ... -> ... x y rib")
        scaled_parallel_acts = einops.einsum(
            scaled_in_acts, self.Cinv_ell, "... x y rib, rib emb -> ... x y emb"
        )
        pre_relu_acts = einops.einsum(
            scaled_in_acts, self.W_hat, "... x y rib, rib mlp -> ... x y mlp"
        )

        post_relu_acts = torch.relu(pre_relu_acts)

        post_relu_concat_acts = torch.concat([scaled_parallel_acts, post_relu_acts], dim=-1)

        if mlp_filter is None:
            post_relu_rib_acts = einops.einsum(
                post_relu_concat_acts,
                self.C_ellp1,
                "... x y mlp, mlp rib -> ... x y rib",
            )
        elif isinstance(mlp_filter, int):
            one_hot = torch.zeros([self.d_concat])
            one_hot[mlp_filter] = 1

            post_relu_rib_acts = einops.einsum(
                post_relu_concat_acts,
                one_hot,
                self.C_ellp1,
                "... x y mlp, mlp, mlp rib -> ... x y rib",
            )
        return post_relu_rib_acts


scaled_rib_mlp = ScaledMLP(W_in)

# %%

# Test that the graph makes sense

batch_size = 100
rib_in_len = rib_acts_extended_embedding.shape[-1]
scale_node_5 = torch.ones([rib_in_len, batch_size])
scale_node_5[5] = torch.linspace(0.2, 5, batch_size)

mlp_out_patched_run = scaled_rib_mlp.forward(scale_node_5)

torch.manual_seed(0)
random_coords = torch.randint(0, 100, (5, 2))
fig, axes = plt.subplots(5, 1, figsize=(6, 6), constrained_layout=True)
for d, (x, y) in enumerate(random_coords):
    ax = axes[d]
    ax.set_title(f"Random data point ({x},{y})")
    for i in range(batch_size):
        cmap = plt.get_cmap("viridis")
        ax.plot(
            mlp_out_patched_run[i, x, y, :] - mlp_out_patched_run[0, x, y, :],
            color=cmap(i / batch_size),
        )
    ax.set_xlim(0, 10)
    ax.set_xlabel("post-ReLU node #")
    ax.set_ylabel("post-ReLU value")
plt.show()


# %%

# Now we know that the mapping is mostly ln2.0#5 --> mlp_out.0#2 let's plot this function
# Note: High RAM usage


def plot_function(scaling, post_relu_rib_acts, axes=None, output=2, x=0, y=0):
    if axes is None:
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    else:
        ax1, ax2 = axes
        fig = ax1.get_figure()
    ax1.scatter(scaling, post_relu_rib_acts[:, x, y, output], s=1, marker=".")
    ax2.scatter(scaling[1:], np.diff(post_relu_rib_acts[:, x, y, output]), s=1, marker=".")
    ax1.set_title(f"Scaling up input number 5, observing output {output} at data point x,y={x},{y}")
    ax1.set_xlabel("Scaling of RIB dimension 5")
    ax2.set_xlabel("Scaling of RIB dimension 5")
    ax1.set_ylabel(f"Output RIB dimension {output}")
    ax2.set_ylabel(f"Derivative")
    ax1.grid()
    ax2.grid()


fig, axes = plt.subplots(6, 4, figsize=(16, 16), constrained_layout=True)
fig.suptitle("RIB 5 --> RIB 3, scaling of RIB dimension 5")

scale_node_5 = torch.ones([rib_in_len, batch_size])
scale_node_5[5] = torch.linspace(0.2, 5, batch_size)
mlp_out_patched_run = scaled_rib_mlp.forward(scale_node_5)
plot_function(scale_node_5[5], mlp_out_patched_run, x=0, y=0, axes=axes[0, :2])

scale_node_5[5] = torch.linspace(0.01, 100, batch_size)
mlp_out_patched_run = scaled_rib_mlp.forward(scale_node_5)
plot_function(scale_node_5[5], mlp_out_patched_run, x=0, y=0, axes=axes[0, 2:])

scale_node_5 = torch.ones([rib_in_len, batch_size])
for d, (x, y) in enumerate(random_coords):
    scale_node_5[5] = torch.linspace(0.2, 5, batch_size)
    mlp_out_patched_run = scaled_rib_mlp.forward(scale_node_5)
    plot_function(scale_node_5[5], mlp_out_patched_run, x=x, y=y, axes=axes[d + 1, :2])

    scale_node_5[5] = torch.linspace(0.01, 100, batch_size)
    mlp_out_patched_run = scaled_rib_mlp.forward(scale_node_5)
    plot_function(scale_node_5[5], mlp_out_patched_run, x=x, y=y, axes=axes[d + 1, 2:])

# %%

# Consider ReLU #1 only
# Note: High RAM usage
fig, axes = plt.subplots(5, 4, figsize=(16, 14), constrained_layout=True)
fig.suptitle("RIB 5 --> RIB 3, mlp_filter #130 only")
for d, (x, y) in enumerate(random_coords):
    scale_node_5[5] = torch.linspace(0.2, 5, batch_size)
    mlp_out_patched_run = scaled_rib_mlp.forward(scale_node_5, mlp_filter=130)
    plot_function(scale_node_5[5], mlp_out_patched_run, x=x, y=y, axes=axes[d, :2])
    scale_node_5[5] = torch.linspace(0.01, 100, batch_size)
    mlp_out_patched_run = scaled_rib_mlp.forward(scale_node_5, mlp_filter=130)
    plot_function(scale_node_5[5], mlp_out_patched_run, x=x, y=y, axes=axes[d, 2:])

# %%

# Consider some ln2.0#5 --> mlp_out.0#other functions
# Note: High RAM usage
fig, axes = plt.subplots(5, 4, figsize=(16, 12), constrained_layout=True)
(x, y) = random_coords[0]
fig.suptitle(f"RIB 5 --> RIB ... for random_coords {x},{y}")
for d, output in enumerate(range(5)):
    scale_node_5[5] = torch.linspace(0.2, 5, batch_size)
    mlp_out_patched_run = scaled_rib_mlp.forward(scale_node_5)
    plot_function(scale_node_5[5], mlp_out_patched_run, x=x, y=y, axes=axes[d, :2], output=output)
    scale_node_5[5] = torch.linspace(0.01, 100, batch_size)
    mlp_out_patched_run = scaled_rib_mlp.forward(scale_node_5)
    plot_function(scale_node_5[5], mlp_out_patched_run, x=x, y=y, axes=axes[d, 2:], output=output)

# %%

# Consider the effect size through each individual neuron, and through MLP vs through Resid
# Make a histogram of the former as well as scatter plot to see if dead neurons are correlated

# Resid
scale_node_5 = torch.ones([rib_in_len, 2])
scale_node_5[5] = torch.linspace(1, 1.01, 2)

indiv_neuron_impacts = []
for n in tqdm(range(642)):
    mlp_out_patched_run = scaled_rib_mlp.forward(scale_node_5, mlp_filter=n)
    indiv_neuron_impacts.append(mlp_out_patched_run)

indiv_neuron_impacts = torch.stack(indiv_neuron_impacts)
resid_impacts, mlp_impacts = indiv_neuron_impacts[:129], indiv_neuron_impacts[129:]
#                    (confirmed it's resid, pre-act)

# %%

mlp_neuron_diff_output2 = mlp_impacts[:, 1, :, :, 2] - mlp_impacts[:, 0, :, :, 2]

# Iterate random datapoints
fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
for d, (x, y) in enumerate(random_coords):
    ax.hist(
        mlp_neuron_diff_output2[:, x, y], bins=100, histtype="step", label=f"Data point {x},{y}"
    )

# Scatter between two random datapoints
plt.figure()
d1 = random_coords[0]
d2 = random_coords[1]
plt.scatter(
    mlp_neuron_diff_output2[:, d1[0], d1[1]],
    mlp_neuron_diff_output2[:, d2[0], d2[1]],
    s=1,
    marker=".",
)

# %%

plt.figure()
plt.scatter(
    mlp_impacts[:, 0, 1, 8, 2] - mlp_impacts[:, 1, 1, 8, 2],
    mlp_impacts[:, 0, 45, 76, 2] - mlp_impacts[:, 1, 45, 76, 2],
    s=1,
    marker=".",
)
plt.xlabel("MLP neuron impact on output 2 at data point (1, 8)")
plt.ylabel("MLP neuron impact on output 2 at data point (45, 76)")
plt.show()


# %%

# Why is the graph sparse?

# First look at all the component matrices, histogram of entries

dtype = torch.float64
_, C_ell_inv = activations.rib_basis_matrices[1]  # extended embedding ln2.0-input "l"
C_ell_inv = C_ell_inv.to("cpu").to(dtype)
C_ellp1, _ = activations.rib_basis_matrices[2]  # post-ReLU mlp_out-input "l + 1"
C_ellp1 = C_ellp1.to("cpu").to(dtype)

W_in_with_id = torch.cat([torch.eye(129), W_in], dim=1)

relu_ops = (sec_1_2_mlp_post_acts + 1e-20) / (sec_1_1_mlp_pre_acts + 1e-20)
relu_means = torch.mean(relu_ops, dim=(0, 1))[0]
relu_means_with_id = torch.cat([torch.ones(129), relu_means], dim=0)


# W_in: 129 x 513, d_embed x d_mlp
# W_in_with_id: 129 x 642, d_embed x d_mlp
# C_ell_inv: 47 x 129, d_rib_l x d_embed
# C_ellp1: 642 x 81, (d_embed + d_mlp) x d_rib_lp1
#                    (confirmed it's resid, pre-act)

matrices = {
    r"$C^{l, -1}$": C_ell_inv,
    r"$C^{l+1}$": C_ellp1,
    r"$W_{\rm in}$": W_in,
    r"$C^{l, -1} W_{\rm in}$": einops.einsum(
        C_ell_inv, W_in, "rib_l d_embed, d_embed d_mlp -> rib_l d_mlp"
    ),
    r"$W_{\rm in} C^{l+1}$": einops.einsum(
        W_in_with_id, C_ellp1[:, :], "d_embed d_mlp, d_mlp rib_lp1 -> d_embed rib_lp1"
    ),
    r"$C^{l, -1} W_{\rm in} C^{l+1}$": einops.einsum(
        C_ell_inv,
        W_in_with_id,
        C_ellp1[:, :],
        "rib_l d_embed, d_embed d_mlp, d_mlp rib_lp1 -> rib_l rib_lp1",
    ),
    r"$C^{l, -1} W_{\rm in} {\rm \langle ReLU \rangle} C^{l+1}$": einops.einsum(
        C_ell_inv,
        W_in_with_id,
        relu_means_with_id,
        C_ellp1[:, :],
        "rib_l d_embed, d_embed d_mlp, d_mlp, d_mlp rib_lp1 -> rib_l rib_lp1",
    ),
}

fig, axes = plt.subplots(4, 2, figsize=(10, 6), constrained_layout=True)
for i, (label, matrix) in enumerate(matrices.items()):
    ax = axes.flatten()[i]
    ax.hist(matrix.flatten(), bins=100)
    ax.hist(matrix.flatten() ** 2, bins=100, histtype="step", color="C1")
    ax.semilogy()
    ax.set_title(f"Entries of {label}")
    # ax.set_xlabel(label)

ax = axes.flatten()[i + 1]
ax.hist(edges["ln2.0"].abs().flatten().log10(), bins=100, color="C0")
ax.hist((edges["ln2.0"] ** 2).flatten().log10(), bins=100, color="C1", histtype="step")
ax.set_title(r"Graph edges $\log_{10} |E_{ij}|$")

# %%

# Now lets imshow that matrix, the effective mapping from RIB to RIB

full_matrix = matrices[r"$C^{l, -1} W_{\rm in} {\rm \langle ReLU \rangle} C^{l+1}$"]


def random_permutation_matrix(n):
    perm = torch.randperm(n)
    identity = torch.eye(n)
    permutation_matrix = identity[perm]
    return permutation_matrix


full_matrix_permuted = einops.einsum(
    C_ell_inv,
    W_in_with_id,
    relu_means_with_id,
    torch.block_diag(torch.eye(129), random_permutation_matrix(513)),
    C_ellp1[:, :],
    "rib_l d_embed, d_embed d_mlp, d_mlp, d_mlp d_mlp, d_mlp rib_lp1 -> rib_l rib_lp1",
)

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
fig.suptitle(r"Plotting RIB-RIB matrix $C^{l, -1} W_{\rm in} {\rm \langle ReLU \rangle} C^{l+1}$")
ax1.imshow(full_matrix, cmap="RdBu", vmin=-1, vmax=1)
im = ax2.imshow(full_matrix_permuted, cmap="RdBu", vmin=-1, vmax=1)
ax1.set_title("Actual network")
ax2.set_title("Random permutation of neurons")
ax1.set_xlabel("RIB_out (81 dim)")
ax1.set_ylabel("RIB_in (47 dim)")
fig.colorbar(im)

# %%

plt.figure(figsize=(20, 5))
vminmax = matrices[r"$C^{l, -1} W_{\rm in}$"].abs().max()
plt.imshow(matrices[r"$C^{l, -1} W_{\rm in}$"], cmap="RdBu", vmin=-vminmax / 5, vmax=vminmax / 5)
plt.title(r"$C^{l, -1} W_{\rm in}$ (colorscale x5, row 13 super large)")

plt.figure(figsize=(5, 20))
vminmax = matrices[r"$C^{l, -1} W_{\rm in}$"].abs().max()
plt.imshow(C_ellp1, cmap="RdBu", vmin=-vminmax, vmax=vminmax)
plt.title(r"$C^{l+1}$")

plt.figure(figsize=(20, 5))
vminmax = matrices[r"$C^{l, -1} W_{\rm in}$"].abs().max()
plt.imshow(
    matrices[r"$C^{l, -1} W_{\rm in}$"] ** 2, cmap="RdBu", vmin=-vminmax / 100, vmax=vminmax / 100
)
plt.title(r"$(C^{l, -1} W_{\rm in})^2$ (colorscale x100, row 13 super large)")

plt.figure(figsize=(5, 20))
vminmax = matrices[r"$C^{l, -1} W_{\rm in}$"].abs().max()
plt.imshow(C_ellp1**2, cmap="RdBu", vmin=-vminmax, vmax=vminmax)
plt.title(r"$(C^{l+1})^2$")
# %%
