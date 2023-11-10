#  %%
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from activations import Activations
from jaxtyping import Float
from plotting import annotated_fft_line_plot, plot_activations, plot_fft_activations
from transformations import fft2, pca_activations, svd_activations

torch.set_grad_enabled(False)

parent_dir = Path(__file__).parent

# %%

activations = Activations(config_path_str=parent_dir.joinpath("mod_arithmetic_config.yaml"))
activations.print_info()

# %%

# Extract all normal activations

attention_scores = activations.get_section_activations(
    section="sections.section_0.1.attention_scores",
)[0]

attention_pattern = torch.softmax(attention_scores, dim=-1)
attention_pattern_p = attention_pattern[:, :, :, -1, :]
attention_pattern_p_to_x = attention_pattern[:, :, :, -1, 0]

sec_pre_2_resid_embed_acts = activations.get_section_activations(section="sections.pre.2")[0]
(
    sec_pre_2_resid_embed_acts_x,
    sec_pre_2_resid_embed_acts_y,
    sec_pre_2_resid_embed_acts_z,
) = einops.rearrange(sec_pre_2_resid_embed_acts, "x y p h -> p x y h")

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

# Extract RIB activations

rib_acts_embedding = activations.get_rib_activations(section="sections.pre.2")
orthog_acts_embedding = activations.get_rib_activations(
    section="sections.pre.2", ablation_type="orthogonal"
)
rib_acts_embedding_x, rib_acts_embedding_y, rib_acts_embedding_z = einops.rearrange(
    rib_acts_embedding, "x y p h -> p x y h"
)

rib_acts_extended_embedding = activations.get_rib_activations(section="sections.section_0.2")

# Note: mlp_post and pre_umembed are basically the same
rib_acts_mlp_post = activations.get_rib_activations(section="sections.section_1.2")
rib_acts_pre_unembed = activations.get_rib_activations(section="sections.section_2.2")

# %%


def get_importance(rib_acts: Float[torch.Tensor, "x y seq node"]) -> Float[torch.Tensor, "node"]:
    return einops.einsum(rib_acts, rib_acts, "x y seq node, x y seq node -> node")


plt.semilogy(get_importance(rib_acts_pre_unembed))
plt.xlim(0, None)
plt.ylim(0, None)

# %%

# Create PCA and SVD versions of normal activations, and FFT these as well as RIB

# Embedding
sec_pre_2_resid_embed_acts_svd = svd_activations(sec_pre_2_resid_embed_acts)

sec_1_2_mlp_post_acts_svd = svd_activations(sec_1_2_mlp_post_acts)
sec_1_2_mlp_post_acts_pca = pca_activations(sec_1_2_mlp_post_acts)
sec_1_2_mlp_post_acts_svd_fft = fft2(sec_1_2_mlp_post_acts_svd)
sec_1_2_mlp_post_acts_pca_fft = fft2(sec_1_2_mlp_post_acts_pca)

# sec_pre_2_resid_embed_acts_xyz_svd = svd_activations(sec_pre_2_resid_embed_acts
#     sec_pre_2_resid_embed_acts.reshape(113, 113, -1)
# ).reshape(113, 113, 3, -1)
# sec_pre_2_resid_embed_acts_xyz_pca = pca_activations(
#     sec_pre_2_resid_embed_acts.reshape(113, 113, -1)
# ).reshape(113, 113, 3, -1)

sec_pre_2_resid_embed_acts_x_svd = svd_activations(sec_pre_2_resid_embed_acts_x)
sec_pre_2_resid_embed_acts_x_pca = pca_activations(sec_pre_2_resid_embed_acts_x)
sec_pre_2_resid_embed_acts_svd_fft = fft2(sec_pre_2_resid_embed_acts_svd)
orthog_acts_embedding_fft = fft2(orthog_acts_embedding)
# sec_pre_2_resid_embed_acts_xyz_svd_fft = fft2(sec_pre_2_resid_embed_acts_xyz_svd)
# sec_pre_2_resid_embed_acts_xyz_pca_fft = fft2(sec_pre_2_resid_embed_acts_xyz_pca)
sec_pre_2_resid_embed_acts_x_svd_fft = fft2(sec_pre_2_resid_embed_acts_x_svd)
sec_pre_2_resid_embed_acts_x_pca_fft = fft2(sec_pre_2_resid_embed_acts_x_pca)

sec_pre_2_resid_embed_acts_y_svd = svd_activations(sec_pre_2_resid_embed_acts_y)
sec_pre_2_resid_embed_acts_y_pca = pca_activations(sec_pre_2_resid_embed_acts_y)
sec_pre_2_resid_embed_acts_y_svd_fft = fft2(sec_pre_2_resid_embed_acts_y_svd)
sec_pre_2_resid_embed_acts_y_pca_fft = fft2(sec_pre_2_resid_embed_acts_y_pca)

rib_acts_embedding_x_fft = fft2(rib_acts_embedding_x)
rib_acts_embedding_y_fft = fft2(rib_acts_embedding_y)

# Attention pattern
attention_pattern_p_to_x_svd = svd_activations(attention_pattern_p_to_x)
attention_pattern_p_to_x_pca = pca_activations(attention_pattern_p_to_x)
attention_pattern_p_to_x_fft = fft2(attention_pattern_p_to_x)
attention_pattern_p_to_x_svd_fft = fft2(attention_pattern_p_to_x_svd)
attention_pattern_p_to_x_pca_fft = fft2(attention_pattern_p_to_x_pca)

# Extended embedding
sec_0_2_resid_post_attn_acts_svd = svd_activations(sec_0_2_resid_post_attn_acts)
sec_0_2_resid_post_attn_acts_pca = pca_activations(sec_0_2_resid_post_attn_acts)
sec_0_2_resid_post_attn_acts_svd_fft = fft2(sec_0_2_resid_post_attn_acts_svd)
sec_0_2_resid_post_attn_acts_pca_fft = fft2(sec_0_2_resid_post_attn_acts_pca)
rib_acts_extended_embedding_fft = fft2(rib_acts_extended_embedding)

# Pre ReLU
sec_1_1_mlp_pre_acts_svd = svd_activations(sec_1_1_mlp_pre_acts)
sec_1_1_mlp_pre_acts_pca = pca_activations(sec_1_1_mlp_pre_acts)
sec_1_1_mlp_pre_acts_svd_fft = fft2(sec_1_1_mlp_pre_acts_svd)
sec_1_1_mlp_pre_acts_pca_fft = fft2(sec_1_1_mlp_pre_acts_pca)

# Post MLP
sec_2_1_resid_post_mlp_acts_svd = svd_activations(sec_2_1_resid_post_mlp_acts)
sec_2_1_resid_post_mlp_acts_pca = pca_activations(sec_2_1_resid_post_mlp_acts)
sec_2_1_resid_post_mlp_acts_svd_fft = fft2(sec_2_1_resid_post_mlp_acts_svd)
sec_2_1_resid_post_mlp_acts_pca_fft = fft2(sec_2_1_resid_post_mlp_acts_pca)
rib_acts_mlp_post_fft = torch.fft.fft2(rib_acts_mlp_post, dim=(0, 1))
rib_acts_mlp_post_fft_z = rib_acts_mlp_post_fft[:, :, -1, :]

# %%

# Plot PCA'ed and SVD'ed activations
plot_activations(
    attention_pattern_p_to_x_svd, nrows=2, center=False, title="Attn patterns, SVD over heads"
)

# Plot FFT of these
plot_fft_activations(
    attention_pattern_p_to_x_svd_fft,
    nrows=4,
    title="Attn patterns, SVD over heads, FFT'ed",
    fftshift=True,
    phaseplot_magnitude_threshold=0.3,
)


# %%


def print_acts_and_phases(ffted_acts, index, p=113, lower=600):
    for x in range(p):
        for y in range(p):
            if ffted_acts[x, y, index].abs() > lower and ffted_acts[x, y, 0].abs() < 800:
                freqs = torch.fft.fftfreq(ffted_acts.shape[0])
                val = ffted_acts[x, y, index].abs().item()
                phase = ffted_acts[x, y, index].angle().item()
                print(
                    f"({freqs[x]:.3f}, {freqs[y]:.3f})",
                    f"Value {val:.4f}",
                    f"Phase {phase/np.pi*180:.1f} deg",
                    f"Phrase as real & imag: e^(i phi) = {np.cos(phase):.3f} + i {np.sin(phase):.3f}",
                )


# %%

print_acts_and_phases(attention_pattern_p_to_x_pca_fft, 0)
# Confirmed same output as play_mod_arithmetic.py notebook

# print_acts_and_phases(rib_acts_mlp_post_fft_z, 0, lower=200000)
# Confirmed same output as play_mod_arithmetic.py notebook

# %%

# Plot embedding activations and compare whether RIB or SVD activations are simpler
fig, axes = plt.subplots(10, 1, figsize=(8, 50))
for i in range(10):
    annotated_fft_line_plot(
        acts=rib_acts_embedding_x_fft[:, :, i].mean(dim=1),
        label=f"RIB, i={i}",
        ax=axes[i],
        title=f"RIB acts at x embedding, i={i}",
        annotation_magnitude_threshold=0.05,
    )

# %%

# Now plot ln2.0 input aka sections.section_0.2 output to compare

plot_fft_activations(
    rib_acts_extended_embedding_fft[:, :, 0, 9:11],
    nrows=2,
    figsize=(10, 10),
    title="FFT of RIB acts at extended embedding, dim9-10",
    fftshift=True,
    annotate=True,
    phaseplot_magnitude_threshold=0.3,
)

# %%

plot_fft_activations(
    rib_acts_extended_embedding_fft[:, :, 0, 6:8],
    nrows=2,
    figsize=(10, 10),
    title="FFT of RIB acts at extended embedding, dim6-7",
    fftshift=True,
    annotate=True,
    phaseplot_magnitude_threshold=0.6,
)

# %%

# look at edges into ln2 node 9

graph_path = "/mnt/ssd-apollo/stefan/rib/modular_arithmetic_interaction_graph.pt"
edges = dict(torch.load(graph_path)["edges"])

plt.plot(dict(edges)["ln1.0"][9, :])  # edges: [layer l + 1 nodes, layer l nodes]
plt.xlim(0, 12)
plt.xlabel("ln 1.0 node index")
plt.ylabel("edge strength to ln2.0 #9")

# %%
# Nix looking at pre-unembed ffts.

plot_fft_activations(rib_acts_mlp_post_fft_z)

# Note: most of the high magnitude entries are on the diagonal. This makes sense!
# Neel's claim was that sin(x+y) and cos(x+y) were the important terms

# %%
num_dims = 10
fig, axs = plt.subplots(num_dims, 2, sharex=True)
freqs = torch.fft.fftshift(torch.fft.fftfreq(113))

# svd
svd_shifted = torch.fft.fftshift(sec_2_1_resid_post_mlp_acts_svd_fft[:, :, 0, :], dim=[0, 1])
svd_diag = svd_shifted[range(113), range(113), :]
for i in range(num_dims):
    axs[i][0].plot(freqs, svd_diag.abs()[:, i])
axs[0][0].set_title("SVD")

# rib
rib_shifted = torch.fft.fftshift(rib_acts_mlp_post_fft_z, dim=[0, 1])
rib_diag = rib_shifted[range(113), range(113), :]

fig.set_size_inches((7, 8))
for i in range(num_dims):
    axs[i][1].plot(freqs, rib_diag.abs()[:, i])
axs[0, 1].set_title("RIB")


axs[-1, 0].set_xlabel("frequency")
axs[-1, 1].set_xlabel("frequency")


# %%
plot_fft_activations(
    sec_1_2_mlp_post_acts_svd_fft[:, :, 0, :],
    nrows=10,
    figsize=(10, 50),
    title="SVD directions, post-ReLU MLP",
)

plot_fft_activations(
    rib_acts_mlp_post_fft_z[:, :, :],
    nrows=10,
    figsize=(10, 50),
    title="RIB directions, post-ReLU MLP",
)
# %%


plot_fft_activations(
    sec_1_1_mlp_pre_acts_svd_fft[:, :, 0, :][:, :, :],
    nrows=10,
    figsize=(10, 50),
    title="SVD directions, pre-ReLU MLP",
)
plot_fft_activations(
    rib_acts_extended_embedding_fft[:, :, 0, :][:, :, :],
    nrows=10,
    figsize=(10, 50),
    title="RIB directions post attn (should be like pre-ReLU MLP)",
)
# %%

# FFT Demo annd experiments
x = np.arange(0, 113)
y = np.arange(0, 113)
x, y = np.meshgrid(x, y)
f = (
    np.sin(2 * np.pi * x * 0.106)
    + np.cos(2 * np.pi * x * 0.1504)
    + np.sin(2 * np.pi * x * 0.2035) * np.cos(2 * np.pi * y * 0.2035)
)
plt.imshow(f)
# FFT
F = np.fft.fft2(f)
F = np.fft.fftshift(F)
freqs = np.fft.fftshift(np.fft.fftfreq(113))
plt.figure(figsize=(10, 10))
plt.imshow(np.abs(F), extent=[-0.5, 0.5, -0.5, 0.5])
plt.axvline(0.106, color="white", alpha=0.1)
plt.axvline(0.1504, color="white", alpha=0.1)
plt.axvline(0.2035, color="white", alpha=0.1)
plt.colorbar()
# Symmetric area
plt.plot(freqs, freqs)
plt.fill_between(freqs, freqs, -np.max(freqs), alpha=0.1)

# Print phases where abs > 100
for i in range(113):
    for j in range(113):
        if np.abs(F[i, j]) > 100:
            print(f"({freqs[i]:.3f}, {freqs[j]:.3f})")
            print(f"Value {np.abs(F[i, j]):.1f}")
            print(f"Phase {np.angle(F[i, j])/np.pi*180:.1f} deg")
            print(
                f"Phrase as real & imag: e^(i phi) = {np.cos(np.angle(F[i, j])):.3f} + i {np.sin(np.angle(F[i, j])):.3f}"
            )
# Convert freqs to sin and cos terms
# F(-x, -y) = F*(x, y)
# F(x, y) term  = cos(x)*cos(y) + i cos(x) sin(y) + i sin(x) cos(y) - sin(x) sin(y)
# F(-x, -y) term = cos(x)*cos(y) - i cos(x) sin(y) - i sin(x) cos(y) + sin(x) sin(y)
# Real part: Same sign in ampltiude, summed. Complex part: Opposite sign in amplitude, subtracted.
# Sum term = 2*F.real * cos(x)*cos(y) - 2*F.real * sin(x)*sin(y) - 2*F.imag * cos(x)*sin(y) - 2*F.imag * sin(x)*cos(y)
# = 2*F.real * (cos(x)*cos(y) - sin(x)*sin(y)) - 2*F.imag * (cos(x)*sin(y) + sin(x)*cos(y))
# There is no further symmetry (for information theory reasons alone)
# F(x, -y) term = cos(x)*cos(y) - i cos(x) sin(y) + i sin(x) cos(y) - sin(x) sin(y)
# F(-x, y) term = cos(x)*cos(y) + i cos(x) sin(y) - i sin(x) cos(y) - sin(x) sin(y)
# Sum term: 2*F.real * cos(x)*cos(y) - 2*F.real * sin(x)*sin(y) + 2*F(x,-y).imag * cos(x)*sin(y) - 2*F(x,-y).imag * sin(x)*cos(y)
# = 2*F.real * (cos(x)*cos(y) - sin(x)*sin(y)) + 2*F(x,-y).imag * (cos(x)*sin(y) - sin(x)*cos(y))
# Sum of both terms
# = 2*(F++.real + F+-.real) * (cos(x)*cos(y) - sin(x)*sin(y)) - 2*(F++.imag - F+-.imag) * ... or so
# It appears that we observe the magnitude fir F(x, y) and F(x, -y) to be almost the same, with different phases though.
# If the phases were equal or so this would imply no sin-cos cross terms but only cos-cos and sin-sin I think (?)

# There exists a formula that takes in F++ F+- F-+ F-- and outputs Fcoscos Fsinsin Fcossin Fsincos
# Do later with clearer math

# %%
