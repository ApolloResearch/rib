#  %%
import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from activations import Activations
from plotting import annotated_fft_line_plot, plot_activations, plot_fft_activations
from transformations import fft2, pca_activations, svd_activations

torch.set_grad_enabled(False)

activations = Activations(config_path_str="mod_arithmetic_config.yaml")

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

# Create PCA and SVD versions of normal activations, and FFT these as well as RIB

# Embedding
sec_pre_2_resid_embed_acts_svd = svd_activations(sec_pre_2_resid_embed_acts)

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


# %%

print_acts_and_phases(attention_pattern_p_to_x_pca_fft, 0)
# Confirmed same output as play_mod_arithmetic.py notebook

print_acts_and_phases(rib_acts_mlp_post_fft_z, 0, lower=200000)
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
