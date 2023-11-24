#  %%
# IPython magic
%reload_ext autoreload
%autoreload 2

# Imports
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from activations import Activations
from jaxtyping import Float
from plotting import *
from transformations import fft2, pca_activations, svd_activations

# Settings
torch.set_grad_enabled(False)
parent_dir = Path(__file__).parent

# %%

# Setup
activations = Activations(
    config_path_str=parent_dir.joinpath("mod_arithmetic_config.yaml"),
    interaction_graph_path="/mnt/ssd-apollo/stefan/rib/modular_arithmetic_interaction_graph.pt",
)
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

sec_pre_2_resid_embed_acts_x_fft = fft2(sec_pre_2_resid_embed_acts_x)
sec_pre_2_resid_embed_acts_y_fft = fft2(sec_pre_2_resid_embed_acts_y)
plot_fft_activations(sec_pre_2_resid_embed_acts_x_fft)
# %%

annotated_fft_line_plot(sec_pre_2_resid_embed_acts_x_fft[:, 0, 2])

# %%
plt.plot(
    torch.fft.fftshift(torch.fft.fftfreq(sec_pre_2_resid_embed_acts_x_fft.shape[0])),
    torch.fft.fftshift(sec_pre_2_resid_embed_acts_x_fft[:, 0, :].abs().sum(dim=-1)),
)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
# %%
for i in range(4):
    plt.plot(torch.fft.fftshift(sec_pre_2_resid_embed_acts_x_fft[:, 0, i].abs()))

# %%


# def fft_plot_cos_phase_1d(acts, rtol=1e-4):
#     r"""
#     Plot the cos phase of a 1D FFT data. The input to the FFT is assumed to be real-valued.

#     Derivation for why we can express the FFT terms \tilde A_f e^(i2π/113 f x) as
#     A_f cos(2π/113 f x + φ), for real-valued input:

#         1. Convert from f=0...112 to f=-56...56 since f=57...112 correspond to f=-56...-1),
#            the difference is a expoent shift by 2π.

#         2. Convert to cos and sin terms, e^(i2π/113 f x) = cos(2π/113 f x) + i sin(2π/113 f x)

#         3. Combine pairs of negative and positive frequencies:
#            (A_f + A_-f) cos(2π/113 f x) + (A_f - A_-f) i sin(2π/113 f x)

#         4. Use that the ampltiudes of a real-valued FFT are complex conjugates A_f = conj(A_-f).
#            (A_f + A_-f) = 2 Re(A_f) and (A_f - A_-f) i = - 2 Im(A_f)

#         5. Finally combine cos and sin terms using "harmonic addition"
#            (https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Sine_and_cosine):
#            a*cos(x) + b*sin(x) = c*cos(x+phi) with c=sign(a)*sqrt(a^2+b^2) and phi=arctan(-b/a)
#            or a*cos(x) + b*sin(x) = c*cos(x+phi) with c=sqrt(a^2+b^2) and phi=-atan2(b, a).

#            Each f term is sqrt(Re(2A_f)^2 + Im(2A_f)^2) cos(2π/113 f x -atan2(Im(-A_f), Re(A_f)))
#            or simply 2*|A_f| cos(2π/113 f x + atan2(Im(A_f), Re(A_f))).

#     Args:
#         acts: nD tensor of activations, complex. 0th dimension assumed to be frequency.

#     Returns:
#         fig, ax: matplotlib figure and axis objects.
#     """
#     # Calculate the FFT
#     n_freqs = acts.shape[0]
#     assert n_freqs % 2 == 1, f"n_freqs={n_freqs} must be odd"

#     # Apply fftshift, makes thinking about the frequencies easier
#     fftshift_freqs = torch.fft.fftshift(torch.fft.fftfreq(n_freqs))
#     fftshift_amplitudes = torch.fft.fftshift(acts)

#     # Central frequency (constant)
#     i_center = n_freqs // 2
#     assert fftshift_freqs[i_center] == 0, f"fftshift_freqs[center] = {fftshift_freqs[i_center]}"

#     # Collect the frequencies and compute the amplitudes and phases
#     freq_labels = []
#     freq_amplitudes = []
#     freq_phases = []
#     for i in np.arange(n_freqs // 2, n_freqs):
#         f = fftshift_freqs[i]
#         if i == i_center:
#             a = fftshift_amplitudes[i]
#             assert f == 0, "central frequency must be 0"
#             assert torch.all(torch.arctan(a.imag / a.real) < rtol), "central frequency must be real"
#             freq_labels.append("const")
#             freq_amplitudes.append(a.real)
#             freq_phases.append(torch.zeros_like(a.real))
#         else:
#             assert f > 0, f"should be iterating over positive frequencies but f={f}"
#             i_pos = i
#             i_neg = i_center - (i - i_center)
#             a_pos = fftshift_amplitudes[i_pos]
#             a_neg = fftshift_amplitudes[i_neg]
#             # Assert complex conjugate of amplitudes (due to real input)
#             assert torch.allclose(
#                 a_pos, torch.conj(a_neg), rtol=rtol
#             ), "amplitude pairs must be complex conjugates (real input?)"
#             assert (n_freqs * f + 0.5) % 1 - 0.5 < rtol, f"{n_freqs}*f={n_freqs*f} must be integer"
#             freq_labels.append(f"cos{n_freqs*f:.0f}")
#             freq_amplitudes.append(2 * torch.sqrt(a_pos.real**2 + a_pos.imag**2))
#             freq_phases.append(-torch.atan2(a_pos.imag, a_pos.real))

#     # Plot the amplitudes and phases
#     freq_labels = np.array(freq_labels)
#     freq_labels_ticks = torch.arange(len(freq_labels))
#     freq_amplitudes = np.array(freq_amplitudes)
#     freq_phases = np.array(freq_phases)
#     fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
#     freq_labels_ticks_nD = (
#         freq_labels_ticks
#         if acts.ndim == 1
#         else torch.repeat_interleave(freq_labels_ticks, acts.shape[1])
#     )
#     im = ax.scatter(
#         freq_labels_ticks_nD,
#         freq_amplitudes,
#         c=freq_phases,
#         cmap="hsv",
#         vmin=-np.pi,
#         vmax=np.pi,
#         s=10,
#     )
#     ax.plot(freq_labels_ticks, freq_amplitudes, lw=0.5)
#     ax.set_xticks(np.arange(len(freq_labels)))
#     ax.set_xticklabels(freq_labels, rotation=-90)
#     ax.set_ylabel("Amplitude A")
#     ax.set_xlabel("Each term A cos f = A*cos(2π/113 f + φ)")
#     cbar = fig.colorbar(im, ax=ax, label="Phase φ")
#     ax.grid(color="grey", linestyle="--", alpha=0.3)
#     return fig, ax


# %%

fft_plot_cos_phase_1d(sec_pre_2_resid_embed_acts_x_fft[:, 0, :4])

from rib.models.utils import get_model_attr

model = activations.hooked_model.model
W_E = get_model_attr(model, "sections.pre.0").W_E.to(torch.float64).to("cpu")
W_E_fft = torch.fft.fft(W_E[:-1], dim=0)
fft_plot_cos_phase_1d((W_E_fft**2).sum(dim=-1))

# %%


plot_fft_activations_cosphase(
    fft2(sec_2_1_resid_post_mlp_acts[:, :, 0, :2]), nrows=2, figsize=(10, 20)
)


# %%

plt.imshow(fft2(sec_2_1_resid_post_mlp_acts[:, :, 0, 0]).abs())
# %%

plot_fft_activations_cosphase(
    (sec_pre_2_resid_embed_acts_x_fft[:, :, :]), nrows=2, figsize=(10, 20)
)

# fft_plot_cos_phase_1d(sec_pre_2_resid_embed_acts_x_fft[:, 0, :4])
# %%

# Attention pattern
attention_pattern_p_to_x_svd = svd_activations(attention_pattern_p_to_x)
attention_pattern_p_to_x_pca = pca_activations(attention_pattern_p_to_x)
attention_pattern_p_to_x_fft = fft2(attention_pattern_p_to_x)
attention_pattern_p_to_x_svd_fft = fft2(attention_pattern_p_to_x_svd)
attention_pattern_p_to_x_pca_fft = fft2(attention_pattern_p_to_x_pca)

# %%


def print_acts_and_phases(ffted_acts, index, p=113, lower=600, upper=np.inf):
    for x in range(p):
        for y in range(p):
            if ffted_acts[x, y, index].abs() > lower and ffted_acts[x, y, 0].abs() < upper:
                freqs = torch.fft.fftfreq(ffted_acts.shape[0])
                val = ffted_acts[x, y, index].abs().item()
                phase = ffted_acts[x, y, index].angle().item()
                print(
                    f"({freqs[x]:.3f}, {freqs[y]:.3f})",
                    f"Value {val:.4f}",
                    f"Phase {phase/np.pi*180:.1f} deg",
                    f"Phrase as real & imag: e^(i phi) = {np.cos(phase):.3f} + i {np.sin(phase):.3f}",
                )


sec_0_2_resid_post_attn_acts_z = sec_0_2_resid_post_attn_acts[:, :, 0, :]
sec_0_2_resid_post_attn_acts_z_svd = svd_activations(sec_0_2_resid_post_attn_acts_z)
sec_0_2_resid_post_attn_acts_z_pca = pca_activations(sec_0_2_resid_post_attn_acts_z)
sec_0_2_resid_post_attn_acts_z_svd_fft = fft2(sec_0_2_resid_post_attn_acts_z_svd)
sec_0_2_resid_post_attn_acts_z_pca_fft = fft2(sec_0_2_resid_post_attn_acts_z_pca)


fft_plot_eikx_2d(sec_0_2_resid_post_attn_acts_z_svd_fft, nrows=2)
fft_plot_cosplusminus(sec_0_2_resid_post_attn_acts_z_svd_fft, nrows=4)
fft_plot_coscos_sinsin(sec_0_2_resid_post_attn_acts_z_svd_fft, nrows=4)


sec_2_1_resid_post_mlp_acts_z = sec_2_1_resid_post_mlp_acts[:, :, 0, :]
sec_2_1_resid_post_mlp_acts_z_svd = svd_activations(sec_2_1_resid_post_mlp_acts_z)
sec_2_1_resid_post_mlp_acts_z_pca = pca_activations(sec_2_1_resid_post_mlp_acts_z)
sec_2_1_resid_post_mlp_acts_z_svd_fft = fft2(sec_2_1_resid_post_mlp_acts_z_svd)
sec_2_1_resid_post_mlp_acts_z_pca_fft = fft2(sec_2_1_resid_post_mlp_acts_z_pca)

fft_plot_eikx_2d(sec_2_1_resid_post_mlp_acts_z_svd_fft, nrows=2)
fft_plot_cosplusminus(sec_2_1_resid_post_mlp_acts_z_svd_fft, nrows=4)
fft_plot_coscos_sinsin(sec_2_1_resid_post_mlp_acts_z_svd_fft, nrows=4)


print_acts_and_phases(attention_pattern_p_to_x_svd_fft, 1, lower=3e3)
# %%
# plot_fft_activations(fft2(sec_0_2_resid_post_attn_acts[:,:,0,:]))
# print_acts_and_phases(fft2(sec_0_2_resid_post_attn_acts[:,:,0,:]), 0, lower=1e4)
print_acts_and_phases(fft2(sec_0_2_resid_post_attn_acts[:, :, 0, :]), 0, lower=1e3)


# %%


# Plot FFT of these
# plot_fft_activations(
#     attention_pattern_p_to_x_svd_fft,
#     nrows=4,
#     title="Attn patterns, SVD over heads, FFT'ed",
#     fftshift=True,
#     phaseplot_magnitude_threshold=0.3,
# )

# plot_fft_activations_cosphase(attention_pattern_p_to_x_svd_fft, nrows=4, figsize=(8, 16))
# plot_fft_activations_cosphase(fft2(svd_activations(sec_0_2_resid_post_attn_acts[:, :, 0, :])))
# %%
plot_fft_activations_cosphase(fft2((sec_1_1_mlp_pre_acts[:, :, 0, :])))
# %%
plot_fft_activations(fft2((sec_1_1_mlp_pre_acts[:, :, 0, :])), nrows=20, figsize=(10, 50))


# %%
plot_fft_activations_coscos(fft2((sec_1_1_mlp_pre_acts[:, :, 0, :])), nrows=20, figsize=(10, 50))

# %%
