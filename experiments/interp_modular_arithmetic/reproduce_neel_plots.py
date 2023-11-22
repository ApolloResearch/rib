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


def fft_plot_cos_phase_1d(acts):
    n_freqs = acts.shape[0]
    assert n_freqs % 2 == 1, f"n_freqs = {n_freqs}"
    fftshift_freqs = torch.fft.fftshift(torch.fft.fftfreq(n_freqs))
    center = n_freqs // 2
    assert fftshift_freqs[center] == 0, f"fftshift_freqs[center] = {fftshift_freqs[center]}"
    fftshift_amplitudes = torch.fft.fftshift(acts)
    freq_labels = []
    freq_amplitudes = []
    freq_phases = []
    for i in np.arange(len(fftshift_freqs) // 2, len(fftshift_freqs)):
        f = fftshift_freqs[i]
        print(i, f)
        if f == 0:
            freq_labels.append("const")
            freq_amplitudes.append(fftshift_amplitudes[i])
            freq_phases.append(0)
        else:
            positive_freq_index = i if f > 0 else center - (i - center)
            negative_freq_index = center - (i - center) if f > 0 else i
            positive_freq_amplitude = fftshift_amplitudes[positive_freq_index]
            negative_freq_amplitude = fftshift_amplitudes[negative_freq_index]
            # Test complex conjugate
            assert torch.allclose(
                positive_freq_amplitude, torch.conj(negative_freq_amplitude), rtol=1e-4
            ), f"positive_freq_amplitude = {positive_freq_amplitude}, negative_freq_amplitude = {negative_freq_amplitude}"
            print(
                "Positive freq index",
                positive_freq_index,
                "freq",
                fftshift_freqs[positive_freq_index],
            )
            print(
                "Negative freq index",
                negative_freq_index,
                "freq",
                fftshift_freqs[negative_freq_index],
            )
            # Current term is (positive_freq_amplitude+negative_freq_amplitude) cos(fx)
            #                 + (positive_freq_amplitude-negative_freq_amplitude) i sin(fx)
            # Because of complex conjugate this is equivalent to
            # (positive_freq_amplitude.real + negative_freq_amplitude.real) cos(fx)
            # - (positive_freq_amplitude.imag - negative_freq_amplitude.imag) sin(fx)
            # And also
            # 2*positive_freq_amplitude.real cos(fx) - 2*positive_freq_amplitude.imag sin(fx)
            # This in turn is equal to
            # c cos(fx+phi) where
            # via acosx+bsinx = c cos(x+phi) with c=sign(a)*sqrt(a^2+b^2) and phi=arctan(-b/a)
            # or c=sqrt(a^2+b^2) and phi=-atan2(b, a)
            period = 1 / f
            # How often the period fits into the full range
            label = f"cos{113/period:.0f}"
            freq_labels.append(label)
            # pi character
            pi_str = "\u03C0"
            freq_amplitudes.append(
                2
                * torch.sqrt(positive_freq_amplitude.real**2 + positive_freq_amplitude.imag**2)
            )
            phi = -torch.atan2(
                positive_freq_amplitude.imag,
                positive_freq_amplitude.real,
            )
            freq_phases.append(phi)
    freq_labels = np.array(freq_labels)
    freq_amplitudes = np.array(freq_amplitudes)
    freq_phases = np.array(freq_phases)
    cmap = plt.get_cmap("hsv")
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.scatter(
        freq_labels,
        freq_amplitudes,
        c=freq_phases,
        cmap=cmap,
        vmin=-np.pi,
        vmax=np.pi,
    )
    # Vertical x labels
    ax.set_xticks(np.arange(len(freq_labels)))
    ax.set_xticklabels(freq_labels, rotation=-90)
    # colkorhar
    ax.set_ylabel("Amplitude A")
    ax.set_title("Each term A cos f = A*cos(2π/113 f + phi)")
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
    sm._A = []
    # grid
    ax.grid()
    fig.colorbar(sm)
    return fig, ax


# fft_plot_cos_phase_1d(sec_pre_2_resid_embed_acts_x_fft[:, 0, 3])
for i in range(10):
    fig, ax = fft_plot_cos_phase_1d(sec_pre_2_resid_embed_acts_x_fft[:, 0, i])
    ax.set_yscale("log")
# %%

from rib.models.utils import get_model_attr

model = activations.hooked_model.model
W_E = get_model_attr(model, "sections.pre.0").W_E.to(torch.float64).to("cpu")
W_E_fft = torch.fft.fft(W_E[:-1], dim=0)
fft_plot_cos_phase_1d((W_E_fft**2).sum(dim=-1))

# %%
