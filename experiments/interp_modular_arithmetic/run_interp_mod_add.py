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

sec_1_2_mlp_post_acts_svd = svd_activations(sec_1_2_mlp_post_acts)
sec_1_2_mlp_post_acts_pca = pca_activations(sec_1_2_mlp_post_acts)
sec_1_2_mlp_post_acts_fft = fft2(sec_1_2_mlp_post_acts)
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

# ReLU operators
relu_ops = (
    (sec_1_2_mlp_post_acts.to(torch.float64)) / (sec_1_1_mlp_pre_acts.to(torch.float64))
).to(torch.float32)[:, :, 0, :]

relu_ops_fft = torch.fft.fft2(relu_ops, dim=(0, 1))

# plot_fft_activations(
#     relu_ops_fft,
#     nrows=15,
#     figsize=(10, 50),
#     title="ReLU operators",
#     fftshift=True,
#     phaseplot_magnitude_threshold=0.2,
# )

# %%

from torch import tensor

relu_blocks = [
    tensor(
        [
            42,
            58,
            91,
            105,
            131,
            207,
            223,
            224,
            265,
            266,
            277,
            311,
            341,
            351,
            369,
            375,
            387,
            388,
            395,
            429,
            435,
            436,
            439,
            441,
            457,
            468,
            485,
            492,
            500,
        ]
    ),
    tensor([36, 90, 94, 238, 282, 372, 385, 421, 449]),
    tensor(
        [
            10,
            31,
            39,
            43,
            60,
            133,
            139,
            141,
            150,
            182,
            235,
            310,
            321,
            340,
            353,
            358,
            380,
            405,
            412,
            416,
            418,
            446,
        ]
    ),
    tensor(
        [
            19,
            20,
            29,
            45,
            72,
            84,
            117,
            125,
            128,
            137,
            147,
            186,
            187,
            216,
            242,
            258,
            278,
            295,
            303,
            315,
            323,
            329,
            342,
            347,
            355,
            364,
            386,
            403,
            417,
            424,
            428,
            465,
            490,
            493,
            506,
            511,
        ]
    ),
    tensor([190]),
    tensor([118, 371]),
    tensor(
        [
            13,
            18,
            37,
            67,
            89,
            106,
            123,
            135,
            143,
            156,
            161,
            165,
            177,
            185,
            189,
            199,
            211,
            221,
            227,
            246,
            250,
            253,
            259,
            268,
            286,
            292,
            325,
            333,
            335,
            345,
            426,
            430,
            433,
            444,
            451,
            476,
            508,
        ]
    ),
    tensor([162, 267]),
    tensor([55]),
    tensor([73, 344, 488]),
    tensor([343]),
    tensor([28, 56, 75, 124, 236, 291, 313, 314, 398]),
    tensor([7, 26, 57, 65, 102, 170, 217, 346, 458, 462, 464, 502]),
    tensor([44, 95, 509]),
    tensor([193, 272]),
    tensor([64, 88, 183, 326, 392, 396, 461, 480]),
    tensor([174, 225, 229, 233, 239, 257, 269, 275, 281, 283, 296, 352, 367, 425, 482, 483, 496]),
    tensor([22, 35, 61, 86, 96, 113, 203, 209, 218, 350, 400, 406, 445, 454, 460]),
    tensor([34, 53, 158, 240, 249, 280, 304, 393]),
    tensor([23, 148, 163, 201, 243, 254, 260, 362, 368, 370, 374]),
    tensor([27, 40, 66, 76, 82, 101, 108, 153, 179, 244, 349, 397, 404, 407, 448, 481, 497]),
    tensor([3, 12, 176, 200, 215, 284, 309, 320, 331, 363, 378, 402, 422, 450, 452, 455, 507]),
    tensor([15, 24, 52, 80, 87, 136, 180, 194, 230, 273, 293, 294, 317, 389, 391, 442]),
    tensor([9, 17, 107, 115, 122, 160, 252, 262, 271, 279, 288, 432, 472, 489]),
    tensor([71, 134, 205, 220, 366, 381, 434]),
    tensor([109, 319]),
    tensor([16, 111, 130, 219, 276, 290, 298, 390, 443, 473, 504]),
    tensor([14, 63, 157, 339, 361, 413, 415]),
    tensor([197, 512]),
    tensor([0, 54, 59, 104, 112, 114, 116, 167, 202, 204, 302, 308, 354, 377, 440, 463, 467, 498]),
    tensor([97, 119, 166, 169, 171, 214, 338, 382, 470, 499]),
    tensor([38, 93, 144, 154, 471, 487]),
    tensor([25, 46, 51, 62, 98, 132, 149, 178, 306, 348, 356, 359, 376, 469, 475, 495, 501]),
    tensor([49, 92, 120, 127, 213, 226, 241, 270, 299, 334, 409, 419, 466, 503]),
    tensor([47, 287, 322]),
    tensor(
        [
            48,
            68,
            110,
            129,
            159,
            181,
            184,
            206,
            208,
            222,
            255,
            285,
            297,
            305,
            316,
            357,
            360,
            427,
            510,
        ]
    ),
    tensor([11, 21, 32, 100, 138, 146, 175, 191, 210, 327, 383, 384, 399, 431, 459, 474, 491, 505]),
    tensor(
        [
            2,
            5,
            6,
            8,
            30,
            50,
            69,
            70,
            74,
            121,
            126,
            155,
            164,
            188,
            195,
            231,
            232,
            237,
            247,
            248,
            251,
            256,
            289,
            300,
            365,
            373,
            394,
            411,
            420,
            486,
            494,
        ]
    ),
    tensor([4, 330, 336]),
    tensor(
        [
            81,
            83,
            99,
            103,
            168,
            172,
            196,
            198,
            212,
            245,
            263,
            264,
            307,
            312,
            324,
            328,
            332,
            337,
            408,
            423,
            447,
            453,
            477,
            478,
            484,
        ]
    ),
    tensor([41, 77, 78, 140, 151, 192, 228, 274, 410, 437, 438, 456]),
    tensor([1, 33, 79, 85, 142, 145, 152, 173, 234, 261, 301, 318, 379, 401, 414, 479]),
]

plot_fft_activations(
    relu_ops_fft[:, :, relu_blocks[0]],
    nrows=15,
    figsize=(10, 50),
    title="ReLU operators",
    fftshift=True,
    phaseplot_magnitude_threshold=0.2,
)

# %%

plot_fft_activations(
    sec_1_2_mlp_post_acts_fft,
    nrows=15,
    figsize=(10, 50),
    title="Neurons post-ReLU",
    fftshift=True,
    phaseplot_magnitude_threshold=0.2,
)

# %%


# Plot FFT of these
plot_fft_activations(
    rib_acts_embedding_y_fft,
    nrows=15,
    figsize=(10, 50),
    title="RIB ln1.0",
    fftshift=True,
    phaseplot_magnitude_threshold=0.2,
)

# %%


def get_importance(rib_acts: Float[torch.Tensor, "x y seq node"]) -> Float[torch.Tensor, "node"]:
    return einops.einsum(rib_acts, rib_acts, "x y seq node, x y seq node -> node")


plt.semilogy(get_importance(rib_acts_pre_unembed))
plt.xlim(0, None)
plt.ylim(0, None)

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
