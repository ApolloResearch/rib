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


# Get required matrices
model = activations.hooked_model.model
W_in = get_model_attr(model, "sections.section_1.1").W_in.to(torch.float64).to("cpu")
W_out = get_model_attr(model, "sections.section_2.0").W_out.to(torch.float64).to("cpu")


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

# Test relation between cancellation and ReLU blocks

W_tilde = einops.einsum(C_ell_inv, W_in_with_id, "rib_l d_embed, d_embed d_mlp -> rib_l d_mlp")

full_cancellation = (
    einops.einsum(
        W_tilde,
        relu_means_with_id,
        C_ellp1[:, :],
        "rib_l d_mlp, d_mlp, d_mlp rib_lp1 -> rib_l rib_lp1",
    )
    ** 2
)

no_cancellation = einops.einsum(
    W_tilde**2,
    relu_means_with_id**2,
    C_ellp1[:, :] ** 2,
    "rib_l d_mlp, d_mlp, d_mlp rib_lp1 -> rib_l rib_lp1",
)

plt.figure()
plt.imshow(full_cancellation, cmap="RdBu", vmin=-1, vmax=1)
plt.figure()
plt.imshow(no_cancellation, cmap="RdBu", vmin=-1, vmax=1)
# %%
relu_blocks = [
    torch.tensor(
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
    torch.tensor([36, 90, 94, 238, 282, 372, 385, 421, 449]),
    torch.tensor(
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
    torch.tensor(
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
    torch.tensor([190]),
    torch.tensor([118, 371]),
    torch.tensor(
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
    torch.tensor([162, 267]),
    torch.tensor([55]),
    torch.tensor([73, 344, 488]),
    torch.tensor([343]),
    torch.tensor([28, 56, 75, 124, 236, 291, 313, 314, 398]),
    torch.tensor([7, 26, 57, 65, 102, 170, 217, 346, 458, 462, 464, 502]),
    torch.tensor([44, 95, 509]),
    torch.tensor([193, 272]),
    torch.tensor([64, 88, 183, 326, 392, 396, 461, 480]),
    torch.tensor(
        [174, 225, 229, 233, 239, 257, 269, 275, 281, 283, 296, 352, 367, 425, 482, 483, 496]
    ),
    torch.tensor([22, 35, 61, 86, 96, 113, 203, 209, 218, 350, 400, 406, 445, 454, 460]),
    torch.tensor([34, 53, 158, 240, 249, 280, 304, 393]),
    torch.tensor([23, 148, 163, 201, 243, 254, 260, 362, 368, 370, 374]),
    torch.tensor([27, 40, 66, 76, 82, 101, 108, 153, 179, 244, 349, 397, 404, 407, 448, 481, 497]),
    torch.tensor(
        [3, 12, 176, 200, 215, 284, 309, 320, 331, 363, 378, 402, 422, 450, 452, 455, 507]
    ),
    torch.tensor([15, 24, 52, 80, 87, 136, 180, 194, 230, 273, 293, 294, 317, 389, 391, 442]),
    torch.tensor([9, 17, 107, 115, 122, 160, 252, 262, 271, 279, 288, 432, 472, 489]),
    torch.tensor([71, 134, 205, 220, 366, 381, 434]),
    torch.tensor([109, 319]),
    torch.tensor([16, 111, 130, 219, 276, 290, 298, 390, 443, 473, 504]),
    torch.tensor([14, 63, 157, 339, 361, 413, 415]),
    torch.tensor([197, 512]),
    torch.tensor(
        [0, 54, 59, 104, 112, 114, 116, 167, 202, 204, 302, 308, 354, 377, 440, 463, 467, 498]
    ),
    torch.tensor([97, 119, 166, 169, 171, 214, 338, 382, 470, 499]),
    torch.tensor([38, 93, 144, 154, 471, 487]),
    torch.tensor([25, 46, 51, 62, 98, 132, 149, 178, 306, 348, 356, 359, 376, 469, 475, 495, 501]),
    torch.tensor([49, 92, 120, 127, 213, 226, 241, 270, 299, 334, 409, 419, 466, 503]),
    torch.tensor([47, 287, 322]),
    torch.tensor(
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
    torch.tensor(
        [11, 21, 32, 100, 138, 146, 175, 191, 210, 327, 383, 384, 399, 431, 459, 474, 491, 505]
    ),
    torch.tensor(
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
    torch.tensor([4, 330, 336]),
    torch.tensor(
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
    torch.tensor([41, 77, 78, 140, 151, 192, 228, 274, 410, 437, 438, 456]),
    torch.tensor([1, 33, 79, 85, 142, 145, 152, 173, 234, 261, 301, 318, 379, 401, 414, 479]),
]

relu_block_filters = []
resid_block = torch.cat([torch.ones(129), torch.zeros(513)])
relu_block_filters.append(resid_block)
for block in relu_blocks:
    one_hot = torch.zeros(513)
    one_hot[block] = 1
    # prepend 129 zeros
    one_hot = torch.cat([torch.zeros(129), one_hot])
    relu_block_filters.append(one_hot)

relu_block_filters = torch.stack(relu_block_filters)

assert torch.equal(relu_block_filters.sum(dim=0), torch.ones(642))

# %%

# Create a fake relu_block_filters with entries shuffled around in similar sized groups

relu_block_filters_shuffled = []
resid_block = torch.cat([torch.ones(129), torch.zeros(513)])
relu_block_filters_shuffled.append(resid_block)
torch.manual_seed(1)
arange_shuffled = torch.arange(513)[torch.randperm(513)]
i = 0
for block in relu_blocks:
    j = i + len(block)
    one_hot = torch.zeros(513)
    one_hot[arange_shuffled[i:j]] = 1
    i = j
    # prepend 129 zeros
    one_hot = torch.cat([torch.zeros(129), one_hot])
    relu_block_filters_shuffled.append(one_hot)

relu_block_filters_shuffled = torch.stack(relu_block_filters_shuffled)
assert torch.equal(relu_block_filters_shuffled.sum(dim=0), torch.ones(642))

# %%

full_cancellation = (
    einops.einsum(
        W_tilde,
        relu_means_with_id,
        C_ellp1[:, :],
        "rib_l d_mlp, d_mlp, d_mlp rib_lp1 -> rib_l rib_lp1",
    )
    ** 2
)

no_cancellation = einops.einsum(
    W_tilde**2,
    relu_means_with_id**2,
    C_ellp1[:, :] ** 2,
    "rib_l d_mlp, d_mlp, d_mlp rib_lp1 -> rib_l rib_lp1",
)

relu_block_intermediate = einops.einsum(
    W_tilde,
    relu_block_filters,
    relu_means_with_id,
    C_ellp1[:, :],
    "rib_l d_mlp, relu_blocks d_mlp, d_mlp, d_mlp rib_lp1 -> rib_l relu_blocks rib_lp1",
)

relu_block_cancellation = einops.einsum(
    relu_block_intermediate**2,
    "rib_l relu_blocks rib_lp1 -> rib_l rib_lp1",
)

relu_block_shuffle_intermediate = einops.einsum(
    W_tilde,
    relu_block_filters_shuffled,
    relu_means_with_id,
    C_ellp1[:, :],
    "rib_l d_mlp, relu_blocks d_mlp, d_mlp, d_mlp rib_lp1 -> rib_l relu_blocks rib_lp1",
)

relu_block_shuffle_cancellation = einops.einsum(
    relu_block_shuffle_intermediate**2,
    "rib_l relu_blocks rib_lp1 -> rib_l rib_lp1",
)

plt.figure()
plt.title("Full cancellation (square of sum)")
vmax = full_cancellation.abs().max()
vmax_full = vmax
plt.imshow(full_cancellation, cmap="RdBu", vmin=-vmax, vmax=vmax)
plt.colorbar()

plt.figure()
plt.title("No cancellation allowed (sum of squares)")
vmax = no_cancellation.abs().max()
plt.imshow(no_cancellation, cmap="RdBu", vmin=-vmax, vmax=vmax)
plt.colorbar()

plt.figure()
plt.title(
    "Cancellation allowed within ReLU-syncing block\n(sum-over-blocks of squares of sums-within-blocks)"
)
vmax = relu_block_cancellation.abs().max()
plt.imshow(relu_block_cancellation, cmap="RdBu", vmin=-vmax, vmax=vmax)
plt.colorbar()

plt.figure()
plt.title("Randomized ReLU blocks")
vmax = relu_block_shuffle_cancellation.abs().max()
plt.imshow(relu_block_shuffle_cancellation, cmap="RdBu", vmin=-vmax, vmax=vmax)
plt.colorbar()

# %%

# The last two plots with the first colorbar

plt.figure()
plt.title(
    "Cancellation allowed within ReLU-syncing block\n(sum-over-blocks of squares of sums-within-blocks)"
)
plt.imshow(relu_block_cancellation, cmap="RdBu", vmin=-vmax_full, vmax=vmax_full)
plt.colorbar()

plt.figure()
plt.title("Randomized ReLU blocks")
plt.imshow(relu_block_shuffle_cancellation, cmap="RdBu", vmin=-vmax_full, vmax=vmax_full)
plt.colorbar()

# %%
