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
from rib.models.utils import get_model_attr

# Settings
torch.set_grad_enabled(False)
parent_dir = Path(__file__).parent

# %%

# Setup
activations = Activations(
    config_path_str=parent_dir.joinpath("mod_arithmetic_config.yaml"),
    interaction_graph_path="/mnt/ssd-apollo/stefan/rib/modular_arithmetic_interaction_graph.pt",
)

model = activations.hooked_model.model

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

# Embeddings

sec_pre_2_resid_embed_acts_x_fft = fft2(sec_pre_2_resid_embed_acts_x)
sec_pre_2_resid_embed_acts_y_fft = fft2(sec_pre_2_resid_embed_acts_y)
sec_pre_2_resid_embed_acts_x_svd = svd_activations(sec_pre_2_resid_embed_acts_x)
sec_pre_2_resid_embed_acts_y_svd = svd_activations(sec_pre_2_resid_embed_acts_y)
sec_pre_2_resid_embed_acts_x_svd_fft = fft2(sec_pre_2_resid_embed_acts_x_svd)
sec_pre_2_resid_embed_acts_y_svd_fft = fft2(sec_pre_2_resid_embed_acts_y_svd)

# %%

fft_plot_cos_phase_1d(sec_pre_2_resid_embed_acts_x_fft[:, 0, :4], title="First four residual stream dimensions")

fft_plot_cos_phase_1d((sec_pre_2_resid_embed_acts_x_fft[:, 0, :].abs()**2).sum(dim=-1)+0j, labels=False, title="Sum over magnitude squared of dimensions")

fft_plot_cos_phase_1d(sec_pre_2_resid_embed_acts_x_svd_fft[:, 0, :2], rtol=1e-2, title="SVD direction 0 and 1")
fft_plot_cos_phase_1d(sec_pre_2_resid_embed_acts_x_svd_fft[:, 0, 2:4], rtol=1e-2, title="SVD direction 2 and 3")
fft_plot_cos_phase_1d(sec_pre_2_resid_embed_acts_x_svd_fft[:, 0, 4:6], rtol=1e-2, title="SVD direction 4 and 5")
fft_plot_cos_phase_1d(sec_pre_2_resid_embed_acts_x_svd_fft[:, 0, 6:8], rtol=1e-2, title="SVD direction 6 and 7")
fft_plot_cos_phase_1d(sec_pre_2_resid_embed_acts_x_svd_fft[:, 0, 8:10], rtol=1e-2, title="SVD direction 8 and 9")

W_E = get_model_attr(model, "sections.pre.0").W_E.to(torch.float64).to("cpu")
W_E_fft = torch.fft.fft(W_E[:-1], dim=0)
fft_plot_cos_phase_1d((W_E_fft**2).sum(dim=-1), labels=False, title="Sum over magnitude squared of embedding directions (1D FFT of W_E)")

# %%

# Attention pattern

attention_pattern = torch.softmax(attention_scores, dim=-1)
attention_pattern_p = attention_pattern[:, :, :, -1, :]
attention_pattern_p_to_x = attention_pattern[:, :, :, -1, 0]

attention_pattern_p_to_x_fft = fft2(attention_pattern_p_to_x)
attention_pattern_p_to_x_svd = svd_activations(attention_pattern_p_to_x)
attention_pattern_p_to_x_pca = pca_activations(attention_pattern_p_to_x)
attention_pattern_p_to_x_svd_fft = fft2(attention_pattern_p_to_x_svd)
attention_pattern_p_to_x_pca_fft = fft2(attention_pattern_p_to_x_pca)

fft_plot_eikx_2d(attention_pattern_p_to_x_svd_fft, title="Attn pattern, SVD over heads, each term corresponds to e^(i2π/113 f_x x + f_y y + φ)")

fft_plot_cosplusminus(attention_pattern_p_to_x_svd_fft, title="SVD Attn pattern, A_{+/-} cos(2π/113 f_x x +/- f_y y + φ_{+/-})", nrows=4)

fft_plot_coscos_sinsin(attention_pattern_p_to_x_svd_fft, title="SVD Attn pattern\n A_cos cos(2π/113 f_x x + φ_x) cos(2π/113 f_y y + φ_y) \n+ A_sin sin(2π/113 f_x x + φ_x) sin(2π/113 f_y y + φ_y)", nrows=4)

# %%

# Resid Mid

sec_0_2_resid_post_attn_acts_z = sec_0_2_resid_post_attn_acts[:, :, 0, :]
sec_0_2_resid_post_attn_acts_z_svd = svd_activations(sec_0_2_resid_post_attn_acts_z)
sec_0_2_resid_post_attn_acts_z_pca = pca_activations(sec_0_2_resid_post_attn_acts_z)
sec_0_2_resid_post_attn_acts_z_svd_fft = fft2(sec_0_2_resid_post_attn_acts_z_svd)
sec_0_2_resid_post_attn_acts_z_pca_fft = fft2(sec_0_2_resid_post_attn_acts_z_pca)

fft_plot_eikx_2d(sec_0_2_resid_post_attn_acts_z_svd_fft, nrows=10, title="Resid Mid, SVD over heads, each term corresponds to e^(i2π/113 f_x x + f_y y + φ)")
fft_plot_cosplusminus(sec_0_2_resid_post_attn_acts_z_svd_fft, title="Resid Mid pattern, A_{+/-} cos(2π/113 f_x x +/- f_y y + φ_{+/-})", nrows=5)
fft_plot_coscos_sinsin(sec_0_2_resid_post_attn_acts_z_svd_fft, title="Resid mid\n A_cos cos(2π/113 f_x x + φ_x) cos(2π/113 f_y y + φ_y) \n+ A_sin sin(2π/113 f_x x + φ_x) sin(2π/113 f_y y + φ_y)", nrows=5)

# %%

# Post MLP

sec_2_1_resid_post_mlp_acts_z = sec_2_1_resid_post_mlp_acts[:, :, 0, :]
sec_2_1_resid_post_mlp_acts_z_svd = svd_activations(sec_2_1_resid_post_mlp_acts_z)
sec_2_1_resid_post_mlp_acts_z_pca = pca_activations(sec_2_1_resid_post_mlp_acts_z)
sec_2_1_resid_post_mlp_acts_z_svd_fft = fft2(sec_2_1_resid_post_mlp_acts_z_svd)
sec_2_1_resid_post_mlp_acts_z_pca_fft = fft2(sec_2_1_resid_post_mlp_acts_z_pca)

fft_plot_eikx_2d(sec_2_1_resid_post_mlp_acts_z_svd_fft, nrows=10, title="Post MLP, SVD over heads, each term corresponds to e^(i2π/113 f_x x + f_y y + φ)")
fft_plot_cosplusminus(sec_2_1_resid_post_mlp_acts_z_svd_fft, title="Post MLP pattern, A_{+/-} cos(2π/113 f_x x +/- f_y y + φ_{+/-})", nrows=5)
fft_plot_coscos_sinsin(sec_2_1_resid_post_mlp_acts_z_svd_fft, title="Post MLP\n A_cos cos(2π/113 f_x x + φ_x) cos(2π/113 f_y y + φ_y) \n+ A_sin sin(2π/113 f_x x + φ_x) sin(2π/113 f_y y + φ_y)", nrows=5)

# %%

# RIB