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

rib_acts_embedding = activations.get_rib_activations(section="sections.pre.2")
rib_acts_extended_embedding = activations.get_rib_activations(section="sections.section_0.2")[:,:,0,:]
# Note: mlp_post and pre_umembed are basically the same
rib_acts_mlp_post = activations.get_rib_activations(section="sections.section_1.2")[:,:,0,:]
rib_acts_pre_unembed = activations.get_rib_activations(section="sections.section_2.2")[:,:,0,:]

# Orthog should be the same as SVD
orthog_acts_extended_embedding = activations.get_rib_activations(
    section="sections.section_0.2", ablation_type="orthogonal"
)
orthog_acts_mlp_post = activations.get_rib_activations(
    section="sections.section_1.2", ablation_type="orthogonal"
)
orthog_acts_pre_unembed = activations.get_rib_activations(
    section="sections.section_2.2", ablation_type="orthogonal"
)
orthog_acts_embedding = activations.get_rib_activations(
    section="sections.pre.2", ablation_type="orthogonal"
)

# Embeddings have token dimension, split up.
rib_acts_embedding_x, rib_acts_embedding_y, rib_acts_embedding_z = einops.rearrange(
    rib_acts_embedding, "x y p h -> p x y h"
)
orthog_acts_embedding_x, orthog_acts_embedding_y, orthog_acts_embedding_z = einops.rearrange(
    orthog_acts_embedding, "x y p h -> p x y h"
)



# %%

# Embeddings

rib_acts_embedding_x_fft = fft2(rib_acts_embedding_x)

fft_plot_cos_phase_1d(rib_acts_embedding_x_fft[:, 0, :2], rtol=1e-2, title="RIB direction 0 and 1")
fft_plot_cos_phase_1d(rib_acts_embedding_x_fft[:, 0, 2:4], rtol=1e-2, title="RIB direction 2 and 3")
fft_plot_cos_phase_1d(rib_acts_embedding_x_fft[:, 0, 4:6], rtol=1e-2, title="RIB direction 4 and 5")
fft_plot_cos_phase_1d(rib_acts_embedding_x_fft[:, 0, 6:8], rtol=1e-2, title="RIB direction 6 and 7")
fft_plot_cos_phase_1d(rib_acts_embedding_x_fft[:, 0, 8:10], rtol=1e-2, title="RIB direction 8 and 9")


# %%

# Resid Mid

rib_acts_extended_embedding_fft = fft2(rib_acts_extended_embedding)

fft_plot_eikx_2d(rib_acts_extended_embedding_fft, nrows=10, title="Resid Mid RIB each term corresponds to e^(i2π/113 f_x x + f_y y + φ)")
fft_plot_cosplusminus(rib_acts_extended_embedding_fft, title="Resid Mid RIB, A_{+/-} cos(2π/113 f_x x +/- f_y y + φ_{+/-})", nrows=5, equalize=True)
fft_plot_coscos_sinsin(rib_acts_extended_embedding_fft, title="Resid mid RIB\n A_cos cos(2π/113 f_x x + φ_x) cos(2π/113 f_y y + φ_y) \n+ A_sin sin(2π/113 f_x x + φ_x) sin(2π/113 f_y y + φ_y)", nrows=5, equalize=True)
# %%

# Post MLP

rib_acts_mlp_post_fft = fft2(rib_acts_mlp_post)

fft_plot_eikx_2d(rib_acts_mlp_post_fft, nrows=10, title="Post MLP RIB, each term corresponds to e^(i2π/113 f_x x + f_y y + φ)")
fft_plot_cosplusminus(rib_acts_mlp_post_fft, title="Post MLP RIB, A_{+/-} cos(2π/113 f_x x +/- f_y y + φ_{+/-})", nrows=5, equalize=True)
fft_plot_coscos_sinsin(rib_acts_mlp_post_fft, title="Post MLP RIB\n A_cos cos(2π/113 f_x x + φ_x) cos(2π/113 f_y y + φ_y) \n+ A_sin sin(2π/113 f_x x + φ_x) sin(2π/113 f_y y + φ_y)", nrows=5, equalize=True)

# %%

# RIB