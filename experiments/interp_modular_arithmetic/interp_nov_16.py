# %%
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

activations = Activations()
activations.print_info()

# %%
print([a[0].shape for a in activations.rib_basis_matrices])
# Plot the MLP hat matrices
# From section 1.0 input or 1.1-input (pre W_in) to section 2.0 output (post W_out)
# For RIB we can use 1.2 inputs and (1.1 outputs) and 1.2 outputs if we have them
# node layers (inputs): ['ln1.0', 'ln2.0', 'mlp_out.0', 'unembed', 'output']
# i.e. use ln2.0 and mlp_out.0, should be 0.2 and 1.2

# 113 113 1 47(from 129)
rib_acts_extended_embedding = activations.get_rib_activations(section="sections.section_0.2")[
    :, :, 0, :
]


# 113 113 1 81(642)
rib_acts_mlp_post = activations.get_rib_activations(section="sections.section_1.2")[:, :, 0, :]

# %%
