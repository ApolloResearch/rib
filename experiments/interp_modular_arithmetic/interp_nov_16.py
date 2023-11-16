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

activations = Activations(
    config_path_str=parent_dir.joinpath("mod_arithmetic_config.yaml"),
    interaction_graph_path="/mnt/ssd-apollo/dan/RIB/rib/experiments/lm_rib_build/out/modular_arithmetic_rib_graph.pt",
    # interaction_graph_path="/mnt/ssd-apollo/stefan/rib/experiments/lm_rib_build/out/modular_arithmetic-nnA_rib_graph.pt",
)
activations.print_info()

# %%

activations.rib_basis_matrices
