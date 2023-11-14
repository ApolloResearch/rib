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
    interaction_graph_path="/mnt/ssd-apollo/dan/RIB/rib/experiments/lm_rib_build/out/modular_arithmetic_rib_graph.pt",
    # interaction_graph_path="/mnt/ssd-apollo/stefan/rib/experiments/lm_rib_build/out/modular_arithmetic-nnA_rib_graph.pt",
)
activations.print_info()

rib_acts_embedding = activations.get_rib_activations(section="sections.pre.2")
rib_acts_embedding_fft = fft2(rib_acts_embedding)


# plot_fft_activations(rib_acts_embedding_fft[:, :, 0, :])


# %%
def plot_fft_1d(acts, n_rows=15, figsize=(16, 40), axes=None, **kwargs):
    if axes is None:
        fig, axes = plt.subplots(n_rows, 2, constrained_layout=True, figsize=figsize)
    else:
        fig = axes[0, 0].get_figure()
    fig.suptitle("FFT of RIB activations, X | Y")
    for i in range(n_rows):
        axes[i, 0].set_ylabel(f"RIB direction {i}")
        axes[i, 0].plot(acts[:, :, 0, i].abs().mean(dim=1), **kwargs)
        axes[i, 1].plot(acts[:, :, 1, i].abs().mean(dim=0), **kwargs)
    return fig, axes


fig, axes = plot_fft_1d(rib_acts_embedding_fft)


# %%

activations_nnA = Activations(
    config_path_str=parent_dir.joinpath("mod_arithmetic_config.yaml"),
    interaction_graph_path="/mnt/ssd-apollo/stefan/rib/experiments/lm_rib_build/out/modular_arithmetic-nnA_rib_graph.pt",
)

rib_acts_embedding_nnA_fft = fft2(activations_nnA.get_rib_activations(section="sections.pre.2"))
# fig, axes = plot_fft_1d(rib_acts_embedding_nnA_fft, axes=axes, ls=":")

# %%

activations_nnB = Activations(
    config_path_str=parent_dir.joinpath("mod_arithmetic_config.yaml"),
    interaction_graph_path="/mnt/ssd-apollo/stefan/mod_add_graph_nnB.pt",
)

rib_acts_embedding_nnB_fft = fft2(activations_nnB.get_rib_activations(section="sections.pre.2"))
fig, axes = plot_fft_1d(rib_acts_embedding_nnB_fft, axes=axes, ls="--")

# %%
fig
