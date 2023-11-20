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

# %%

activations = Activations()
activations.print_info()

# %%

# 113 113 1 47(from 129)
rib_acts_extended_embedding = activations.get_rib_activations(section="sections.section_0.2")[
    :, :, 0, :
]

# 113 113 1 81(642)
rib_acts_mlp_post = (
    activations.get_rib_activations(section="sections.section_1.2")[:, :, 0, :]
    .to("cpu")
    .to(torch.float64)
)

# %%

sec_0_2_resid_post_attn_acts = (
    activations.get_section_activations(section="sections.section_0.2")[0][:, :, 0, :]
    .to(torch.float64)
    .to("cpu")
)

sec_1_1_resid_pre_mlp_acts, sec_1_1_mlp_pre_acts = activations.get_section_activations(
    section="sections.section_1.1",
)
sec_1_1_mlp_pre_acts = sec_1_1_mlp_pre_acts[:, :, 0, :].to(torch.float64).to("cpu")

sec_1_2_resid_parallel_acts, sec_1_2_mlp_post_acts = activations.get_section_activations(
    section="sections.section_1.2",
)
sec_1_2_mlp_post_acts = sec_1_2_mlp_post_acts[:, :, 0, :].to(torch.float64).to("cpu")
sec_1_2_resid_parallel_acts = sec_1_2_resid_parallel_acts[:, :, 0, :].to(torch.float64).to("cpu")

sec_2_1_resid_post_mlp_acts = (
    activations.get_section_activations(section="sections.section_2.1")[0][:, :, 0, :]
    .to(torch.float64)
    .to("cpu")
)

# %%
from rib.models.utils import get_model_attr

model = activations.hooked_model.model
W_in = get_model_attr(model, "sections.section_1.1").W_in.to(torch.float64).to("cpu")
W_out = get_model_attr(model, "sections.section_2.0").W_out.to(torch.float64).to("cpu")

# %%


def rib_ln20_to_mlpout0(five, rib_ln20_acts=rib_acts_extended_embedding, dtype=torch.float64):
    # Take in RIB acts
    ln20_acts = rib_ln20_acts.to(dtype)
    ln20_acts[:, :, 5] = five
    # Convert to normal activations
    C, Cinv = activations.rib_basis_matrices[1]
    Cinv = Cinv.to("cpu")
    Cinv = Cinv.to(dtype)
    normal_acts = einops.einsum(ln20_acts, Cinv, "x y rib, rib embed -> x y embed")
    # TODO Is the inaccuracy due to truncation?
    assert five != 1 or torch.allclose(normal_acts, sec_0_2_resid_post_attn_acts, atol=0.01)
    # Apply W_in (could be skipped if we had chosed a nearer node_layer)
    mlp_pre_act = einops.einsum(normal_acts, W_in, "x y embed, embed mlp -> x y mlp")
    assert five != 1 or torch.allclose(mlp_pre_act, sec_1_1_mlp_pre_acts, atol=0.01)
    # Apply ReLU
    mlp_post_act = torch.relu(mlp_pre_act)
    assert five != 1 or torch.allclose(mlp_post_act, sec_1_2_mlp_post_acts, atol=0.01)

    # Convert back to RIB
    C, Cinv = activations.rib_basis_matrices[2]
    C = C.to("cpu").to(dtype)
    concat = torch.concat([sec_1_2_resid_parallel_acts, mlp_post_act], dim=-1)
    rib_acts_mlp_post_comparison = einops.einsum(concat, C, "x y mlp, mlp rib -> x y rib")
    assert five != 1 or torch.allclose(rib_acts_mlp_post, rib_acts_mlp_post_comparison, atol=0.01)
    return rib_acts_mlp_post_comparison, rib_acts_mlp_post
    # Also calculate resid_post for completeness
    resid_post = einops.einsum(mlp_post_act, W_out, "x y mlp, mlp embed -> x y embed")
    assert five != 1 or torch.allclose(
        resid_post + sec_1_2_resid_parallel_acts, sec_2_1_resid_post_mlp_acts, atol=0.01
    )

    # plt.figure()
    # scaled_out, orig_out = rib_ln20_to_mlpout0(1)
    # # plt.plot(scaled_out[0,0], color="k")
    # data_point = (0, 0)
    # plt.title(f"Scaling up input number 5, observing first 20 outputs at data point x,y={data_point}")
    # plt.ylabel("Absolute change in output")
    # plt.xticks(range(20))
    # cmap = plt.get_cmap("viridis")
    # for scale in np.geomspace(0.1, 50, 100):
    #     scaled_out, _ = rib_ln20_to_mlpout0(scale)
    plt.plot((scaled_out - orig_out)[data_point[0], data_point[1], :20], color=cmap(scale / 50))


# %%


class run_rib_model_scaling:
    def __init__(self, dtype=torch.float64):
        # 1 is ln2.0
        self.C_ell, self.Cinv_ell = activations.rib_basis_matrices[1]
        # self.C_ell = self.C_ell.to("cpu").to(dtype)
        self.Cinv_ell = self.Cinv_ell.to("cpu").to(dtype)
        self.C_ellp1, _ = activations.rib_basis_matrices[2]
        self.C_ellp1 = self.C_ellp1.to("cpu").to(dtype)

        self.in_acts = rib_acts_extended_embedding.to(dtype)
        self.parallel_acts = sec_1_2_resid_parallel_acts
        # self.parallel_acts = torch.unsqueeze(self.parallel_acts, 0)
        self.W_hat = einops.einsum(W_in, self.Cinv_ell, "embed mlp, rib embed -> rib mlp")

    def forward(self, scaling):
        scaled_in_acts = einops.einsum(self.in_acts, scaling, "x y rib, rib ... -> ... x y rib")
        scaled_parallel_acts = einops.einsum(
            scaled_in_acts, self.Cinv_ell, "... x y rib, rib emb -> ... x y emb"
        )
        # Fake
        scaled_parallel_acts = einops.einsum(
            self.parallel_acts,
            torch.ones([self.parallel_acts.shape[-1], scaling.shape[-1]]),
            "x y emb, emb ... -> ... x y emb",
        )
        pre_relu_acts = einops.einsum(
            scaled_in_acts, self.W_hat, "... x y rib, rib mlp -> ... x y mlp"
        )

        post_relu_acts = torch.relu(pre_relu_acts)

        post_relu_concat_acts = torch.concat([post_relu_acts, scaled_parallel_acts], dim=-1)

        post_relu_rib_acts = einops.einsum(
            post_relu_concat_acts, self.C_ellp1, "... x y mlp, mlp rib -> ... x y rib"
        )
        return post_relu_rib_acts


rib_model = run_rib_model_scaling()
rib_in_len = rib_acts_extended_embedding.shape[-1]
batch_size = 100
five_scaling = torch.ones([rib_in_len, batch_size])
five_scaling[5] = torch.linspace(0.3, 23, batch_size)

res_by_neuron = rib_model.forward(five_scaling)
res = res_by_neuron  # .sum(dim=-1)
# %%

plt.scatter(five_scaling[5], res[:, 1, 8, 2], marker=".", s=1)
plt.figure()
plt.scatter(five_scaling[5, 1:], np.diff(res[:, 1, 8, 2]), marker=".", s=1)

# %%
