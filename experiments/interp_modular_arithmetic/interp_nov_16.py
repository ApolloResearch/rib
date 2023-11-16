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
rib_acts_mlp_post = (
    activations.get_rib_activations(section="sections.section_1.2")[:, :, 0, :]
    .to("cpu")
    .to(torch.float64)
)

# %%
from rib.models.utils import get_model_attr

model = activations.hooked_model.model
W_in = get_model_attr(model, "sections.section_1.1").W_in.to(torch.float64).to("cpu")
W_out = get_model_attr(model, "sections.section_2.0").W_out.to(torch.float64).to("cpu")
# %%
# ln2.0 node 5 to mlp_out.0 2 seems single strong connection --> plot

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


def rib_ln20_to_mlpout0(five, rib_ln20_acts=rib_acts_extended_embedding, dtype=torch.float64):
    # Take in RIB acts
    ln20_acts = rib_ln20_acts.to(dtype)
    ln20_acts[:, :, 5] *= five
    # Convert to normal activations
    C, Cinv = activations.rib_basis_matrices[1]
    Cinv = Cinv.to("cpu")
    Cinv = Cinv.to(dtype)
    normal_acts = einops.einsum(ln20_acts, Cinv, "x y rib, rib embed -> x y embed")
    # TODO Is the inaccuracy due to truncation?
    assert torch.allclose(normal_acts, sec_0_2_resid_post_attn_acts, atol=0.01)
    # Apply W_in (could be skipped if we had chosed a nearer node_layer)
    mlp_pre_act = einops.einsum(normal_acts, W_in, "x y embed, embed mlp -> x y mlp")
    assert torch.allclose(mlp_pre_act, sec_1_1_mlp_pre_acts, atol=0.01)
    # Apply ReLU
    mlp_post_act = torch.relu(mlp_pre_act)
    assert torch.allclose(mlp_post_act, sec_1_2_mlp_post_acts, atol=0.01)

    # Convert back to RIB
    C, Cinv = activations.rib_basis_matrices[2]
    C = C.to("cpu").to(dtype)
    concat = torch.concat([sec_1_2_resid_parallel_acts, mlp_post_act], dim=-1)
    rib_acts_mlp_post_comparison = einops.einsum(concat, C, "x y mlp, mlp rib -> x y rib")
    assert torch.allclose(rib_acts_mlp_post, rib_acts_mlp_post_comparison, atol=0.01)

    # Also calculate resid_post for completeness
    resid_post = einops.einsum(mlp_post_act, W_out, "x y mlp, mlp embed -> x y embed")
    assert torch.allclose(
        resid_post + sec_1_2_resid_parallel_acts, sec_2_1_resid_post_mlp_acts, atol=0.01
    )


# %%

plt.figure()
plt.hist((activations.rib_basis_matrices[1][1].cpu().to(torch.float64) @ W_in).flatten(), bins=100)
plt.semilogy()
plt.xlabel(r"$C^{\ell_1} W_{\rm in}$")

plt.figure()
plt.hist(activations.rib_basis_matrices[2][0].cpu().flatten(), bins=100)
plt.semilogy()
plt.xlabel(r"$C^{\ell_2}$")

# %%

plt.figure()
plt.hist(
    (activations.rib_basis_matrices[1][1].cpu().to(torch.float64) @ W_in[:, :-1]).flatten(),
    bins=100,
)
plt.semilogy()
plt.xlabel(r"$C^{\ell_1} W_{\rm in}$ (without bias)")

plt.figure()
plt.hist((W_in[:, :-1]).flatten(), bins=100)
plt.semilogy()
plt.xlabel(r"$W_{\rm in}$")

plt.figure()
plt.hist((activations.rib_basis_matrices[1][1].cpu().to(torch.float64)).flatten(), bins=100)
plt.semilogy()
plt.xlabel(r"$C^{\ell_1}$")

# %%

np.where((activations.rib_basis_matrices[1][1].cpu().to(torch.float64) @ W_in).abs() > 0.1)
# Seem to come mostly from RIB entry 13
# (array([13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
#         13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 16]),
#  array([ 15,  24,  49,  52,  80,  87,  92, 120, 127, 136, 180, 194, 213,
#         226, 230, 241, 270, 273, 293, 294, 299, 317, 334, 389, 391, 409,
#         419, 442, 466, 503, 512]))

# %%

# Now let's do the naive multiplication assuming ReLU == 1
C_l1_inv = activations.rib_basis_matrices[1][1].cpu().to(torch.float64)
Wtilde = C_l1_inv @ W_in
C_l2 = activations.rib_basis_matrices[2][0].cpu().to(torch.float64)
naive_mlp = Wtilde @ C_l2[:-129, :]

plt.figure()
plt.hist(naive_mlp.flatten(), bins=100)
plt.semilogy()
plt.xlabel(r"$\tilde{W}_{\rm in} C^{\ell_2}$")

np.where(naive_mlp.abs() > 0.3)

plt.figure()
plt.imshow(naive_mlp)
plt.xlabel("RIB out")
plt.ylabel("RIB in")
plt.colorbar()
# plt.xlim(0, 15)
# plt.ylim(00, 40)
# %%

# %%

plt.hist(activations.rib_basis_matrices[1][1].flatten().cpu(), bins=100)

# neuron in RIBin  -> direction in resid_mid -> direction in ReLU-in
# neuron in RIBout -> direction in ReLU-out
# 2(give) = sum C_i ReLU_i preact_i = sum_ij C_i ReLU_i W_in_i5 RIBin_5 + other RIBs

# Plot C_i entries (the non MLP one but ln2)


normal_acts = rib_ln20_to_mlpout0(1)
# sec_0_2_resid_post_attn_acts = sec_0_2_resid_post_attn_acts.to(torch.float64)

# plt.plot(normal_acts[34][67])
# plt.plot(sec_0_2_resid_post_attn_acts[34][67], ls=":")


# %%
