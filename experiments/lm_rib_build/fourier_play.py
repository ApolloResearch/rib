# This data was calculated by running the following code at a breakpoint inside
# M_dash_and_Lambda_dash_pre_forward_hook_fn after the f_hat is calculated and when
# if isinstance(module[0], MLPIn):
#     print("INSIDE seciton 2")

# import numpy as np

# DEVICE = "cuda"
# p = 113
# fourier_basis = []
# fourier_basis.append(torch.ones(p) / np.sqrt(p))
# fourier_basis_names = ["Const"]
# # Note that if p is even, we need to explicitly add a term for cos(kpi), ie
# # alternating +1 and -1
# for i in range(1, p // 2 + 1):
#     fourier_basis.append(torch.cos(2 * torch.pi * torch.arange(p) * i / p))
#     fourier_basis.append(torch.sin(2 * torch.pi * torch.arange(p) * i / p))
#     fourier_basis[-2] /= fourier_basis[-2].norm()
#     fourier_basis[-1] /= fourier_basis[-1].norm()
#     fourier_basis_names.append(f"cos {i}")
#     fourier_basis_names.append(f"sin {i}")
# fourier_basis = torch.stack(fourier_basis, dim=0).to(DEVICE)


# f_hat_trunc = f_hat[:, -1, :]
# plot_data = []
# for hidden_idx in range(f_hat_trunc.shape[-1]):
#     f_hat_trunc_idx = f_hat_trunc[:, hidden_idx].view(113, 113)
#     transformed = fourier_basis @ f_hat_trunc_idx @ fourier_basis.T
#     plot_data.append(transformed)

# # Save plot_data
# with open("/mnt/ssd-apollo/dan/RIB/fourier_data.pt", "wb") as f:
#     torch.save(plot_data, f)

import matplotlib.pyplot as plt

# %%
import torch

plot_data = torch.load("/mnt/ssd-apollo/dan/RIB/fourier_data.pt")
plot_data = [x.detach().cpu() for x in plot_data]

# %%
vmax = plot_data[0].abs().max()
print(vmax)
plt.imshow(plot_data[0], cmap="RdBu", vmax=1, vmin=-1)
# %%
N = 5
p = 113
fig, ax = plt.subplots(N, N, figsize=(12, 12))

for i, hidden_idx_data in enumerate(plot_data):
    v_max = hidden_idx_data.abs().max()
    row = i % N
    col = i // N
    if i == N**2:
        break
    ax[row][col].imshow(hidden_idx_data.numpy(), cmap="RdBu", vmax=v_max, vmin=-v_max)

fig.suptitle("TODO")
plt.show()
# %%
