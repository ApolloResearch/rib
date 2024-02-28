# %%
import numpy as np
import torch

result = torch.load(
    "/mnt/ssd-interp/stefan/rib/rib_scripts/rib_build/out/block_diagonal_nnib_10k_v2_cross_rib_graph.pt"
)

C = result["interaction_rotations"][0]["C"]
# Shjape: orig, rib
import matplotlib.pyplot as plt

plt.matshow(np.log(np.abs(C)))

# %%
# 7, 1

# 7 10 12 13 14 1... 18 20
plt.figure(figsize=(10, 5))
plt.title("RIB directions in layer 0")
plt.plot(C[:, 7], label="RIB dim 7", color="C0")
plt.plot(C[:, 10], label="RIB dim 10", color="C0")
plt.plot(C[:, 12], label="RIB dim 12", color="C0")
plt.plot(C[:, 13], label="RIB dim 13", color="C0")
plt.plot(C[:, 14], label="RIB dim 14", color="C0")
plt.plot(C[:, 15], label="RIB dim 15", color="C0")
plt.plot(C[:, 16], label="RIB dim 16", color="C0")
plt.plot(C[:, 17], label="RIB dim 17", color="C0")
plt.plot(C[:, 18], label="RIB dim 18", color="C0")
plt.plot(C[:, 20], label="RIB dim 20", color="C0")

# others, tuquoise
plt.plot(C[:, 1], label="RIB dim 1", color="turquoise")
plt.plot(C[:, 2], label="RIB dim 2", color="turquoise")
plt.plot(C[:, 3], label="RIB dim 3", color="turquoise")
plt.plot(C[:, 4], label="RIB dim 4", color="turquoise")
plt.plot(C[:, 5], label="RIB dim 5", color="turquoise")
plt.plot(C[:, 6], label="RIB dim 6", color="turquoise")
plt.plot(C[:, 8], label="RIB dim 8", color="turquoise")
plt.plot(C[:, 9], label="RIB dim 9", color="turquoise")
plt.plot(C[:, 11], label="RIB dim 11", color="turquoise")
plt.plot(C[:, 19], label="RIB dim 19", color="turquoise")

plt.legend()


plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Neuron basis")
# %%
