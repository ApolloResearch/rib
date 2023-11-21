"""
Analysis for pythia edges.
Change the number in the results file to get a different set of edges.
0 = edges across ln1 (pre-attention)
1 = edges across attn
2 = edges across ln2 (pre-mlp)
3 = edges across mlp
"""

# %%
import matplotlib.pyplot as plt
import torch

# %%


results_file = (
    "/mnt/ssd-apollo/nix/rib/experiments/lm_rib_build/pythia_edges/out/edges_4_rib_graph.pt"
)

results = torch.load(results_file)
print("keys:")
for k in results.keys():
    print("\t", k)
print("config")
for k, v in results["config"].items():
    print("\t", k.ljust(30), v)

input_mname, output_mname = results["config"]["node_layers"]

# %%

edge_name, edges = results["edges"][0]
edges.shape  # [out, in]

# %%
extent = None
data = edges[:extent, :extent].T
vmax = data.abs().max()
plt.matshow(data, cmap="PiYG", norm="symlog", vmin=-vmax, vmax=vmax)

plt.ylabel(f"input ({input_mname})")
plt.xlabel(f"output ({output_mname})")
plt.colorbar()
plt.gcf().set_size_inches(10, 10)

# %%
extent = 50
data = edges[:extent, :extent].T
vmax = 1  # data.abs().max()
plt.matshow(data, cmap="PiYG", vmin=-1, vmax=1)

plt.ylabel(f"input ({input_mname})")
plt.xlabel(f"output ({output_mname})")
plt.colorbar()

# %%

extent = 80
data = edges[:extent, :extent].T
vmax = 1  # data.abs().max()
plt.matshow(data.abs().log(), cmap="Blues", vmin=-8, vmax=0)

plt.ylabel(f"input ({input_mname})")
plt.xlabel(f"output ({output_mname})")
plt.colorbar()

# %%

plt.semilogy(edges.sum(0))
plt.ylabel("sum of edge values")
plt.xlabel(f"node in {input_mname}")

# %%
## investigating a weird column in the ln1 edges
# plt.plot(edges[128], "-o", ms=3)
# plt.gcf().set_size_inches(8, 4)

# attn_rots_dict = [
#     d for d in results["interaction_rotations"] if d["node_layer_name"] == "attn_in.5"
# ][0]
# C = attn_rots_dict["C"]
# C.shape  # [full_dim, rotated_dim]
# C[128]
# %%
