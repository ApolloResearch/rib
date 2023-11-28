"""
Analysis for pythia edges.
Change the number in the results file to get a different set of edges.
0 = edges across ln1 (pre-attention)
1 = edges across attn
2 = edges across ln2 (pre-mlp)
3 = edges across mlp
"""

# %%
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from fancy_einsum import einsum
from tqdm import tqdm

from rib.data import HFDatasetConfig
from rib.data_accumulator import run_dataset_through_model
from rib.hook_fns import rotate_pre_forward_hook_fn
from rib.hook_manager import Hook, HookedModel
from rib.loader import create_data_loader, load_dataset, load_sequential_transformer

# %%


results_file = (
    "/mnt/ssd-apollo/nix/rib/experiments/lm_rib_build/pythia_edges/out/edges_3_rib_graph.pt"
)

results = torch.load(results_file)
print("keys:")
for k in results.keys():
    print("\t", k)
print("config")
rib_config = results["config"]
for k, v in rib_config.items():
    print("\t", k.ljust(30), v)

input_mname, output_mname = rib_config["node_layers"]

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


# load pythia
model, _ = load_sequential_transformer(
    node_layers=rib_config["node_layers"],
    last_pos_module_type=rib_config["last_pos_module_type"],
    tlens_pretrained=rib_config["tlens_pretrained"],
    tlens_model_path=rib_config["tlens_pretrained"],
    eps=None,
    fold_bias=True,
    dtype=torch.float64,
    device="cuda",
)
model.to("cuda")
hooked_model = HookedModel(model)


# %%
dataset_config = HFDatasetConfig(**rib_config["dataset"])
dataset_config.return_set_frac = 0.01
dataset = load_dataset(
    dataset_config=dataset_config,
    return_set="train",
    tlens_model_path=None,
)
print("total length:", len(dataset))
data_loader = create_data_loader(dataset, shuffle=False, batch_size=32, seed=0)

# %%


def get_c(module_id):
    c_info = [d for d in results["interaction_rotations"] if d["node_layer_name"] == module_id][0]
    return c_info["C"]


hooks = [
    Hook(
        name="rotated_acts",
        data_key=module_id,
        fn=rotate_pre_forward_hook_fn,
        module_name=model.module_id_to_section_id[module_id],
        fn_kwargs={"rotation_matrix": get_c(module_id).to("cuda"), "output_rotated": False},
    )
    for module_id in [input_mname, output_mname]
]

run_dataset_through_model(
    hooked_model, data_loader, hooks, dtype=torch.float64, device="cuda", use_tqdm=True
)

rib_acts = {
    module_id: torch.concatenate(hooked_model.hooked_data["rotated_acts"][module_id], dim=0)
    for module_id in [input_mname, output_mname]
}

# %%

fig, axs = plt.subplots(4, 4, figsize=(10, 10))
for i in range(16):
    ax = axs[i // 4, i % 4]
    acts = rib_acts[input_mname][:, :, i].flatten()
    ax.hist(acts, bins=100)


# %%
MAX_BATCHES = 10
model_ids_of_interest = [input_mname, output_mname]

with torch.inference_mode():
    acts = defaultdict(list)
    for batch in tqdm(data_loader):
        data, _ = batch
        out, cache = hooked_model.run_with_cache(data)
        for model_id in model_ids_of_interest:
            x = cache[model.module_id_to_section_id[model_id]]["acts"][0].cpu()
            acts[model_id].append(x)

    acts = {k: torch.concat(v, axis=0) for k, v in acts.items()}


# %%
acts_in_rib = {
    c_info["node_layer_name"]: einsum(
        "orig rib, ... orig -> ... rib", c_info["C"], acts[c_info["node_layer_name"]]
    )
    for c_info in results["interaction_rotations"]
    if c_info["node_layer_name"] in acts
}

acts_in_rib
# TODO: input / output mismatch??
# %%

acts_in_rib = {
    c_info["node_layer_name"]: (c_info["C"].shape, acts[c_info["node_layer_name"]].shape)
    for c_info in results["interaction_rotations"]
    if c_info["node_layer_name"] in acts
}

# %%
