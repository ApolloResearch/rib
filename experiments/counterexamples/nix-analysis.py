# %%
import os

from rib.data_accumulator import collect_gram_matrices, collect_interaction_edges
from rib.hook_manager import HookedModel
from rib.interaction_algos import calculate_interaction_rotations
from rib.models import MLP, MLPLayer
from rib.plotting import plot_interaction_graph

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch.utils.data import DataLoader, TensorDataset

# %%

N_SAMPLES = 100_000
DEVICE = "cpu"
DTYPE = torch.float32
IN_SIZE = 2


def get_model():
    model = MLP([2, 2], input_size=IN_SIZE, output_size=2, bias=True, activation_fn="identity")

    for l in model.layers:
        l: MLPLayer
        l.W.data = torch.where(
            torch.arange(l.in_features)[:, None] == torch.arange(l.in_features)[None, :],
            torch.rand_like(l.W.diagonal()) * 0.2 + 0.9,
            torch.zeros_like(l.W.data),
        )
        l.b.data.zero_()
        print("", l.W, l.b, sep="\n")

    model.fold_bias()
    print("folded")
    for l in model.layers:
        print("", l.W, l.b, sep="\n")

    return model


model = get_model()
hooked_model = HookedModel(model)

# %%
rand_vectors = torch.randn((N_SAMPLES, IN_SIZE))
dataset = TensorDataset(rand_vectors, torch.zeros(N_SAMPLES))
dataloader = DataLoader(dataset, batch_size=1_000)
node_layers = [f"layers.{i}" for i in range(len(model.layers))] + ["output"]

caches = []
for X, y in dataloader:
    out, cache = hooked_model.run_with_cache(X)
    caches.append(cache)

# %%

gram_matrices = collect_gram_matrices(
    hooked_model=hooked_model,
    module_names=node_layers[:-1],
    data_loader=dataloader,
    dtype=DTYPE,
    device=DEVICE,
    collect_output_gram=True,
)

for l in node_layers:
    print(l, "\n", gram_matrices[l])
# %%
Cs, Us = calculate_interaction_rotations(
    gram_matrices=gram_matrices,
    section_names=node_layers[:-1],
    node_layers=node_layers,
    hooked_model=hooked_model,
    data_loader=dataloader,
    dtype=DTYPE,
    device=DEVICE,
    n_intervals=0,
    truncation_threshold=1e-6,
    rotate_final_node_layer=False,
)

for u_info in Us:
    print(u_info.node_layer_name, "\n", u_info.U)


for c_info in Cs:
    #  [orig_acts, out_acts]
    # or new basis in columns
    # so x @ C is the rotated activations
    print(c_info.node_layer_name, c_info.C, sep="\n")

# %%
E_hats = collect_interaction_edges(
    Cs=Cs,
    hooked_model=hooked_model,
    n_intervals=0,
    section_names=node_layers[:-1],
    data_loader=dataloader,
    dtype=DTYPE,
    device=DEVICE,
)

# %%

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(10, 3))

for layer in range(3):
    ax = axs[layer]
    acts = caches[0][f"layers.{layer}{'.activation' if layer < 2 else ''}"]["acts"][0][:, :2]
    ax.scatter(acts[:, 0], acts[:, 1], alpha=0.5, color="k", s=2)

    c_info = Cs[layer]
    assert c_info.node_layer_name == f"layers.{layer}"
    C = c_info.C
    for i, color in zip([0, 1], ["mediumblue", "deeppink"]):
        dx, dy = C[[0, 1], i]
        ax.axline((0, 0), (dx, dy), color=color)
    extent = -3, 5
    ax.set_xlim(extent)
    ax.set_ylim(extent)
    ax.set_aspect("equal")

    ax.set_title(f"layer {layer}")

# %%


fig, axs = plt.subplots(1, 3, figsize=(9, 3))

for layer in range(3):
    edges = E_hats[f"layers.{layer}"]  # [out, in]
    axs[layer].imshow(edges, cmap="Blues", vmin=0, vmax=10)
    axs[layer].set_title(f"layer {layer}")
    axs[layer].set_ylabel(f"input (layer {layer})")
    axs[layer].set_xlabel(f"output (layer {layer +1})")
plt.suptitle("edges")
plt.tight_layout()

# %%

plot_interaction_graph(
    E_hats.items(),
    node_layers,
    "idenity",
    nodes_per_layer=2,
    out_file="experiments/counterexamples/out/identity.png",
    colors=[["mediumblue", "deeppink"]] * 4,
)

# %%
