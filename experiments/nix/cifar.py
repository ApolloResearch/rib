# %%
from pathlib import Path

import torch
import yaml
from torch import nn

from experiments.mnist_ablations.plot_mnist_ablations import main as plot_ablations
from experiments.mnist_ablations.run_mnist_ablations import Config as AblationConfig
from experiments.mnist_ablations.run_mnist_ablations import main as ablation_main
from experiments.mnist_rib_build.plot_mnist_graph import main as plot_graph
from experiments.mnist_rib_build.run_mnist_rib_build import Config as RibConfig
from experiments.mnist_rib_build.run_mnist_rib_build import main as rib_main
from rib.data import VisionDatasetConfig
from rib.loader import create_data_loader, load_dataset, load_mlp

# %%

path = Path(
    "/mnt/ssd-apollo/nix/rib/experiments/train_mnist/.checkpoints/mnist/lr-0.001_bs-64_2023-11-24_16-17-40/model_epoch_12.pt"
)
with open(path.parent / "config.yaml", "r") as f:
    model_config_dict = yaml.safe_load(f)

mlp = load_mlp(model_config_dict, path, device="cuda")
mlp.to("cuda")

# %%

test_dataset = load_dataset(VisionDatasetConfig(**model_config_dict["data"]), "test")
test_loader = create_data_loader(test_dataset, shuffle=True, batch_size=10_000, seed=0)


class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

criterion = nn.CrossEntropyLoss(reduction="none")


def examine_loss():
    for images, labels in test_loader:
        images, labels = images.to("cuda"), labels.to("cuda")

        outputs = mlp(images)
        loss = criterion(outputs, labels)
        print(f"test set loss: {loss.mean().item():.4f} \n")

        print("test set performance:")
        for i, name in enumerate(class_names):
            class_loss = loss[labels == i].mean().item()
            print(f"{name}: {class_loss:.4f}")


examine_loss()
# %%
# run rib on the model

config_str = """
exp_name: cifar-4-node-layers
mlp_path: experiments/train_mnist/.checkpoints/mnist/lr-0.001_bs-64_2023-11-24_16-17-40/model_epoch_12.pt
batch_size: 10000
seed: 0
truncation_threshold: 1e-6
rotate_final_node_layer: false
n_intervals: 0
dtype: float32
node_layers:
  - layers.0
  - layers.1
  - layers.2
  - output
dataset:
  name: CIFAR10
  return_set_frac: null
  seed: 0
"""

rib_results_path = Path("experiments/mnist_rib_build/out/cifar-4-node-layers_rib_results.json")

if rib_results_path.exists():
    print("rib results already exist")
    rib_results = torch.load(rib_results_path)
else:
    config = RibConfig(**yaml.safe_load(config_str))
    rib_results = rib_main(config)

# %%

plot_graph("experiments/mnist_rib_build/out/cifar-4-node-layers_rib_graph.pt")
# %%

ablation_config_str = """
exp_name: cifar-rib_4-node-layers
ablation_type: rib
interaction_graph_path: experiments/mnist_rib_build/out/cifar-4-node-layers_rib_graph.pt
schedule:
  schedule_type: exponential
  early_stopping_threshold: null
  ablate_every_vec_cutoff: 10
  exp_base: 2.0
dtype: float32
ablation_node_layers:  # Rotate input to these modules
  - layers.0
  - layers.1
  - layers.2
batch_size: 2048
dataset:
  name: CIFAR10
  return_set_frac: null
  seed: 0
seed: 0
"""

ablation_config = AblationConfig(**yaml.safe_load(ablation_config_str))
ablation_results = ablation_main(ablation_config)
# %%

plot_ablations("experiments/mnist_ablations/out/cifar-rib_4-node-layers_ablation_results.json")
# %%
