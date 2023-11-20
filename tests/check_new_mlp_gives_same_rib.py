"""
Not currently a proper reproducible test file. Will need to be changed before merging into main.
Named check_ so pytest doesn't try to run it.
"""
# %%
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from experiments.mnist_ablations.plot_mnist_ablations import main as ablate_plot
from experiments.mnist_ablations.run_mnist_ablations import Config as AblateConfig
from experiments.mnist_ablations.run_mnist_ablations import main as ablate
from experiments.mnist_rib_build.plot_mnist_graph import main as rib_plot
from experiments.mnist_rib_build.run_mnist_rib_build import Config as RibConfig
from experiments.mnist_rib_build.run_mnist_rib_build import main as rib_build
from rib.models import MLP, MLPLayer
from rib.models.utils import save_model
from rib.utils import REPO_ROOT

# %%

# load weights from old model into new class

old_model_dir = "experiments/train_mnist/sample_checkpoints/lr-0.001_bs-64_2023-08-13_16-23-59/"
old_model_state_dict = torch.load(old_model_dir + "model_epoch_3.pt")
with open(old_model_dir + "config.yaml") as f:
    train_config = yaml.safe_load(f)

print(train_config)
new_model = MLP(
    hidden_sizes=train_config["model"]["hidden_sizes"],
    input_size=784,
    output_size=10,
    activation_fn=train_config["model"]["activation_fn"],
    bias=True,
    fold_bias=False,
)

for i, layer in enumerate(new_model.layers):
    # [out_features, (in_features + bias)]
    old_weight = old_model_state_dict[f"layers.{i}.linear.weight"]
    layer.W.data = old_weight[:, :-1].T
    layer.b.data = old_weight[:, -1]

new_model.cuda()


# check the model gets good accuracy
@torch.inference_mode()
def evaluate_model(model: MLP, device: str) -> float:
    # Load the MNIST train dataset
    transform = transforms.ToTensor()
    data = datasets.MNIST(root=REPO_ROOT / ".data", train=True, download=True, transform=transform)
    data_loader = DataLoader(data, batch_size=256, shuffle=True)

    # Test the model
    model.eval()
    correct = 0
    total = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


assert evaluate_model(new_model, "cuda") > 95


save_dir = "experiments/train_mnist/sample_checkpoints/old_reproduction/"
# save_model(train_config, Path(save_dir), new_model, 2)
# %%
with open("experiments/mnist_rib_build/4-node-layers.yaml") as f:
    rib_config = yaml.safe_load(f)
rib_config["exp_name"] = "new-for-repo"
rib_config["mlp_path"] = save_dir + "model_epoch_3.pt"
rib_build(RibConfig(**rib_config))

# %%
# ran rib_build on old model while on main
old_graph_path = "experiments/mnist_rib_build/out/old-for-repo_rib_graph.pt"
new_graph_path = "experiments/mnist_rib_build/out/new-for-repo_rib_graph.pt"

old = torch.load(old_graph_path)
new = torch.load(new_graph_path)

for layer in range(3):
    o, n = old["edges"][layer][1], new["edges"][layer][1]
    print(o.shape, n.shape)
    # print((o - n).abs().mean().item())


rib_plot(old_graph_path)
rib_plot(new_graph_path)


# %%
with open("experiments/mnist_ablations/rib_4-node-layers.yaml") as f:
    ablate_config = yaml.safe_load(f)

old_ablate_config = ablate_config.copy()
old_ablate_config["exp_name"] = "old-for-repo"
old_ablate_config["interaction_graph_path"] = old_graph_path
ablate(AblateConfig(**old_ablate_config))

new_ablate_config = ablate_config.copy()
new_ablate_config["exp_name"] = "new-for-repo"
new_ablate_config["interaction_graph_path"] = new_graph_path
ablate(AblateConfig(**new_ablate_config))

# %%

ablate_plot(
    "experiments/mnist_ablations/out/old-for-repo_ablation_results.json",
    "experiments/mnist_ablations/out/new-for-repo_ablation_results.json",
)
