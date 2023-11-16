#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pydantic import BaseModel, field_validator
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import json
from dataclasses import asdict
from pathlib import Path

import fire
import yaml

from rib.data_accumulator import collect_gram_matrices, collect_interaction_edges
from rib.hook_manager import HookedModel
from rib.interaction_algos import calculate_interaction_rotations
from rib.log import logger
from rib.models import MLP
from rib.types import TORCH_DTYPES
from rib.utils import REPO_ROOT, load_config, overwrite_output, set_seed
from dataclasses import dataclass, asdict
#%%
class SmallDNN(nn.Module):
    def __init__(self, layers = 4, n=4, k=2, dtype = torch.float32):
        super(SmallDNN, self).__init__()
        self.dtype = dtype
        # Define layers
        self.fc = nn.ModuleList([nn.Linear(n, n,dtype=self.dtype) for _ in range(layers)])

        # Hardcode weights and biases
        for i in range(layers):
            self.fc[i].weight = nn.Parameter(random_block_diagonal_matrix(n, k, dtype=self.dtype))
            self.fc[i].bias = nn.Parameter(torch.randn(n,dtype=self.dtype))

    def forward(self, x):
        for i in range(len(self.fc)-1):
            x = F.relu(self.fc[i](x))
        return F.softmax(self.fc[-1](x),-1)
    
def random_block_diagonal_matrix(n, k, dtype = torch.float32):
    """Generate a random block diagonal matrix of size n x n with two blocks of size k x k and n - k x n - k."""
    # Generate random matrix
    A = torch.randn(n, n, dtype=dtype)

    # Zero out the blocks
    A[0:k, k:n] = 0
    A[k:n, 0:k] = 0

    return A

class RandomVectorDataset(Dataset):
    def __init__(self, n, size, dtype = torch.float32):
        """
        Args:
            n (int): Length of each vector.
            size (int): Total number of vectors in the dataset.
        """
        self.n = n
        self.size = size
        self.data = torch.randn(size, n, dtype=dtype)
        self.labels = torch.randint(0, n, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Generates a random vector of length n.
        """
        vector = self.data[idx]
        return vector, self.labels[idx]

def node_layers(hookedmodel: HookedModel) -> list[str]:
    """Return the names of the layers that are nodes in the graph."""
    out = [name for name, _ in hookedmodel.model.fc.named_modules() if name != ""]
    out[-1] = "output"
    return out

@dataclass
class Config:
    exp_name: str
    n: int
    k: int
    layers: int
    dataset_size: int
    batch_size: int
    seed: int
    truncation_threshold: float # Remove eigenvectors with eigenvalues below this threshold.
    n_intervals: float # The number of intervals to use for integrated gradients.
    dtype: type # Data type of all tensors (except those overriden in certain functions).
    rotate_final_node_layer: bool = True # Whether to rotate the final node layer.
#%%
mlp = SmallDNN(4,4,2)
dataset = RandomVectorDataset(4,128)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
criterion = nn.CrossEntropyLoss()
for batch, labels in train_loader:
    print(criterion(mlp(batch),labels))

mlphooked = HookedModel(mlp)
print(mlphooked)


node_layers(mlphooked)


#%%
config = Config(
    exp_name="small_modular_dnn",
    n=4,
    k=2,
    layers=3,
    dataset_size=128,
    batch_size=32,
    seed=0,
    truncation_threshold=1e-3,
    n_intervals=50,
    dtype=torch.float32
)

def main(config: Config) -> None:
    """Implement the main algorithm and store the graph to disk."""
    set_seed(config.seed)

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{config.exp_name}_rib_graph.pt"
    if out_file.exists() and not overwrite_output(out_file):
        logger.info("Exiting.")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = TORCH_DTYPES[config.dtype]
    mlp = SmallDNN(config.layers, config.n,config.k)
    mlp.eval()
    mlp.to(device=torch.device(device), dtype=config.dtype)
    hooked_mlp = HookedModel(mlp)

    dataset = RandomVectorDataset(config.n, config.dataset_size, dtype=config.dtype)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    node_layers = node_layers(hooked_mlp)
    non_output_node_layers = [layer for layer in node_layers if layer != "output"]
    # Only need gram matrix for logits if we're rotating the final node layer
    collect_output_gram = node_layers[-1] == "output" and config.rotate_final_node_layer

    gram_matrices = collect_gram_matrices(
        hooked_model=hooked_mlp,
        module_names=non_output_node_layers,
        data_loader=dataloader,
        dtype=dtype,
        device=device,
        collect_output_gram=collect_output_gram,
    )

    Cs, Us = calculate_interaction_rotations(
        gram_matrices=gram_matrices,
        section_names=non_output_node_layers,
        node_layers=node_layers,
        hooked_model=hooked_mlp,
        data_loader=dataloader,
        dtype=dtype,
        device=device,
        n_intervals=config.n_intervals,
        truncation_threshold=config.truncation_threshold,
        rotate_final_node_layer=config.rotate_final_node_layer,
    )

    E_hats = collect_interaction_edges(
        Cs=Cs,
        hooked_model=hooked_mlp,
        n_intervals=config.n_intervals,
        section_names=node_layers,
        data_loader=train_loader,
        dtype=dtype,
        device=device,
    )

    # Move interaction matrices to the cpu and store in dict
    interaction_rotations = []
    for C_info in Cs:
        info_dict = asdict(C_info)
        info_dict["C"] = info_dict["C"].cpu() if info_dict["C"] is not None else None
        info_dict["C_pinv"] = info_dict["C_pinv"].cpu() if info_dict["C_pinv"] is not None else None
        interaction_rotations.append(info_dict)

    eigenvectors = [asdict(U_info) for U_info in Us]

    results = {
        "exp_name": config.exp_name,
        "gram_matrices": {k: v.cpu() for k, v in gram_matrices.items()},
        "interaction_rotations": interaction_rotations,
        "eigenvectors": eigenvectors,
        "edges": [(module, E_hats[module].cpu()) for module in E_hats],
        "model_config_dict": asdict(config),
    }

    # Save the results (which include torch tensors) to file
    torch.save(results, out_file)
    logger.info("Saved results to %s", out_file)


if __name__ == "__main__":
    fire.Fire(main)


# # Parameters
# n = 4  # length of the vector, same as used in SmallDNN
# dataset_size = 1000  # total number of vectors in the dataset

# # Create the dataset
# dataset = RandomVectorDataset(n, dataset_size)

# # Create the DataLoader
# batch_size = 32  # Define your batch size
# shuffle = True   # Shuffle the data every epoch
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# # Example of using the dataloader
# # Create the network
# k = n//2
# net = SmallDNN()

# %%
