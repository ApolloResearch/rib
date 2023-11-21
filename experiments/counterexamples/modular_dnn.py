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
from rib.utils import REPO_ROOT, load_config, check_outfile_overwrite, set_seed
from dataclasses import dataclass, asdict
from rib.plotting import plot_interaction_graph
from rib.interaction_algos import InteractionRotation
#%%
def random_block_diagonal_matrix(n, k, dtype = torch.float32):
    """Generate a random block diagonal matrix of size n x n with two blocks of size k x k and n - k x n - k."""
    # Generate random matrix
    A = torch.randn(n, n, dtype=dtype)

    # Zero out the blocks
    A[0:k, k:n] = 0
    A[k:n, 0:k] = 0

    return A

class BlockDiagonalDNN(MLP):
    def __init__(self, layers = 4, n=4, k=2, dtype = torch.float32, bias = None, activation_fn = 'relu'):
        super(BlockDiagonalDNN, self).__init__(hidden_sizes = [n] * layers, 
                                       input_size = n, output_size = n,
                                       dtype = dtype, fold_bias = False,
                                       activation_fn=activation_fn)
        # self.dtype = dtype
        # # Define layers
        # self.fc = nn.ModuleList([nn.Linear(n, n,dtype=self.dtype) for _ in range(layers)])

        # Hardcode weights and biases
        for i in range(layers+1):
            self.layers[i].W = nn.Parameter(random_block_diagonal_matrix(n, k, dtype=dtype))
            if bias is not None:
                self.layers[i].b = nn.Parameter(bias * torch.ones(n,dtype=dtype))
        self.fold_bias()

# print(test.layers[0].W)
#%%
    

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

class RandomStrongCorrelatedDataset(Dataset):
    def __init__(self, n, k, size, dtype = torch.float32):
        """
        Args:
            n (int): Length of each vector.
            size (int): Total number of vectors in the dataset.
        """
        self.n = n
        self.size = size
        self.data = torch.zeros(size, n, dtype=dtype)
        self.data[:,0:k] = torch.randn(size, dtype=dtype).unsqueeze(-1)
        self.data[:,k:n] = torch.randn(size, dtype=dtype).unsqueeze(-1)
        self.labels = torch.randint(0, n, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Generates a random vector of length n.
        """
        vector = self.data[idx]
        return vector, self.labels[idx]
    
def get_node_layers(n_layers: int) -> list[str]:
    """Return the names of the layers that are nodes in the graph."""
    out = [f'layers.{i}' for i in range(n_layers+1)]

    out.append("output")
    return out

def cs_to_identity(Cs: list[InteractionRotation]):
    """Set all the Cs to identity matrices."""
    for C_info in Cs:
        print(C_info.C.shape, C_info.node_layer_name)
        if C_info.C.shape[0] != C_info.C.shape[1]:
            print("Warning: C is not square.")
            new_C = torch.eye(len(C_info.C), dtype=C_info.C.dtype, device=C_info.C.device)
            C_info.C = new_C[:C_info.C.shape[0],:C_info.C.shape[1]]
            C_info.C_pinv = new_C[:C_info.C_pinv.shape[0],:C_info.C_pinv.shape[1]] if C_info.C_pinv is not None else None
        else:
            C_info.C =  torch.eye(len(C_info.C), dtype=C_info.C.dtype, device=C_info.C.device)
            C_info.C_pinv = torch.eye(len(C_info.C_pinv), dtype=C_info.C_pinv.dtype, device=C_info.C_pinv.device) if C_info.C_pinv is not None else None
            
    return Cs

def cs_to_us(Cs: list[InteractionRotation], Us: list):
    """Replace Cs with Us, the matrices which diagonalise the gram matrix of functions in each layer."""
    for C_info, U_info in zip(Cs, Us):
        print(C_info.C.shape, C_info.node_layer_name)
        C_info.C = U_info.U
        C_info.C_pinv = U_info.U.T
    return Cs
#%%

class Config:
    def __init__(self,
                 exp_name: str = "small_modular_dnn",
                 n: int = 4,
                 k: int = 2,
                 layers: int = 3,
                 dataset_size: int = 128,
                 batch_size: int = 32,
                 seed: int = None,
                 truncation_threshold: float = 1e-20,
                 n_intervals: int = 0,
                 dtype: type = torch.float32,
                 node_layers: list = None,
                 datatype: str = 'random',
                 rotate_final_node_layer: bool = True,
                 force: bool = True,
                 hardcode_bias = None,
                 activation_fn = 'relu',
                #  basis: str = 'rib',
                 ):
        """
        Initializes the configuration for the experiment.

        :param exp_name: Name of the experiment.
        :param n: [Description of n]
        :param k: [Description of k]
        :param layers: Number of layers.
        :param dataset_size: Size of the dataset.
        :param batch_size: Batch size for training.
        :param seed: Seed for random number generators. 
                     A random seed is generated if None is provided.
        :param truncation_threshold: Remove eigenvectors with eigenvalues below this threshold.
        :param n_intervals: The number of intervals to use for integrated gradients.
        :param dtype: Data type of all tensors (except those overridden in certain functions).
        :param node_layers: The names of the layers that are nodes in the graph. 
                            Defaults are set based on 'layers' if None is provided.
        :param datatype: The type of data to use for the dataset.
        :param rotate_final_node_layer: Whether to rotate the final node layer.
        """
        self.exp_name = exp_name
        self.n = n
        self.k = k
        self.layers = layers
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed if seed is not None else np.random.randint(0, 1000)
        self.truncation_threshold = truncation_threshold
        self.n_intervals = n_intervals
        self.dtype = dtype
        self.node_layers = node_layers if node_layers is not None else get_node_layers(layers)
        self.datatype = datatype
        self.rotate_final_node_layer = rotate_final_node_layer
        self.force = force
        self.hardcode_bias = hardcode_bias
        self.activation_fn = activation_fn
        # self.basis = basis
    def to_dict(self):
        return vars(self)

def main(config: Config) -> None:
    """Implement the main algorithm and store the graph to disk."""
    if config.seed is None:
        config.seed = np.random.randint(0,1000)
        print(f"Seed not specified, using {config.seed}")
    set_seed(config.seed)
    exp_name = config.exp_name
    exp_name += f'_seed{config.seed}_n{config.n}_k{config.k}_layers{config.layers}_{config.datatype}_act_{config.activation_fn}_dsize_{config.dataset_size}'#_basis{config.basis}'
    if config.hardcode_bias is not None:
        exp_name += f'_bias{config.hardcode_bias}'
    assert config.datatype in ['strongcorrelated', 'random']
    # assert config.basis in ['rib', 'pca', 'neuron']

    out_dir = Path(__file__).parent / "results"
    out_file = out_dir / exp_name / "rib_graph.pt"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists() and not check_outfile_overwrite(out_file):
        logger.info("Exiting.")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = config.dtype
    mlp = BlockDiagonalDNN(config.layers, config.n,config.k, dtype = dtype, bias = config.hardcode_bias, activation_fn = config.activation_fn)
    mlp.eval()
    mlp.to(device=torch.device(device), dtype=config.dtype)
    hooked_mlp = HookedModel(mlp)
    if config.datatype == 'strongcorrelated':
        dataset = RandomStrongCorrelatedDataset(config.n, 
                                                config.k,
                                                config.dataset_size,
                                                dtype=config.dtype)
    else:
        dataset = RandomVectorDataset(config.n, config.dataset_size, dtype=config.dtype)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    node_layers = config.node_layers
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
    
    E_hats_rib = collect_interaction_edges(
        Cs=Cs,
        hooked_model=hooked_mlp,
        n_intervals=config.n_intervals,
        section_names=node_layers,
        data_loader=dataloader,
        dtype=dtype,
        device=device,
    )
    
    neuron_cs = cs_to_identity(Cs)
    E_hats_neuron = collect_interaction_edges(
        Cs=neuron_cs,
        hooked_model=hooked_mlp,
        n_intervals=config.n_intervals,
        section_names=node_layers,
        data_loader=dataloader,
        dtype=dtype,
        device=device,
    )
    pca_cs = cs_to_us(Cs, Us)
    E_hats_pca = collect_interaction_edges(
        Cs=pca_cs,
        hooked_model=hooked_mlp,
        n_intervals=config.n_intervals,
        section_names=node_layers,
        data_loader=dataloader,
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
        "exp_name": exp_name,
        "gram_matrices": {k: v.cpu() for k, v in gram_matrices.items()},
        "interaction_rotations": interaction_rotations,
        "eigenvectors": eigenvectors,
        "rib_edges": [(module, E_hats_rib[module].cpu()) for module in E_hats_rib],
        "neuron_edges": [(module, E_hats_neuron[module].cpu()) for module in E_hats_neuron],
        "pca_edges": [(module, E_hats_pca[module].cpu()) for module in E_hats_pca],
        "model_config_dict": config.to_dict(),
        "mlp": mlp.cpu().state_dict()
    }
    # Save the results (which include torch tensors) to file
    torch.save(results, out_file)
    logger.info("Saved results to %s", out_file)

    # Plot the graphs
    parent_dir = out_file.parent
    out_file_graph = parent_dir / "rib_graph.png"
    out_file_mlp = parent_dir / "mlp_graph.png"
    out_file_neuron_basis = parent_dir / "neuron_basis_graph.png"
    out_file_pca = parent_dir / "pca_graph.png"

    mlp = mlp.cpu()
    mlp_edges = []
    for name in results["model_config_dict"]["node_layers"][:-1]:
        num = int(name.split(".")[1])
        mlp_edges.append((name, mlp.layers[num].W.data.T))

    if not check_outfile_overwrite(out_file_graph, config.force):
        return

    nodes_per_layer = config.n + 1
    layer_names = results["model_config_dict"]["node_layers"] + ["output"]

    #make rib graph
    plot_interaction_graph(
        raw_edges=results["rib_edges"],
        layer_names=layer_names,
        exp_name=results["exp_name"],
        nodes_per_layer=nodes_per_layer,
        out_file=out_file_graph,
    )

    #make graph of model weights
    plot_interaction_graph(
        raw_edges=mlp_edges,
        layer_names=layer_names,
        exp_name=results["exp_name"],
        nodes_per_layer=nodes_per_layer,
        out_file=out_file_mlp,
    )
    
    #make pca graph
    plot_interaction_graph(
        raw_edges=results["pca_edges"],
        layer_names=layer_names,
        exp_name=results["exp_name"],
        nodes_per_layer=nodes_per_layer,
        out_file=out_file_pca,
    )
    
    #make neuron basis graph
    plot_interaction_graph(
        raw_edges=results["neuron_edges"],
        layer_names=layer_names,
        exp_name=results["exp_name"],
        nodes_per_layer=nodes_per_layer,
        out_file=out_file_neuron_basis,
    )
    


if __name__ == "__main__":
    fire.Fire(lambda **kwargs: main(Config(**kwargs)))


# %%
