from pathlib import Path
from typing import Literal, Optional

import einops
import numpy as np
import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from rib.ablations import load_basis_matrices
from rib.data import ModularArithmeticDatasetConfig
from rib.hook_manager import HookedModel
from rib.loader import create_data_loader, load_dataset, load_sequential_transformer
from rib.types import TORCH_DTYPES
from rib.utils import load_config


class ModelConfig(BaseModel):
    n_layers: int
    d_model: int
    d_head: int
    n_heads: int
    d_mlp: int
    d_vocab: int
    n_ctx: int
    act_fn: str
    normalization_type: Optional[str]


class Config(BaseModel):
    seed: int
    model: ModelConfig
    dataset: ModularArithmeticDatasetConfig


class Activations:
    """Class to get activations from a hooked model.

    Loads a model from a config file, and computes normal
    and RIB activations.
    """

    # Mapping from sections to between node_layers, the outputs of the sections are
    # the inputs to the node_layers
    node_layer_dict = {
        "sections.pre.2": "ln1.0",
        "sections.section_0.2": "ln2.0",
        "sections.section_1.2": "mlp_out.0",
        "sections.section_2.2": "unembed",
    }

    def __init__(
        self,
        config_path_str: str | Path = "mod_arithmetic_config.yaml",
        internal_device: Optional[str] = None,
        return_device: str = "cpu",
        dtype: Optional[str] = None,
        modulus: int = 113,
        tlens_model_path: str
        | Path = Path(
            "/mnt/ssd-apollo/checkpoints/rib/modular_arthimetic/lr-0.001_bs-10000_norm-None_2023-09-27_18-19-33/model_epoch_60000.pt"
        ),
        interaction_graph_path: str
        | Path = Path(
            "/mnt/ssd-apollo/stefan/rib/experiments/lm_rib_build/out/modular_arithmetic-fp64-alpha50-83fcf85_rib_graph.pt"
        ),
    ) -> None:
        """Initialize the Activations class.

        Args:
            config_path_str: The path to the config file. Defaults to "mod_arithmetic_config.yaml".
            internal_device: The device to use for internal computations such as model calls.
                Defaults to "cuda" if available, otherwise "cpu".
            return_device: The device to use for the return values. Defaults to "cpu".
            dtype: The dtype to use. Defaults to "float32".
            modulus: The mod add modulus (p) to use. Defaults to 113.
            tlens_model_path: The path to the TL model checkpoint to use. Defaults to
                "/mnt/ssd-apollo/checkpoints/rib/modular_arthimetic/lr-0.001_bs-10000_norm-None_2023-09-27_18-19-33/model_epoch_60000.pt".
            interaction_graph_path: The path to the interaction graph. Defaults to
                "/mnt/ssd-apollo/dan/RIB/rib/experiments/lm_rib_build/out/modular_arithmetic_interaction_graph.pt".

        """
        self.dtype = TORCH_DTYPES["float32"] if dtype is None else TORCH_DTYPES[dtype]
        self.p = modulus
        self.tlens_model_path = (
            tlens_model_path if isinstance(tlens_model_path, Path) else Path(tlens_model_path)
        )
        self.interaction_graph_path = interaction_graph_path
        self.internal_device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
            if internal_device is None
            else internal_device
        )
        self.return_device = return_device

        self.config_path = Path(config_path_str)
        self.config = load_config(self.config_path, config_model=Config)

        seq_model, tlens_cfg_dict = load_sequential_transformer(
            node_layers=["ln1.0", "ln2.0", "mlp_out.0", "unembed"],
            last_pos_module_type="add_resid1",  # module type in which to only output the last position index
            tlens_pretrained=None,
            tlens_model_path=self.tlens_model_path,
            eps=1e-5,
            dtype=self.dtype,
            device=self.internal_device,
        )

        seq_model.eval()
        seq_model.to(self.internal_device)
        seq_model.fold_bias()
        self.hooked_model = HookedModel(seq_model)

        self.datasets = load_dataset(
            dataset_config=self.config.dataset, return_set=self.config.dataset.return_set
        )
        self.train_loader: DataLoader
        self.test_loader: DataLoader
        self.train_loader, self.test_loader = create_data_loader(
            self.datasets, shuffle=True, batch_size=11, seed=0
        )

        self.interaction_graph_info = torch.load(self.interaction_graph_path)
        self.node_layers = self.interaction_graph_info["config"]["node_layers"]
        self.ablation_node_layers = [l for l in self.node_layers if l != "output"]

        self.rib_basis_matrices = load_basis_matrices(
            interaction_graph_info=self.interaction_graph_info,
            ablation_node_layers=self.ablation_node_layers,
            ablation_type="rib",
            dtype=self.dtype,
            device=self.internal_device,
        )
        self.orthog_basis_matrices = load_basis_matrices(
            interaction_graph_info=self.interaction_graph_info,
            ablation_node_layers=self.ablation_node_layers,
            ablation_type="orthogonal",
            dtype=self.dtype,
            device=self.internal_device,
        )

    def print_info(self):
        """Print some info about the model and dataset."""
        random_input, random_label = self.train_loader.dataset[np.random.randint(0, self.p)]
        print("Model", self.hooked_model)
        print("Input", random_input, "Label", random_label)
        assert random_input[0] + random_input[1] % self.p == random_label
        assert random_input[2] == self.p
        logits, cache = self.hooked_model.run_with_cache(random_input)
        logits = logits[0]
        assert logits.argmax(dim=-1).item() == random_label.item()
        for key, value in cache.items():
            for index, stream in enumerate(value["acts"]):
                print(key, index, cache[key]["acts"][index].shape)

    def _get_activation_shapes(self, section: str) -> list:
        """Get the shapes of activations in a section.

        Run the model once (dummy inputs, batch size 1) and return the shapes of all activations
        (there may be multiple because our modules have multiple inputs and outputs).

        Args:
            section: The section to get activation sizes for.

        Returns:
            A list of shapes of activations in the section, should be of the form (p, p, ...).
        """
        _, cache = self.hooked_model.run_with_cache(torch.tensor([[0, 0, 0]]))
        batch_index = 0
        acts = cache[section]["acts"]
        # acts is a n_acts-long list with elements of shape (batch_size, *shape)
        # where shape is e.g. [token, d_embed] for resid, or [n_head, query, key, d_head] for attn
        return [act[batch_index].shape for act in acts]

    def get_section_activations(
        self,
        section: str,
        batch_size: int = 32,
        sizes: Optional[list] = None,
        device: Optional[str] = None,
    ) -> tuple[torch.Tensor, ...]:
        """Get activations for a section from run_with_cache.

        Args:
            section: The section to get activations for.
            batch_size: The number of samples to process in a batch.
            sizes: The sizes of activations in the section. If None, this is determined automatically.
            device: The device to use for the activations. Defaults to self.return_device.

        Returns:
            A tensor (if concat) or tuple of tensors (if not concat) containing the activations
            of the section. The tensor(s) have shape (p, p, *act_shape) where p is the vocab size,
            and act_shape is the shape of the activations in the section, including
                token dimension but not batch dimension. Examples: Residual stream
                activations have shape (token, d_embed), attention activations have shape
                (n_head, query, key, d_head).
        """
        if sizes is None:
            sizes = self._get_activation_shapes(section=section)
            print("Determined sizes of activations as", sizes)

        device = self.return_device if device is None else device

        # Initialize the tensor to hold all activations
        return_acts = [torch.empty([self.p, self.p, *size]) for size in sizes]

        # Generatw combinations for x and y
        n_combinations = self.p * self.p
        combinations = [(x, y) for x in range(self.p) for y in range(self.p)]

        # Process combinations in batches
        with torch.no_grad():
            for i in tqdm(
                range(0, n_combinations, batch_size), desc=f"Getting activations for {section}"
            ):
                batch = combinations[i : i + batch_size]
                batch_input = torch.tensor([[x, y, self.p] for x, y in batch])

                # Run the batch through the model
                _, cache = self.hooked_model.run_with_cache(batch_input)
                acts = cache[section]["acts"]

                # Make sure the `sizes` argument matches the actual sizes
                assert len(acts) == len(sizes), f"len(sizes) mismatch {len(sizes)} != {len(acts)}"

                for batch_idx, (x, y) in enumerate(batch):
                    for i, act in enumerate(acts):
                        act_shape = act[batch_idx].shape
                        assert act_shape == sizes[i], f"{act_shape} != {sizes[i]}"
                        return_acts[i][x, y] = act[batch_idx]

        return tuple([r.to(device) for r in return_acts])

    def get_rib_activations(
        self,
        section: str,
        logits_node_layer: bool = False,
        ablation_type: Literal["rib", "orthogonal"] = "rib",
    ):
        """Collect RIB-transformed activations for a section.

        Obtains activations from cache and transforms them using the RIB basis matrices.

        Args:
            section: The section to get activations for.
            logits_node_layer: TODO. Defaults to False.
            ablation_type: The type of ablation to perform. Defaults to "rib".

        Returns:
            A tensor of shape (p, p, *act_shape) containing the transformed activations.
                Note that this is concatenated over n_acts by RIB. p is the vocab size,
                act_shape is the shape of the activations in the section, containing
                token dimension but not batch dimension. Examples: Activations during the MLP
                have the shape (token, d_embed + d_mlp), attention pattern activations have shape
                (n_head, query, key, d_head).
        """

        node_layer = self.node_layer_dict[section]
        node_index = np.where(node_layer == np.array(self.node_layers))[0][0]
        basis_matrices = (
            self.rib_basis_matrices if ablation_type == "rib" else self.orthog_basis_matrices
        )
        assert (
            len(basis_matrices[node_index]) == 2
        ), f"basis_matrices should contain C and C_inv matrices, but contains {len(basis_matrices[node_index])} elements"
        basis_matrices_index = 0  # get C rather than C_inv

        print(
            "Got rotation matrix of shape",
            basis_matrices[node_index][basis_matrices_index].shape,
        )

        acts = torch.cat(
            self.get_section_activations(section=section, device=self.internal_device), dim=-1
        )
        print("Got activations of shape", acts.shape)

        return einops.einsum(
            acts,
            basis_matrices[node_index][basis_matrices_index],
            "... hidden, hidden rib -> ... rib",
        ).to(self.return_device)

    def get_section_activations_unbatched(
        self,
        section: str,
        sizes: Optional[list] = None,
    ) -> torch.Tensor:
        """[Better use the batched version!] Get activations for a section from run_with_cache.

        Args:
            section: The section to get activations for.
            sizes: The sizes of activations in the section. If None, this is determined
                automatically.
        """
        if sizes is None:
            sizes = self._get_activation_shapes(section=section)
            print("Determined sizes of activations as", sizes)

        batch_index = 0

        return_acts = []
        for size in sizes:
            return_acts.append(torch.empty([self.p, self.p, *size]))

        with torch.no_grad():
            for x in tqdm(range(self.p), desc=f"Getting activations for {section}"):
                for y in range(self.p):
                    _, cache = self.hooked_model.run_with_cache(torch.tensor([[x, y, self.p]]))
                    acts = cache[section]["acts"]
                    # Make sure the `sizes` argument matches the actual sizes
                    assert len(acts) == len(
                        sizes
                    ), f"len(sizes) mismatch {len(sizes)} != {len(acts)}"
                    for i, act in enumerate(acts):
                        assert (
                            act[batch_index].shape == sizes[i]
                        ), f"{act[batch_index].shape} != {sizes[i]}"
                        return_acts[i][x, y] = act[batch_index]
        return torch.stack(return_acts, dim=0)  # (n_acts, p, p, *act_shape)
