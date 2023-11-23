from pathlib import Path
from typing import Any, Union

import fire
import torch
import torch.nn as nn
import yaml
from jaxtyping import Float
from pydantic import BaseModel, field_validator
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from experiments.relu_interactions.relu_interaction_utils import (
    edit_weights_fn,
    relu_plot_and_cluster,
    swap_all_layers_using_clusters,
)
from rib.data_accumulator import (
    collect_gram_matrices,
    collect_relu_interactions,
)
from rib.hook_manager import HookedModel
from rib.interaction_algos import (
    Eigenvectors,
    InteractionRotation,
    calculate_interaction_rotations,
)
from rib.loader import load_mlp
from rib.models import MLP
from rib.types import TORCH_DTYPES
from rib.utils import REPO_ROOT, load_config, set_seed


class Config(BaseModel):
    exp_name: str
    mlp_path: Path
    batch_size: int
    seed: int
    truncation_threshold: float  # Remove eigenvectors with eigenvalues below this threshold.
    rotate_final_node_layer: bool  # Whether to rotate the output layer to its eigenbasis.
    n_intervals: int  # The number of intervals to use for integrated gradients.
    dtype: str  # Data type of all tensors (except those overriden in certain functions).
    activation_layers: list[str]
    node_layers: list[str]
    relu_metric_type: int
    edit_weights: bool
    threshold: float # For dendrogram distance cutting with fcluster to make clusters
    use_residual_stream: bool

    @field_validator("dtype")
    def dtype_validator(cls, v):
        assert v in TORCH_DTYPES, f"dtype must be one of {TORCH_DTYPES}"
        return v


def load_mnist_dataloader(train: bool = False, batch_size: int = 64) -> DataLoader:
    transform = transforms.ToTensor()
    test_data = datasets.MNIST(
        root=REPO_ROOT / ".data", train=train, download=True, transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return test_loader


# Helper functions for main ========================================================

def get_Cs(
    model: nn.Module,
    config: Config,
    file_path: str,
) -> dict[str,
    Union[list[InteractionRotation],
    list[Eigenvectors],
    list[Float[Tensor, "d_hidden_trunc d_hidden_extra_trunc"]],
    list[Float[Tensor, "d_hidden_extra_trunc d_hidden_trunc"]]],
]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device=torch.device(device), dtype=TORCH_DTYPES[config.dtype])
    hooked_model = HookedModel(model)
    assert model.has_folded_bias

    train_loader = load_mnist_dataloader(train=True, batch_size=config.batch_size)

    non_output_node_layers = [layer for layer in config.node_layers if layer != "output"]
    # Only need gram matrix for logits if we're rotating the final node layer
    collect_output_gram = config.node_layers[-1] == "output" and config.rotate_final_node_layer

    gram_matrices = collect_gram_matrices(
        hooked_model=hooked_model,
        module_names=non_output_node_layers,
        data_loader=train_loader,
        dtype=TORCH_DTYPES[config.dtype],
        device=device,
        collect_output_gram=collect_output_gram,
    )

    # Calls on collect_M_dash_and_Lambda_dash
    # Builds sqrt sorted Lambda matrix and its inverse
    Cs, Us, Lambda_abs_sqrts, Lambda_abs_sqrt_pinvs, U_D_sqrt_pinv_Vs, U_D_sqrt_Vs, Lambda_dashes = calculate_interaction_rotations(
        gram_matrices=gram_matrices,
        section_names=non_output_node_layers,
        node_layers=config.node_layers,
        hooked_model=hooked_model,
        data_loader=train_loader,
        dtype=TORCH_DTYPES[config.dtype],
        device=device,
        n_intervals=config.n_intervals,
        truncation_threshold=config.truncation_threshold,
        rotate_final_node_layer=config.rotate_final_node_layer,
    )

    C_list = [C_info.C for C_info in Cs]
    C_pinv_list = [C_info.C_pinv for C_info in Cs]

    with open(file_path, "wb") as f:
        torch.save({"C": C_list, "C_pinv": C_pinv_list, "Lambda_abs_sqrts": Lambda_abs_sqrts, "Lambda_abs_sqrt_pinvs": Lambda_abs_sqrt_pinvs, "U_D_sqrt_pinv_Vs": U_D_sqrt_pinv_Vs, "U_D_sqrt_Vs": U_D_sqrt_Vs, "Lambda_dashes": Lambda_dashes, "Cs raw": Cs, "Us raw": Us, "gram matrices": gram_matrices,}, f)

    return {"C": C_list, "C_pinv": C_pinv_list, "Lambda_abs_sqrts": Lambda_abs_sqrts, "Lambda_abs_sqrt_pinvs": Lambda_abs_sqrt_pinvs, "U_D_sqrt_pinv_Vs": U_D_sqrt_pinv_Vs, "U_D_sqrt_Vs": U_D_sqrt_Vs, "Lambda_dashes": Lambda_dashes, "Cs raw": Cs, "Us raw": Us, "gram matrices": gram_matrices}


def get_relu_similarities(
    model: nn.Module,
    config: Config,
    file_path: Path,
    Cs_list: list[Float[Tensor, "d_hidden1 d_hidden2"]],
) -> dict[str, Float[Tensor, "d_hidden d_hidden"]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer_list = ["layers.0.linear", "layers.1.linear"]
    model.to(device=torch.device(device), dtype=TORCH_DTYPES[config.dtype])
    hooked_mlp = HookedModel(model)

    train_loader = load_mnist_dataloader(
        train=True, batch_size=config.batch_size)

    relu_similarity_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]] = collect_relu_interactions(
        hooked_model=hooked_mlp,
        module_names=config.activation_layers,
        data_loader=train_loader,
        dtype=TORCH_DTYPES[config.dtype],
        device=device,
        relu_metric_type=config.relu_metric_type,
        Cs_list=Cs_list,
        layer_module_names=config.node_layers,
        n_intervals=config.n_intervals,
        unhooked_model=model,
        use_residual_stream=config.use_residual_stream,
        is_lm=False
    )

    with open(file_path, "wb") as f:
        torch.save(relu_similarity_matrices, f)

    return relu_similarity_matrices


def check_and_open_file(
    file_path: Path,
    get_var_fn: callable,
    model: nn.Module,
    config: Config,
    device: str,
    **kwargs
) -> Union[Any, tuple[Any, ...]]:
    """Load information from pickle file into a variable and return it.

    Note the return type is overloaded to allow for tuples.
    """
    if file_path.exists():
        with file_path.open("rb") as f:
            var = torch.load(f, map_location=device)
    else:
        var = get_var_fn(model, config, file_path, **kwargs)

    return var

# Main ========================================================

def mlp_relu_main(config_path_str: str) -> None:
    """MAIN FUNCTION 1. Test for ReLU interactions (separate to main RIB algorithm)."""
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)
    set_seed(config.seed)

    relu_matrices_save_file = Path(__file__).parent / f"similarity_metric_{config.relu_metric_type}_mlp"
    Cs_save_file = Path(__file__).parent / "Cs_mlp"

    out_dir = Path(__file__).parent / f"out_mlp_relu/type_{config.relu_metric_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config.mlp_path.parent / "config.yaml", "r") as f:
        model_config_dict = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_mlp(model_config_dict, config.mlp_path, device=device)
    model.to(device=torch.device(device), dtype=TORCH_DTYPES[config.dtype])
    hooked_model = HookedModel(model)
    # print_all_modules(model) # Check module names were correctly defined
    model.eval()  # Run in inference only

    # dict of InteractionRotation objects or Tensors
    Cs_and_Lambdas: dict[str, list[Union[InteractionRotation, Float[Tensor, ...]]]] = check_and_open_file(
        file_path=Cs_save_file,
        get_var_fn=get_Cs,
        config=config,
        model=model,
        device=device,
    )

    relu_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]] = check_and_open_file(
        file_path=relu_matrices_save_file,
        get_var_fn=get_relu_similarities,
        config=config,
        model=model,
        device=device,
        Cs_list=Cs_and_Lambdas["C"],
    )

    replacement_idxs_from_cluster, num_valid_swaps_from_cluster, all_cluster_idxs = relu_plot_and_cluster(relu_matrices, out_dir, config)

    swap_all_layers_using_clusters(
        replacement_idxs_from_cluster=replacement_idxs_from_cluster,
        num_valid_swaps_from_cluster=num_valid_swaps_from_cluster,
        hooked_model=hooked_model,
        config=config,
        data_loader=load_mnist_dataloader(train=True, batch_size=config.batch_size),
        device=device,
    )


if __name__ == "__main__":
    fire.Fire(mlp_relu_main)
    """Run above: python run_relu_interactions.py relu_interactions.yaml
    Check ReLU metric used and whether weights are edited in yaml file
    """