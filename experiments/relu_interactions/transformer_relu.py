"""This script builds a RIB graph for a language model.
We build the graph using a SequentialTransformer model, with weights ported over from a
transformer-lens model.

Steps to build the graph:
1. Load a model from transformerlens (either from_pretrained or via ModelConfig).
2. Fold in the biases into the weights.
3. Convert the model to a SequentialTransformer model, which has nn.Modules corresponding to each
    node layer.
5. Collect the gram matrices at each node layer.
6. Calculate the interaction basis matrices (labelled C in the paper) for each node layer, starting
    from the final node layer and working backwards.
7. Calculate the edges of the interaction graph between each node layer.

Usage:
    python run_lm_rib_build.py <path/to/config.yaml>

The config.yaml should contain the `node_layers` field. This describes the sections of the
graph that will be built: A graph layer will be built on the inputs to each specified node layer,
as well as the output of the final node layer. For example, if `node_layers` is ["attn.0",
"mlp_act.0"], this script will build the following graph layers:
- One on the inputs to the "attn.0" node layer. This will include the residual stream concatenated
    with the output of ln1.0.
- One on the input to "mlp_act.0". This will include the residual stream concatenated with the
    output of "mlp_in.0".
- One on the output of "mlp_act.0". This will include the residual stream concatenated with the
    output of "mlp_act.0".
"""
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Literal, Optional, Union, cast

import fire
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from rib.data import HFDatasetConfig, ModularArithmeticDatasetConfig
from rib.data_accumulator import (
    calculate_all_swapped_iterative_relu_loss,
    calculate_swapped_relu_loss,
    collect_gram_matrices,
    collect_relu_interactions,
)
from rib.hook_manager import HookedModel
from rib.interaction_algos import (
    Eigenvectors,
    InteractionRotation,
    calculate_interaction_rotations,
)
from rib.loader import create_data_loader, load_dataset, load_sequential_transformer
from rib.log import logger
from rib.types import TORCH_DTYPES
from rib.utils import (
    eval_cross_entropy_loss,
    eval_model_accuracy,
    load_config,
    overwrite_output,
    set_seed,
)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exp_name: str = Field(..., description="The name of the experiment")
    seed: int = Field(..., description="The random seed value for reproducibility")
    tlens_pretrained: Optional[Literal["gpt2", "pythia-14m"]] = Field(
        None, description="Pretrained transformer lens model."
    )
    tlens_model_path: Optional[Path] = Field(
        None, description="Path to saved transformer lens model."
    )
    node_layers: list[str] = Field(
        ..., description="Names of the modules whose inputs correspond to node layers in the graph."
    )
    activation_layers: list[str] = Field(
        ..., description="Names of activation modules to hook ReLUs in."
    )
    logits_node_layer: bool = Field(
        ...,
        description="Whether to build an extra output node layer for the logits.",
    )
    rotate_final_node_layer: bool = Field(
        ...,
        description="Whether to rotate the final node layer to its eigenbasis or not.",
    )
    dataset: Union[ModularArithmeticDatasetConfig, HFDatasetConfig] = Field(
        ...,
        discriminator="source",
        description="The dataset to use to build the graph.",
    )
    batch_size: int = Field(..., description="The batch size to use when building the graph.")
    gram_batch_size: Optional[int] = Field(
        None,
        description="The batch size to use when calculating the gram matrices. If None, use the same"
        "batch size as the one used to build the graph.",
    )
    truncation_threshold: float = Field(
        ...,
        description="Remove eigenvectors with eigenvalues below this threshold.",
    )
    last_pos_module_type: Optional[Literal["add_resid1", "unembed"]] = Field(
        None,
        description="Module type in which to only output the last position index.",
    )

    n_intervals: int = Field(
        ...,
        description="The number of intervals to use for the integrated gradient approximation.",
    )

    dtype: str = Field(..., description="The dtype to use when building the graph.")

    eps: float = Field(
        1e-5,
        description="The epsilon value to use for numerical stability in layernorm layers.",
    )
    calculate_edges: bool = Field(
        True,
        description="Whether to calculate the edges of the interaction graph.",
    )
    eval_type: Optional[Literal["accuracy", "ce_loss"]] = Field(
        None,
        description="The type of evaluation to perform on the model before building the graph."
        "If None, skip evaluation.",
    )
    relu_metric_type: Literal[0, 1, 2, 3] = Field(
        None,
        description="Which similarity metric to use to calculate whether ReLUs are synced."
    )
    edit_weights: bool = Field(
        False,
        description="Whether to edit weights to push biases up so all ReLUs are synced - for debugging. Typically turned off."
    )

    @field_validator("dtype")
    def dtype_validator(cls, v):
        assert v in TORCH_DTYPES, f"dtype must be one of {TORCH_DTYPES}"
        return v

    @model_validator(mode="after")
    def verify_model_info(self) -> "Config":
        if sum(1 for val in [self.tlens_pretrained, self.tlens_model_path] if val is not None) != 1:
            raise ValueError(
                "Exactly one of [tlens_pretrained, tlens_model_path] must be specified"
            )
        return self


# Helper functions for main ========================================================

def print_all_modules(mlp):
    """Use for choosing which modules go in config file."""
    for name, module in mlp.named_modules():
        print(name, ":", module)


def relu_plotting(similarity_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]], out_dir: Path, config: Config) -> None:
    for i, similarity_matrix in enumerate(list(similarity_matrices.values())):
        # Plot raw matrix values before clustering
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, annot=False,
                    cmap="YlGnBu", cbar=True, square=True)
        plt.savefig(
            out_dir / f"mat_{i}_type_{config.relu_metric_type}_editweights_{config.edit_weights}.png")

        # Plot histogram
        plt.figure(figsize=(10,8))
        flat_similarity = np.array(similarity_matrix).flatten()
        sns.histplot(flat_similarity, bins=100, kde=False)
        plt.savefig(
            out_dir / f"hist_{i}_type_{config.relu_metric_type}_editweights_{config.edit_weights}.png")

        match config.relu_metric_type:
            case 0:
                # Threshold
                threshold = 0.95
                similarity_matrix[similarity_matrix < threshold] = 0
                distance_matrix = 1 - similarity_matrix
            case 1 | 2:
                distance_matrix = torch.max(similarity_matrix, similarity_matrix.T) # Make symmetric
                threshold = 15
                distance_matrix[distance_matrix > threshold] = threshold
            case 3:
                distance_matrix = torch.max(torch.abs(similarity_matrix), torch.abs(similarity_matrix).T)

        # Deal with zeros on diagonal
        distance_matrix = distance_matrix.fill_diagonal_(0)

        # Create linkage matrix using clustering algorithm
        linkage_matrix = linkage(squareform(
            distance_matrix), method='complete')
        order = leaves_list(linkage_matrix)
        rearranged_similarity_matrix = similarity_matrix[order, :][:, order]

        # Plot sorted similarity matrix via indices obtained from distance matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(rearranged_similarity_matrix, annot=False,
                    cmap="YlGnBu", cbar=True, square=True)
        plt.title("Reordered Similarity Matrix")
        plt.savefig(
            out_dir / f"rearr_mat_{i}_type_{config.relu_metric_type}_editweights_{config.edit_weights}.png")


def load_local_config(config_path_str: str) -> dict:
    """Load config (specifically, including MLP config) from local config file.

    Rest of RIB loads MLP config from FluidStack servers.
    """
    config_path = Path(__file__).parent / config_path_str
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_Cs(
    model: nn.Module,
    config: Config,
    file_path: str,
    dataset: Dataset,
    device: str,
    hooked_model: HookedModel = None,
) -> dict[str,
    Union[list[InteractionRotation],
    list[Eigenvectors],
    list[Float[Tensor, "d_hidden_trunc d_hidden_extra_trunc"]],
    list[Float[Tensor, "d_hidden_extra_trunc d_hidden_trunc"]]],
]:
    """Depending on which is easier, one of model or hooked_model are used.

    Hooked model should already be on device.
    """
    dtype = TORCH_DTYPES[config.dtype]
    return_set = cast(Literal["train", "test", "all"], config.dataset.return_set)
    dataset = load_dataset(
        dataset_config=config.dataset,
        return_set=return_set,
        tlens_model_path=config.tlens_model_path,
    )
    gram_train_loader = create_data_loader(
        dataset, shuffle=True, batch_size=config.gram_batch_size or config.batch_size
    )
    if config.eval_type is not None:
        # Test model accuracy/loss before graph building, to be sure
        if config.eval_type == "accuracy":
            accuracy = eval_model_accuracy(
                hooked_model, gram_train_loader, dtype=dtype, device=device
            )
            logger.info("Model accuracy on dataset: %.2f%%", accuracy * 100)
        elif config.eval_type == "ce_loss":
            loss = eval_cross_entropy_loss(
                hooked_model, gram_train_loader, dtype=dtype, device=device
            )
            logger.info("Model per-token loss on dataset: %.2f", loss)

    # Don't build the graph for the section of the model before the first node layer
    # The names below are *defined from* the node_layers dictionary in the config file
    graph_module_names = [f"sections.{sec}" for sec in model.sections if sec != "pre"]
    # Only need gram matrix for logits if we're rotating the final node layer
    collect_output_gram = config.logits_node_layer and config.rotate_final_node_layer

    logger.info("Collecting gram matrices for %d batches.", len(gram_train_loader))
    gram_matrices = collect_gram_matrices(
        hooked_model=hooked_model,
        module_names=graph_module_names,
        data_loader=gram_train_loader,
        dtype=dtype,
        device=device,
        collect_output_gram=collect_output_gram,
        hook_names=config.node_layers,
    )

    graph_train_loader = create_data_loader(dataset, shuffle=True, batch_size=config.batch_size)
    # Calls on collect_M_dash_and_Lambda_dash
    # Builds sqrt sorted Lambda matrix and its inverse
    Cs, Us, Lambda_abs_sqrts, Lambda_abs_sqrt_pinvs, U_D_sqrt_pinv_Vs, U_D_sqrt_Vs, Lambda_dashes = calculate_interaction_rotations(
        gram_matrices=gram_matrices,
        module_names=graph_module_names,
        hooked_model=hooked_model,
        data_loader=graph_train_loader,
        dtype=dtype,
        device=device,
        n_intervals=config.n_intervals,
        logits_node_layer=config.logits_node_layer,
        truncation_threshold=config.truncation_threshold,
        rotate_final_node_layer=config.rotate_final_node_layer,
        hook_names=config.node_layers,
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
    dataset: Dataset,
    device: str,
    hooked_model: HookedModel,
    Cs: list[Float[Tensor, "d_hidden1 d_hidden2"]],
    Lambda_dashes: list[Float[Tensor, "d_hidden d_hidden"]],
) -> dict[str, Float[Tensor, "d_hidden d_hidden"]]:
    return_set = cast(Literal["train", "test", "all"], config.dataset.return_set)
    dataset = load_dataset(
        dataset_config=config.dataset,
        return_set=return_set,
        tlens_model_path=config.tlens_model_path,
    )

    # # Weight editing stuff
    # print_all_modules(model) # Check module names were correctly defined
    # layer_list = ["layers.0.linear", "layers.1.linear"]
    # if config.edit_weights:
    #     for layer in layer_list:
    #         edit_weights_fn(model, layer)

    graph_train_loader = create_data_loader(dataset, shuffle=True, batch_size=config.batch_size)

    # Don't build the graph for the section of the model before the first node layer
    # The names below are *defined from* the node_layers dictionary in the config file
    graph_module_names = [f"sections.{sec}" for sec in model.sections if sec != "pre"]

    relu_similarity_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]] = collect_relu_interactions(
        hooked_model=hooked_model,
        module_names=config.activation_layers,
        data_loader=graph_train_loader,
        dtype=TORCH_DTYPES[config.dtype],
        device=device,
        relu_metric_type=config.relu_metric_type,
        Cs=Cs,
        Lambda_dashes=Lambda_dashes,
        layer_module_names=graph_module_names,
        n_intervals=config.n_intervals,
        unhooked_model=model,
    )

    with open(file_path, "wb") as f:
        torch.save(relu_similarity_matrices, f)

    return relu_similarity_matrices


def check_and_open_file(
    file_path: Path,
    get_var_fn: callable,
    model: nn.Module,
    config: Config,
    dataset: Dataset,
    device: str,
    hooked_model: HookedModel = None,
    **kwargs
) -> Union[Any, tuple[Any, ...]]:
    """Load information from pickle file into a variable and return it.

    Note the return type is overloaded to allow for tuples.
    """
    if file_path.exists():
        with file_path.open("rb") as f:
            var = torch.load(f, map_location=device)
    else:
        var = get_var_fn(model, config, file_path, dataset, device, hooked_model, **kwargs)

    return var


# ============================================================================

def transformer_relu_main(config_path_str: str):
    """Build the interaction graph and store it on disk."""
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)
    set_seed(config.seed)

    relu_matrices_save_file = Path(__file__).parent / f"transformer_{config.relu_metric_type}_edit_weights_{config.edit_weights}"
    Cs_save_file = Path(__file__).parent / "Cs"

    out_dir = Path(__file__).parent / f"out_transformer_relu / type_{config.relu_metric_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = TORCH_DTYPES[config.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seq_model, tlens_cfg_dict = load_sequential_transformer(
        node_layers=config.node_layers,
        last_pos_module_type=config.last_pos_module_type,
        tlens_pretrained=config.tlens_pretrained,
        tlens_model_path=config.tlens_model_path,
        eps=config.eps,
        dtype=dtype,
        device=device,
    )
    seq_model.eval()
    seq_model.to(device=torch.device(device), dtype=dtype)
    seq_model.fold_bias()

    print_all_modules(seq_model) # Check module names were correctly defined

    hooked_model = HookedModel(seq_model)

    # This script doesn't need both train and test sets
    return_set = cast(Literal["train", "test", "all"], config.dataset.return_set)
    dataset = load_dataset(
        dataset_config=config.dataset,
        return_set=return_set,
        tlens_model_path=config.tlens_model_path,
    )

    # dict of InteractionRotation objects or Tensors
    Cs_and_Lambdas: dict[str, list[Union[InteractionRotation, Float[Tensor, ...]]]] = check_and_open_file(
        file_path=Cs_save_file,
        get_var_fn=get_Cs,
        model=seq_model,
        config=config,
        dataset=dataset,
        device=device,
        hooked_model=hooked_model,
    )

    relu_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]] = check_and_open_file(
        file_path=relu_matrices_save_file,
        get_var_fn=get_relu_similarities,
        model=seq_model,
        config=config,
        dataset=dataset,
        device=device,
        hooked_model=hooked_model,
        Cs=Cs_and_Lambdas["C"],
        Lambda_dashes=Cs_and_Lambdas["Lambda_dashes"],
    )

    relu_plotting(relu_matrices, out_dir, config)


if __name__ == "__main__":
    fire.Fire(transformer_relu_main)
