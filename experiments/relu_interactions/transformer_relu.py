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
import re
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Union, cast

import fire
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import Tensor
from torch.utils.data import Dataset

from experiments.relu_interactions.relu_interaction_utils import (
    print_all_modules,
    relu_plot_and_cluster,
    swap_all_layers_using_clusters,
    plot_eigenvalues,
    plot_eigenvectors,
)
from rib.data import HFDatasetConfig, ModularArithmeticDatasetConfig
from rib.data_accumulator import (
    collect_clustered_relu_P_mats,
    collect_clustered_relu_P_mats_no_W,
    collect_gram_matrices,
    collect_relu_interactions,
    collect_test_edges,
    collect_cluster_grams,
)
from rib.hook_manager import HookedModel
from rib.interaction_algos import (
    Eigenvectors,
    InteractionRotation,
    calculate_interaction_rotations,
)
from rib.linalg import eigendecompose
from rib.loader import create_data_loader, load_dataset, load_sequential_transformer
from rib.log import logger
from rib.models.utils import get_model_attr, get_model_weight
from rib.types import TORCH_DTYPES
from rib.utils import load_config, set_seed


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exp_name: str = Field(..., description="The name of the experiment")
    seed: int = Field(...,
                      description="The random seed value for reproducibility")
    tlens_pretrained: Optional[Literal["gpt2", "pythia-14m"]] = Field(
        None, description="Pretrained transformer lens model."
    )
    tlens_model_path: Optional[Path] = Field(
        None, description="Path to saved transformer lens model."
    )
    interaction_matrices_path: Optional[Path] = Field(
        None, description="Path to pre-saved interaction matrices. If provided, we don't recompute."
    )
    node_layers: list[str] = Field(
        ..., description="Names of the modules whose inputs correspond to node layers in the graph."
    )
    activation_layers: list[str] = Field(
        ..., description="Names of activation modules to hook ReLUs in."
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
    batch_size: int = Field(...,
                            description="The batch size to use when building the graph.")
    gram_batch_size: Optional[int] = Field(
        None,
        description="The batch size to use when calculating the gram matrices. If None, use the same"
        "batch size as the one used to build the graph.",
    )
    edge_batch_size: Optional[int] = Field(
        None,
        description="The batch size to use when calculating the edges. If None, use the same batch"
        "size as the one used to build the graph.",
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
    out_dim_chunk_size: Optional[int] = Field(
        None,
        description="The size of the chunks to use for calculating the jacobian. If none, calculate"
        "the jacobian on all output dimensions at once.",
    )
    dtype: str = Field(...,
                       description="The dtype to use when building the graph.")
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
    threshold: float = Field(
        None,
        description="Dendrogram distance cutting with fcluster to make clusters"
    )
    relu_metric_type: Literal[0, 1, 2, 3] = Field(
        None,
        description="Which similarity metric to use to calculate whether ReLUs are synced."
    )
    edit_weights: bool = Field(
        False,
        description="Whether to edit weights to push biases up so all ReLUs are synced - for debugging. Typically turned off."
    )
    use_residual_stream: bool = Field(
        False,
        description="Whether to count residual stream in ReLU clustering alongside the MLP neurons."
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


def _verify_compatible_configs(config: Config, loaded_config: Config) -> None:
    """Ensure that the config for calculating edges is compatible with that used to calculate Cs."""

    assert config.node_layers == loaded_config.node_layers[-len(config.node_layers):], (
        "node_layers in the config must be a subsequence of the node layers in the config used to"
        "calculate the C matrices, ending at the final node layer. Otherwise, the C matrices won't"
        "match those needed to correctly calculate the edges."
    )

    # The following attributes must exactly match across configs
    for attr in [
        "tlens_model_path",
        "tlens_pretrained",
    ]:
        assert getattr(config, attr) == getattr(loaded_config, attr), (
            f"{attr} in config ({getattr(config, attr)}) does not match "
            f"{attr} in loaded matrices ({getattr(loaded_config, attr)})"
        )

    # Verify that, for huggingface datasets, we're not trying to calculate edges on data that
    # wasn't used to calculate the Cs
    assert config.dataset.name == loaded_config.dataset.name, "Dataset names must match"
    assert config.dataset.return_set == loaded_config.dataset.return_set, "Return sets must match"
    if isinstance(config.dataset, HFDatasetConfig):
        assert isinstance(loaded_config.dataset, HFDatasetConfig)
        if config.dataset.return_set_frac is not None:
            assert loaded_config.dataset.return_set_frac is not None
            assert (
                config.dataset.return_set_frac <= loaded_config.dataset.return_set_frac
            ), "Cannot use a larger return_set_frac for edges than to calculate the Cs"
        elif config.dataset.return_set_n_samples is not None:
            assert loaded_config.dataset.return_set_n_samples is not None
            assert (
                config.dataset.return_set_n_samples <= loaded_config.dataset.return_set_n_samples
            ), "Cannot use a larger return_set_n_samples for edges than to calculate the Cs"


def load_interaction_rotations(
    config: Config,
) -> tuple[
    dict[str, Float[Tensor, "d_hidden d_hidden"]
         ], list[InteractionRotation], list[Eigenvectors]
]:
    logger.info("Loading pre-saved C matrices from %s",
                config.interaction_matrices_path)
    assert config.interaction_matrices_path is not None
    matrices_info = torch.load(config.interaction_matrices_path)

    loaded_config = Config(**matrices_info["config"])
    _verify_compatible_configs(config, loaded_config)

    gram_matrices = matrices_info["gram_matrices"]
    Cs = [InteractionRotation(**data)
          for data in matrices_info["interaction_rotations"]]
    Us = [Eigenvectors(**data) for data in matrices_info["eigenvectors"]]
    return gram_matrices, Cs, Us


# Helper functions for main ========================================================


def plot_and_save_Ps(
    tensor_dict: dict[str, dict[Int[Tensor, "cluster_size"],
                                Float[Tensor, "d_hidden_next_layer d_hidden"]]],
    out_dir: Path
) -> None:
    for outer_key, inner_dict in tensor_dict.items():
        for i, (inner_key, tensor) in enumerate(inner_dict.items()):
            tensor = tensor.detach().cpu()
            plt.figure()
            ax = plt.gca()
            plt.imshow(tensor, cmap='viridis')

            # Adjust subplot parameters to create space for annotation
            plt.subplots_adjust(top=0.85)
            ax.text(0.5, 1.05, f'Inner Key: {inner_key}', transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=8, bbox=dict(boxstyle='round,pad=0.3'))

            plt.colorbar()
            file_name = out_dir / f"{outer_key}_cluster_{i}.png"
            plt.savefig(file_name, bbox_inches='tight')
            plt.close()


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
    section_names = [f"sections.{sec}" for sec in model.sections if sec != "pre"]

    if config.interaction_matrices_path is None:
        # Only need gram matrix for output if we're rotating the final node layer
        collect_output_gram = config.node_layers[-1] == "output" and config.rotate_final_node_layer

        gram_train_loader = create_data_loader(
            dataset,
            shuffle=False,
            batch_size=config.gram_batch_size or config.batch_size,
            seed=config.seed,
        )

        logger.info("Collecting gram matrices for %d batches.", len(gram_train_loader))
        gram_matrices = collect_gram_matrices(
            hooked_model=hooked_model,
            module_names=section_names,
            data_loader=gram_train_loader,
            dtype=dtype,
            device=device,
            collect_output_gram=collect_output_gram,
            hook_names=[layer_name for layer_name in config.node_layers if layer_name != "output"],
        )

        graph_train_loader = create_data_loader(dataset, shuffle=False, batch_size=config.batch_size, seed=config.seed)

        logger.info("Calculating interaction rotations (Cs).")
        # Calls on collect_M_dash_and_Lambda_dash
        # Builds sqrt sorted Lambda matrix and its inverse
        Cs, Us, Lambda_abs_sqrts, Lambda_abs_sqrt_pinvs, U_D_sqrt_pinv_Vs, U_D_sqrt_Vs, Lambda_dashes = calculate_interaction_rotations(
            gram_matrices=gram_matrices,
            section_names=section_names,
            node_layers=config.node_layers,
            hooked_model=hooked_model,
            data_loader=graph_train_loader,
            dtype=dtype,
            device=device,
            n_intervals=config.n_intervals,
            truncation_threshold=config.truncation_threshold,
            rotate_final_node_layer=config.rotate_final_node_layer,
        )
    else:
        gram_matrices, Cs, Us = load_interaction_rotations(config=config)

    # C_info may be None for final (output) layer if rotate_output_logits=False
    C_list = [C_info.C for C_info in Cs if C_info is not None]
    C_pinv_list = [C_info.C_pinv for C_info in Cs if C_info is not None]

    with open(file_path, "wb") as f:
        torch.save({"C": C_list, "C_pinv": C_pinv_list, "Lambda_abs_sqrts": Lambda_abs_sqrts, "Lambda_abs_sqrt_pinvs": Lambda_abs_sqrt_pinvs,
                   "U_D_sqrt_pinv_Vs": U_D_sqrt_pinv_Vs, "U_D_sqrt_Vs": U_D_sqrt_Vs, "Lambda_dashes": Lambda_dashes, "Cs raw": Cs, "Us raw": Us, "gram matrices": gram_matrices, }, f)

    return {"C": C_list, "C_pinv": C_pinv_list, "Lambda_abs_sqrts": Lambda_abs_sqrts, "Lambda_abs_sqrt_pinvs": Lambda_abs_sqrt_pinvs, "U_D_sqrt_pinv_Vs": U_D_sqrt_pinv_Vs, "U_D_sqrt_Vs": U_D_sqrt_Vs, "Lambda_dashes": Lambda_dashes, "Cs raw": Cs, "Us raw": Us, "gram matrices": gram_matrices}


def get_relu_similarities(
    model: nn.Module,
    config: Config,
    file_path: Path,
    dataset: Dataset,
    device: str,
    hooked_model: HookedModel,
    Cs: list[Float[Tensor, "d_hidden1 d_hidden2"]],
) -> dict[str, Float[Tensor, "d_hidden d_hidden"]]:
    graph_train_loader = create_data_loader(dataset, shuffle=True, batch_size=config.batch_size)
    graph_section_names = [f"sections.{sec}" for sec in model.sections if sec != "pre"]

    relu_similarities: dict[str, Float[Tensor, "d_hidden d_hidden"]] = collect_relu_interactions(
        hooked_model=hooked_model,
        module_names=config.activation_layers,
        data_loader=graph_train_loader,
        dtype=TORCH_DTYPES[config.dtype],
        device=device,
        relu_metric_type=config.relu_metric_type,
        Cs=Cs,
        layer_module_names=graph_section_names,
        n_intervals=config.n_intervals,
        unhooked_model=model,
        use_residual_stream=config.use_residual_stream,
    )

    with open(file_path, "wb") as f:
        torch.save(relu_similarities, f)

    return relu_similarities


def get_rotated_Ws(
    model: nn.Module,
    C_pinv_list: list[Float[Tensor, "d_hidden1 d_hidden2"]],
) -> dict[str, Float[Tensor, "layer_count d_hidden d_hidden"]]:
    """Extract W^l, perform paper-equivalent right multiplication of psuedoinverse of C^l.

    Has to be called on config with node_layers edited to include the weight matrix layers.
    This must start at the same place as the node layers to calculate the C_pinvs did.

    Quick explanation on how this works:
    - graph_section_names passes in automatically named sections based on node_layers in config.
    - These are MultiSequential objects with modules inside, and they are iterable.
    - Iterate through and extract weights, put in dictionary with key being the submodules in that
    section
    - NOTE: This code is not happy if you have more than one submodule in that section with a valid weight
    - Append returned weight to weight list only if it is not None.
    - Zip with the C list, cutting off first C since we match weight to C in next layer.
    - Append identity to bottom of weight matrix for residual stream and left-multiply by C_pinv

    Returns:
        rotated_weights_dict: Keys are module names from section we extracted weight from. Should
            never have two submodules in section with valid weights, making this function only valid for
            MLPs and not attention layers. Values are weight tensors, with dimensions [in_dim, out_dim]
    """
    graph_section_names = [f"sections.{sec}" for sec in model.sections if sec != "pre"]

    weights_dict: dict[list[str], float[Tensor, "d_hidden1, d_hidden2"]] = {}
    for section_name in graph_section_names:
        section: Iterable = get_model_attr(model, section_name)  # From rib.models.utils
        str_module_names: str = ", ".join([submodule.__class__.__name__ for submodule in section])
        weight = get_model_weight(model, attr_path=section_name)
    # Return None if module_classes instance doesn't match module, only consider mlp_out for now
    # Regex is needed to specify exactly which module (e.g. not unembed matrix)
        if weight is not None and re.search(r'\bMLPOut\b', str_module_names):
            weights_dict[str_module_names] = weight

    rotated_weights = {}
    for (str_module_name, weight), C_pinv in zip(list(weights_dict.items()), C_pinv_list[1:]):
        # Cut off first C matrix because need to use C in next layer
        resid_stream_size = C_pinv.shape[1] - weight.shape[0]
        assert resid_stream_size > 0, "Weight and C mats don't match"
        weight = torch.cat((weight, torch.eye(resid_stream_size)))
        rotated_weights[str_module_name] = C_pinv @ weight

    return rotated_weights


def get_cluster_gram(
    model: nn.Module,
    config: Config,
    file_path: Path,
    dataset: Dataset,
    device: str,
    hooked_model: HookedModel,
    all_cluster_idxs: list[list[Int[Tensor, "d_cluster"]]],
) -> dict[str, list[Float[Tensor, "d_cluster d_cluster"]]]:
    graph_train_loader = create_data_loader(dataset, shuffle=True, batch_size=config.batch_size)

    cluster_grams: dict[str, list[Float[Tensor, "d_cluster d_cluster"]]] = collect_cluster_grams(
        hooked_model=hooked_model,
        module_names=config.activation_layers,
        data_loader=graph_train_loader,
        dtype=TORCH_DTYPES[config.dtype],
        device=device,
        all_cluster_idxs=all_cluster_idxs,
        use_residual_stream=config.use_residual_stream,
        dataset_size=len(dataset)
    )

    with open(file_path, "wb") as f:
        torch.save(cluster_grams, f)


    return cluster_grams

# def get_P_matrices(
#     model: nn.Module,
#     config: Config,
#     file_path: Path,
#     dataset: Dataset,
#     device: str,
#     hooked_model: HookedModel,
#     C_list: list[Float[Tensor, "d_hidden d_hidden"]],
#     W_hat_list: list[Float[Tensor,  "d_hidden d_hidden"]],
#     all_cluster_idxs: list[list[np.ndarray]],
# ) -> dict[str, dict[Int[Tensor, "cluster_size"], Float[Tensor, "d_hidden_next_layer d_hidden"]]]:
#     """Helper function for P matrix collection method.

#     In some cases this might include usage of W_hat_list, and in others it won't
#     """
#     graph_section_names = [f"sections.{sec}" for sec in model.sections if sec != "pre"]

#     Ps: dict[str, Float[Tensor, "d_hidden d_hidden"]] = collect_clustered_relu_P_mats_no_W(
#         module_names=graph_section_names,
#         C_list=C_list,
#         all_cluster_idxs=all_cluster_idxs,
#     )

#     with open(file_path, "wb") as f:
#         torch.save(Ps, f)

#     return Ps


# def get_edges(
#     model: nn.Module,
#     config: Config,
#     file_path: str,
#     dataset: Dataset,
#     device: str,
#     hooked_model: HookedModel,
#     C_list: list[Float[Tensor, "d_hidden_out d_hidden_in"]],
#     C_unscaled_list: list[Float[Tensor, "d_hidden_out d_hidden_in"]],
#     W_hat_list: list[Float[Tensor, "d_hidden_in d_hidden_out"]],
# ) -> dict[str, Float[Tensor, "d_hidden_trunc_curr d_hidden_trunc_next"]]:
#     graph_train_loader = create_data_loader(dataset, shuffle=True, batch_size=config.batch_size)

#     """NOTE CODE BELOW HAS C_UNSCALED_LIST = C_LIST. THIS SHOULD NOT ALWAYS BE TRUE."""
#     edges: dict[str, Float[Tensor, "d_hidden_trunc_1 d_hiddn_trunc_2"]] = collect_test_edges(
#         C_unscaled_list=C_list,
#         C_list=C_list,
#         W_hat_list=W_hat_list,
#         hooked_model=hooked_model,
#         module_names=config.activation_layers,
#         data_loader=graph_train_loader,
#         dtype=TORCH_DTYPES[config.dtype],
#         device=device,
#     )

#     with open(file_path, "wb") as f:
#         torch.save(edges, f)

#     return edges


def check_and_open_file(
    get_var_fn: callable,
    model: nn.Module,
    config: Config,
    file_path: Path,
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
        var = get_var_fn(model, config, file_path, dataset,
                         device, hooked_model, **kwargs)

    return var


# ============================================================================

def transformer_relu_main(config_path_str: str):
    """Build the interaction graph and store it on disk."""
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)
    set_seed(config.seed)

    relu_matrices_save_file = Path(__file__).parent / f"similarity_metric_{config.relu_metric_type}_transformer"
    Cs_save_file = Path(__file__).parent / "Cs_transformer"
    Ws_save_file = Path(__file__).parent / "Ws_transformer"
    Ps_save_file = Path(__file__).parent / "Ps_transformer"
    edges_save_file = Path(__file__).parent / "edges_transformer"
    cluster_gram_save_file = Path(__file__).parent / "cluster_gram_transformer"

    out_dir = Path(__file__).parent / f"out_transformer_relu / type_{config.relu_metric_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = TORCH_DTYPES[config.dtype]
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

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
    print_all_modules(seq_model)  # Check module names were correctly defined
    hooked_model = HookedModel(seq_model)

    # This script doesn't need both train and test sets
    return_set = cast(Literal["train", "test", "all"], config.dataset.return_set)
    dataset = load_dataset(
        dataset_config=config.dataset,
        return_set=return_set,
        tlens_model_path=config.tlens_model_path,
    )
    graph_train_loader = create_data_loader(
        dataset, shuffle=True, batch_size=config.batch_size)

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

    C_list, C_pinv_list, Lambda_abs_sqrts_list, Lambda_abs_sqrt_pinvs_list, U_D_sqrt_pinv_Vs_list, U_D_sqrt_Vs_list, Cs, Us, gram_matrices = Cs_and_Lambdas["C"], Cs_and_Lambdas["C_pinv"], Cs_and_Lambdas[
        "Lambda_abs_sqrts"], Cs_and_Lambdas["Lambda_abs_sqrt_pinvs"], Cs_and_Lambdas["U_D_sqrt_pinv_Vs"], Cs_and_Lambdas["U_D_sqrt_Vs"], Cs_and_Lambdas["Cs raw"], Cs_and_Lambdas["Us raw"], Cs_and_Lambdas["gram matrices"]

    relu_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]] = check_and_open_file(
        get_var_fn=get_relu_similarities,
        model=seq_model,
        config=config,
        file_path=relu_matrices_save_file,
        dataset=dataset,
        device=device,
        hooked_model=hooked_model,
        Cs=C_list,
    )

    replacement_idxs_from_cluster, num_valid_swaps_from_cluster, all_cluster_idxs = relu_plot_and_cluster(
        relu_matrices, out_dir, config)

    swap_all_layers_using_clusters(
        replacement_idxs_from_cluster=replacement_idxs_from_cluster,
        num_valid_swaps_from_cluster=num_valid_swaps_from_cluster,
        hooked_model=hooked_model,
        config=config,
        data_loader=graph_train_loader,
        device=device,
    )

    # Keys: module name for layer; values: list of gram matrices
    cluster_grams: dict[str, list[Float[Tensor, "d_cluster d_cluster"]]] = check_and_open_file(
        get_var_fn=get_cluster_gram,
        model=seq_model,
        config=config,
        file_path=cluster_gram_save_file,
        dataset=dataset,
        device=device,
        hooked_model=hooked_model,
        all_cluster_idxs=all_cluster_idxs,
    )

    for (module_name, layer_gram_list) in list(cluster_grams.items()):
        for (cluster_idx, matrix) in enumerate(layer_gram_list):
            sorted_eigenvalues, sorted_eigenvectors = eigendecompose(matrix)
            plot_eigenvalues(sorted_eigenvalues, out_dir, title=f"{module_name}_{cluster_idx}")
            plot_eigenvectors(sorted_eigenvectors, out_dir, title=f"{module_name}_{cluster_idx}")


    # Separate part of main code ===================================================
    # Redefine layers required for weight matrices

    # seq_model, tlens_cfg_dict = load_sequential_transformer(
    #     # CHANGE LAYERS FOR MLP - NOTE THIS AT LEAST NEEDS TO START WHERE THE PREVIOUS MODEL LIST DID, SO THAT MULTIPLYING BY THE PREVIOUSLY CALCULATED C WITH ZIP WORKS
    #     node_layers=["mlp_out.0", "unembed"],
    #     last_pos_module_type=config.last_pos_module_type,
    #     tlens_pretrained=config.tlens_pretrained,
    #     tlens_model_path=config.tlens_model_path,
    #     eps=config.eps,
    #     dtype=dtype,
    #     device=device,
    # )
    # seq_model.eval()
    # seq_model.to(device=torch.device(device), dtype=dtype)
    # hooked_model = HookedModel(seq_model)

    # # Has to be after editing node_layers so this contains the linear layers to extract weights from
    # W_hat_dict: dict[str, float[Tensor, "d_trunc_C_pinv, d_out_W"]] = get_rotated_Ws(
    #     model=seq_model,
    #     C_pinv_list=C_pinv_list,
    # )

    # edges_dict = check_and_open_file(
    #     get_var_fn=get_edges,
    #     model=seq_model,
    #     config=config,
    #     file_path=edges_save_file,
    #     dataset=dataset,
    #     device=device,
    #     hooked_model=hooked_model,
    #     C_list=C_list,
    #     C_unscaled_list=U_D_sqrt_pinv_Vs_list,
    #     W_hat_list=list(W_hat_dict.values()),
    # )
    # edges = list(edges_dict.values())
    # plot_matrix_list(edges, "edges", out_dir)

    # # config.node_layers can now be used in hooks in this function
    # # But first need to be converted into graph node layers
    # P_matrices: dict[str, dict[Int[Tensor, "cluster_size"], Float[Tensor, "d_hidden_next_layer d_hidden"]]] = check_and_open_file(
    #     file_path=Ps_save_file,
    #     get_var_fn=get_P_matrices,
    #     model=seq_model,
    #     config=config,
    #     dataset=dataset,
    #     device=device,
    #     hooked_model=hooked_model,
    #     C_list=Cs_and_Lambdas["C"],
    #     W_hat_list=list(W_hat_dict.values()),
    #     all_cluster_idxs=all_cluster_idxs,
    # )
    # plot_and_save_Ps(P_matrices, out_dir)


if __name__ == "__main__":
    fire.Fire(transformer_relu_main)
