import re
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Union, cast

import fire
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import Dataset

from experiments.relu_interactions.config_schemas import (
    LMConfig,
    _verify_compatible_configs,
)
from experiments.relu_interactions.relu_interaction_utils import (
    plot_eigenvalues,
    plot_eigenvectors,
    print_all_modules,
    relu_plot_and_cluster,
    swap_all_layers_using_clusters,
)
from rib.data_accumulator import (
    collect_cluster_grams,
    collect_clustered_relu_P_mats,
    collect_clustered_relu_P_mats_no_W,
    collect_gram_matrices,
    collect_relu_interactions,
    collect_test_edges,
    collect_cluster_fns,
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


def load_interaction_rotations(
    config: LMConfig,
) -> tuple[
    dict[str, Float[Tensor, "d_hidden d_hidden"]
         ], list[InteractionRotation], list[Eigenvectors]
]:
    logger.info("Loading pre-saved C matrices from %s", config.interaction_matrices_path)
    assert config.interaction_matrices_path is not None
    matrices_info = torch.load(config.interaction_matrices_path)

    loaded_config = LMConfig(**matrices_info["config"])
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
    config: LMConfig,
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
    config: LMConfig,
    file_path: Path,
    dataset: Dataset,
    device: str,
    hooked_model: HookedModel,
    Cs_list: list[Float[Tensor, "d_hidden1 d_hidden2"]],
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
        Cs_list=Cs_list,
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


def get_cluster_grams(
    model: nn.Module,
    config: LMConfig,
    file_path: Path,
    dataset: Dataset,
    device: str,
    hooked_model: HookedModel,
    all_cluster_idxs: list[list[Int[Tensor, "d_cluster"]]],
) -> dict[str, list[Float[Tensor, "d_cluster d_cluster"]]]:
    graph_train_loader = create_data_loader(dataset, shuffle=True, batch_size=config.batch_size)

    cluster_grams = collect_cluster_grams(
        hooked_model=hooked_model,
        module_names=["sections.section_0.0"], # If you start defining node layers from mlp_in
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


def get_cluster_fns(
    model: nn.Module,
    config: LMConfig,
    file_path: Path,
    dataset: Dataset,
    device: str,
    hooked_model: HookedModel,
    all_cluster_idxs: list[list[Int[Tensor, "d_cluster"]]],
) -> dict[str, list[Float[Tensor, "d_cluster"]]]:
    graph_train_loader = create_data_loader(dataset, shuffle=True, batch_size=config.batch_size, drop_last=True)

    cluster_fns = collect_cluster_fns(
        hooked_model=hooked_model,
        module_names=["sections.section_1.0"], # If you start defining node layers from mlp_in
        data_loader=graph_train_loader,
        dtype=TORCH_DTYPES[config.dtype],
        device=device,
        all_cluster_idxs=all_cluster_idxs,
        use_residual_stream=config.use_residual_stream,
    )

    # with open(file_path, "wb") as f:
    #     torch.save(cluster_fns, f)

    return cluster_fns


def check_and_open_file(
    get_var_fn: callable,
    model: nn.Module,
    config: LMConfig,
    file_path: Path,
    dataset: Dataset,
    device: str,
    hooked_model: Optional[HookedModel] = None,
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
    config = load_config(config_path, config_model=LMConfig)
    set_seed(config.seed)

    relu_matrices_save_file = Path(__file__).parent / f"similarity_metric_{config.relu_metric_type}_transformer"
    Cs_save_file = Path(__file__).parent / "Cs_transformer"
    Ws_save_file = Path(__file__).parent / "Ws_transformer"
    Ps_save_file = Path(__file__).parent / "Ps_transformer"
    edges_save_file = Path(__file__).parent / "edges_transformer"
    cluster_gram_save_file = Path(__file__).parent / "cluster_gram_transformer"
    cluster_fn_save_file = Path(__file__).parent / "cluster_fn_transformer"

    out_dir = Path(__file__).parent / f"out_transformer_relu / type_{config.relu_metric_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = TORCH_DTYPES[config.dtype]
    device = "cpu" if torch.cuda.is_available() else "cpu"

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
    print_all_modules(seq_model)
    hooked_model = HookedModel(seq_model)

    return_set = cast(Literal["train", "test", "all"], config.dataset.return_set)
    dataset = load_dataset(
        dataset_config=config.dataset,
        return_set=return_set,
        tlens_model_path=config.tlens_model_path,
    )
    graph_train_loader = create_data_loader(dataset, shuffle=True, batch_size=config.batch_size)

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

    C_list, C_pinv_list, Lambda_abs_sqrts_list, Lambda_abs_sqrt_pinvs_list, U_D_sqrt_pinv_Vs_list, U_D_sqrt_Vs_list, Cs, Us, gram_matrices = Cs_and_Lambdas["C"], Cs_and_Lambdas["C_pinv"], Cs_and_Lambdas["Lambda_abs_sqrts"], Cs_and_Lambdas["Lambda_abs_sqrt_pinvs"], Cs_and_Lambdas["U_D_sqrt_pinv_Vs"], Cs_and_Lambdas["U_D_sqrt_Vs"], Cs_and_Lambdas["Cs raw"], Cs_and_Lambdas["Us raw"], Cs_and_Lambdas["gram matrices"]

    relu_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]] = check_and_open_file(
        get_var_fn=get_relu_similarities,
        model=seq_model,
        config=config,
        file_path=relu_matrices_save_file,
        dataset=dataset,
        device=device,
        hooked_model=hooked_model,
        Cs_list=C_list,
    )

    replacement_idxs_from_cluster, num_valid_swaps_from_cluster, all_cluster_idxs, all_centroid_idxs = relu_plot_and_cluster(relu_matrices, out_dir, config)

    swap_all_layers_using_clusters(
        replacement_idxs_from_cluster=replacement_idxs_from_cluster,
        num_valid_swaps_from_cluster=num_valid_swaps_from_cluster,
        hooked_model=hooked_model,
        config=config,
        data_loader=graph_train_loader,
        device=device,
    )

    # Separate part of main code ===================================================
    # Redefine layers and instantiate new transformer

    mode = "weights"
    new_node_layer_dict = {'weights': ["mlp_act.0", "mlp_out.0", "unembed"], "cluster_grams": ["mlp_in.0", "mlp_act.0"]}
    seq_model, tlens_cfg_dict = load_sequential_transformer(
        # CHANGE LAYERS FOR MLP - NOTE for w_hat this needs to start where previous model node
        # layers did, so using zip with previously saved C matrices list works
        node_layers=new_node_layer_dict[mode],
        last_pos_module_type=config.last_pos_module_type,
        tlens_pretrained=config.tlens_pretrained,
        tlens_model_path=config.tlens_model_path,
        eps=config.eps,
        dtype=dtype,
        device=device,
    )
    seq_model.eval()
    seq_model.to(device=torch.device(device), dtype=dtype)
    print_all_modules(seq_model)
    hooked_model = HookedModel(seq_model)

    # Keys: module name for layer; values: list of gram matrices
    cluster_grams: dict[str, list[list[Float[Tensor, "d_cluster d_cluster"]]]] = check_and_open_file(
        get_var_fn=get_cluster_grams,
        model=seq_model,
        config=config,
        file_path=cluster_gram_save_file,
        dataset=dataset,
        device=device,
        hooked_model=hooked_model,
        all_cluster_idxs=all_cluster_idxs,
    )

    cluster_fns: dict[str, list[list[Float[Tensor, "d_cluster"]]]] = check_and_open_file(
        get_var_fn=get_cluster_fns,
        model=seq_model,
        config=config,
        file_path=cluster_fn_save_file,
        dataset=dataset,
        device=device,
        hooked_model=hooked_model,
        all_cluster_idxs=all_cluster_idxs,
    )

    for (module_name, layer_gram_list) in list(cluster_grams.items()):
        # cluster_fns_layer = cluster_fns[module_name]
        # cluster_layer_similarities = {}

        for (cluster_idx, matrix) in enumerate(layer_gram_list):
            sorted_eigenvalues, sorted_eigenvectors = eigendecompose(matrix)
            plot_eigenvalues(sorted_eigenvalues, out_dir, title=f"{module_name}_{cluster_idx}")
            plot_eigenvectors(sorted_eigenvectors, out_dir, title=f"{module_name}_{cluster_idx}", num_vectors=2)
        #     similarities = [
        #         (torch.div(torch.dot(fn, sorted_eigenvectors[0]), torch.dot(sorted_eigenvectors[0], sorted_eigenvectors[0])),
        #         torch.div(torch.dot(fn, sorted_eigenvectors[1]), torch.dot(sorted_eigenvectors[1], sorted_eigenvectors[1])))
        #         for fn in cluster_fns_layers
        #     ]
        #     cluster_layer_similarities[cluster_idx] = similarities
        # print(cluster_layer_similarities)

    ## For whole layer gram instead
    # sorted_eigenvalues, sorted_eigenvectors = eigendecompose(whole_layer_gram)
    # plot_eigenvalues(sorted_eigenvalues, out_dir, title=f"whole_layer_{module_name}")


if __name__ == "__main__":
    fire.Fire(transformer_relu_main)

    # def get_P_matrices(
#     model: nn.Module,
#     config: LMConfig,
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
#     config: LMConfig,
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

    ## BELOW IS IRRELEVANT CODE FOR NOW - relic from when we wanted to compare P mats to edges
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
