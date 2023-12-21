import time
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Union, cast

import fire
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import Dataset

from experiments.relu_interactions.config_schemas import (
    LMConfig,
    Config,
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
    collect_gram_matrices,
    collect_relu_interactions,
    collect_cluster_fns,
)
from rib.hook_manager import HookedModel
from rib.interaction_algos import (
    Eigenvectors,
    InteractionRotation,
    calculate_interaction_rotations,
)
from rib.log import logger
from rib.distributed_utils import adjust_logger_dist, get_device_mpi, get_dist_info
from rib.linalg import eigendecompose
from rib.loader import get_dataset_chunk, load_dataset, load_sequential_transformer
from rib.models.utils import get_model_attr, get_model_weight
from rib.types import TORCH_DTYPES
from rib.utils import (
    check_outfile_overwrite,
    eval_cross_entropy_loss,
    eval_model_accuracy,
    load_config,
    set_seed,
)


def load_interaction_rotations(
    config: Config,
) -> tuple[
    dict[str, Float[Tensor, "d_hidden d_hidden"]], list[InteractionRotation], list[Eigenvectors]
]:
    logger.info("Loading pre-saved C matrices from %s", config.interaction_matrices_path)
    assert config.interaction_matrices_path is not None
    matrices_info = torch.load(config.interaction_matrices_path)

    config_dict = config.model_dump()
    # The loaded config might have a different schema. Only pass fields that are still valid.
    valid_fields = list(config_dict.keys())

    # If not all fields are valid, log a warning
    loaded_config_dict: dict = {}
    for loaded_key in matrices_info["config"]:
        if loaded_key in valid_fields:
            loaded_config_dict[loaded_key] = matrices_info["config"][loaded_key]
        else:
            logger.warning(
                "The following field in the loaded config is no longer supported and will be ignored:"
                f" {loaded_key}"
            )

    loaded_config = Config(**loaded_config_dict)
    _verify_compatible_configs(config, loaded_config)

    Cs = [InteractionRotation(**data) for data in matrices_info["interaction_rotations"]]
    Us = [Eigenvectors(**data) for data in matrices_info["eigenvectors"]]
    return matrices_info["gram_matrices"], Cs, Us


# Helper functions for main ========================================================

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

        gram_train_loader = DataLoader(dataset=dataset, batch_size=config.gram_batch_size or config.batch_size, shuffle=False)

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

        graph_train_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False)
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
    graph_train_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False)
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


def get_cluster_grams(
    model: nn.Module,
    config: LMConfig,
    file_path: Path,
    dataset: Dataset,
    device: str,
    hooked_model: HookedModel,
    all_cluster_idxs: list[list[Int[Tensor, "d_cluster"]]],
) -> dict[str, list[Float[Tensor, "d_cluster d_cluster"]]]:
    graph_train_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False)

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
    graph_train_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False)

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

def main(config_path_or_obj: Union[str, LMConfig], force: bool = False, n_pods: int = 1, pod_rank: int = 0):
    """Build the interaction graph and store it on disk."""
    config = load_config(config_path_or_obj, config_model=LMConfig)
    set_seed(config.seed)

    dist_info = get_dist_info(n_pods=n_pods, pod_rank=pod_rank)
    adjust_logger_dist(dist_info)
    device = get_device_mpi(dist_info)

    relu_matrices_save_file = Path(__file__).parent / f"similarity_metric_{config.relu_metric_type}_pythia"
    Cs_save_file = Path(__file__).parent / "Cs_pythia"
    cluster_gram_save_file = Path(__file__).parent / "cluster_gram_pythia"

    if config.out_dir is not None:
        config.out_dir.mkdir(parents=True, exist_ok=True)
        obj_name = "graph" if config.calculate_edges else "Cs"
        global_rank_suffix = (
            f"_global_rank{dist_info.global_rank}" if dist_info.global_size > 1 else ""
        )
        f_name = f"{config.exp_name}_rib_{obj_name}{global_rank_suffix}.pt"
        out_file = config.out_dir / f_name
        if not check_outfile_overwrite(out_file, force):
            dist_info.local_comm.Abort()  # stop this and other processes

    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(__file__).parent / f"out_pythia_relu / type_{config.relu_metric_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = TORCH_DTYPES[config.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seq_model, tlens_cfg_dict = load_sequential_transformer(
        node_layers=config.node_layers,
        last_pos_module_type=config.last_pos_module_type,
        tlens_pretrained=config.tlens_pretrained,
        tlens_model_path=config.tlens_model_path,
        fold_bias=True,
        dtype=dtype,
        device=device,
    )
    seq_model.eval()
    seq_model.to(device=torch.device(device), dtype=dtype)
    hooked_model = HookedModel(seq_model)

    return_set = cast(Literal["train", "test", "all"], config.dataset.return_set)
    dataset = load_dataset(
        dataset_config=config.dataset,
        return_set=return_set,
        tlens_model_path=config.tlens_model_path,
        model_n_ctx=seq_model.cfg.n_ctx,
    )
    graph_train_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False)
    gram_matrices, Cs, Us = load_interaction_rotations(config=config)
    edge_Cs = [C for C in Cs if C.node_layer_name in config.node_layers]

    # # dict of InteractionRotation objects or Tensors
    # Cs_and_Lambdas: dict[str, list[Union[InteractionRotation, Float[Tensor, ...]]]] = check_and_open_file(
    #     file_path=Cs_save_file,
    #     get_var_fn=get_Cs,
    #     model=seq_model,
    #     config=config,
    #     dataset=dataset,
    #     device=device,
    #     hooked_model=hooked_model,
    # )

    # C_list, C_pinv_list, Lambda_abs_sqrts_list, Lambda_abs_sqrt_pinvs_list, U_D_sqrt_pinv_Vs_list, U_D_sqrt_Vs_list, Cs, Us, gram_matrices = Cs_and_Lambdas["C"], Cs_and_Lambdas["C_pinv"], Cs_and_Lambdas["Lambda_abs_sqrts"], Cs_and_Lambdas["Lambda_abs_sqrt_pinvs"], Cs_and_Lambdas["U_D_sqrt_pinv_Vs"], Cs_and_Lambdas["U_D_sqrt_Vs"], Cs_and_Lambdas["Cs raw"], Cs_and_Lambdas["Us raw"], Cs_and_Lambdas["gram matrices"]

    # relu_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]] = check_and_open_file(
    #     get_var_fn=get_relu_similarities,
    #     model=seq_model,
    #     config=config,
    #     file_path=relu_matrices_save_file,
    #     dataset=dataset,
    #     device=device,
    #     hooked_model=hooked_model,
    #     Cs_list=C_list,
    # )

    # replacement_idxs_from_cluster, num_valid_swaps_from_cluster, all_cluster_idxs, all_centroid_idxs = relu_plot_and_cluster(relu_matrices, out_dir, config)

    # swap_all_layers_using_clusters(
    #     replacement_idxs_from_cluster=replacement_idxs_from_cluster,
    #     num_valid_swaps_from_cluster=num_valid_swaps_from_cluster,
    #     hooked_model=hooked_model,
    #     config=config,
    #     data_loader=graph_train_loader,
    #     device=device,
    # )

    # # Separate part of main code ===================================================
    # # Redefine layers and instantiate new transformer

    # mode = "weights"
    # new_node_layer_dict = {'weights': ["mlp_act.0", "mlp_out.0", "unembed"], "cluster_grams": ["mlp_in.0", "mlp_act.0"]}
    # seq_model, tlens_cfg_dict = load_sequential_transformer(
    #     # CHANGE LAYERS FOR MLP - NOTE for w_hat this needs to start where previous model node
    #     # layers did, so using zip with previously saved C matrices list works
    #     node_layers=new_node_layer_dict[mode],
    #     last_pos_module_type=config.last_pos_module_type,
    #     tlens_pretrained=config.tlens_pretrained,
    #     tlens_model_path=config.tlens_model_path,
    #     eps=config.eps,
    #     dtype=dtype,
    #     device=device,
    # )
    # seq_model.eval()
    # seq_model.to(device=torch.device(device), dtype=dtype)
    # print_all_modules(seq_model)
    # hooked_model = HookedModel(seq_model)

    # # Keys: module name for layer; values: list of gram matrices
    # cluster_grams: dict[str, list[list[Float[Tensor, "d_cluster d_cluster"]]]] = check_and_open_file(
    #     get_var_fn=get_cluster_grams,
    #     model=seq_model,
    #     config=config,
    #     file_path=cluster_gram_save_file,
    #     dataset=dataset,
    #     device=device,
    #     hooked_model=hooked_model,
    #     all_cluster_idxs=all_cluster_idxs,
    # )

    # for (module_name, layer_gram_list) in list(cluster_grams.items()):
    #     for (cluster_idx, matrix) in enumerate(layer_gram_list):
    #         sorted_eigenvalues, sorted_eigenvectors = eigendecompose(matrix)
    #         plot_eigenvalues(sorted_eigenvalues, out_dir, title=f"{module_name}_{cluster_idx}")
    #         plot_eigenvectors(sorted_eigenvectors, out_dir, title=f"{module_name}_{cluster_idx}", num_vectors=2)


if __name__ == "__main__":
    fire.Fire(main)