from pathlib import Path
from typing import Any, Literal, Optional, Union, cast

import fire
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset

from experiments.relu_interactions.config_schemas import LMConfig
from experiments.relu_interactions.relu_interaction_utils import relu_plot_and_cluster
from rib.data_accumulator import (
    calculate_delete_cluster_duplicate_loss,
    collect_relu_interactions,
)
from rib.hook_manager import HookedModel
from rib.loader import create_data_loader, load_dataset, load_sequential_transformer
from rib.types import TORCH_DTYPES
from rib.utils import load_config, set_seed


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
        var = get_var_fn(model, config, file_path, dataset,
                         device, hooked_model, **kwargs)

    return var


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


def del_neurons_main(config_path_str: str):
    """Note config node layers are hard coded into this function, as are the module names for the hook."""
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=LMConfig)
    set_seed(config.seed)

    relu_matrices_save_file = Path(__file__).parent / f"similarity_metric_{config.relu_metric_type}_transformer"

    out_dir = Path(__file__).parent / f"out_transformer_del_neurons"
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = TORCH_DTYPES[config.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seq_model, tlens_cfg_dict = load_sequential_transformer(
        node_layers=["mlp_out.0, unembed"],
        last_pos_module_type=config.last_pos_module_type,
        tlens_pretrained=config.tlens_pretrained,
        tlens_model_path=config.tlens_model_path,
        eps=config.eps,
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
    )
    graph_train_loader = create_data_loader(dataset, shuffle=True, batch_size=config.batch_size)

    relu_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]] = check_and_open_file(
        get_var_fn=get_relu_similarities,
        model=seq_model,
        config=config,
        file_path=relu_matrices_save_file,
        dataset=dataset,
        device=device,
        hooked_model=hooked_model,
        Cs_list=None, # This can be none if you're sure the relu matrices already exist in a file
    )

    replacement_idxs_from_cluster, num_valid_swaps_from_cluster, all_cluster_idxs = relu_plot_and_cluster(relu_matrices, out_dir, config)

    unhooked_loss, hooked_loss, unhooked_accuracy, hooked_accuracy = calculate_delete_cluster_duplicate_loss(
        hooked_model=hooked_model,
        module_names=["sections.section_0.0"],
        data_loader=graph_train_loader,
        dtype=dtype,
        device=device,
        all_cluster_idxs=all_cluster_idxs,
        use_residual_stream=config.use_residual_stream,
    )

    print(
        f"unhooked loss {unhooked_loss}",
        f"hooked loss {hooked_loss}\n",
        f"unhooked accuracy {unhooked_accuracy}",
        f"hooked accuracy {hooked_accuracy}\n",
    )


if __name__ == "__main__":
    fire.Fire(del_neurons_main)