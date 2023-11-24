from pathlib import Path
from typing import Literal, cast

import fire
import torch
import torch.nn as nn

from experiments.relu_interactions.config_schemas import LMConfig
from rib.loader import create_data_loader, load_dataset, load_sequential_transformer
from rib.rlct_helpers import estimate_rlcts_training
from rib.types import TORCH_DTYPES
from rib.utils import load_config, set_seed


def rlct_main(config_path_str: str):
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=LMConfig)
    set_seed(config.seed)

    out_dir = Path(__file__).parent / f"out_rlct"
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

    # This script doesn't need both train and test sets
    return_set = cast(Literal["train", "test", "all"], config.dataset.return_set)
    dataset = load_dataset(
        dataset_config=config.dataset,
        return_set=return_set,
        tlens_model_path=config.tlens_model_path,
    )
    graph_train_loader = create_data_loader(
        dataset, shuffle=True, batch_size=config.batch_size)

    slgd_mean, sgld_std, sgnht_mean, sgnht_std, trace_sgld, trace_sgnht = estimate_rlcts_training(
        model=seq_model,
        config=config,
        criterion=nn.CrossEntropyLoss(),
        dataloader=graph_train_loader,
        device=torch.device(device),
    )

    with open(out_dir / f"rlct_{config.rlct_config.sample_layer}.txt", "w") as f:
        f.write(f"SGLD: {slgd_mean} +/- {sgld_std}\n")
        f.write(f"SGNHT: {sgnht_mean} +/- {sgnht_std}\n")


if __name__ == "__main__":
    fire.Fire(rlct_main)