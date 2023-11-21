from pathlib import Path
from typing import List, Literal, Optional, Union, cast

import fire
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rib.data import HFDatasetConfig, ModularArithmeticDatasetConfig
from rib.loader import create_data_loader, load_dataset, load_sequential_transformer
from rib.rlct_helpers import estimate_rlcts_training
from rib.types import TORCH_DTYPES
from rib.utils import (
    load_config,
    set_seed,
)


class SGLDKwargs(BaseModel):
    lr: float
    noise_level: float
    weight_decay: float
    elasticity: float
    temperature: Union[str, float]
    num_samples: int


class SGNHTKwargs(BaseModel):
    lr: float
    diffusion_factor: float
    bounding_box_size: float
    num_samples: int


class RLCTConfig(BaseModel):
    sampling_method: str
    sigma: float
    sgld_kwargs: SGLDKwargs
    sgnht_kwargs: SGNHTKwargs
    num_chains: int
    num_draws: int
    num_burnin_steps: int
    num_steps_bw_draws: int
    batch_size: int
    cores: int
    seed: Optional[Union[int, List[int]]]
    pbar: bool
    verbose: bool
    return_weights: bool
    use_distill_loss: bool
    save_results: bool
    sample_layer: Optional[str] = Field(
        None, description="Layer name to estimate RLCT for. All other layers will be frozen in sampler optimiser."
    )


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exp_name: str = Field(..., description="The name of the experiment")
    force_overwrite_output: Optional[bool] = Field(
        False, description="Don't ask before overwriting the output file."
    )
    seed: int = Field(..., description="The random seed value for reproducibility")
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
        ...,
        description="Names of the modules whose inputs correspond to node layers in the graph."
        "`output` is a special node layer that corresponds to the output of the model.",
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
        description="Module type in which to only output the last position index. For modular"
        "arithmetic only.",
    )
    n_intervals: int = Field(
        ...,
        description="The number of intervals to use for the integrated gradient approximation."
        "If 0, we take a point estimate (i.e. just alpha=1).",
    )
    out_dim_chunk_size: Optional[int] = Field(
        None,
        description="The size of the chunks to use for calculating the jacobian. If none, calculate"
        "the jacobian on all output dimensions at once.",
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
    out_dir: Optional[Path] = Field(
        None,
        description="Directory for the output files. If not provided it is `./out/` relative to this file.",
    )
    rlct_config: RLCTConfig = Field(
        None,
        description="For layer-wise RLCT estimation",
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


def rlct_main(config_path_str: str):
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)
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