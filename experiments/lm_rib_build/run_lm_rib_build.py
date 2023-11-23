"""This script builds a RIB graph for a language model.
We build the graph using a SequentialTransformer model, with weights ported over from a
transformer-lens model.

Steps to build the graph:
1. Load a model from transformerlens (either from_pretrained or via ModelConfig).
2. Convert the model to a SequentialTransformer model, which has nn.Modules corresponding to each
    node layer.
3. Fold in the biases into the weights.
4. Collect the gram matrices at each node layer. If interaction_matrices_path is provided, we skip
    this step.
5. Calculate the interaction basis matrices (labelled C in the paper) for each node layer, starting
    from the final node layer and working backwards. If interaction_matrices_path is provided, we
    load the pre-saved matrices instead of calculating them.
6. Calculate the edges of the interaction graph between each node layer.

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
- (If logits_node_layer is True:) One on the output of the model, i.e. the logits.

This file also support parallelization to compute edge values across multiple processes using mpi. To enable this, just preface the command with `mpirun -n [num_processes]`. These processes will distribute as evenly as possible across all availible GPUs. The rank-0 process will gather all data and output it as a single file.
"""
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Dict, Literal, Optional, Union, cast

import fire
import torch
from jaxtyping import Float
from mpi4py import MPI
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from rib.data import HFDatasetConfig, ModularArithmeticDatasetConfig
from rib.data_accumulator import collect_gram_matrices, collect_interaction_edges
from rib.hook_manager import HookedModel
from rib.interaction_algos import (
    Eigenvectors,
    InteractionRotation,
    calculate_interaction_rotations,
)
from rib.loader import (
    create_data_loader,
    get_dataset_chunk,
    load_dataset,
    load_sequential_transformer,
)
from rib.log import logger
from rib.mpi_utils import (
    adjust_logger_mpi,
    check_sizes_mpi,
    get_device_mpi,
    get_mpi_info,
)
from rib.types import TORCH_DTYPES, RootPath, StrDtype
from rib.utils import (
    check_outfile_overwrite,
    eval_cross_entropy_loss,
    eval_model_accuracy,
    load_config,
    set_seed,
)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exp_name: str = Field(..., description="The name of the experiment")
    out_dir: Optional[RootPath] = Field(
        Path(__file__).parent / "out",
        description="Directory for the output files. Defaults to `./out/`. If None, no output is written.",
    )
    seed: int = Field(..., description="The random seed value for reproducibility")
    tlens_pretrained: Optional[Literal["gpt2", "pythia-14m"]] = Field(
        None, description="Pretrained transformer lens model."
    )
    tlens_model_path: Optional[RootPath] = Field(
        None, description="Path to saved transformer lens model."
    )
    interaction_matrices_path: Optional[RootPath] = Field(
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
        "If 0, we take a point estimate (i.e. just alpha=0.5).",
    )

    dtype: StrDtype = Field(..., description="The dtype to use when building the graph.")

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

    @model_validator(mode="after")
    def verify_model_info(self) -> "Config":
        if sum(1 for val in [self.tlens_pretrained, self.tlens_model_path] if val is not None) != 1:
            raise ValueError(
                "Exactly one of [tlens_pretrained, tlens_model_path] must be specified"
            )
        return self


def _verify_compatible_configs(config: Config, loaded_config: Config) -> None:
    """Ensure that the config for calculating edges is compatible with that used to calculate Cs.

    TODO: It would be nice to unittest this, but awkward to avoid circular imports and keep the
    path management nice with this Config being defined in this file in the experiments dir.
    """

    # config.node_layers must be a subsequence of loaded_config.node_layers
    assert "|".join(config.node_layers) in "|".join(loaded_config.node_layers), (
        "node_layers in the config must be a subsequence of the node layers in the config used to"
        "calculate the C matrices. Otherwise, the C matrices won't match those needed to correctly"
        "calculate the edges."
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


def main(config_path_or_obj: Union[str, Config], force: bool = False) -> Dict[str, Any]:
    """Build the interaction graph and store it on disk.

    Note that we may be calculating the Cs and E_hats (edges) in different scripts. When calculating
    E_hats using pre-saved Cs, we need to ensure, among other things, that the pre-saved Cs were
    calculated for the same node_layers that we wish to draw edges between (i.e. config.node_layers
    should be a subsequence of the node_layers used to calculate the Cs).

    We use the variable edge_Cs to indicate the Cs that are needed to calculate the edges. If
    the Cs were pre-calculated and loaded from file, edge_Cs may be a subsequence of Cs.

    Args:
        config: a str or Config object. If str must point at YAML config file
        kwargs: modifications to passed config
    """
    config = load_config(config_path_or_obj, config_model=Config)
    set_seed(config.seed)

    mpi_info = get_mpi_info()
    adjust_logger_mpi(logger)
    device = get_device_mpi(logger)

    if config.out_dir is not None:
        config.out_dir.mkdir(parents=True, exist_ok=True)
        f_name = f"{config.exp_name}_rib_{'graph' if config.calculate_edges else 'Cs'}.pt"
        out_file = config.out_dir / f_name
        if not check_outfile_overwrite(out_file, force):
            mpi_info.comm.Abort()  # stop this and other processes

    dtype = TORCH_DTYPES[config.dtype]
    calc_C_time = None
    calc_edges_time = None

    # Time each stage
    load_model_data_start_time = time.time()
    seq_model, tlens_cfg_dict = load_sequential_transformer(
        node_layers=config.node_layers,
        last_pos_module_type=config.last_pos_module_type,
        tlens_pretrained=config.tlens_pretrained,
        tlens_model_path=config.tlens_model_path,
        eps=config.eps,
        fold_bias=True,
        dtype=dtype,
        device=device,
    )
    seq_model.eval()
    seq_model.to(device=torch.device(device), dtype=dtype)
    hooked_model = HookedModel(seq_model)

    # This script doesn't need both train and test sets
    return_set = cast(Literal["train", "test", "all"], config.dataset.return_set)
    dataset = load_dataset(
        dataset_config=config.dataset,
        return_set=return_set,
        tlens_model_path=config.tlens_model_path,
    )

    logger.info("Time to load model and dataset: %.2f", time.time() - load_model_data_start_time)
    if config.eval_type is not None:
        eval_loader = create_data_loader(
            dataset, shuffle=False, batch_size=config.batch_size, seed=config.seed
        )
        # Test model accuracy/loss before graph building, ta be sure
        if config.eval_type == "accuracy":
            accuracy = eval_model_accuracy(hooked_model, eval_loader, dtype=dtype, device=device)
            logger.info("Model accuracy on dataset: %.2f%%", accuracy * 100)
        elif config.eval_type == "ce_loss":
            loss = eval_cross_entropy_loss(hooked_model, eval_loader, dtype=dtype, device=device)
            logger.info("Model per-token loss on dataset: %.2f", loss)

    # Don't build the graph for the section of the model before the first node layer
    section_names = [f"sections.{sec}" for sec in seq_model.sections if sec != "pre"]

    if config.interaction_matrices_path is None:
        # Only need gram matrix for output if we're rotating the final node layer
        collect_output_gram = config.node_layers[-1] == "output" and config.rotate_final_node_layer

        gram_train_loader = create_data_loader(
            dataset,
            shuffle=False,
            batch_size=config.gram_batch_size or config.batch_size,
            seed=config.seed,
        )
        collect_gram_start_time = time.time()
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

        logger.info("Time to collect gram matrices: %.2f", time.time() - collect_gram_start_time)

        graph_train_loader = create_data_loader(
            dataset, shuffle=False, batch_size=config.batch_size, seed=config.seed
        )
        c_start_time = time.time()
        logger.info("Calculating interaction rotations (Cs).")
        Cs, Us = calculate_interaction_rotations(
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
        # Cs used to calculate edges
        edge_Cs = Cs

        calc_C_time = f"{(time.time() - c_start_time) / 60:.1f} minutes"
        logger.info("Time to calculate Cs: %s", calc_C_time)
    else:
        gram_matrices, Cs, Us = load_interaction_rotations(config=config)
        edge_Cs = [C for C in Cs if C.node_layer_name in config.node_layers]

    if not config.calculate_edges:
        logger.info("Skipping edge calculation.")
        E_hats = {}
    else:
        full_dataset_len = len(dataset)  # type: ignore
        # no-op if only 1 process
        data_subset = get_dataset_chunk(
            dataset, chunk_idx=mpi_info.rank, total_chunks=mpi_info.size
        )

        edge_train_loader = create_data_loader(
            data_subset,
            shuffle=False,
            batch_size=config.edge_batch_size or config.batch_size,
            seed=config.seed,
        )

        logger.info("Calculating edges.")
        edges_start_time = time.time()
        E_hats = collect_interaction_edges(
            Cs=edge_Cs,
            hooked_model=hooked_model,
            n_intervals=config.n_intervals,
            section_names=section_names,
            data_loader=edge_train_loader,
            dtype=dtype,
            device=device,
            data_set_size=full_dataset_len,  # includes data for other processes
        )

        if mpi_info.is_parallelised:
            for m_name, edge_vals in E_hats.items():
                check_sizes_mpi(edge_vals)
                receiver_tensor = None
                if mpi_info.is_main_process:
                    receiver_tensor = torch.empty_like(edge_vals).cpu()
                # we sum across the edge_val tensors in each process, putting the result in
                # the root process's reciever_tensor.
                mpi_info.comm.Reduce(edge_vals.cpu(), receiver_tensor, op=MPI.SUM, root=0)
                if mpi_info.is_main_process:
                    assert receiver_tensor is not None
                    E_hats[m_name] = receiver_tensor

        calc_edges_time = f"{(time.time() - edges_start_time) / 60:.1f} minutes"
        logger.info("Time to calculate edges: %s", calc_edges_time)

    # Move interaction matrices to the cpu and store in dict
    interaction_rotations = []
    for C_info in Cs:
        info_dict = asdict(C_info)
        info_dict["C"] = info_dict["C"].cpu() if info_dict["C"] is not None else None
        info_dict["C_pinv"] = info_dict["C_pinv"].cpu() if info_dict["C_pinv"] is not None else None
        interaction_rotations.append(info_dict)

    eigenvectors = [asdict(U_info) for U_info in Us]

    results = {
        "exp_name": config.exp_name,
        "gram_matrices": {k: v.cpu() for k, v in gram_matrices.items()},
        "interaction_rotations": interaction_rotations,
        "eigenvectors": eigenvectors,
        "edges": [(node_layer, E_hats[node_layer].cpu()) for node_layer in E_hats],
        "config": json.loads(config.model_dump_json()),
        "model_config_dict": tlens_cfg_dict,
        "calc_C_time": calc_C_time,
        "calc_edges_time": calc_edges_time,
    }

    if config.out_dir is not None and mpi_info.is_main_process:
        # Save the results (which include torch tensors) to file
        torch.save(results, out_file)
        logger.info("Saved results to %s", out_file)

    return results


if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: "")
