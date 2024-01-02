"""Build a RIB graph.

Steps to build the graph:
1. Load a SequentialTransformer from a transformerlens model (either from_pretrained or via a
    saved model) or a saved MLP or create a modular MLP (which does not need training).
2. Collect the gram matrices at each node layer. If interaction_matrices_path is provided, we skip
    this step (only supported for transformer models).
5. Calculate the interaction basis matrices (labelled C in the paper) for each node layer, starting
    from the final node layer and working backwards. If interaction_matrices_path is provided, we
    load the pre-saved matrices instead of calculating them (only supported for transformer models).
6. Calculate the edges of the interaction graph between each node layer.

Supports passing a path to a config.yaml or a RibConfig object. This config should contain the
`node_layers` field. This describes the sections of the graph that will be built: A graph layer will
be built on the inputs to each specified node layer, as well as the output of the final node layer.
For example, if `node_layers` is ["attn.0", "mlp_act.0", "output"], this script will build the
following graph layers:
- One on the inputs to the "attn.0" node layer. This will include the residual stream concatenated
    with the output of ln1.0.
- One on the input to "mlp_act.0". This will include the residual stream concatenated with the
    output of "mlp_in.0".
- One on the output of the model. This is a special keyword that does not correspond to a module_id.

This file also support parallelization to compute edge values across multiple processes using mpi.
To enable this, just preface the command with `mpirun -n [num_processes]`. These processes will
distribute as evenly as possible across all availible GPUs. The rank-0 process will gather all data
and output it as a single file.
"""
import time
from pathlib import Path
from typing import Literal, Optional, Union

import torch
import yaml
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor
from torch.utils.data import DataLoader

from rib.data import (
    BlockVectorDatasetConfig,
    HFDatasetConfig,
    ModularArithmeticDatasetConfig,
    VisionDatasetConfig,
)
from rib.data_accumulator import (
    collect_dataset_means,
    collect_gram_matrices,
    collect_interaction_edges,
)
from rib.distributed_utils import (
    DistributedInfo,
    adjust_logger_dist,
    get_device_mpi,
    get_dist_info,
)
from rib.hook_manager import HookedModel
from rib.interaction_algos import (
    Eigenvectors,
    InteractionRotation,
    calculate_interaction_rotations,
)
from rib.loader import (
    get_dataset_chunk,
    load_dataset,
    load_mlp,
    load_sequential_transformer,
)
from rib.log import logger
from rib.models import (
    MLP,
    MLPConfig,
    ModularMLPConfig,
    SequentialTransformer,
    SequentialTransformerConfig,
)
from rib.settings import REPO_ROOT
from rib.types import TORCH_DTYPES, RootPath, StrDtype
from rib.utils import (
    check_outfile_overwrite,
    eval_cross_entropy_loss,
    eval_model_accuracy,
    load_config,
    set_seed,
)


class RibBuildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    exp_name: str = Field(..., description="The name of the experiment")
    out_dir: Optional[RootPath] = Field(
        REPO_ROOT / "rib_scripts/rib_build/out",
        description="Directory for the output files. Defaults to `./out/`. If None, no output "
        "is written. If a relative path, it is relative to the root of the rib repo.",
    )
    seed: Optional[int] = Field(0, description="The random seed value for reproducibility")
    tlens_pretrained: Optional[Literal["gpt2", "pythia-14m"]] = Field(
        None, description="Pretrained transformer lens model."
    )
    tlens_model_path: Optional[RootPath] = Field(
        None, description="Path to saved transformer lens model."
    )
    mlp_path: Optional[RootPath] = Field(
        None,
        description="Path to the saved MLP model. If None, we expect the MLP class to be "
        "initialized with manual weights (such as in the case of a modular MLP).",
    )
    modular_mlp_config: Optional[ModularMLPConfig] = Field(
        None,
        description="The model to use. If None, we expect mlp_path to be set.",
    )
    interaction_matrices_path: Optional[RootPath] = Field(
        None,
        description="Path to pre-saved interaction matrices. If provided, we don't recompute."
        "Only supported for transformer models.",
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
    dataset: Union[
        ModularArithmeticDatasetConfig,
        HFDatasetConfig,
        VisionDatasetConfig,
        BlockVectorDatasetConfig,
    ] = Field(
        ...,
        discriminator="dataset_type",
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
    calculate_edges: bool = Field(
        True,
        description="Whether to calculate the edges of the interaction graph.",
    )
    eval_type: Optional[Literal["accuracy", "ce_loss"]] = Field(
        None,
        description="The type of evaluation to perform on the model before building the graph."
        "If None, skip evaluation.",
    )
    basis_formula: Literal["(1-alpha)^2", "(1-0)*alpha", "svd", "neuron"] = Field(
        "(1-0)*alpha",
        description="The integrated gradient formula to use to calculate the basis. If 'svd', will"
        "use Us as Cs, giving the eigendecomposition of the gram matrix. If 'neuron', will use "
        "the neuron-basis.",
    )
    edge_formula: Literal["functional", "squared"] = Field(
        "functional",
        description="The attribution method to use to calculate the edges.",
    )
    center: bool = Field(
        False,
        description="Whether to center the activations before performing rib. Currently only"
        "supported for basis_formula='svd', which gives the 'pca' basis.",
    )

    @model_validator(mode="after")
    def verify_model_info(self) -> "RibBuildConfig":
        """Exactly one of tlens_pretrained, tlens_model_path, mlp_path, modular_mlp_config must be
        set.

        In addition, we don't support loading interaction matrices for mlp models (they're so small
        we shouldn't need to).
        """
        model_options = [
            self.tlens_pretrained,
            self.tlens_model_path,
            self.mlp_path,
            self.modular_mlp_config,
        ]
        if sum(1 for val in model_options if val is not None) != 1:
            raise ValueError(f"Exactly one of {model_options} must be set")

        if self.interaction_matrices_path is not None:
            assert (
                self.mlp_path is None and self.modular_mlp_config is None
            ), "We don't support loading interaction matrices for mlp models"
        return self


class RibBuildResults(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    exp_name: str = Field(..., description="The name of the experiment")
    gram_matrices: dict[str, torch.Tensor] = Field(description="Gram matrices at each node layer.")
    interaction_rotations: list[InteractionRotation] = Field(
        description="Interaction rotations (Cs) at each node layer."
    )
    eigenvectors: list[Eigenvectors] = Field(description="Eigenvectors at each node layer.")
    edges: list[tuple[str, torch.Tensor]] = Field(description="The edges between each node layer.")
    dist_info: DistributedInfo = Field(
        description="Information about the parallelisation setup used for the run."
    )
    contains_all_edges: bool = Field(
        description="True if there is no parallelisation or if the edges have been combined (as is "
        "done in rib.utils.combine_edges)."
    )
    config: RibBuildConfig = Field(description="The config used to build the graph.")
    ml_model_config: Union[MLPConfig, ModularMLPConfig, SequentialTransformerConfig] = Field(
        discriminator="config_type", description="The config of the model used to build the graph."
    )
    calc_C_time: Optional[str] = Field(
        None, description="The time taken to calculate the interaction rotations."
    )
    calc_edges_time: Optional[str] = Field(
        None, description="The time taken to calculate the edges."
    )


def _verify_compatible_configs(config: RibBuildConfig, loaded_config: RibBuildConfig) -> None:
    """Ensure that the config for calculating edges is compatible with that used to calculate Cs.

    TODO: It would be nice to unittest this, but awkward to avoid circular imports and keep the
    path management nice with this Config being defined in this file in the rib_scripts dir.
    """
    assert (
        config.dataset.dataset_type == loaded_config.dataset.dataset_type
    ), "Dataset types must match"

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

    # Verify that, for huggingface and torchvision datasets, we're not trying to calculate edges on
    # data that wasn't used to calculate the Cs
    if hasattr(config.dataset, "name"):
        assert hasattr(loaded_config.dataset, "name"), "loaded_config doesn't have a dataset name"
        assert config.dataset.name == loaded_config.dataset.name, "Dataset names must match"
    assert config.dataset.return_set == loaded_config.dataset.return_set, "Return sets must match"
    if isinstance(config.dataset, HFDatasetConfig):
        assert isinstance(loaded_config.dataset, HFDatasetConfig)
        if config.dataset.return_set_frac is not None:
            assert (
                loaded_config.dataset.return_set_frac is not None
            ), "Can't set return_set_frac if the loaded config didn't use it"
            assert (
                config.dataset.return_set_frac <= loaded_config.dataset.return_set_frac
            ), "Cannot use a larger return_set_frac for edges than to calculate the Cs"
        elif config.dataset.return_set_n_samples is not None:
            assert loaded_config.dataset.return_set_n_samples is not None
            assert (
                config.dataset.return_set_n_samples <= loaded_config.dataset.return_set_n_samples
            ), "Cannot use a larger return_set_n_samples for edges than to calculate the Cs"


def load_interaction_rotations(
    config: RibBuildConfig,
) -> tuple[
    dict[str, Float[Tensor, "d_hidden d_hidden"]], list[InteractionRotation], list[Eigenvectors]
]:
    """Load pre-saved C matrices from file. Useful for just calculating edges on large models."""
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

    loaded_config = RibBuildConfig(**loaded_config_dict)
    _verify_compatible_configs(config, loaded_config)

    Cs = [InteractionRotation(**data) for data in matrices_info["interaction_rotations"]]
    Us = [Eigenvectors(**data) for data in matrices_info["eigenvectors"]]
    return matrices_info["gram_matrices"], Cs, Us


def rib_build(
    config_path_or_obj: Union[str, RibBuildConfig],
    force: bool = False,
    n_pods: int = 1,
    pod_rank: int = 0,
) -> RibBuildResults:
    """Build the interaction graph and store it on disk.

    Note that we may be calculating the Cs and E_hats (edges) in different scripts. When calculating
    E_hats using pre-saved Cs, we need to ensure, among other things, that the pre-saved Cs were
    calculated for the same node_layers that we wish to draw edges between (i.e. config.node_layers
    should be a subsequence of the node_layers used to calculate the Cs).

    We use the variable edge_Cs to indicate the Cs that are needed to calculate the edges. If
    the Cs were pre-calculated and loaded from file, edge_Cs may be a subsequence of Cs.

    Args:
        config_path_or_obj: a str or Config object. If str must point at YAML config file
        force: whether to overwrite existing output files
        n_pods: number of pods/processes to use for distributed computing
        pod_rank: rank of this pod/process

    Returns:
        Results of the graph build
    """
    config = load_config(config_path_or_obj, config_model=RibBuildConfig)
    set_seed(config.seed)

    dist_info = get_dist_info(n_pods=n_pods, pod_rank=pod_rank)

    adjust_logger_dist(dist_info)
    device = get_device_mpi(dist_info)
    dtype = TORCH_DTYPES[config.dtype]

    out_file: Optional[Path] = None
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

    calc_C_time = None
    calc_edges_time = None

    # Load model
    model: Union[SequentialTransformer, MLP]
    if config.mlp_path is not None or config.modular_mlp_config is not None:
        mlp_config: Union[MLPConfig, ModularMLPConfig]
        if config.mlp_path is not None:
            with open(config.mlp_path.parent / "config.yaml", "r") as f:
                raw_model_config_dict = yaml.safe_load(f)
            mlp_config = MLPConfig(**raw_model_config_dict["model"])
        else:
            assert config.modular_mlp_config is not None
            mlp_config = config.modular_mlp_config

        model = load_mlp(
            mlp_config,
            node_layers=config.node_layers,
            mlp_path=config.mlp_path,
            fold_bias=True,
            device=device,
            seed=config.seed,
        ).to(device=torch.device(device), dtype=dtype)
        assert model.has_folded_bias, "MLP must have folded bias to run RIB"
    else:
        model = load_sequential_transformer(
            node_layers=config.node_layers,
            last_pos_module_type=config.last_pos_module_type,
            tlens_pretrained=config.tlens_pretrained,
            tlens_model_path=config.tlens_model_path,
            fold_bias=True,
            dtype=dtype,
            device=device,
        )
    model.eval()
    hooked_model = HookedModel(model)

    # Load dataset
    model_n_ctx = model.cfg.n_ctx if isinstance(model, SequentialTransformer) else None
    dataset = load_dataset(
        dataset_config=config.dataset,
        return_set=config.dataset.return_set,
        model_n_ctx=model_n_ctx,
        tlens_model_path=config.tlens_model_path,
    )
    logger.info(f"Dataset length: {len(dataset)}")  # type: ignore

    # Evaluate model on dataset for sanity check
    if config.eval_type is not None:
        eval_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False)
        # Test model accuracy/loss before graph building, ta be sure
        if config.eval_type == "accuracy":
            accuracy = eval_model_accuracy(hooked_model, eval_loader, dtype=dtype, device=device)
            logger.info("Model accuracy on dataset: %.2f%%", accuracy * 100)
        elif config.eval_type == "ce_loss":
            loss = eval_cross_entropy_loss(hooked_model, eval_loader, dtype=dtype, device=device)
            logger.info("Model per-token loss on dataset: %.2f", loss)

    # Run RIB
    # Don't build the graph for the section of the model before the first node layer
    if isinstance(model, SequentialTransformer):
        section_names = [f"sections.{sec}" for sec in model.sections if sec != "pre"]
    else:
        section_names = [layer for layer in config.node_layers if layer != "output"]

    if config.interaction_matrices_path is None:
        # Only need gram matrix for output if we're rotating the final node layer
        collect_output_gram = config.node_layers[-1] == "output" and config.rotate_final_node_layer
        gram_train_loader = DataLoader(
            dataset=dataset, batch_size=config.gram_batch_size or config.batch_size, shuffle=False
        )

        means: Optional[dict[str, Float[Tensor, "d_hidden"]]] = None
        bias_positions: Optional[dict[str, Int[Tensor, "segments"]]] = None
        if config.center:
            logger.info("Collecting dataset means")
            means, bias_positions = collect_dataset_means(
                hooked_model=hooked_model,
                module_names=section_names,
                data_loader=gram_train_loader,
                dtype=dtype,
                device=device,
                collect_output_dataset_means=collect_output_gram,
                hook_names=[module_id for module_id in config.node_layers if module_id != "output"],
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
            hook_names=[module_id for module_id in config.node_layers if module_id != "output"],
            means=means,
            bias_positions=bias_positions,
        )
        logger.info("Time to collect gram matrices: %.2f", time.time() - collect_gram_start_time)

        graph_train_loader = DataLoader(
            dataset=dataset, batch_size=config.batch_size, shuffle=False
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
            basis_formula=config.basis_formula,
            center=config.center,
            means=means,
            bias_positions=bias_positions,
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
            dataset, chunk_idx=dist_info.global_rank, total_chunks=dist_info.global_size
        )

        edge_train_loader = DataLoader(
            data_subset, batch_size=config.edge_batch_size or config.batch_size, shuffle=False
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
            edge_formula=config.edge_formula,
        )

        calc_edges_time = f"{(time.time() - edges_start_time) / 60:.1f} minutes"
        logger.info("Time to calculate edges: %s", calc_edges_time)

    results = RibBuildResults(
        exp_name=config.exp_name,
        gram_matrices={k: v.cpu() for k, v in gram_matrices.items()},
        interaction_rotations=Cs,
        eigenvectors=Us,
        edges=[(node_layer, E_hats[node_layer].cpu()) for node_layer in E_hats],
        dist_info=dist_info,
        contains_all_edges=dist_info.global_size == 1,  # True if no parallelisation
        config=config,
        ml_model_config=model.cfg,
        calc_C_time=calc_C_time,
        calc_edges_time=calc_edges_time,
    )

    if out_file is not None:
        # Save the results (which include torch tensors) to file
        torch.save(results.model_dump(), out_file)
        logger.info("Saved results to %s", out_file)

    return results
