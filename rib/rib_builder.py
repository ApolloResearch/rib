"""Build a RIB graph.

Steps to build the graph:
1. Load a SequentialTransformer from a transformerlens model (either from_pretrained or via a
    saved model) or a saved MLP or create a modular MLP (which does not need training).
2. Collect the gram matrices at each node layer. If interaction_matrices_path is provided, we skip
    this step (only supported for transformer models).
5. Calculate the interaction basis matrices (labelled C in the paper) for each node layer, starting
    from the final node layer and working backwards. If interaction_matrices_path is provided, we
    load the pre-saved matrices instead of calculating them (only supported for transformer models).
6. Calculate the edges of the RIB graph between each node layer.

Supports passing a path to a config yaml file or a RibConfig object. This config should contain the
`node_layers` field, which describes the sections of the graph that will be built. It contains a
list of module_ids. A graph layer will be built on the inputs to each specified node layer, as well
as the output of the final node layer. For example, if `node_layers` is
["attn.0", "mlp_act.0", "output"] for a SequentialTransformer, this script will build the following
graph layers:
- One on the inputs to the "attn.0" node layer. This will include the residual stream concatenated
    with the output of ln1.0.
- One on the input to "mlp_act.0". This will include the residual stream concatenated with the
    output of "mlp_in.0".
- One on the output of the model. This is a special keyword that does not correspond to a module_id.

For the above node layers, the "section_ids" of the model will be:
- sections.pre (containing everything up to but not including attn.0)
- sections.section_0 (containing attn.0 and all layers up to but not including mlp_act.0)
- sections.section_1 (containing mlp_act.0 and the rest of the model)
For calculating the Cs and edges, we ignore the "sections.pre" section since it won't be part of the
graph.

This file also support parallelization to compute edge values across multiple processes using mpi.
To enable this, just preface the command with `mpirun -n [num_processes]`. These processes will
distribute as evenly as possible across all availible GPUs. The rank-0 process will gather all data
and output it as a single file.
"""

import tempfile
import time
from pathlib import Path
from typing import Literal, Optional, Union

import torch
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from rib.data import (
    BlockVectorDatasetConfig,
    HFDatasetConfig,
    ModularArithmeticDatasetConfig,
    VisionDatasetConfig,
)
from rib.data_accumulator import (
    Edges,
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
from rib.interaction_algos import InteractionRotation, calculate_interaction_rotations
from rib.loader import load_dataset, load_model
from rib.log import logger
from rib.models import (
    MLPConfig,
    ModularMLPConfig,
    SequentialTransformer,
    SequentialTransformerConfig,
)
from rib.settings import REPO_ROOT
from rib.types import TORCH_DTYPES, IntegrationMethod, RootPath, StrDtype
from rib.utils import (
    check_out_file_overwrite,
    eval_cross_entropy_loss,
    eval_model_accuracy,
    get_chunk_indices,
    handle_overwrite_fail,
    load_config,
    replace_pydantic_model,
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
    tlens_pretrained: Optional[
        Literal[
            "gpt2",
            "tiny-stories-1M",
            "pythia-14m",
            "pythia-31m",
            "pythia-70m",
            "pythia-160m",
            "pythia-410m",
            "pythia-1b",
            "pythia-1.4b",
            "pythia-2.8b",
        ]
    ] = Field(None, description="Pretrained transformer lens model.")
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
    gram_matrices_path: Optional[RootPath] = Field(
        None,
        description="Path to pre-saved mean and gram matrices. If provided, we don't recompute."
        "Only supported for transformer models.",
    )
    calculate_Cs: bool = Field(
        True,
        description="Whether to compute Cs. If false skips Cs and edges computation. Note that if"
        "interaction_matrices_path is provided, we don't actually compute the Cs but we load them.",
    )
    calculate_edges: bool = Field(
        True,
        description="Whether to calculate the edges of the RIB graph.",
    )
    node_layers: list[str] = Field(
        ...,
        description="Module ids whose inputs correspond to node layers in the graph."
        "`output` is a special node layer that corresponds to the output of the model.",
    )
    rotate_final_node_layer: bool = Field(
        ...,
        description="Whether to rotate the final node layer to its eigenbasis or not.",
    )
    dataset: Optional[
        Union[
            ModularArithmeticDatasetConfig,
            HFDatasetConfig,
            VisionDatasetConfig,
            BlockVectorDatasetConfig,
        ]
    ] = Field(
        ...,
        discriminator="dataset_type",
        description="The dataset to use to build the graph. Is allowed to be None if computing"
        "gram matrices only.",
    )
    gram_dataset: Optional[
        Union[
            ModularArithmeticDatasetConfig,
            HFDatasetConfig,
            VisionDatasetConfig,
            BlockVectorDatasetConfig,
        ]
    ] = Field(
        None,
        discriminator="dataset_type",
        description="The dataset to use for the gram matrix. Defaults to the same dataset as the"
        "one used to build the graph.",
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
        0,
        description="The number of intervals to use for the integrated gradient approximation."
        "If 0, we take a point estimate (i.e. just alpha=0.5).",
    )
    integration_method: Union[IntegrationMethod, dict[str, IntegrationMethod]] = Field(
        "gradient",
        description="The integration method to choose. Valid integration methods are 'gradient',"
        "which replaces Integrated Gradients with Gradients (and is much faster),"
        "'trapezoidal' which estimates the IG integral using the trapezoidal rule, and"
        "'gauss-legendre' which estimates the integral using the G-L quadrature points."
        "A dictionary can be used to select different methods for different node layers."
        "The keys are names of node layers, optionally excluding `.[block-num]` suffix."
        "These are checked against the node layers used in the graph.",
    )
    dtype: StrDtype = Field(..., description="The dtype to use when building the graph.")
    eval_type: Optional[Literal["accuracy", "ce_loss"]] = Field(
        None,
        description="The type of evaluation to perform on the model before building the graph."
        "If None, skip evaluation.",
    )
    basis_formula: Literal["jacobian", "(1-alpha)^2", "(1-0)*alpha", "svd", "neuron"] = Field(
        "jacobian",
        description="The integrated gradient formula to use to calculate the basis. If 'svd', will"
        "use Us as Cs, giving the eigendecomposition of the gram matrix. If 'neuron', will use "
        "the neuron-basis. Defaults to '(1-0)*alpha'",
    )
    edge_formula: Literal["functional", "squared"] = Field(
        "squared",
        description="The attribution method to use to calculate the edges.",
    )
    n_stochastic_sources_basis_pos: Optional[int] = Field(
        None,
        description="The number of stochastic sources in the out_pos direction to use when"
        "calculating  stochastic Cs. When None, no stochasticity over position is used.",
    )
    n_stochastic_sources_basis_hidden: Optional[int] = Field(
        None,
        description="The number of stochastic sources in the out_hat_hidden direction to use when"
        "calculating stochastic Cs. When None, no stochasticity over hidden dim is used.",
    )
    n_stochastic_sources_edges: Optional[int] = Field(
        None,
        description="The number of stochastic sources to use when calculating squared edges. Uses"
        "normal deterministic formula when None. Must be None for other edge formulas.",
    )
    center: bool = Field(
        True,
        description="Whether to center the activations before performing rib.",
    )
    dist_split_over: Literal["out_dim", "dataset"] = Field(
        "dataset",
        description="For distributed edge runs, whether to split over out_dim or dataset.",
    )
    isolate_ln_var: bool = Field(
        True,
        description="Whether to special case the LN-variance function as a separate node. Otherwise"
        "the LN variance will be potentially mixed with other RIB functions.",
    )
    naive_gradient_flow: bool = Field(
        False,
        description="Use gradient flow (naive version), running repeated RIB builds with pairs of"
        "node layers.",
    )

    @model_validator(mode="after")
    def verify_model_info(self) -> "RibBuildConfig":
        """Checks:
        - Exactly one of tlens_pretrained, tlens_model_path, mlp_path, modular_mlp_config must
            be set.
        - We don't try to load interaction matrices for mlp models (they're so small we
            shouldn't need to).
        - `n_stochastic_sources_edges` is None for non-squared edge_formula.
        - `n_intervals` must be 0 for gradient integration rule.
        - `naive_gradient_flow` is not compatible with MLP
        - `naive_gradient_flow` is not compatible out_dir is None
        """
        model_options = [
            self.tlens_pretrained,
            self.tlens_model_path,
            self.mlp_path,
            self.modular_mlp_config,
        ]
        if sum(1 for val in model_options if val is not None) != 1:
            raise ValueError(f"Exactly one of {model_options} must be set")

        if self.dataset is None:
            if self.calculate_Cs or self.calculate_edges:
                raise ValueError("dataset must be set if calculate_Cs or calculate_edges is True")
            if self.gram_dataset is None:
                raise ValueError("dataset must be set if gram_dataset is None")
            if self.eval_type is not None:
                raise ValueError("dataset must be set if eval_type is not None")

        if self.calculate_edges and not self.calculate_Cs:
            raise ValueError("calculate_edges=True requires calculate_Cs=True")

        if self.gram_matrices_path and not self.calculate_Cs:
            raise ValueError("gram_matrices_path given requires calculate_Cs=True")

        if self.interaction_matrices_path and not self.calculate_Cs:
            raise ValueError("interaction_matrices_path given requires calculate_Cs=True")

        if self.interaction_matrices_path is not None and not self.calculate_edges:
            raise ValueError("interaction_matrices_path given requires calculate_edges=True")

        if self.interaction_matrices_path is not None:
            assert (
                self.mlp_path is None and self.modular_mlp_config is None
            ), "We don't support loading interaction matrices for mlp models"

        if self.n_stochastic_sources_basis_pos is not None:
            assert (
                self.basis_formula == "jacobian"
            ), "n_stochastic_sources_basis_pos only implemented for jacobian basis_formula"

        if self.n_stochastic_sources_basis_hidden is not None:
            assert (
                self.basis_formula == "jacobian"
            ), "n_stochastic_sources_basis_hidden only implemented for jacobian basis_formula"

        if self.edge_formula != "squared":
            assert (
                self.n_stochastic_sources_edges is None
            ), "n_stochastic_sources_edges must be None for non-squared edge_formula"

        if self.n_stochastic_sources_edges is not None:
            assert (
                self.edge_formula == "squared"
            ), "n_stochastic_sources_edges must be None for non-squared edge_formula"

        if self.edge_formula == "functional" and self.dist_split_over == "out_dim":
            raise ValueError("Cannot use functional edge formula with out_dim split")

        if self.integration_method == "gradient":
            assert self.n_intervals == 0, "n_intervals must be 0 for gradient integration rule"

        if isinstance(self.integration_method, dict):
            for node_layer in self.node_layers:
                prefix = node_layer.split(".")[0]
                assert (
                    prefix in self.integration_method or node_layer in self.integration_method
                ), f"Integration method not specified for node layer {node_layer}"
            node_layer_prefixes = set(node_layer.split(".")[0] for node_layer in self.node_layers)
            for key in self.integration_method:
                assert (
                    key in self.node_layers or key in node_layer_prefixes
                ), f"Integration method specified for non-existent node layer {key}"

        if self.naive_gradient_flow:
            assert (
                self.mlp_path is None and self.modular_mlp_config is None
            ), "Naive gradient flow not compatible with MLP"

        return self

    def get_integration_method(self, node_layer: str) -> IntegrationMethod:
        """Get the integration method for a given node layer."""
        if isinstance(self.integration_method, dict):
            if node_layer in self.integration_method:
                return self.integration_method[node_layer]
            prefix = node_layer.split(".")[0]
            return self.integration_method[prefix]
        return self.integration_method


class RibBuildResults(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    exp_name: str = Field(..., description="The name of the experiment")
    mean_vectors: Optional[dict[str, torch.Tensor]] = Field(
        default_factory=lambda: {},
        description="Mean vectors at each node layer. Set to None if using center=False. Set to {} "
        "if loading old Cs where we didn't store mean vectors.",
        repr=False,
    )
    gram_matrices: dict[str, torch.Tensor] = Field(
        description="Gram matrices at each node layer.", repr=False
    )
    interaction_rotations: list[InteractionRotation] = Field(
        description="Interaction rotation matrices (e.g. Cs, Us) at each node layer.", repr=False
    )
    edges: list[Edges] = Field(
        description="The edges between each node layer. Set to [] if no edges calculated.",
        repr=False,
    )
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
    calc_grams_time: Optional[float] = Field(
        None, description="The time taken (in minutes) to calculate the gram matrices."
    )
    calc_C_time: Optional[float] = Field(
        None, description="The time taken (in minutes) to calculate the interaction rotations."
    )
    calc_edges_time: Optional[float] = Field(
        None, description="The time taken (in minutes) to calculate the edges."
    )


def _getattr_recursive(obj, attr):
    """Get the value of an attribute-path (where the path is separated by ".")."""
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _get_all_subfields(config):
    field_paths = []
    fields = set(list(config.model_fields.keys()))
    for field in fields:
        if not isinstance(getattr(config, field), BaseModel):
            field_paths.append(field)
        else:
            subfields = _get_all_subfields(getattr(config, field))
            field_paths.extend([f"{field}.{subfield}" for subfield in subfields])
    return field_paths


def _verify_compatible_configs(
    config: RibBuildConfig, loaded_config: RibBuildConfig, whitelist=None
):
    """Ensure that the config for calculating edges is compatible with that used to calculate Cs.

    Args:
        config: The current config
        loaded_config: The config that was used to calculate the loaded file
        whitelist: A list of attributes that are allowed to differ between the two configs. Defaults
            to an empty list.

    TODO: It would be nice to unittest this, but awkward to avoid circular imports and keep the
    path management nice with this Config being defined in this file in the rib_scripts dir.
    """
    # Fields that we don't need to verify. For this specific call (e.g. basis & edges stuff when
    # loading gram matrices) + general fields that are allowed to differ (like batch size).
    whitelist = whitelist or []
    whitelist += [
        "exp_name",
        "out_dir",
        "gram_matrices_path",
        "interaction_matrices_path",
        "eval_type",
        "calculate_Cs",
        "calculate_edges",
        "batch_size",
        "gram_batch_size",
        "edge_batch_size",
        "dist_split_over",
        "node_layers",
    ]

    # config.node_layers must be a subsequence of loaded_config.node_layers
    assert "|".join(config.node_layers) in "|".join(loaded_config.node_layers), (
        "node_layers in the config must be a subsequence of the node layers in the config used to"
        "calculate the C matrices. Otherwise, the C matrices won't match those needed to correctly"
        "calculate the edges."
    )

    list_of_fields = set(_get_all_subfields(config) + _get_all_subfields(loaded_config))
    for field in list_of_fields:
        if field not in whitelist:
            # Both configs should have the same fields because they're both RibBuildConfigs so
            # no need to check has_attr.
            val1 = _getattr_recursive(config, field)
            val2 = _getattr_recursive(loaded_config, field)
            if val1 != val2:
                logger.warning(
                    f"{field} in config ({val1}) does not match {field} in loaded_config ({val2})"
                )
                # TODO Raise an error here once we're sure it won't disrupt a big run due to some
                # mismatched fields from old files where we didn't pay attention to this.


def load_partial_results(
    config: RibBuildConfig,
    device: Union[torch.device, str],
    path: Union[str, Path],
    return_interaction_rotations: bool = True,
) -> tuple[
    dict[str, Float[Tensor, "orig"]],
    dict[str, Float[Tensor, "orig orig"]],
    list[InteractionRotation],
]:
    """Load pre-saved mean vectors, gram matrices and interaction rotation matrices from file.

    Useful for just calculating Cs or edges on large models.

    Args:
        config: The config used to calculate the C matrices.
        device: The device to move the matrices to.
        return_interaction_rotations: Whether to return the interaction rotations. Set to False to
            load mean vectors and gram matrices only.

    Returns:
        The mean vectors, gram matrices and interaction rotations
    """
    logger.info("Loading pre-saved results from %s", path)
    matrices_info = torch.load(path)

    # The loaded config might have a different schema. Only pass fields that are still valid.
    # FIXME: Move the code below into it's own function, or into _verify_compatible_configs.
    config_dict = config.model_dump()
    valid_fields = list(config_dict.keys())
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
    # Verify configs
    _verify_compatible_configs(config, loaded_config)

    interaction_rotations = (
        [InteractionRotation(**data) for data in matrices_info["interaction_rotations"]]
        if return_interaction_rotations
        else []
    )
    mean_vectors = matrices_info["mean_vectors"] if "mean_vectors" in matrices_info else {}
    gram_matrices = matrices_info["gram_matrices"]

    # Move to device
    mean_vectors = {k: v.to(device) for k, v in mean_vectors.items()} if mean_vectors else None
    gram_matrices = {k: v.to(device) for k, v in gram_matrices.items()}

    return mean_vectors, gram_matrices, interaction_rotations


def _get_out_file_path(
    config: RibBuildConfig,
    dist_info: Optional[DistributedInfo] = None,
) -> Optional[Path]:
    if config.out_dir is None:
        return None
    if config.calculate_edges:
        if dist_info is not None and dist_info.global_size > 1:
            # Multiple output files so we store in a dedicated directory for the experiment
            out_dir = config.out_dir / f"distributed_{config.exp_name}"
            out_file = out_dir / f"rib_graph_global_rank{dist_info.global_rank}.pt"
        else:
            out_file = config.out_dir / f"{config.exp_name}_rib_graph.pt"
    elif config.calculate_Cs:
        # all processes will compute the same Cs so we only have rank 0 write output
        if dist_info is not None and dist_info.global_size > 1 and dist_info.global_rank != 0:
            return None
        else:
            out_file = config.out_dir / f"{config.exp_name}_rib_Cs.pt"
    else:
        assert (
            dist_info is None or dist_info.global_size == 1
        ), "Distributed grams not implemented yet"
        out_file = config.out_dir / f"{config.exp_name}_rib_grams.pt"
    return out_file


def _check_out_file_path(
    out_file: Optional[Path], force: bool, dist_info: Optional[DistributedInfo] = None
) -> None:
    if out_file is not None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if not check_out_file_overwrite(out_file, force):
            handle_overwrite_fail(dist_info)


def gradient_flow_loop(
    config: RibBuildConfig,
    force: bool = False,
) -> RibBuildResults:
    """Call rib_build() repeatedly to implement a naive version of the gradient flow basis

    There is three procedures we handle:
        1. Calculate edges only. This happens if interaction_matrices_path is provided. we assume
           these to be the gradient flow basis and just run the normal edges calculation. We still
           need to wrap this here because we we want to write naive_gradient_flow into the config
           of the final output.
        2a. Calculate bases only (interaction_matrices_path is None and calculate_edges is False).
        2b. Calculate bases and then edges (the full run).


    """
    # Case 1, skip directly to rib_build for (potential) edge calculation
    if config.interaction_matrices_path is not None:
        out_file = _get_out_file_path(config)
        _check_out_file_path(out_file, force)

        edges_config = replace_pydantic_model(
            config, {"naive_gradient_flow": False, "out_dir": None}
        )
        edges_results = rib_build(edges_config, force=force)
        edges_results.config = config
        if out_file is not None:
            torch.save(edges_results.model_dump(), out_file)
            logger.info("Saved gradient flow results (pre-saved Cs, and edges) to %s", out_file)
        return edges_results

    # Case 2, we need a directory to store the Cs because gradient flow works with multiple calls.
    Cs_out_dir = config.out_dir or Path(tempfile.TemporaryDirectory().name)
    Cs_out_file = Cs_out_dir / f"{config.exp_name}_rib_Cs.pt"
    _check_out_file_path(Cs_out_file, force)
    # Case 2b, check the edges output file already before going through the Cs calculation.
    if config.calculate_edges:
        Cs_and_edges_out_file = _get_out_file_path(config)
        _check_out_file_path(Cs_and_edges_out_file, force)

    # Run inidivual rib_builds for each node layer. Iterate through node_layers backwards (from
    # output to input). In the first step save rotations for last and second to last layer, in
    # the other steps save rotations for the current layer only (append at beginning of list).
    node_layers = config.node_layers
    final_node_layer = node_layers[-1]
    results = None
    for current_node_layer in node_layers[:-1][::-1]:
        updates = {
            "node_layers": [current_node_layer, final_node_layer],
            "exp_name": f"{config.exp_name}_{current_node_layer}",
            "naive_gradient_flow": False,
            "calculate_edges": False,
            "interaction_matrices_path": None,
            "out_dir": None,
        }
        config_i = replace_pydantic_model(config, updates)
        result_i = rib_build(config_i, force=force)
        if results is None:
            results = result_i
        else:
            results.gram_matrices.update(result_i.gram_matrices)
            results.calc_C_time += result_i.calc_C_time
            # result_i.interaction_rotations contains both the current and the final layer. In all
            # but the first iteration (above) we only want the rotations for the current layer.
            assert len(result_i.interaction_rotations) == 2
            assert result_i.interaction_rotations[-1].node_layer == final_node_layer
            results.interaction_rotations.insert(0, result_i.interaction_rotations[0])
    # Merge results into a single result and save to file
    assert isinstance(results, RibBuildResults)
    # Update exp_name and config to the one of the full run (rather than the last individual run)
    results.exp_name = config.exp_name
    results.config = replace_pydantic_model(
        config, {"calculate_edges": False, "out_dir": Cs_out_dir}
    )
    # This save is necessary. In case 2a it constitutes the final results, in case 2b we use this
    # file to pass the Cs to the edge calculation.
    torch.save(results.model_dump(), Cs_out_file)
    logger.info("Saved gradient flow results (Cs) to %s", Cs_out_file)

    if not config.calculate_edges:
        return results
    else:
        edges_config = replace_pydantic_model(
            config,
            {
                "naive_gradient_flow": False,
                "interaction_matrices_path": Cs_out_file,
                "out_dir": None,
            },
        )
        edges_results = rib_build(edges_config, force=force)
        results.edges = edges_results.edges
        results.calc_edges_time = edges_results.calc_edges_time
        # Save the edges results to file
        if Cs_and_edges_out_file is not None:
            torch.save(results.model_dump(), Cs_and_edges_out_file)
            logger.info("Saved gradient flow results (Cs + edges) to %s", Cs_and_edges_out_file)
        return results


def rib_build(
    config_path_or_obj: Union[str, RibBuildConfig],
    force: bool = False,
    n_pods: int = 1,
    pod_rank: int = 0,
) -> RibBuildResults:
    """Build the RIB graph and store it on disk.

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

    dist_info = get_dist_info(n_pods=n_pods, pod_rank=pod_rank)

    # we increment the seed between processes so we generate different phis. This is because
    # each process will calculate a fraction of the total sources, and we need these sources
    # (phis) to be different.
    if config.dist_split_over == "out_dim" and config.seed is not None:
        random_increment = 9594
        # chosen by fair dice roll, guaranteed to be random (https://xkcd.com/221/)
        # for real, the reason we add an increment is to avoid correlation between
        # different global seeds, i.e. seed=0 pod=1 and seed=1 pod=0
        set_seed(config.seed + random_increment * dist_info.global_rank)
    else:
        set_seed(config.seed)

    adjust_logger_dist(dist_info)
    device = get_device_mpi(dist_info)
    dtype = TORCH_DTYPES[config.dtype]

    if config.naive_gradient_flow:
        if n_pods > 1 or pod_rank > 0:
            raise NotImplementedError("Distributed naive gradient flow not implemented yet")
        return gradient_flow_loop(config, force)

    if config.calculate_Cs:
        assert n_pods == 1, "Cannot parallelize Cs calculation between pods"
        if dist_info.global_size > 1 and config.dist_split_over == "dataset":
            if config.interaction_matrices_path is None:
                raise NotImplementedError("Cannot parallelize Cs calculation over dataset")

    out_file = _get_out_file_path(config, dist_info)
    _check_out_file_path(out_file, force, dist_info)

    calc_grams_time = None
    calc_C_time = None
    calc_edges_time = None

    model = load_model(config, device=device, dtype=dtype)

    if config.dataset is not None:
        dataset = load_dataset(
            dataset_config=config.dataset,
            model_n_ctx=model.cfg.n_ctx if isinstance(model, SequentialTransformer) else None,
            tlens_model_path=config.tlens_model_path,
        )
        logger.info(f"Dataset length: {len(dataset)}")  # type: ignore

    if config.gram_dataset is None:
        gram_dataset = dataset
    elif config.gram_matrices_path is None and config.interaction_matrices_path is None:
        # Load the gram_dataset, except for in the cases where we skip gram matrix calculation.
        # If config.graph_matrix_path is given we will skip gram matrix calculation and load the
        # pre-saved gram matrices.  If config.interaction_matrices_path is given we will skip gram
        # and Cs calculation and load the pre-saved gram and Cs matrices.
        gram_dataset = load_dataset(
            dataset_config=config.gram_dataset,
            model_n_ctx=model.cfg.n_ctx if isinstance(model, SequentialTransformer) else None,
            tlens_model_path=config.tlens_model_path,
        )
        logger.info(f"Gram dataset length: {len(gram_dataset)}")  # type: ignore

    model.eval()
    hooked_model = HookedModel(model)

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
    if isinstance(model, SequentialTransformer):
        # Don't build the graph for the section of the model before the first node layer
        section_names = [f"sections.{sec}" for sec in model.sections if sec != "pre"]
    else:
        # MLP "sections" are simply the model layers specified in config.node_layers
        section_names = [layer for layer in config.node_layers if layer != "output"]

    integration_methods = [
        config.get_integration_method(node_layer) for node_layer in config.node_layers
    ]

    # 1) Compute or load gram matrices (load if neither gram matrices nor Cs are provided)
    if config.gram_matrices_path is None and config.interaction_matrices_path is None:
        # Note that we use shuffle=False because we already shuffled the dataset when we loaded it
        gram_train_loader = DataLoader(
            dataset=gram_dataset,
            batch_size=config.gram_batch_size or config.batch_size,
            shuffle=False,
        )
        # Only need gram matrix for output if we're rotating the final node layer
        collect_output_gram = config.node_layers[-1] == "output" and config.rotate_final_node_layer
        mean_vectors: Optional[dict[str, Float[Tensor, "orig"]]] = None
        if config.center:
            logger.info("Collecting dataset means")
            mean_vectors = collect_dataset_means(
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
            means=mean_vectors,
        )
        calc_grams_time = (time.time() - collect_gram_start_time) / 60
        logger.info("Time to collect gram matrices: %.2f minutes", calc_grams_time)
    elif config.interaction_matrices_path is not None:
        # If we have interaction_matrices_path we won't need to compute Cs and thus don't
        # need to load means and grams here. Means won't be needed for edges and grams will
        # be loaded from interaction_matrices_path.
        pass
    elif config.gram_matrices_path is not None:
        logger.info("Skipping gram matrix calculation, loading pre-saved gram matrices")
        mean_vectors, gram_matrices, _ = load_partial_results(
            config,
            device,
            path=config.gram_matrices_path,
            return_interaction_rotations=False,
        )
    else:
        assert False, "This else should never be reached"

    # 2) Compute or load Cs (this happens only if calculate_Cs)
    if config.calculate_Cs and config.interaction_matrices_path is None:
        graph_train_loader = DataLoader(
            dataset=dataset, batch_size=config.batch_size, shuffle=False
        )
        c_start_time = time.time()
        logger.info(
            "Calculating interaction rotations (Cs) for %s for %d batches.",
            config.node_layers,
            len(graph_train_loader),
        )
        interaction_rotations = calculate_interaction_rotations(
            gram_matrices=gram_matrices,
            section_names=section_names,
            node_layers=config.node_layers,
            hooked_model=hooked_model,
            data_loader=graph_train_loader,
            dtype=dtype,
            device=device,
            n_intervals=config.n_intervals,
            integration_methods=integration_methods,
            truncation_threshold=config.truncation_threshold,
            rotate_final_node_layer=config.rotate_final_node_layer,
            basis_formula=config.basis_formula,
            center=config.center,
            means=mean_vectors,
            n_stochastic_sources_pos=config.n_stochastic_sources_basis_pos,
            n_stochastic_sources_hidden=config.n_stochastic_sources_basis_hidden,
            out_dim_n_chunks=dist_info.global_size if config.dist_split_over == "out_dim" else 1,
            out_dim_chunk_idx=dist_info.global_rank if config.dist_split_over == "out_dim" else 0,
            isolate_ln_var=config.isolate_ln_var,
        )
        calc_C_time = (time.time() - c_start_time) / 60
        logger.info("Time to calculate Cs: %.2f minutes", calc_C_time)
        if "cuda" in device:
            logger.info(
                "Max memory allocated for Cs: %.2f GB", torch.cuda.max_memory_allocated() / 1e9
            )
            torch.cuda.reset_peak_memory_stats()
    elif config.calculate_Cs and config.interaction_matrices_path is not None:
        logger.info("Skipping Cs calculation, loading pre-saved Cs")
        mean_vectors, gram_matrices, interaction_rotations = load_partial_results(
            config, device, path=config.interaction_matrices_path
        )
    else:
        logger.info("Not computing or loading Cs.")
        interaction_rotations = []

    # 3) Compute edges
    if config.calculate_edges and config.calculate_Cs:
        edge_interaction_rotations = [
            obj for obj in interaction_rotations if obj.node_layer in config.node_layers
        ]
        logger.info("Calculating edges.")
        full_dataset_len = len(dataset)  # type: ignore
        if config.dist_split_over == "dataset":
            # no-op if only 1 process
            start_idx, end_idx = get_chunk_indices(
                data_size=full_dataset_len,
                chunk_idx=dist_info.global_rank,
                n_chunks=dist_info.global_size,
            )
            dataset = Subset(dataset, range(start_idx, end_idx))

        edge_train_loader = DataLoader(
            dataset, batch_size=config.edge_batch_size or config.batch_size, shuffle=False
        )

        logger.info(
            "Calculating edges for %s for %d batches.", config.node_layers, len(edge_train_loader)
        )
        edges_start_time = time.time()
        E_hats = collect_interaction_edges(
            interaction_rotations=edge_interaction_rotations,
            hooked_model=hooked_model,
            n_intervals=config.n_intervals,
            integration_methods=integration_methods[:-1],
            section_names=section_names,
            data_loader=edge_train_loader,
            dtype=dtype,
            device=device,
            data_set_size=full_dataset_len,  # includes data for other processes
            edge_formula=config.edge_formula,
            n_stochastic_sources=config.n_stochastic_sources_edges,
            out_dim_n_chunks=dist_info.global_size if config.dist_split_over == "out_dim" else 1,
            out_dim_chunk_idx=dist_info.global_rank if config.dist_split_over == "out_dim" else 0,
        )

        calc_edges_time = (time.time() - edges_start_time) / 60
        logger.info("Time to calculate edges: %.2f minutes", calc_edges_time)
        logger.info(
            "Max memory allocated for edges: %.2f GB", torch.cuda.max_memory_allocated() / 1e9
        )
    else:
        logger.info("Skipping edge calculation because calculate_edges or calculate_Cs were False")
        E_hats = []

    # Note that mean_vectors can be the empty dict if we loaded an old Cs file. It is None if
    # center=False.
    results = RibBuildResults(
        exp_name=config.exp_name,
        mean_vectors={k: v.cpu() for k, v in mean_vectors.items()} if mean_vectors else None,
        gram_matrices={k: v.cpu() for k, v in gram_matrices.items()},
        interaction_rotations=interaction_rotations,
        edges=E_hats,
        dist_info=dist_info,
        contains_all_edges=dist_info.global_size == 1,  # True if no parallelisation
        config=config,
        ml_model_config=model.cfg,
        calc_grams_time=calc_grams_time,
        calc_C_time=calc_C_time,
        calc_edges_time=calc_edges_time,
    )

    if out_file is not None:
        torch.save(results.model_dump(), out_file)
        logger.info("Saved results to %s", out_file)

    return results


ResultsLike = Union[RibBuildResults, Path, str]


def to_results(results: ResultsLike) -> RibBuildResults:
    if isinstance(results, RibBuildResults):
        return results
    elif isinstance(results, (str, Path)):
        return RibBuildResults(**torch.load(results))
    else:
        raise ValueError(f"Invalid results type: {type(results)}")
