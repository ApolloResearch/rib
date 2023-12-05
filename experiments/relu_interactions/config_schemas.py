from pathlib import Path
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rib.data import HFDatasetConfig, ModularArithmeticDatasetConfig
from rib.types import TORCH_DTYPES, RibBuildResults, RootPath, StrDtype


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
    num_chains: int = Field(
        None, description="Number of chains to use for each RLCT estimate."
    )
    num_draws: int = Field(
        100, description="Number of draws to use for estimating RLCT in each chain. Takes into account number of steps between draws."
    )
    num_burnin_steps: int
    num_steps_bw_draws: int = Field(
        1, description="Monte Carlo sampling requires i.i.d samples for good convergence properties of the estimator. However, by construction, Markov chains have dependency between successive draws. For best results, this value should therefore be > 1."
    )
    batch_size: int
    cores: int
    seed: Optional[Union[int, List[int]]]
    pbar: bool = Field(
        True, description="Whether to show a progress bar for chain sampling."
    )
    verbose: bool
    return_weights: bool
    use_distill_loss: bool = Field(
        False, description="Current RLCT estimation assumes comparison to ideal true q distribution at a local minimum. This leads to negative lambdas if using a non-local minimum checkpoint. To fix this, we can compare to the outputs the model gives at the start point of the chain instead."
    )
    save_results: bool = Field(
        False, description="Used in current implementation to save RLCT mean/var estimates to JSON for SGNHT and SGLD."
    )
    sample_layer: Optional[str] = Field(
        None, description="Layer name to estimate RLCT for. All other layers will be frozen in sampler optimiser."
    )


class MLPConfig(BaseModel):
    exp_name: str
    mlp_path: Path
    batch_size: int
    seed: int
    truncation_threshold: float  # Remove eigenvectors with eigenvalues below this threshold.
    rotate_final_node_layer: bool  # Whether to rotate the output layer to its eigenbasis.
    n_intervals: int  # The number of intervals to use for integrated gradients.
    dtype: str  # Data type of all tensors (except those overriden in certain functions).
    activation_layers: list[str]
    node_layers: list[str]
    relu_metric_type: int
    edit_weights: bool
    threshold: float # For dendrogram distance cutting with fcluster to make clusters
    use_residual_stream: bool

    @field_validator("dtype")
    def dtype_validator(cls, v):
        assert v in TORCH_DTYPES, f"dtype must be one of {TORCH_DTYPES}"
        return v


class LMConfig(BaseModel):
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
    use_residual_stream: bool = Field(
        False,
        description="Whether to count residual stream in ReLU clustering alongside the MLP neurons."
    )
    rlct_config: Optional[RLCTConfig] = Field(
        None,
        description="For layer-wise RLCT estimation",
    )
    force_overwrite_output: bool = False,

    @field_validator("dtype")
    def dtype_validator(cls, v):
        assert v in TORCH_DTYPES, f"dtype must be one of {TORCH_DTYPES}"
        return v

    @model_validator(mode="after")
    def verify_model_info(self) -> "LMConfig":
        if sum(1 for val in [self.tlens_pretrained, self.tlens_model_path] if val is not None) != 1:
            raise ValueError(
                "Exactly one of [tlens_pretrained, tlens_model_path] must be specified"
            )
        return self


def _verify_compatible_configs(config: Union[MLPConfig, LMConfig], loaded_config: Union[MLPConfig, LMConfig]) -> None:
    """Ensure that the config for calculating edges is compatible with that used to calculate Cs."""

    assert config.node_layers == loaded_config.node_layers[-len(config.node_layers):], (
        "node_layers in the config must be a subsequence of the node layers in the config used to"
        " calculate the C matrices, ending at the final node layer. Otherwise, the C matrices won't"
        " match those needed to correctly calculate the edges."
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


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    exp_name: str = Field(..., description="The name of the experiment")
    out_dir: Optional[RootPath] = Field(
        Path(__file__).parent / "out",
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

    calculate_edges: bool = Field(
        True,
        description="Whether to calculate the edges of the interaction graph.",
    )
    eval_type: Optional[Literal["accuracy", "ce_loss"]] = Field(
        None,
        description="The type of evaluation to perform on the model before building the graph."
        "If None, skip evaluation.",
    )
    basis_formula: Literal["(1-alpha)^2", "(1-0)*alpha", "svd"] = Field(
        "(1-0)*alpha",
        description="The integrated gradient formula to use to calculate the basis. If 'svd', will"
        "use Us as Cs, giving the eigendecomposition of the gram matrix.",
    )
    edge_formula: Literal["functional", "squared"] = Field(
        "functional",
        description="The attribution method to use to calculate the edges.",
    )

    @model_validator(mode="after")
    def verify_model_info(self) -> "Config":
        if sum(1 for val in [self.tlens_pretrained, self.tlens_model_path] if val is not None) != 1:
            raise ValueError(
                "Exactly one of [tlens_pretrained, tlens_model_path] must be specified"
            )
        return self