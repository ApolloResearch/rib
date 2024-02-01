from typing import Optional

import torch
import yaml
from pydantic.v1.utils import deep_update
from torch.nn.functional import cosine_similarity
from torch.testing import assert_close

from rib.interaction_algos import InteractionRotation
from rib.rib_builder import RibBuildConfig


def get_modular_arithmetic_config(*updates: dict) -> RibBuildConfig:
    config_str = f"""
    exp_name: test
    seed: 0
    tlens_pretrained: null
    tlens_model_path: rib_scripts/train_modular_arithmetic/sample_checkpoints/lr-0.001_bs-10000_norm-None_2023-11-28_16-07-19/model_epoch_60000.pt
    dataset:
        dataset_type: modular_arithmetic
        return_set: train
        n_samples: 10
    node_layers:
        - ln1.0
        - mlp_in.0
        - unembed
        - output
    batch_size: 6
    truncation_threshold: 1e-15  # we've been using 1e-6 previously but this increases needed atol
    rotate_final_node_layer: false
    last_pos_module_type: add_resid1
    n_intervals: 0
    dtype: float64
    eval_type: accuracy
    out_dir: null
    basis_formula: (1-0)*alpha
    edge_formula: squared
    n_stochastic_sources_edges: null
    """
    config_dict = deep_update(yaml.safe_load(config_str), *updates)
    return RibBuildConfig(**config_dict)


def get_pythia_config(*updates: dict) -> RibBuildConfig:
    config_str = f"""
    exp_name: test
    seed: 0
    tlens_pretrained: pythia-14m
    tlens_model_path: null
    dataset:
        dataset_type: huggingface
        name: NeelNanda/pile-10k
        tokenizer_name: EleutherAI/pythia-14m
        return_set: train
        return_set_frac: null
        n_documents: 20
        n_samples: 3
        return_set_portion: first
        n_ctx: 128
        seed: 0
    node_layers:
        - ln2.1
        - unembed
    batch_size: 2
    truncation_threshold: 1e-15  # we've been using 1e-6 previously but this increases needed atol
    rotate_final_node_layer: false
    n_intervals: 0
    dtype: float64
    calculate_edges: false
    edge_formula: squared
    eval_type: ce_loss
    out_dir: null
    basis_formula: (1-0)*alpha
    """
    config_dict = deep_update(yaml.safe_load(config_str), *updates)
    return RibBuildConfig(**config_dict)


def get_tinystories_config(*updates: dict) -> RibBuildConfig:
    config_str = f"""
    exp_name: test
    out_dir: null
    seed: 0
    tlens_pretrained: tiny-stories-1M
    tlens_model_path: null
    dataset:
        dataset_type: huggingface
        name: roneneldan/TinyStories # or skeskinen/TinyStories-GPT4, but not clear if part of training
        tokenizer_name: EleutherAI/gpt-neo-125M
        return_set: train
        return_set_frac: null
        n_documents: 1 # avg ~235 toks / story
        n_samples: 15
        return_set_portion: first
        n_ctx: 10 # needs to be <= 511 for the model to behave reasonably
    node_layers:
        - ln1.3
        - ln1.5
        - unembed
    gram_batch_size: 100
    batch_size: 50
    edge_batch_size: 500
    truncation_threshold: 1e-15
    rotate_final_node_layer: true
    n_intervals: 0
    dtype: float64
    center: true
    calculate_edges: true
    eval_type: ce_loss
    basis_formula: jacobian
    edge_formula: squared
    """
    config_dict = deep_update(yaml.safe_load(config_str), *updates)
    return RibBuildConfig(**config_dict)


def get_mnist_config(*updates: dict) -> RibBuildConfig:
    config_str = f"""
    exp_name: test
    mlp_path: "rib_scripts/train_mlp/sample_checkpoints/lr-0.001_bs-64_2023-11-29_14-36-29/model_epoch_12.pt"
    batch_size: 256
    seed: 0
    truncation_threshold: 1e-15  # we've been using 1e-6 previously but this increases needed atol
    rotate_final_node_layer: false
    n_intervals: 0
    dtype: float64
    node_layers:
        - layers.0
        - layers.1
        - layers.2
        - output
    dataset:
        dataset_type: torchvision
        name: MNIST
        return_set_frac: 0.01  # 3 batches (with batch_size=256)
    out_dir: null
    basis_formula: (1-0)*alpha
    edge_formula: squared
    """
    config_dict = deep_update(yaml.safe_load(config_str), *updates)
    return RibBuildConfig(**config_dict)


def get_modular_mlp_config(*updates: dict) -> RibBuildConfig:
    config_str = f"""
    exp_name: test
    out_dir: null
    node_layers:
        - layers.0
        - layers.1
        - layers.2
        - output
    modular_mlp_config:
        n_hidden_layers: 2
        width: 10
        weight_variances: [1,1]
        weight_equal_columns: false
        bias: 0
        activation_fn: relu
    dataset:
        dataset_type: block_vector
        size: 1000
        length: 10
        data_variances: [1,1]
        data_perfect_correlation: false
    seed: 123
    batch_size: 256
    n_intervals: 0
    truncation_threshold: 1e-15
    dtype: float64
    rotate_final_node_layer: false
    basis_formula: (1-0)*alpha
    edge_formula: squared
    """
    config_dict = deep_update(yaml.safe_load(config_str), *updates)
    config = RibBuildConfig(**config_dict)
    return config


def assert_is_close(a, b, atol, rtol, **kwargs):
    """Customized version of torch.testing.assert_close(). **kwargs added in msg output"""
    kwargs_str = "\n".join([f"{k}={v}" for k, v in kwargs.items()])
    msg_modifier = lambda m: m + "\n" + kwargs_str
    b = torch.as_tensor(b)
    b = torch.broadcast_to(b, a.shape)
    assert_close(
        a, b, atol=atol, rtol=rtol, msg=msg_modifier, check_device=False, check_dtype=False
    )


def assert_is_ones(tensor, atol, **kwargs):
    """Assert that all elements of a tensor are 1. **kwargs added in msg output"""
    assert_is_close(tensor, 1.0, atol=atol, rtol=0, **kwargs)


def assert_is_zeros(tensor, atol, **kwargs):
    """Assert that all elements of a tensor are 0. **kwargs added in msg output"""
    assert_is_close(tensor, 0.0, atol=atol, rtol=0, **kwargs)


def _assignment_permutations(sim: torch.Tensor) -> tuple[list[int], list[int]]:
    """Return the indices of an assignment between rows and cols using a greedy algorithm.

    For each column in order chooses the maximal row that hasn't been chosen yet.
    Will return lists of even length, equal to the minimium of the number of rows and columns.

    A replacement for `scipy.optimize.linear_sum_assignment` without a scipy dependancy."""
    rows_selected = torch.zeros(sim.shape[0], dtype=torch.bool)
    row_idxs = []
    col_idxs = []
    for col_idx in range(min(sim.shape)):
        masked_col = torch.where(rows_selected, -torch.inf, sim[:, col_idx])
        row_idx = masked_col.argmax().item()
        row_idxs.append(row_idx)
        col_idxs.append(col_idx)
        rows_selected[row_idx] = True
    return row_idxs, col_idxs


def assert_basis_similarity(
    ir_A: InteractionRotation, ir_B: InteractionRotation, error: Optional[float] = 0.02
):
    """
    Compare two InteractionRotations and assert similarity, allowing for permutations.

    Returns:
        dir_sims: cosine similarities of the permuted basis vectors
        dir_norm_ratios: the ratio of basis vector norms
        lambda_ratios: the ratio of lambda values for the basis directions
    """
    assert ir_A.node_layer == ir_B.node_layer
    if ir_A.C is None:
        assert ir_B.C is None
        return None, None, None
    sim = cosine_similarity(ir_A.C.unsqueeze(1), ir_B.C.unsqueeze(2), dim=0).abs()
    a_order, b_order = _assignment_permutations(sim)
    dir_sims = sim[a_order, b_order]
    dir_norm_ratios = torch.norm(ir_A.C, dim=0)[a_order] / torch.norm(ir_B.C, dim=0)[b_order]
    lambda_ratios = ir_A.Lambda[a_order] / ir_B.Lambda[b_order]
    if error is not None:
        assert_is_ones(dir_sims.mean(), atol=error, node_layer=ir_A.node_layer)
        assert_is_zeros(dir_sims.std(), atol=error, node_layer=ir_A.node_layer)
        assert_is_ones(dir_norm_ratios.mean(), atol=error, node_layer=ir_A.node_layer)
        assert_is_zeros(dir_norm_ratios.std(), atol=error, node_layer=ir_A.node_layer)
        assert_is_ones(lambda_ratios.mean(), atol=error, node_layer=ir_A.node_layer)
        assert_is_zeros(lambda_ratios.std(), atol=error, node_layer=ir_A.node_layer)
    return dir_sims, dir_norm_ratios, lambda_ratios
