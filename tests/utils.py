import torch
import yaml
from pydantic.v1.utils import deep_update
from torch.testing import assert_close

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
        return_set_n_samples: 10
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
    n_stochastic_sources: null
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
        return_set_n_documents: 20
        return_set_n_samples: 3
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
    eval_type: ce_loss
    out_dir: null
    basis_formula: (1-0)*alpha
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
