"""Test various properties of an lm and mnist interaction graph.

Properties tested:
1. The size of the sum of activations in the interaction basis (calculated with
    (C.T @ gram_matrices[module_name] @ C).diag()) is equal to both the outgoing edges of a node
    (calculated E_hats[i].sum(0).abs()), and the absolute sorted Lambdas of that layer.
2. The output rotation (C) is an identity matrix (i.e. it should be the eigenspace)

Note that, when comparing tensors, we normalize by their max value to account for tensors of
various scales. This is because combining atol and rtol does not work particularly well for tensors
that have a small set of large numbers and a large set of small numbers.
"""

import sys
from pathlib import Path
from typing import Callable, Union
from unittest.mock import patch

import pytest
import torch
import yaml

# Append the root directory to sys.path
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

from experiments.lm_rib_build.run_lm_rib_build import Config as LMRibConfig
from experiments.lm_rib_build.run_lm_rib_build import main as lm_build_graph_main
from experiments.mlp_rib_build.run_mlp_rib_build import Config as MlpRibConfig
from experiments.mlp_rib_build.run_mlp_rib_build import main as mlp_build_graph_main
from rib.interaction_algos import build_sorted_lambda_matrices


def build_get_lambdas(config: Union[LMRibConfig, MlpRibConfig], build_graph_main_fn: Callable):
    """Build the graph but extracting the lambdas"""
    Lambda_abs: list[torch.Tensor] = []

    def mock_build_sorted_lambda_matrices(Lambda_abs_arg, *args, **kwargs):
        # Call the original function to get the real lambdas
        Lambda_abs.append(Lambda_abs_arg.cpu())
        return build_sorted_lambda_matrices(Lambda_abs_arg, *args, **kwargs)

    with patch(
        "rib.interaction_algos.build_sorted_lambda_matrices",
        side_effect=mock_build_sorted_lambda_matrices,
    ):
        results = build_graph_main_fn(config)

    # Sort each row, and reverse the order of the Lambda_abs
    Lambdas = [torch.sort(lambda_row, descending=True).values for lambda_row in Lambda_abs[::-1]]

    return results, Lambdas


def graph_build_test(
    config: Union[LMRibConfig, MlpRibConfig],
    build_graph_main_fn: Callable,
    atol: float,
):
    results, Lambdas = build_get_lambdas(config, build_graph_main_fn)

    grams = results["gram_matrices"]
    Cs = results["interaction_rotations"]
    E_hats = results["edges"]

    # The output interaction matrix should be None if rotate_final_node_layer is False
    if not config.rotate_final_node_layer:
        assert (
            Cs[-1]["C"] is None
        ), "The output interaction matrix should be None if rotate_final_node_layer is False"

    # We don't have edges or lambdas for the final layer in node_layers
    comparison_layers = config.node_layers[:-1]
    for i, module_name in enumerate(comparison_layers):
        # Get the module names from the grams
        act_size = (Cs[i]["C"].T @ grams[module_name] @ Cs[i]["C"]).diag()
        if E_hats:
            # E_hats[i] is a tuple (name, tensor)
            if config.edge_formula == "squared":
                assert (E_hats[i][1] >= 0).all(), f"edges not >= 0 for {module_name}"
            assert (E_hats[i][1] != 0).any(), f"edges all zero for {module_name}"
            if config.edge_formula == "functional" and config.basis_formula == "(1-alpha)^2":
                # Check that the size of the sum of activations in the interaction basis is equal
                # to the outgoing edges of a node. The relation should hold only in this one config
                # case.
                edge_size = E_hats[i][1].sum(0).abs()
                assert (
                    act_size.shape == edge_size.shape
                ), f"act_size and edge_size not same shape for {module_name}"
                assert torch.allclose(
                    act_size / act_size.abs().max(),
                    edge_size / edge_size.abs().max(),
                    atol=atol,
                ), f"act_size not equal to edge_size for {module_name}"

        # Check that the Lambdas are also the same as the act_size and edge_size
        # Note that the Lambdas need to be truncated to edge_size/act_size (this happens in
        # `rib.interaction_algos.build_sort_lambda_matrix)
        Lambdas_trunc = Lambdas[i][: len(act_size)]
        assert torch.allclose(
            act_size / act_size.abs().max(),
            Lambdas_trunc / Lambdas_trunc.max(),
            atol=atol,
        ), f"act_size not equal to Lambdas for {module_name}"


@pytest.mark.slow
@pytest.mark.parametrize(
    "basis_formula, edge_formula",
    [
        ("(1-alpha)^2", "functional"),
        ("(1-0)*alpha", "functional"),
        ("(1-alpha)^2", "squared"),
        ("(1-0)*alpha", "squared"),
    ],
)
def test_modular_arithmetic_build_graph(basis_formula, edge_formula):
    dtype_str = "float32"
    atol = 1e-5  # Works with 1e-7 for float32 and 1e-12 for float64. NEED 1e-5 for CPU

    config_str = f"""
    exp_name: test
    seed: 0
    tlens_pretrained: null
    tlens_model_path: experiments/train_modular_arithmetic/sample_checkpoints/lr-0.001_bs-10000_norm-None_2023-11-28_16-07-19/model_epoch_60000.pt
    node_layers:
        - ln1.0
        - mlp_in.0
        - unembed
        - output
    dataset:
        source: custom
        name: modular_arithmetic
        return_set: train
        return_set_n_samples: 10
    batch_size: 6
    truncation_threshold: 1e-15  # we've been using 1e-6 previously but this increases needed atol
    rotate_final_node_layer: false
    last_pos_module_type: add_resid1
    n_intervals: 0
    dtype: {dtype_str}
    eval_type: accuracy
    out_dir: null
    basis_formula: "{basis_formula}"
    edge_formula: "{edge_formula}"
    """
    config_dict = yaml.safe_load(config_str)
    config = LMRibConfig(**config_dict)

    graph_build_test(config=config, build_graph_main_fn=lm_build_graph_main, atol=atol)


@pytest.mark.slow
def test_pythia_14m_build_graph():
    dtype_str = "float64"
    atol = 0  # Works with 1e-7 for float32 and 0 for float64

    config_str = f"""
    exp_name: test
    seed: 0
    tlens_pretrained: pythia-14m
    tlens_model_path: null
    dataset:
      source: huggingface
      name: NeelNanda/pile-10k
      tokenizer_name: EleutherAI/pythia-14m
      return_set: train
      return_set_frac: null
      return_set_n_samples: 10  # 10 samples gives 3x2048 tokens
      return_set_portion: first
    node_layers:
        - ln2.1
        - unembed
    batch_size: 2
    truncation_threshold: 1e-15  # we've been using 1e-6 previously but this increases needed atol
    rotate_final_node_layer: false
    n_intervals: 0
    dtype: {dtype_str}
    calculate_edges: false
    eval_type: ce_loss
    out_dir: null
    """
    config_dict = yaml.safe_load(config_str)
    config = LMRibConfig(**config_dict)

    graph_build_test(
        config=config,
        build_graph_main_fn=lm_build_graph_main,
        atol=atol,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "basis_formula, edge_formula",
    [
        ("(1-alpha)^2", "functional"),
        ("(1-0)*alpha", "functional"),
        ("(1-alpha)^2", "squared"),
        ("(1-0)*alpha", "squared"),
    ],
)
def test_mnist_build_graph(basis_formula, edge_formula):
    dtype_str = "float32"
    # Works with 1e-7 for float32 and 1e-15 (and maybe smaller) for float64. Need 1e-6 for CPU
    atol = 1e-6

    config_str = f"""
    exp_name: test
    mlp_path: "experiments/train_mlp/sample_checkpoints/lr-0.001_bs-64_2023-11-29_14-36-29/model_epoch_12.pt"
    batch_size: 256
    seed: 0
    truncation_threshold: 1e-15  # we've been using 1e-6 previously but this increases needed atol
    rotate_final_node_layer: false
    n_intervals: 0
    dtype: {dtype_str}
    node_layers:
        - layers.0
        - layers.1
        - layers.2
        - output
    dataset:
        return_set_frac: 0.01  # 3 batches (with batch_size=256)
    out_dir: null
    basis_formula: "{basis_formula}"
    edge_formula: "{edge_formula}"
    """

    config_dict = yaml.safe_load(config_str)
    config = MlpRibConfig(**config_dict)

    graph_build_test(
        config=config,
        build_graph_main_fn=mlp_build_graph_main,
        atol=atol,
    )


def rotate_final_layer_invariance(
    config_str_rotated: str,
    config_cls: Union["LMRibConfig", "MlpRibConfig"],
    build_graph_main_fn: Callable,
    rtol: float = 1e-7,
    atol: float = 0,
):
    config_str_not_rotated = config_str_rotated.replace(
        "rotate_final_node_layer: true", "rotate_final_node_layer: false"
    )

    config_rotated = config_cls(**yaml.safe_load(config_str_rotated))
    config_not_rotated = config_cls(**yaml.safe_load(config_str_not_rotated))

    edges_rotated = build_graph_main_fn(config_rotated)["edges"]
    edges_not_rotated = build_graph_main_fn(config_not_rotated)["edges"]

    # -1 has no edges, -2 is the final layer and changes
    comparison_layers = config_rotated.node_layers[:-2]
    for i, module_name in enumerate(comparison_layers):
        # E_hats[i] is a tuple (name, tensor)
        print("Comparing", module_name)
        # Check shape
        assert (
            edges_not_rotated[i][1].shape == edges_rotated[i][1].shape
        ), f"edges_not_rotated and edges_rotated not same shape for {module_name}"
        # Check values
        assert torch.allclose(
            edges_not_rotated[i][1],
            edges_rotated[i][1],
            rtol=rtol,
            atol=atol,
        ), f"edges_not_rotated not equal to shape of edges_rotated for {module_name}. Biggest relative deviation: {(edges_not_rotated[i][1] / edges_rotated[i][1]).min()}, {(edges_not_rotated[i][1] / edges_rotated[i][1]).max()}"


@pytest.mark.slow
@pytest.mark.parametrize(
    "basis_formula, edge_formula",
    [
        ("(1-alpha)^2", "functional"),
        ("(1-0)*alpha", "functional"),
        ("(1-alpha)^2", "squared"),
        ("(1-0)*alpha", "squared"),
    ],
)
def test_mnist_rotate_final_layer_invariance(basis_formula, edge_formula, rtol=1e-7, atol=1e-8):
    """Test that the non-final edges are the same for MNIST whether or not we rotate the final layer."""
    config_str_rotated = f"""
    exp_name: test
    mlp_path: experiments/train_mlp/sample_checkpoints/lr-0.001_bs-64_2023-11-29_14-36-29/model_epoch_12.pt
    batch_size: 256
    seed: 0
    truncation_threshold: 1e-6
    rotate_final_node_layer: true  # Gets overridden by rotate_final_layer_invariance
    n_intervals: 0
    dtype: float64 # in float32 the truncation changes between both runs
    dataset:
        return_set_frac: 0.01  # 3 batches (with batch_size=256)
    node_layers:
    - layers.1
    - layers.2
    - output
    out_dir: null
    basis_formula: "{basis_formula}"
    edge_formula: "{edge_formula}"
    """

    rotate_final_layer_invariance(
        config_str_rotated=config_str_rotated,
        config_cls=MlpRibConfig,
        build_graph_main_fn=mlp_build_graph_main,
        rtol=rtol,
        atol=atol,
    )


# Mod add tests are slow because return_set_n_samples is not implemented yet
@pytest.mark.xfail
@pytest.mark.slow
@pytest.mark.parametrize(
    "basis_formula, edge_formula, dtype_str",
    [
        # functional fp32 currently fails with these tolerances
        # ("(1-alpha)^2", "functional", "float32"),
        # ("(1-0)*alpha", "functional", "float32"),
        ("(1-alpha)^2", "functional", "float64"),
        ("(1-0)*alpha", "functional", "float64"),
        ("(1-alpha)^2", "squared", "float32"),
        ("(1-0)*alpha", "squared", "float32"),
        ("(1-alpha)^2", "squared", "float64"),
        ("(1-0)*alpha", "squared", "float64"),
    ],
)
def test_modular_arithmetic_rotate_final_layer_invariance(
    basis_formula,
    edge_formula,
    dtype_str,
    rtol=1e-3,
    atol=1e-3,
):
    """Test that the non-final edges are independent of final layer rotation for modadd.

    Note that atol is necessary as the less important edges do deviate. The largest edges are
    between 1e3 and 1e5 large.
    """
    config_str_rotated = f"""
    exp_name: test
    seed: 0
    tlens_pretrained: null
    tlens_model_path: experiments/train_modular_arithmetic/sample_checkpoints/lr-0.001_bs-10000_norm-None_2023-11-28_16-07-19/model_epoch_60000.pt
    dataset:
        source: custom
        name: modular_arithmetic
        return_set: train
        return_set_frac: null
        return_set_n_samples: 10
    node_layers:
        - mlp_out.0
        - unembed
        - output
    batch_size: 6
    gram_batch_size: 6
    edge_batch_size: 6
    truncation_threshold: 1e-15
    rotate_final_node_layer: true  # Gets overridden by rotate_final_layer_invariance
    last_pos_module_type: add_resid1
    n_intervals: 2
    dtype: {dtype_str}
    eval_type: accuracy
    out_dir: null
    basis_formula: "{basis_formula}"
    edge_formula: "{edge_formula}"
    """
    rotate_final_layer_invariance(
        config_str_rotated=config_str_rotated,
        config_cls=LMRibConfig,
        build_graph_main_fn=lm_build_graph_main,
        rtol=rtol,
        atol=atol,
    )


def test_mnist_build_graph_invalid_node_layers():
    """Test that non-sequential node_layers raises an error."""
    mock_config = """
    exp_name: test
    mlp_path: "experiments/train_mlp/sample_checkpoints/lr-0.001_bs-64_2023-11-29_14-36-29/model_epoch_12.pt"
    batch_size: 256
    seed: 0
    truncation_threshold: 1e-15
    rotate_final_node_layer: false
    n_intervals: 0
    dtype: float32
    node_layers:
        - layers.0
        - layers.2
    out_dir: null
    """

    config_dict = yaml.safe_load(mock_config)
    config = MlpRibConfig(**config_dict)

    with pytest.raises(AssertionError):
        graph_build_test(
            config=config,
            build_graph_main_fn=mlp_build_graph_main,
            atol=0,
        )


@pytest.mark.slow
def test_svd_basis():
    dtype_str = "float64"

    config_str = f"""
    exp_name: test
    seed: 0
    tlens_pretrained: pythia-14m
    tlens_model_path: null
    dataset:
      source: huggingface
      name: NeelNanda/pile-10k
      tokenizer_name: EleutherAI/pythia-14m
      return_set: train
      return_set_frac: null
      return_set_n_samples: 50
      return_set_portion: first
    node_layers:
        - ln2.1
        - unembed
    batch_size: 2
    truncation_threshold: 1e-15  # we've been using 1e-6 previously but this increases needed atol
    rotate_final_node_layer: false
    n_intervals: 0
    dtype: {dtype_str}
    calculate_edges: false
    eval_type: ce_loss
    out_dir: null
    basis_formula: svd
    """
    config_dict = yaml.safe_load(config_str)
    config = LMRibConfig(**config_dict)
    results = lm_build_graph_main(config)
    for c_info, u_info in zip(results["interaction_rotations"], results["eigenvectors"]):
        C = c_info["C"]
        U = u_info["U"]
        assert (C is None) == (U is None)
        if C is not None:
            assert torch.allclose(C, U, atol=0)
