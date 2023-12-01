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
                # edges must be positive >= 0
                assert (E_hats[i][1] >= 0).all()
            # edges should not all be zero
            assert (E_hats[i][1] != 0).any()
            if config.edge_formula == "functional" and config.basis_formula == "(1-alpha)^2":
                # Check that the size of the sum of activations in the interaction basis is equal
                # to the outgoing edges of a node. The relation should hold only in this one config
                # case.
                edge_size = E_hats[i][1].sum(0).abs()
                # Test shapes
                assert (
                    act_size.shape == edge_size.shape
                ), f"act_size and edge_size not same shape for {module_name}"
                # Test sum of edges == function size
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
    batch_size: 128
    truncation_threshold: 1e-15  # we've been using 1e-6 previously but this increases needed atol
    rotate_final_node_layer: false
    last_pos_module_type: add_resid1
    n_intervals: 0
    dtype: {dtype_str}
    eval_type: accuracy
    use_analytic_integrad: false
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
    "basis_formula, edge_formula, use_analytic_integrad",
    [
        ("(1-alpha)^2", "functional", True),
        ("(1-0)*alpha", "functional", True),
        ("(1-0)*alpha", "functional", False),
        ("(1-alpha)^2", "squared", True),
        ("(1-0)*alpha", "squared", True),
        ("(1-0)*alpha", "squared", False),
    ],
)
def test_mnist_build_graph(basis_formula, edge_formula, use_analytic_integrad):
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
    use_analytic_integrad: {use_analytic_integrad}
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
