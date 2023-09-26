"""Test various properties of an lm interaction graph.

Properties tested:
1. The size of the sum of activations in the interaction basis (calculated with
    (C.T @ gram_matrices[module_name] @ C).diag()) is equal to both the outgoing edges of a node
    (calculated E_hats[i].sum(0).abs()), and the absolute sorted Lambdas of that layer.
2. The output rotation (C) is an identity matrix (i.e. it should be the eigenspace)
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

# Append the root directory to sys.path
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR))


from experiments.lm_rib_build.lm_build_rib_graph import main as build_graph_main
from rib.interaction_algos import build_sorted_lambda_matrices
from rib.utils import load_config

MOCK_CONFIG = """
exp_name: test
seed: 0
tlens_pretrained: null
tlens_model_path: OVERWRITE/IN/MOCK
dataset: modular_arithmetic
batch_size: 16
truncation_threshold: 1e-20
rotate_output: false
last_pos_only: true
dtype: float32
node_layers:
  - ln1.0
  - mlp_in.0
  - unembed
"""


def mock_load_config(*args, **kwargs):
    # Load the config as normal but set the mlp_path using a relative path
    config = load_config(*args, **kwargs)
    config.tlens_model_path = (
        Path(__file__).parent.parent
        / "experiments/train_modular_arithmetic/sample_checkpoints/lr-0.001_bs-10000_2023-08-31_09-33-30/model_epoch_60000.pt"
    )
    return config


@pytest.mark.slow
def test_modular_arithmetic_build_graph():
    Lambda_sqrts: list[torch.Tensor] = []

    def mock_build_sorted_lambda_matrices(Lambda_diag_abs_sqrt, **kwargs):
        # Call the original function to get the real lambdas
        Lambda_sqrts.append(Lambda_diag_abs_sqrt.cpu())
        return build_sorted_lambda_matrices(Lambda_diag_abs_sqrt, **kwargs)

    # Create a temporary file and write the mock config to it
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml") as temp_config:
        temp_config.write(MOCK_CONFIG)
        temp_config.flush()

        with patch("torch.save"), patch(
            "experiments.lm_rib_build.lm_build_rib_graph.load_config",
            side_effect=mock_load_config,
        ), patch(
            "rib.interaction_algos.build_sorted_lambda_matrices",
            side_effect=mock_build_sorted_lambda_matrices,
        ):
            results = build_graph_main(temp_config.name)
            grams = results["gram_matrices"]
            Cs = results["interaction_rotations"]
            E_hats = results["edges"]

            # Square, sort, and reverse the order of the lambda_sqrts
            Lambdas = [
                torch.sort(Lambda_sqrt**2, descending=True).values
                for Lambda_sqrt in Lambda_sqrts[::-1]
            ]
            # Check that the output layer rotation is an identity matrix
            assert torch.allclose(
                Cs[-1]["C"],
                torch.eye(Cs[-1]["C"].shape[0], device=Cs[-1]["C"].device, dtype=Cs[-1]["C"].dtype),
            )

            for i, module_name in enumerate(edge_info[0] for edge_info in E_hats):
                # Check that the size of the sum of activations in the interaction basis is equal
                # to the outgoing edges of a node
                act_size = (Cs[i]["C"].T @ grams[module_name] @ Cs[i]["C"]).diag()
                edge_size = E_hats[i][1].sum(0).abs()
                assert torch.allclose(
                    act_size, edge_size, atol=1e-2
                ), f"act_size not equal to edge_size for {module_name}"
                # Check that the Lambdas are also the same
                assert torch.allclose(
                    act_size, Lambdas[i], atol=1e-2
                ), f"act_size not equal to Lambdas for {module_name}"
