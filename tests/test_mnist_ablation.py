"""Run the mnist orthog ablation script and check the below properties:

1. There are accuracies for each layer in the config.
2. There are accuracies listed for the expected number of ablated vectors (as per
   `ablate_every_vec_cutoff`)
3. The accuracy of ablating 0 vectors is higher than ablating all vectors.
"""

import sys
import tempfile
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest

# Append the root directory to sys.path
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

from experiments.mnist_orthog_ablation.run_orthog_ablations import (
    main as orthog_ablations_main,
)
from experiments.mnist_orthog_ablation.run_orthog_ablations import (
    run_ablations as run_orthog_ablations,
)
from experiments.mnist_rib_ablation.run_rib_ablations import main as rib_ablations_main
from experiments.mnist_rib_ablation.run_rib_ablations import (
    run_ablations as run_rib_ablations,
)
from rib.utils import load_config


def mock_load_config_orthog(*args, **kwargs):
    # Load the config as normal but set the mlp_path using a relative path
    config = load_config(*args, **kwargs)
    config.mlp_path = (
        Path(__file__).parent.parent
        / "experiments/train_mnist/sample_checkpoints/lr-0.001_bs-64_2023-08-13_16-23-59/model_epoch_3.pt"
    )
    return config


def mock_load_config_rib(*args, **kwargs):
    # Load the config as normal but set the mlp_path using a relative path
    config = load_config(*args, **kwargs)
    config.interaction_graph_path = (
        Path(__file__).parent.parent
        / "experiments/mnist_rib_ablation/sample_graphs/3-node-layers_interaction_graph_sample.pt"
    )
    return config


def mock_run_ablations_orthog(*args, **kwargs):
    # Call the original function to get the real ablation results
    accuracies = run_orthog_ablations(*args, **kwargs)
    mock_run_ablations_orthog.accuracies = accuracies  # Store the accuracies on the mock itself
    return accuracies


def mock_run_ablations_rib(*args, **kwargs):
    # Call the original function to get the real ablation results
    accuracies = run_rib_ablations(*args, **kwargs)
    mock_run_ablations_rib.accuracies = accuracies  # Store the accuracies on the mock itself
    return accuracies


def ablation_mock_run(
    mock_config: str,
    run_ablations_path: str,
    load_config_path: str,
    mock_load_config_fn: Callable,
    mock_run_ablations_fn: Callable,
    mock_main_fn: Callable,
    expected_layer_1_schedule: list[int],
):
    # Create a temporary file and write the mock config to it
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml") as temp_config:
        temp_config.write(mock_config)
        temp_config.flush()

        with patch(
            run_ablations_path,
            side_effect=mock_run_ablations_fn,
        ), patch(
            load_config_path,
            side_effect=mock_load_config_fn,
        ):
            # Call the main function
            mock_main_fn(temp_config.name)

            # Check that there are accuracies for both layers
            assert (
                len(mock_run_ablations_fn.accuracies) == 2
            ), "We don't have results for the layers specified in the config"

            # Check that layers.1 has the correct ablation schedule
            layer_1_schedule = sorted(
                list(mock_run_ablations_fn.accuracies["layers.1"].keys()), reverse=True
            )
            assert (
                layer_1_schedule == expected_layer_1_schedule
            ), "layers.1 does not have the expected ablation schedule"

            # Check that ablating 0 vectors performs better than ablating all vectors
            n_vecs = max(layer_1_schedule)
            assert (
                mock_run_ablations_fn.accuracies["layers.1"][0]
                > mock_run_ablations_fn.accuracies["layers.1"][n_vecs]
            )


@pytest.mark.slow
def test_run_orthog_ablations():
    mock_orthog_config = """
    exp_name: null  # Prevent saving output
    mlp_path: OVERWRITE/IN/MOCK
    ablate_every_vec_cutoff: 2
    module_names:
        - layers.1
        - layers.2
    """

    ablation_mock_run(
        mock_config=mock_orthog_config,
        run_ablations_path="experiments.mnist_orthog_ablation.run_orthog_ablations.run_ablations",
        load_config_path="experiments.mnist_orthog_ablation.run_orthog_ablations.load_config",
        mock_load_config_fn=mock_load_config_orthog,
        mock_run_ablations_fn=mock_run_ablations_orthog,
        mock_main_fn=orthog_ablations_main,
        expected_layer_1_schedule=[101, 100, 99, 98, 96, 92, 84, 68, 36, 0],
    )


@pytest.mark.slow
def test_run_rib_ablations():
    """Test our ablation criteria for RIB.

    Unlike the test_run_orthog_ablations test, this test takes as input a precomputed graph path
    so that we have access to the rotation matrix for each layer. These rotation matrices assume
    that basis vectors with very small values have already been removed. I.e. the shape of our
    interaction matrices may be (101, 90), meaning that 11 nodes have already been removed.

    This means that our ablation schedule will depend on the number of nodes remaining in the graph,
    and not the layer size. In this test, layers.1 has 100 nodes remaining and layers.2 has 94 nodes
    remaining.
    """
    mock_rib_config = """
    exp_name: null  # Prevent saving output
    interaction_graph_path: OVERWRITE/IN/MOCK
    ablate_every_vec_cutoff: 2
    module_names:
        - layers.1  # 100 non-zero basis vectors remaining in the graph
        - layers.2  # 94 non-zero basis vectors remaining in the graph
    batch_size: 64
    """

    ablation_mock_run(
        mock_config=mock_rib_config,
        run_ablations_path="experiments.mnist_rib_ablation.run_rib_ablations.run_ablations",
        load_config_path="experiments.mnist_rib_ablation.run_rib_ablations.load_config",
        mock_load_config_fn=mock_load_config_rib,
        mock_run_ablations_fn=mock_run_ablations_rib,
        mock_main_fn=rib_ablations_main,
        expected_layer_1_schedule=[100, 99, 98, 97, 95, 91, 83, 67, 35, 0],
    )
