"""Run the mnist orthog ablation script and check the below properties:

1. There are accuracies for each layer in the config.
2. There are accuracies listed for the expected number of ablated vectors (as per
   `ablate_every_vec_cutoff`)
3. The accuracy of ablating 0 vectors is higher than ablating all vectors.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Append the root directory to sys.path
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

from experiments.mnist_orthog_ablation.run_orthog_ablations import (
    main as ablations_main,
)
from experiments.mnist_orthog_ablation.run_orthog_ablations import run_ablations
from rib.utils import load_config

MOCK_CONFIG = """
exp_name: null  # Prevent saving output
mlp_path: OVERWRITE/IN/MOCK
ablate_every_vec_cutoff: 2
module_names:
    - layers.1
    - layers.2
"""


def mock_load_config(*args, **kwargs):
    # Load the config as normal but set the mlp_path using a relative path
    config = load_config(*args, **kwargs)
    config.mlp_path = (
        Path(__file__).parent.parent
        / "experiments/train_mnist/sample_checkpoints/lr-0.001_bs-64_2023-08-13_16-23-59/model_epoch_3.pt"
    )
    return config


def mock_run_ablations(*args, **kwargs):
    # Call the original function to get the real ablation results
    accuracies = run_ablations(*args, **kwargs)
    mock_run_ablations.accuracies = accuracies  # Store the accuracies on the mock itself
    return accuracies


@pytest.mark.slow
def test_run_ablations():
    # Create a temporary file and write the mock config to it
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml") as temp_config:
        temp_config.write(MOCK_CONFIG)
        temp_config.flush()

        with patch(
            "experiments.mnist_orthog_ablation.run_orthog_ablations.run_ablations",
            side_effect=mock_run_ablations,
        ), patch(
            "experiments.mnist_orthog_ablation.run_orthog_ablations.load_config",
            side_effect=mock_load_config,
        ):
            # Call the main function
            ablations_main(temp_config.name)

            # Check that there are accuracies for both layers
            assert (
                len(mock_run_ablations.accuracies) == 2
            ), "We don't have results for the layers specified in the config"

            # Check that layers.1 has the correct ablation schedule
            layer_1_schedule = sorted(
                list(mock_run_ablations.accuracies["layers.1"].keys()), reverse=True
            )
            expected_schedule = [101, 100, 99, 98, 96, 92, 84, 68, 36, 0]
            assert (
                layer_1_schedule == expected_schedule
            ), "layers.1 does not have the expected ablation schedule"

            # Check that ablating 0 vectors performs better than ablating all vectors
            n_vecs = max(layer_1_schedule)
            assert (
                mock_run_ablations.accuracies["layers.1"][0]
                > mock_run_ablations.accuracies["layers.1"][n_vecs]
            )
