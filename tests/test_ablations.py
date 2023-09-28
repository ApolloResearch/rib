"""Run the mnist orthog and rib ablation scripts and check the below properties:

1. The accuracy when no vectors are ablated is higher than a set threshold (e.g. 95%)
2. Ablating all vectors gives an accuracy lower than 50% (arbitrarily chosen)
3. There are accuracies for all ablated vectors.
4. The accuracies are sorted roughly in descending order of the number of ablated vectors.

This is currently very hacky. In particular, for the rib ablation script we need to mock
torch.load to return an interaction graph with an updated MLP path. This is necessary because the
interaction graph is saved with an absolute path to the MLP, and a github action will not have
access to the same absolute path.


"""

import sys
import tempfile
from pathlib import Path
from typing import Callable, Union
from unittest.mock import patch

import pytest
import torch

# Append the root directory to sys.path
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

from experiments.lm_orthog_ablation.run_lm_orthog_ablations import (
    main as lm_orthog_ablations_main,
)
from experiments.lm_orthog_ablation.run_lm_orthog_ablations import (
    run_ablations as run_lm_orthog_ablations,
)
from experiments.lm_rib_ablation.run_lm_rib_ablations import (
    main as lm_rib_ablations_main,
)
from experiments.lm_rib_ablation.run_lm_rib_ablations import (
    run_ablations as run_lm_rib_ablations,
)
from experiments.mnist_orthog_ablation.run_orthog_ablations import (
    main as mnist_orthog_ablations_main,
)
from experiments.mnist_orthog_ablation.run_orthog_ablations import (
    run_ablations as run_mnist_orthog_ablations,
)
from experiments.mnist_rib_ablation.run_rib_ablations import (
    main as mnist_rib_ablations_main,
)
from experiments.mnist_rib_ablation.run_rib_ablations import (
    run_ablations as run_minst_rib_ablations,
)
from rib.utils import load_config


def mock_load_config_mnist_orthog(*args, **kwargs):
    # Load the config as normal but set the mlp_path using a relative path
    config = load_config(*args, **kwargs)
    config.mlp_path = (
        Path(__file__).parent.parent
        / "experiments/train_mnist/sample_checkpoints/lr-0.001_bs-64_2023-08-13_16-23-59/model_epoch_3.pt"
    )
    return config


def mock_load_config_mnist_rib(*args, **kwargs):
    # Load the config as normal but set the interaction_graph_path using a relative path
    config = load_config(*args, **kwargs)
    config.interaction_graph_path = (
        Path(__file__).parent.parent
        / "experiments/mnist_rib_build/sample_graphs/3-node-layers_interaction_graph_sample.pt"
    )
    return config


def mock_load_config_lm_rib(*args, **kwargs):
    # Load the config as normal but set the interaction_graph_path using a relative path
    config = load_config(*args, **kwargs)
    config.interaction_graph_path = (
        Path(__file__).parent.parent
        / "experiments/lm_rib_build/sample_graphs/modular_arithmetic_interaction_graph_sample.pt"
    )
    return config


def mock_load_config_lm_orthog(*args, **kwargs):
    # Load the config as normal but set the mlp_path using a relative path
    config = load_config(*args, **kwargs)
    config.tlens_model_path = (
        Path(__file__).parent.parent
        / "experiments/train_modular_arithmetic/sample_checkpoints/lr-0.001_bs-10000_norm-None_2023-09-27_18-19-33/model_epoch_60000.pt"
    )
    return config


def mock_run_ablations_mnist_orthog(*args, **kwargs):
    # Call the original function to get the real ablation results
    accuracies = run_mnist_orthog_ablations(*args, **kwargs)
    mock_run_ablations_mnist_orthog.accuracies = (
        accuracies  # Store the accuracies on the mock itself
    )
    return accuracies


def mock_run_ablations_mnist_rib(*args, **kwargs):
    # Call the original function to get the real ablation results
    accuracies = run_minst_rib_ablations(*args, **kwargs)
    mock_run_ablations_mnist_rib.accuracies = accuracies  # Store the accuracies on the mock itself
    return accuracies


def mock_run_ablations_lm_rib(*args, **kwargs):
    # Call the original function to get the real ablation results
    accuracies = run_lm_rib_ablations(*args, **kwargs)
    mock_run_ablations_lm_rib.accuracies = accuracies  # Store the accuracies on the mock itself
    return accuracies


def mock_run_ablations_lm_orthog(*args, **kwargs):
    # Call the original function to get the real ablation results
    accuracies = run_lm_orthog_ablations(*args, **kwargs)
    mock_run_ablations_lm_orthog.accuracies = accuracies  # Store the accuracies on the mock itself
    return accuracies


original_torch_load = torch.load


def mock_torch_load(*args, **kwargs):
    """Mock torch.load to return an interaction graph with an updated MLP path.

    This is necessary because the interaction graph is saved with an absolute path to the MLP, and
    a github action will not have access to the same absolute path.

    This is especially hacky because torch load gets called multiple times, and we only update the
    mlp_path on calls which return a dictionary with a config and mlp_path.
    """
    # Call the original function to get the real ablation results
    interaction_graph_info = original_torch_load(*args, **kwargs)
    # If the load outputs a dictionary with a config, set the mlp_path using a relative path
    if "config" in interaction_graph_info:
        # MNIST
        if "mlp_path" in interaction_graph_info["config"]:
            interaction_graph_info["config"]["mlp_path"] = (
                Path(__file__).parent.parent
                / "experiments/train_mnist/sample_checkpoints/lr-0.001_bs-64_2023-08-13_16-23-59/model_epoch_3.pt"
            )
    return interaction_graph_info


def _is_roughly_sorted(lst: list[Union[int, float]], k: int = 1):
    """
    Check if a list is roughly sorted within a tolerance of k out-of-order pairs.

    Args:
        - The list to check.
        - The number of out-of-order pairs to tolerate.

    Returns:
        - True if the list is roughly sorted, otherwise False.
    """

    count_out_of_order = 0
    for i in range(len(lst) - 1):
        if lst[i] > lst[i + 1]:
            count_out_of_order += 1
            if count_out_of_order > k:
                return False
    return True


def ablation_mock_run(
    mock_config: str,
    script_path: str,
    mock_load_config_fn: Callable,
    mock_run_ablations_fn: Callable,
    mock_main_fn: Callable,
    layer_keys: list[str],
    max_accuracy_threshold: float,
    sort_tolerance: int = 10,
) -> None:
    """Run the ablation script with a mock config and check the results.

    Args:
        mock_config: The mock config to use.
        script_path: The path to the run ablation script
        mock_load_config_fn: The function to mock load_config with.
        mock_run_ablations_fn: The function to mock run_ablations with.
        mock_main_fn: The function to mock main with.
        layer_keys: The keys to check for in the accuracies dictionary.
        max_accuracy_threshold: Lower bound on accuracy to expect with 0 ablated vectors.
        sort_tolerance: The number of out-of-order pairs to tolerate when checking if the
            accuracies are roughly sorted.
    """
    # Create a temporary file and write the mock config to it
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml") as temp_config:
        temp_config.write(mock_config)
        temp_config.flush()

        with patch(
            script_path + ".run_ablations",
            side_effect=mock_run_ablations_fn,
        ), patch(
            script_path + ".load_config",
            side_effect=mock_load_config_fn,
        ), patch(
            script_path + ".torch.load",
            side_effect=mock_torch_load,
        ):
            # Call the main function
            mock_main_fn(temp_config.name)

            # Check that there are accuracies returned
            assert list(mock_run_ablations_fn.accuracies.keys()) == layer_keys, (
                f"Expected accuracies for {layer_keys}, but got "
                f"{list(mock_run_ablations_fn.accuracies.keys())}"
            )

            for layer_key in layer_keys:
                ablated_vecs = list(mock_run_ablations_fn.accuracies[layer_key].keys())
                accuracies = list(mock_run_ablations_fn.accuracies[layer_key].values())

                # Check that the accuracies are ordered by their number of ablated vectors
                assert ablated_vecs == sorted(ablated_vecs, reverse=True)

                # Check that ablating 0 vectors gives at least max_accuracy_threshold
                assert accuracies[-1] >= max_accuracy_threshold

                # Check that ablating all vectors gives less than 50% accuracy (arbitrarily chosen)
                assert accuracies[0] < 0.5

                # Check that the accuracies are sorted in descending order of the number of ablated
                # vectors
                assert _is_roughly_sorted(accuracies, k=sort_tolerance)


@pytest.mark.slow
def test_run_mnist_orthog_ablations():
    """Test various ablation result properties for orthogonal ablations on MNIST."""
    mock_orthog_config = """
    exp_name: null  # Prevent saving output
    mlp_path: OVERWRITE/IN/MOCK
    ablate_every_vec_cutoff: 2
    dtype: float32
    module_names:
        - layers.1
        - layers.2
    seed: 0
    """

    ablation_mock_run(
        mock_config=mock_orthog_config,
        script_path="experiments.mnist_orthog_ablation.run_orthog_ablations",
        mock_load_config_fn=mock_load_config_mnist_orthog,
        mock_run_ablations_fn=mock_run_ablations_mnist_orthog,
        mock_main_fn=mnist_orthog_ablations_main,
        layer_keys=["layers.1", "layers.2"],
        max_accuracy_threshold=0.95,  # Model should converge to at least 95% accuracy
    )


@pytest.mark.slow
def test_run_mnist_rib_ablations():
    """Test various ablation result properties for RIB on MNIST.

    Unlike the test_run_orthog_ablations test, this test takes as input a precomputed graph path
    so that we have access to the rotation matrix for each layer. These rotation matrices assume
    that basis vectors with very small values have already been removed. I.e. the shape of our
    interaction matrices may be (101, 90), meaning that 11 nodes have already been removed.

    This means that our ablation schedule will depend on the number of nodes remaining in the graph,
    and not the layer size. In this test, layers.1 has 100 nodes remaining and layers.2 has 94 nodes
    remaining.

    We also need to modify the mlp_path that is a value in the dictionary in interaction_graph_path.
    To do this, we mock torch.load and replace the mlp_path with a relative path to a checkpoint.
    """
    mock_rib_config = """
    exp_name: null  # Prevent saving output
    interaction_graph_path: OVERWRITE/IN/MOCK
    ablate_every_vec_cutoff: 2
    dtype: float32
    module_names:
        - layers.1  # 100 non-zero basis vectors remaining in the graph
        - layers.2  # 94 non-zero basis vectors remaining in the graph
    batch_size: 64
    seed: 0
    """

    ablation_mock_run(
        mock_config=mock_rib_config,
        script_path="experiments.mnist_rib_ablation.run_rib_ablations",
        mock_load_config_fn=mock_load_config_mnist_rib,
        mock_run_ablations_fn=mock_run_ablations_mnist_rib,
        mock_main_fn=mnist_rib_ablations_main,
        layer_keys=["layers.1", "layers.2"],
        max_accuracy_threshold=0.95,  # Model should converge to at least 95% accuracy
    )


@pytest.mark.slow
def test_run_modular_arithmetic_rib_ablations():
    """Test various ablation result properties for RIB on modular arithmetic."""

    mock_rib_config = """
    exp_name: null  # Prevent saving output
    interaction_graph_path: OVERWRITE/IN/MOCK
    ablate_every_vec_cutoff: 2
    node_layers:
        - ln1.0
        - mlp_in.0
        - unembed
    batch_size: 64
    dtype: float32
    seed: 0
    """

    ablation_mock_run(
        mock_config=mock_rib_config,
        script_path="experiments.lm_rib_ablation.run_lm_rib_ablations",
        mock_load_config_fn=mock_load_config_lm_rib,
        mock_run_ablations_fn=mock_run_ablations_lm_rib,
        mock_main_fn=lm_rib_ablations_main,
        layer_keys=["ln1.0", "mlp_in.0", "unembed"],
        max_accuracy_threshold=0.998,  # Model should converge to 100% accuracy
    )


@pytest.mark.slow
def test_run_modular_arithmetic_orthog_ablations():
    """Test various ablation result properties for orthogonal ablations on modular arithmetic."""

    mock_orthog_config = """
    exp_name: null  # Prevent saving output
    tlens_pretrained: null
    tlens_model_path: OVERWRITE/IN/MOCK
    dataset: modular_arithmetic
    batch_size: 16
    last_pos_only: true
    ablate_every_vec_cutoff: 10
    dtype: float32
    node_layers:
        - ln1.0
        - mlp_in.0
        - unembed
    seed: 0
    """

    ablation_mock_run(
        mock_config=mock_orthog_config,
        script_path="experiments.lm_orthog_ablation.run_lm_orthog_ablations",
        mock_load_config_fn=mock_load_config_lm_orthog,
        mock_run_ablations_fn=mock_run_ablations_lm_orthog,
        mock_main_fn=lm_orthog_ablations_main,
        layer_keys=["ln1.0", "mlp_in.0", "unembed"],
        max_accuracy_threshold=0.998,  # Model should converge to 100% accuracy
    )
