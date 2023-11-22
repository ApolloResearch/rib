"""Run the mnist orthog and rib ablation scripts and check the below properties:

1. The accuracy when no vectors are ablated is higher than a set threshold (e.g. 95%)
2. Ablating all vectors gives an accuracy lower than 50% (arbitrarily chosen)
3. There are accuracies for all ablated vectors.
4. The accuracies are sorted roughly in descending order of the number of ablated vectors.

This is currently quite hacky. In particular, we mock torch.load to return an interaction graph
with an updated MLP path. This is necessary because the interaction graph is saved with an
absolute path to the MLP, and a github action will not have access to the same absolute path.
"""

import sys
from pathlib import Path
from typing import Union

import pytest
import yaml

from rib.ablations import AblationAccuracies

# Append the root directory to sys.path
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

from experiments.lm_ablations.run_lm_ablations import Config as LMAblationConfig
from experiments.lm_ablations.run_lm_ablations import main as lm_ablations_main
from experiments.mnist_ablations.run_mnist_ablations import (
    Config as MNISTAblationConfig,
)
from experiments.mnist_ablations.run_mnist_ablations import main as mnist_ablations_main
from rib.loader import load_mlp


def _is_roughly_sorted(lst: list[Union[int, float]], k: int = 1, reverse: bool = False) -> bool:
    """
    Check if a list is roughly sorted within a tolerance of k out-of-order pairs.

    Args:
        lst: The list to check.
        k: The number of out-of-order pairs to tolerate.
        reverse: If True, check that the list is roughly sorted in descending order.


    Returns:
        - True if the list is roughly sorted, otherwise False.
    """

    if reverse:
        lst = lst[::-1]
    count_out_of_order = 0
    for i in range(len(lst) - 1):
        if lst[i] > lst[i + 1]:
            count_out_of_order += 1
            if count_out_of_order > k:
                return False
    return True


def check_accuracies(
    accuracies: AblationAccuracies,
    config: Union[MNISTAblationConfig, LMAblationConfig],
    max_accuracy_threshold: float,
    sort_tolerance: int = 10,
) -> None:
    """Run the ablation script with a mock config and check the results.

    Args:
        script_path: The path to the run ablation script
        mock_load_config_fn: The function to mock load_config with.
        mock_main_fn: The function to mock main with.
        layer_keys: The keys to check for in the accuracies dictionary.
        max_accuracy_threshold: Lower bound on accuracy to expect with 0 ablated vectors.
        sort_tolerance: The number of out-of-order pairs to tolerate when checking if the
            accuracies are roughly sorted.
        specific_points: The specific ablation points to check for.
    """
    # Check that there are accuracies returned
    assert list(accuracies.keys()) == config.ablation_node_layers, (
        f"Expected accuracies for {config.ablation_node_layers}, but got "
        f"{list(accuracies.keys())}"
    )
    for layer_key in config.ablation_node_layers:
        vecs_remaining = list(accuracies[layer_key].keys())
        accuracy_vals = list(accuracies[layer_key].values())

        if config.schedule.specific_points is not None:
            for point in config.schedule.specific_points:
                assert (
                    point in vecs_remaining
                ), f"Expected specific point {point} in vecs remaining, but it isn't there"

        # Check that the accuracies are ordered by their number of ablated vectors
        assert vecs_remaining == sorted(vecs_remaining, reverse=True)

        # Check that ablating 0 vectors gives at least max_accuracy_threshold
        assert accuracy_vals[0] >= max_accuracy_threshold

        if config.schedule.early_stopping_threshold is None:
            # This means the final accuracy_val corresponds to all vecs ablated.
            # Check that this is < 50%
            assert accuracy_vals[-1] < 0.5
        else:
            # Check that the run which ablated the most vectors is all vectors is at least
            # early_stopping_threshold worse than the max accuracy
            assert accuracy_vals[0] - accuracy_vals[-1] >= config.schedule.early_stopping_threshold

        # Check that the accuracies are sorted in descending order of the number of ablated
        # vectors
        assert _is_roughly_sorted(accuracy_vals, k=sort_tolerance, reverse=True)


@pytest.mark.slow
@pytest.mark.parametrize("ablation_type", ["orthogonal", "rib"])
def test_run_mnist_ablations(ablation_type):
    """Test various ablation result properties for orthogonal ablations on MNIST."""
    config_str = f"""
    exp_name: null  # Prevent saving output
    ablation_type: {ablation_type}
    interaction_graph_path: experiments/mnist_rib_build/sample_graphs/4-node-layers_rib_graph_sample.pt
    schedule:
        schedule_type: exponential
        early_stopping_threshold: 0.05
        ablate_every_vec_cutoff: 2
        exp_base: 4.0
    dtype: float32
    ablation_node_layers:
        - layers.1
        - layers.2
    batch_size: 64
    seed: 0
    eval_type: accuracy
    """
    config_dict = yaml.safe_load(config_str)
    config = MNISTAblationConfig(**config_dict)
    accuracies = mnist_ablations_main(config)
    check_accuracies(accuracies, config, max_accuracy_threshold=0.95)


@pytest.mark.slow
@pytest.mark.parametrize("ablation_type", ["orthogonal", "rib"])
def test_run_modular_arithmetic_rib_ablations(ablation_type):
    """Test various ablation result properties for RIB on modular arithmetic."""

    config_str = f"""
    exp_name: null  # Prevent saving output
    ablation_type: {ablation_type}
    interaction_graph_path: experiments/lm_rib_build/sample_graphs/modular_arithmetic_rib_graph_sample.pt
    schedule:
        schedule_type: exponential
        early_stopping_threshold: 0.2
        ablate_every_vec_cutoff: 2
        exp_base: 2.0
        specific_points: [30, 31]
    dataset:
        source: custom
        name: modular_arithmetic
        return_set: test
    ablation_node_layers:
        - ln1.0
        - unembed
    batch_size: 64
    dtype: float32
    seed: 0
    eval_type: accuracy
    """
    config_dict = yaml.safe_load(config_str)
    config = LMAblationConfig(**config_dict)
    accuracies = lm_ablations_main(config)
    check_accuracies(accuracies, config, max_accuracy_threshold=0.998)
