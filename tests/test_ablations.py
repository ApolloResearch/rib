"""Run the mnist orthog and rib ablation scripts and check the below properties:

1. The accuracy when no vectors are ablated is higher than a set threshold (e.g. 95%)
2. Ablating all vectors gives an accuracy lower than 50% (arbitrarily chosen)
3. There are accuracies for all ablated vectors.
4. The accuracies are sorted roughly in descending order of the number of ablated vectors.

This is currently quite hacky. In particular, we mock torch.load to return an interaction graph
with an updated MLP path. This is necessary because the interaction graph is saved with an
absolute path to the MLP, and a github action will not have access to the same absolute path.
"""

from pathlib import Path
from typing import Union

import pytest
import torch
import yaml

from experiments.lm_ablations.run_lm_ablations import Config as LMAblationConfig
from experiments.lm_ablations.run_lm_ablations import main as lm_ablations_main
from experiments.mnist_ablations.run_mnist_ablations import (
    Config as MNISTAblationConfig,
)
from experiments.mnist_ablations.run_mnist_ablations import main as mnist_ablations_main
from rib.ablations import AblationAccuracies


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
    """Check the results of an ablation experiment

    Args:
        accuracies: The result of the ablation experiment
        config: The config the ablation was run with
        max_accuracy_threshold: Lower bound on accuracy to expect with 0 ablated vectors.
        sort_tolerance: The number of out-of-order pairs to tolerate when checking if the
            accuracies are roughly sorted.
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
    """Test various ablation result properties for ablations on MNIST.

    The ablation experiments load model from the config of the interaction graph. To run on ci
    we need this path to be local. If that isn't the case you can manually fix this with:
    ```
    import torch
    rib_graph = torch.load("experiments/mnist_rib_build/sample_graphs/4-node-layers_rib_graph_sample.pt")
    rib_graph['config']['mlp_path'] = "experiments/train_mnist/sample_checkpoints/lr-0.001_bs-64_2023-11-22_13-05-08/model_epoch_3.pt"
    torch.save(rib_graph, "experiments/mnist_rib_build/sample_graphs/4-node-layers_rib_graph_sample.pt")
    ```
    """
    rib_graph = torch.load(
        "experiments/mnist_rib_build/sample_graphs/4-node-layers_rib_graph_sample.pt"
    )
    model_path = Path(rib_graph["config"]["mlp_path"])
    assert not model_path.is_absolute(), "must be relative to run in ci, see docstring"

    config_str = f"""
    exp_name: "test_ablation_mnist"
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
    out_dir: null
    """
    config_dict = yaml.safe_load(config_str)
    config = MNISTAblationConfig(**config_dict)
    accuracies = mnist_ablations_main(config)
    check_accuracies(accuracies, config, max_accuracy_threshold=0.95)


@pytest.mark.slow
@pytest.mark.parametrize("ablation_type", ["orthogonal", "rib"])
def test_run_modular_arithmetic_rib_ablations(ablation_type):
    """Test various ablation result properties on modular arithmetic.

    The ablation experiments load model from the config of the interaction graph. To run on ci
    we need this path to be local. If that isn't the case you can manually fix this with:
    ```
    import torch
    rib_graph = torch.load("experiments/lm_rib_build/sample_graphs/modular_arithmetic_rib_graph_sample.pt")
    rib_graph['config']['tlens_model_path'] = "experiments/train_modular_arithmetic/sample_checkpoints/lr-0.001_bs-10000_norm-None_2023-11-28_16-07-19/model_epoch_60000.pt"
    torch.save(rib_graph, "experiments/lm_rib_build/sample_graphs/modular_arithmetic_rib_graph_sample.pt")
    ```
    """
    rib_graph = torch.load(
        "experiments/lm_rib_build/sample_graphs/modular_arithmetic_rib_graph_sample.pt"
    )
    model_path = Path(rib_graph["config"]["tlens_model_path"])
    assert not model_path.is_absolute(), "must be relative to run in ci, see docstring"

    config_str = f"""
    exp_name: "test_ablation_mod_add"
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
    out_dir: null
    """
    config_dict = yaml.safe_load(config_str)
    config = LMAblationConfig(**config_dict)
    accuracies = lm_ablations_main(config)
    check_accuracies(accuracies, config, max_accuracy_threshold=0.998)
