"""Run MNIST and modular artihmetic ablation scripts and check the below properties:

1. The accuracy when no vectors are ablated is higher than a set threshold (e.g. 95%)
2. Ablating all vectors gives an accuracy lower than 50% (arbitrarily chosen)
3. There are accuracies for all ablated vectors.
4. The accuracies are sorted roughly in descending order of the number of ablated vectors.
"""

from typing import Union

import pytest
import torch
import yaml

from rib.ablations import (
    AblationAccuracies,
    AblationConfig,
    _get_edge_mask,
    load_bases_and_ablate,
)
from rib.rib_builder import rib_build
from tests.utils import get_mnist_config, get_modular_arithmetic_config


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
    config: AblationConfig,
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
    expected_layers = config.ablation_node_layers
    if config.ablation_type == "edge":
        expected_layers = expected_layers[:-1]
    assert list(accuracies.keys()) == expected_layers, (
        f"Expected accuracies for {expected_layers}, but got " f"{list(accuracies.keys())}"
    )
    for layer_key in expected_layers:
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


@pytest.mark.parametrize("ablation_type", ["orthogonal", "rib", "edge"])
def test_run_mnist_ablations(ablation_type, tmp_path):
    build_config = get_mnist_config(
        {"node_layers": ["layers.1", "layers.2", "output"], "batch_size": 1024, "dtype": "float32"}
    )
    results = rib_build(build_config)
    tempfile = tmp_path / "mnist_rib_graph.pt"
    torch.save(results.model_dump(), tempfile)

    ablation_config_str = f"""
    exp_name: "test_ablation_mnist"
    ablation_type: {ablation_type}
    rib_results_path: {tempfile}
    schedule:
        schedule_type: exponential
        early_stopping_threshold: 0.05
        ablate_every_vec_cutoff: 1
        exp_base: 4.0
    dtype: float32
    ablation_node_layers:
        - layers.1
        - layers.2
    dataset:
        dataset_type: torchvision
        name: MNIST
        return_set_n_samples: 100
    batch_size: 64  # 2 batches
    seed: 0
    out_dir: null
    eval_type: accuracy
    """
    ablation_config_dict = yaml.safe_load(ablation_config_str)
    ablation_config = AblationConfig(**ablation_config_dict)
    accuracies = load_bases_and_ablate(ablation_config)
    check_accuracies(accuracies, ablation_config, max_accuracy_threshold=0.95)


@pytest.mark.slow
@pytest.mark.parametrize("ablation_type", ["orthogonal", "rib", "edge"])
def test_run_modular_arithmetic_rib_ablations(ablation_type, tmp_path):
    build_config = get_modular_arithmetic_config(
        {
            "node_layers": ["ln1.0", "mlp_out.0", "unembed", "output"],
            "batch_size": 10,
            "dataset": {"return_set_n_samples": 10},
        }
    )
    results = rib_build(build_config)
    tempfile = tmp_path / "modular_arithmetic_rib_graph.pt"
    torch.save(results.model_dump(), tempfile)

    ablation_config_str = f"""
    exp_name: "test_ablation_mod_add"
    ablation_type: {ablation_type}
    rib_results_path: {tempfile}
    schedule:
        schedule_type: exponential
        early_stopping_threshold: 0.3
        ablate_every_vec_cutoff: 2
        exp_base: 2.0
        specific_points: [101, 100]
    dataset:
        dataset_type: modular_arithmetic
        return_set: train
        return_set_n_samples: 10
    ablation_node_layers:
        - ln1.0
        - mlp_out.0
        - unembed
    batch_size: 10
    dtype: float32
    seed: 0
    eval_type: accuracy
    out_dir: null
    """
    ablation_config_dict = yaml.safe_load(ablation_config_str)
    ablation_config = AblationConfig(**ablation_config_dict)
    accuracies = load_bases_and_ablate(ablation_config)
    check_accuracies(accuracies, ablation_config, max_accuracy_threshold=0.998)


class TestEdgeMask:
    def test_all_edges_kept(self):
        edge_weights = torch.rand(5, 5)
        assert _get_edge_mask(edge_weights, 26, False).all()
        assert _get_edge_mask(edge_weights, 20, True).all()

    def test_no_edges_kept(self):
        edge_weights = torch.rand(5, 5)
        assert not _get_edge_mask(edge_weights, 0, False).any()

    def test_large_edges_kept(self):
        edge_weights = torch.rand(5, 5)
        edge_weights[range(5), range(5)] += 1.0
        mask = _get_edge_mask(edge_weights, 5, False)
        assert mask.diag().all()
        assert mask.sum() == 5

        mask_keep_const = _get_edge_mask(edge_weights, 4, True)
        assert mask_keep_const.diag().all()
        assert mask_keep_const[:, 0].all()
        assert mask_keep_const[:, 1:].sum() == 4
