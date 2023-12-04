# %%

import sys
from pathlib import Path
from typing import Callable, Literal, Union
from unittest.mock import patch

import fire
import torch
import yaml

from experiments.modular_dnn_build.run_modular_dnn_rib_build import (
    Config as modular_dnn_build_Config,
)
from experiments.modular_dnn_build.run_modular_dnn_rib_build import (
    main as modular_dnn_build_main,
)
from rib.plotting import plot_interaction_graph

# %%


def main(
    # datatype -> perfect_data_correlation
    # hardcode_bias -> bias
    # variances / data_variances
    # ribmethods --> basis_formula + edge_formula
    # Removed binarise
    exp_name: str = "small_modular_dnn",
    bias: float = 0,
    width: int = 4,
    layers: int = 4,
    variances: list[float] = [1.0, 1.0],
    equal_rows: bool = False,
    perfect_data_correlation: bool = False,
    basis_formula: Literal["(1-alpha)^2", "(1-0)*alpha", "svd"] = "(1-0)*alpha",
    dataset_size: int = 10000,
    rotate_final_node_layer: bool = True,
    dtype: Literal["float32", "float64"] = "float32",
    seed: int = 0,
    force: bool = False,
    data_variances: list[float] = [1.0, 1.0],
    edge_formula: Literal["functional", "squared"] = "functional",
):
    node_layers_str = "\n        - ".join([f"layers.{i}" for i in range(layers + 1)])

    config_str = f"""
    exp_name: {exp_name}
    seed: {seed}
    batch_size: 256
    truncation_threshold: 1e-6
    n_intervals: 0
    rotate_final_node_layer: {rotate_final_node_layer}
    dtype: {dtype}
    model:
        n_hidden_layers: {layers}
        width: {width}
        weight_variances: {variances}
        equal_rows: {equal_rows}
        bias: {bias}
        first_block_width: null
        activation_fn: relu
        dtype: {dtype}
    dataset:
        size: {dataset_size}
        data_variances: {data_variances}
    node_layers:
        - {node_layers_str}
        - output
    perfect_data_correlation: {perfect_data_correlation}
    basis_formula: {basis_formula}
    edge_formula: {edge_formula}
    """

    print(config_str)

    config_dict = yaml.safe_load(config_str)
    config = modular_dnn_build_Config(**config_dict)
    results = modular_dnn_build_main(config, force=force)

    nodes_per_layer = 10
    layer_names = results["config"]["node_layers"] + ["output"]

    out_dir = Path(__file__).parent / "out"
    out_file = out_dir / f"{results['exp_name']}_rib_graph.png"

    plot_interaction_graph(
        raw_edges=results["edges"],
        layer_names=layer_names,
        exp_name=results["exp_name"],
        nodes_per_layer=nodes_per_layer,
        out_file=out_file,
    )

    print("Saved plot to", out_file)


if __name__ == "__main__":
    fire.Fire(lambda **kwargs: main(**kwargs))
