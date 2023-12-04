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
from rib.utils import check_outfile_overwrite

# %%

# Mapping from Jake's old script:
# exp_name --> exp_name
# n --> width
# k --> N/A
# layers --> n_hidden_layers
# batch_size --> N/A
# seed --> seed
# truncation_threshold --> N/A
# n_intervals --> N/A
# dtype --> dtype
# node_layers --> N/A
# datatype -> perfect_data_correlation
# rotate_final_node_layer --> rotate_final_node_layer
# force --> force
# hardcode_bias -> bias
# activation_fn --> N/A
# variances --> weight_variances
# data_variances --> data_variances
# binarise --> What??
# ribmethods --> basis_formula + edge_formula
# column_equal --> weight_equal_columns
# N/A --> dataset_size


def main(
    exp_name: str = "small_modular_dnn",
    n_hidden_layers: int = 3,
    width: int = 4,
    weight_variances: list[float] = [1.0, 1.0],
    weight_equal_columns: bool = False,
    bias: float = 0,
    dataset_size: int = 128,
    data_variances: list[float] = [1.0, 1.0],
    perfect_data_correlation: bool = False,
    basis_formula: Literal["(1-alpha)^2", "(1-0)*alpha", "svd"] = "(1-0)*alpha",
    rotate_final_node_layer: bool = False,
    dtype: Literal["float32", "float64"] = "float32",
    seed: int = 0,
    force: bool = True,
    edge_formula: Literal["functional", "squared"] = "functional",
):
    node_layers_str = "\n            - ".join(
        [f"layers.{i}" for i in range(n_hidden_layers + 1)] + ["output"]
    )

    config_str = f"""
        exp_name: {exp_name}
        out_dir: null
        node_layers:
            - {node_layers_str}
        model:
            n_hidden_layers: {n_hidden_layers}
            width: {width}
            weight_variances: {weight_variances}
            weight_equal_columns: {weight_equal_columns}
            bias: {bias}
        dataset:
            size: {dataset_size}
            length: {width}
            data_variances: {data_variances}
            data_perfect_correlation: {perfect_data_correlation}
        seed: {seed}
        batch_size: 256
        n_intervals: 0
        truncation_threshold: 1e-6
        dtype: {dtype}
        rotate_final_node_layer: {rotate_final_node_layer}
        basis_formula: {basis_formula}
        edge_formula: {edge_formula}
    """

    config_dict = yaml.safe_load(config_str)
    config = modular_dnn_build_Config(**config_dict)

    for type in ["rib", "svd", "neuron"]:
        if type == "rib":
            config.basis_formula = config_dict["basis_formula"]
            results = modular_dnn_build_main(config, force=force)
        elif type == "svd":
            config.basis_formula = "svd"
            results = modular_dnn_build_main(config, force=force)
        elif type == "neuron":
            config.basis_formula = "neuron"
            results = modular_dnn_build_main(config, force=force)

        out_dir = Path(__file__).parent / "out"
        out_file = out_dir / f"{results['exp_name']}_graph_{type}.png"

        plot_interaction_graph(
            raw_edges=results["edges"],
            layer_names=results["config"]["node_layers"] + ["output"],
            exp_name=results["exp_name"],
            nodes_per_layer=10,
            out_file=out_file,
        )

        print("Saved plot to", out_file)


if __name__ == "__main__":
    fire.Fire(lambda **kwargs: main(**kwargs))
