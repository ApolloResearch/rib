# %%

from pathlib import Path
from typing import Literal

import fire
import yaml

from experiments.modular_mlp_build.run_modular_mlp_rib_build import (
    Config as modular_mlp_build_Config,
)
from experiments.modular_mlp_build.run_modular_mlp_rib_build import (
    main as modular_mlp_build_main,
)
from rib.plotting import plot_interaction_graph

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
# activation_fn --> activation_fn
# variances --> weight_variances
# data_variances --> data_variances
# binarise --> What??
# ribmethods --> basis_formula + edge_formula
# column_equal --> weight_equal_columns
# N/A --> dataset_size


def main(
    exp_name: str = "small_modular_mlp",
    n_hidden_layers: int = 3,
    width: int = 4,
    weight_variances: list[float] = [1.0, 1.0],
    weight_equal_columns: bool = False,
    bias: float = 0,
    dataset_size: int = 2048,
    data_variances: list[float] = [1.0, 1.0],
    perfect_data_correlation: bool = False,
    basis_formula: Literal["(1-alpha)^2", "(1-0)*alpha", "svd"] = "(1-0)*alpha",
    rotate_final_node_layer: bool = False,
    dtype: Literal["float32", "float64"] = "float32",
    seed: int = 0,
    force: bool = True,
    edge_formula: Literal["functional", "squared"] = "functional",
    activation_fn: Literal["relu", "identity"] = "relu",
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
            activation_fn: {activation_fn}
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

    # # Out file that combines all args
    # out_dir = f"nhl={n_hidden_layers}_w={width}_wv={weight_variances}_wec={weight_equal_columns}_b={bias}_ds={dataset_size}_dv={data_variances}_pdc={perfect_data_correlation}_bf={basis_formula}_r={rotate_final_node_layer}_dt={dtype}_s={seed}_ef={edge_formula}"

    config_dict = yaml.safe_load(config_str)
    config = modular_mlp_build_Config(**config_dict)

    for type in ["rib", "svd", "neuron"]:
        if type == "rib":
            config.basis_formula = basis_formula
            results = modular_mlp_build_main(config, force=force)
        elif type == "svd":
            config.basis_formula = "svd"
            results = modular_mlp_build_main(config, force=force)
        elif type == "neuron":
            config.basis_formula = "neuron"
            results = modular_mlp_build_main(config, force=force)

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
    fire.Fire(main)
