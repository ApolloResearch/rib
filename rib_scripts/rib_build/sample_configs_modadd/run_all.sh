#!/bin/bash
set -e
set -x
function run_modularity_pipeline {
    type="$1"
    config_dir="/mnt/ssd-interp/stefan/rib/rib_scripts/rib_build/sample_configs_modadd/"
    base_dir="/mnt/ssd-interp/stefan"
    build_dir="${base_dir}/rib/rib_scripts/rib_build"
    ablation_dir="${base_dir}/rib/rib_scripts/ablations"
    interp_dir="${base_dir}/interp/interp/mod_add_interp"
    prefix="modular_arithmetic_"

    cd "${build_dir}"
    python run_rib_build.py "${config_dir}/${prefix}${type}.yaml"

    cd "${interp_dir}"
    python generate_labels_mod_add.py "${build_dir}/out/${prefix}${type}_rib_graph.pt" --out_file="${prefix}${type}.csv" --max_dim 140 -f

    cd "${ablation_dir}"
    python run_ablations.py "${config_dir}/ablations_edge_bisect_${prefix}${type}.yaml"

    cd "${build_dir}"
    OMP_NUM_THREADS=1 python run_modularity.py "${build_dir}/out/${prefix}${type}_rib_graph.pt" --ablation_path "${ablation_dir}/out/${prefix}${type}_edge_ablation_results.json" --labels_file "${interp_dir}/${prefix}${type}.csv" --gamma 1 --nodes_per_layer 30 --seed 0 --hide_const_edges True --plot_norm abs --figsize 10,14 --sorting cluster
}

run_modularity_pipeline rib
run_modularity_pipeline pca
run_modularity_pipeline neuron
