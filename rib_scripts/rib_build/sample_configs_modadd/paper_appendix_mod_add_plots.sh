#!/bin/bash
set -e
set -x
function run_modularity_pipeline {
    type="$1"
    config_dir="/data/stefan_heimersheim/s3/rib/rib_scripts/rib_build/sample_configs_modadd"
    base_dir="/data/stefan_heimersheim/s3/"
    build_dir="${base_dir}/rib/rib_scripts/rib_build"
    ablation_dir="${base_dir}/rib/rib_scripts/ablations"
    interp_dir="${base_dir}/interp/interp/mod_add_interp"
    prefix="modular_arithmetic_"
    suffix="_seed$2"

    # cd "${build_dir}"
    # python run_rib_build.py "${config_dir}/${prefix}${type}${suffix}.yaml" -f

    # cd "${ablation_dir}"
    # python run_ablations.py "${config_dir}/ablations_edge_bisect_${prefix}${type}${suffix}.yaml" -f

    # cd "${interp_dir}"
    # python generate_labels_mod_add.py "${build_dir}/out/${prefix}${type}${suffix}_rib_graph.pt" --out_file="${prefix}${type}${suffix}.csv" --max_dim 50 -f

    cd "${build_dir}"
    OMP_NUM_THREADS=1 python run_modularity.py "${build_dir}/out/${prefix}${type}${suffix}_rib_graph.pt" --ablation_path "${ablation_dir}/out/${prefix}${type}${suffix}_edge_ablation_results.json" --labels_file "${interp_dir}/${prefix}${type}${suffix}.csv" --gamma 1 --nodes_per_layer 25 --seed 0 --hide_const_edges True --plot_norm abs --figsize 6,10 --sorting rib --hide_singleton_nodes False --hide_output_layer False --line_width_factor 50 --merge_output_nodes_into_one True --adjustable_spacing_per_layer False --hide_0th_node True
}

for i in {0..4}
do
    run_modularity_pipeline rib $i
    run_modularity_pipeline pca $i
done