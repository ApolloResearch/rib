#!/bin/bash
set -e
set -x
function run_modularity_pipeline {
    type="$1"
    base_dir="/data/stefan_heimersheim/projects/RIB"
    config_dir="${base_dir}/rib/rib_scripts/rib_build/sample_configs_modadd"
    build_dir="${base_dir}/rib/rib_scripts/rib_build"
    ablation_dir="${base_dir}/rib/rib_scripts/ablations"
    out_dir="${config_dir}/out"
    prefix="modular_arithmetic_"
    suffix="_seed$2"
    run_name="${prefix}${type}${suffix}"

    python ${build_dir}/run_rib_build.py "${config_dir}/${run_name}.yaml" -f
    python ${build_dir}/run_generate_labels_mod_add.py "${out_dir}/${run_name}_rib_graph.pt" --out_file="$out_dir/${run_name}_labels.csv" --max_dim 50 -f
    python ${ablation_dir}/run_ablations.py "${config_dir}/ablations_edge_bisect_${run_name}.yaml" -f
    OMP_NUM_THREADS=1 python ${build_dir}/run_modularity.py "${out_dir}/${run_name}_rib_graph.pt" --ablation_path "${out_dir}/${run_name}_edge_ablation_results.json" --labels_file "$out_dir/${run_name}_labels.csv" --gamma 1 --nodes_per_layer 25 --seed 0 --hide_const_edges True --plot_norm abs --figsize 6,10 --sorting rib --hide_singleton_nodes False --hide_output_layer False --line_width_factor 50 --merge_output_nodes_into_one True --adjustable_spacing_per_layer False --hide_0th_node True
}

for i in {0..4}
do
    run_modularity_pipeline rib $i
    run_modularity_pipeline pca $i
done
