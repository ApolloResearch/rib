#!/bin/bash
set -e
set -x

base_dir="/data/stefan_heimersheim/projects/RIB"
config_dir="${base_dir}/rib/rib_scripts/rib_build/sample_configs_modadd"
build_dir="${base_dir}/rib/rib_scripts/rib_build"
ablation_dir="${base_dir}/rib/rib_scripts/ablations"
out_dir="${config_dir}/out"
prefix="modular_arithmetic_"
suffix="_seed$2"

function run_modularity_pipeline {
    type="$1"
    run_name="${prefix}${type}${suffix}"

    python ${build_dir}/run_rib_build.py "${config_dir}/${run_name}.yaml" -f
    python ${ablation_dir}/run_ablations.py "${config_dir}/ablations_node_exp_${run_name}.yaml" -f
}

for i in {0..4}
do
    run_modularity_pipeline rib $i
    run_modularity_pipeline pca $i
done

# Make sure to run at least this command on branch paper/mod_add_custom_ablation_plots
python ${ablation_dir}/plot_ablations.py ${out_dir}/modular_arithmetic_node_{rib,pca}_seed*_rib_ablation_results.json -f
