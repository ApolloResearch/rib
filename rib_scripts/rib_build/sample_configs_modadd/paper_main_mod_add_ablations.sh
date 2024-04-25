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

    cd "${ablation_dir}"
    python run_ablations.py "${config_dir}/ablations_node_exp_${prefix}${type}${suffix}.yaml" -f

    # On branch paper/mod_add_custom_ablation_plots
    python ${ablation_dir}/plot_ablations.py ${ablation_dir}/out/modular_arithmetic_node_*_rib_ablation_results.json -f
}

for i in {0..4}
do
    run_modularity_pipeline rib $i
    run_modularity_pipeline pca $i
done
