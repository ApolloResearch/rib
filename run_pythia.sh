#!/bin/bash
set -e
python /mnt/ssd-apollo/stefan/rib/experiments/lm_rib_build/run_lm_rib_build.py /mnt/ssd-apollo/stefan/rib/experiments/lm_rib_build/pythia-14m.yaml
python /mnt/ssd-apollo/stefan/rib/experiments/lm_ablations/run_lm_ablations.py /mnt/ssd-apollo/stefan/rib/experiments/lm_ablations/rib_pythia-14m.yaml
python /mnt/ssd-apollo/stefan/rib/experiments/lm_ablations/plot_lm_ablations.py /mnt/ssd-apollo/stefan/rib/experiments/lm_ablations/out/rib_pythia-14m-test1_ablation_results.json