#!/bin/bash

# Runs `mpirun -n N_GPUS python run_lm_rib_build.py CONFIG_FILE --n_pods=N --pod_rank=R`
# where CONFIG_FILE is the configuration file, n_pods is the number of pods, and pod_rank is the rank of the current pod, given as named arguments to the script.

N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

# Initialize variables
N_PODS=""
POD_RANK=""
CONFIG_FILE=""

# Parse named arguments
for i in "$@"
do
case $i in
    --n_pods=*)
    N_PODS="${i#*=}"
    shift # past argument=value
    ;;
    --pod_rank=*)
    POD_RANK="${i#*=}"
    shift # past argument=value
    ;;
    --config_file=*)
    CONFIG_FILE="${i#*=}"
    shift # past argument=value
    ;;
    *)
          # unknown option
    ;;
esac
done

# Check if N_PODS, POD_RANK, and CONFIG_FILE are set
if [ -z "$N_PODS" ] || [ -z "$POD_RANK" ] || [ -z "$CONFIG_FILE" ]; then
    echo "Error: Arguments --n_pods, --pod_rank, and --config_file are required."
    exit 1
fi

# Get the current directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mpirun -n $N_GPUS python $DIR/run_lm_rib_build.py $CONFIG_FILE --n_pods=$N_PODS --pod_rank=$POD_RANK
