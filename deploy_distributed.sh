#!/bin/bash

# This script deploys a distributed job to the Kubernetes cluster. It should be run from your
# local machine, not from within the cluster. It now takes four arguments:
#   1. The job spec file (e.g. specs/dan-a5000-job.yaml)
#   2. Path to the environment activation script (e.g. /mnt/ssd-apollo/dan/RIB/rib-env/bin/activate)
#   3. Path to script on the cluster (e.g. /mnt/ssd-apollo/dan/RIB/rib/experiments/lm_rib_build/distributed_edges.sh)
#   4. The config file for the script (e.g. /mnt/ssd-apollo/dan/RIB/rib/experiments/lm_rib_build/pythia-14m.yaml)
#   5. The number of jobs to deploy (e.g. 4)

# Each job will be associated with a single pod. Depending on your job spec file and the script you
# are running, each job/pod may split the job across all available GPUs on that pod.

### Example usage:
#   ./deploy_distributed.sh \
#       specs/dan-a5000-job.yaml \
#       /mnt/ssd-apollo/dan/RIB/rib-env/bin/activate \
#       /mnt/ssd-apollo/dan/RIB/rib/experiments/lm_rib_build/distributed_edges.sh \
#       /mnt/ssd-apollo/dan/RIB/rib/experiments/lm_rib_build/pythia-14m.yaml \
#       4


# To delete all jobs, run:
#   kubectl delete job dan-a5000-job-0 dan-a5000-job-1 dan-a5000-job-2 dan-a5000-job-3
#
#  (replace the job names with the ones you used, which you can get from kubectl get jobs)

# Check if the correct number of arguments is provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <job_spec_file> <env_activation_path> <script_path> <config_file> <num_jobs>"
    exit 1
fi

# Job template, environment activation path, script path, configuration file, and number of jobs
JOB_SPEC=$1
ENV_ACTIVATION_PATH=$2
SCRIPT_PATH=$3
CONFIG_FILE=$4
NUM_JOBS=$5

# Extract the base name from the job template
JOB_BASE_NAME=$(basename "$JOB_SPEC" .yaml)

# Command to delete jobs
DELETE_CMD="kubectl delete job"

# Loop to create and deploy jobs
for ((i=0; i<NUM_JOBS; i++)); do
    # Create a temporary job file
    JOB_FILE="job_$i.yaml"

    # Unique job name
    JOB_NAME="${JOB_BASE_NAME}-${i}"

    # Replace <JOB_NAME> and <COMMAND> in the job spec
    sed -e "s|<JOB_NAME>|$JOB_NAME|g" \
        -e "s|<COMMAND>|source $ENV_ACTIVATION_PATH \&\& bash $SCRIPT_PATH --config_file=$CONFIG_FILE --n_pods=$NUM_JOBS --pod_rank=$i|g" \
        $JOB_SPEC > $JOB_FILE


    # Deploy the job
    kubectl apply -f $JOB_FILE

    if [ $? -ne 0 ]; then
        echo "Error deploying job $JOB_NAME."
        exit 1
    fi

    # Add to delete command
    DELETE_CMD+=" $JOB_NAME"

    # Clean up the temporary job file
    rm $JOB_FILE
done

echo "Completed deployment of $NUM_JOBS jobs at $(date)"
echo "To delete all jobs, run:"
echo $DELETE_CMD