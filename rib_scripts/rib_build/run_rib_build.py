"""Script for building a rib graph.

This script can be run in parallel across multiple gpus. Each gpu will build the edges for a
subset of the dataset, which can later be combined using `run_combine_edges.py`. Note that
parallelism is currently only supported for the edge calculation and not the calculation of Cs.

Usage:
    Single GPU:
        # python run_rib_build.py <path/to/config.yaml> [--force] [--n_pods <int>] [--pod_rank <int>]
        python run_rib_build.py <path/to/config.yaml> [--force]
    Multiple GPUs:
        mpirun -n <n_gpus> python run_rib_build.py <path/to/config.yaml> [--force] [--n_pods <int>]
        [--pod_rank <int>]

    Args:
        path/to/config.yaml: Path to the config file for building the rib graph.
        --force: Flag for forcing the script to overwrite existing files.
        --n_pods: The total number of pods in which this script is being run. Will be 1 unless this
            script was called as part of a distributed job over e.g. kubernetes.
        --pod_rank: Rank of the current pod. Will be 0 unless this script was called as part of a
            distributed job over e.g. kubernetes.
"""
import fire

from rib.rib_builder import rib_build

if __name__ == "__main__":
    fire.Fire(rib_build, serialize=lambda _: "")
