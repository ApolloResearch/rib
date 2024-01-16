"""
NOTE: Should be run from your local machine in an environment with the kuber cli installed.

Deploys a distributed edges run using the [kuber](https://github.com/ApolloResearch/kuber) cli.

Note that each job corresponds to a single pod, so we use the terms interchangeably.

Usage:
    python deploy_distributed_edges.py
        --n_pods=2
        --n_gpus=2
        --gpu_type=a5000
        --env_path=/mnt/ssd-interp/dan/RIB/rib-env
        --script_path=/mnt/ssd-interp/dan/RIB/rib/rib_scripts/rib_build/run_rib_build.py
        --config_file=/mnt/ssd-interp/dan/RIB/rib/rib_scripts/rib_build/edges_pythia-14m.yaml
"""
import argparse
import asyncio
import os
import sys

PYTHON_EXECUTABLE_PATH = sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy distributed edges")
    parser.add_argument("-n", "--n_pods", type=int, default=1, help="number of pods")
    parser.add_argument("-g", "--n_gpus", type=int, default=1, help="number of gpus per pod")
    parser.add_argument(
        "--gpu_type",
        type=str,
        default="a5000",
        help="type of gpu to use. 'cpu' for cpu only",
    )
    parser.add_argument(
        "-e",
        "--env_path",
        type=str,
        default="/mnt/ssd-interp/dan/RIB/rib-env",
        help="path to virtual environment on the pod",
    )
    parser.add_argument(
        "-s",
        "--script_path",
        type=str,
        default="/mnt/ssd-interp/dan/RIB/rib/rib_scripts/rib_build/run_rib_build.py",
        help="path to the script on the pod that sets off the edges run",
    )
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="/mnt/ssd-interp/dan/RIB/rib/rib_scripts/rib_build/edges_pythia-14m.yaml",
        help="path to config file on the pod",
    )
    args = parser.parse_args()
    return args


async def print_stream(stream: asyncio.StreamReader, prefix: str):
    while True:
        line = await stream.readline()
        if line:
            print(f"{prefix}: {line.decode().strip()}")
        else:
            break


async def deploy_pod(args: argparse.Namespace, pod_idx: int):
    job_name = f"edges-{pod_idx}"

    post_init_command = (
        f"source {args.env_path}/bin/activate && mpirun -n {args.n_gpus} python {args.script_path} "
        f"{args.config_file} --n_pods={args.n_pods} --pod_rank={pod_idx}"
    )
    deploy_command = (
        f"{PYTHON_EXECUTABLE_PATH} cli.py deploy {args.gpu_type} --name {job_name} "
        f"--kill_time_london instantly --n_gpus {args.n_gpus} "
        f"--post_init_command '{post_init_command}'"
    )

    # Set the python output is not buffered so that we can see it in real time.
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = await asyncio.create_subprocess_shell(
        deploy_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env
    )

    assert process.stdout is not None
    assert process.stderr is not None
    await asyncio.gather(
        print_stream(process.stdout, prefix=f"Pod {pod_idx}"),
        print_stream(process.stderr, prefix=f"Pod {pod_idx}"),
    )


async def main():
    args = parse_args()
    await asyncio.gather(*(deploy_pod(args, pod_idx) for pod_idx in range(args.n_pods)))


if __name__ == "__main__":
    asyncio.run(main())
