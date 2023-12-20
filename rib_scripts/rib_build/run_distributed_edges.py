import subprocess
from pathlib import Path
from typing import Optional

import fire
import torch

from rib.log import logger


def main(config_file: str, n_pods: int, pod_rank: int, n_processes: Optional[int] = None):
    """
    Start multiple processes to compute the edges in parallel.

    Args:
        config_file (str): path to config
        n_pods (int): total number of pods the script is running on
        pod_rank (int): the rank of this pod, 0 indexed.
        n_processes (Optional[int]): how many processes to start on this pod. If None will start 1
            for each gpu
    """
    n_processes = torch.cuda.device_count() if n_processes is None else n_processes
    run_command = [
        "mpirun",
        "-n",
        str(n_processes),
        "python",
        str(Path(__file__).parent / "run_rib_build.py"),
        config_file,
        "--n_pods=" + str(n_pods),
        "--pod_rank=" + str(pod_rank),
    ]

    process_return_info = subprocess.run(run_command, text=True, capture_output=True)
    try:
        process_return_info.check_returncode()
    except:
        # subprocess failed
        # log line by line so easier to read stack trace
        for err_line in process_return_info.stderr.splitlines():
            logger.error(err_line)


if __name__ == "__main__":
    fire.Fire(main)
