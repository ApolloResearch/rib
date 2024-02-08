"""
Utilities for dealing with parallel processes across pods and also across GPUs within a pod.
"""

import warnings
from logging import WARNING
from typing import Optional

import torch
from mpi4py import MPI
from pydantic import BaseModel, ConfigDict, Field

from rib.log import logger


class DistributedInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)
    global_size: int  # Total number of processes. Equivalent to n_pods * n_gpus_per_pod.
    global_rank: int  # The rank of this process amongst all processes, from 0 to world_size.
    n_pods: int  # The number of pods (i.e. machines) used to parallelize the script.
    pod_rank: int  # The rank of this pod, from 0 to n_pods.
    local_comm: MPI.Intracomm = Field(
        default_factory=lambda: MPI.COMM_WORLD, exclude=True
    )  # MPI communicator used within this pod.
    local_rank: int  # The rank of this process within this pod, from 0 to n_gpus_per_pod.
    local_size: int  # The number of processes in this pod.
    is_parallelised: bool  # Whether the script is parallelised.
    is_main_process: bool  # True if local_rank == 0 and false otherwise.


def get_dist_info(n_pods: int, pod_rank: int) -> DistributedInfo:
    """Get information about the distributed setup.

    Args:
        n_pods: The number of pods (i.e. machines) used to parallelize the script.
        pod_rank: The rank of this pod, from 0 to n_pods.

    Returns:
        A dataclass containing information about the distributed setup.
    """
    local_comm = MPI.COMM_WORLD
    local_rank, local_size = local_comm.Get_rank(), local_comm.Get_size()
    is_parallelised = local_size > 1
    is_main_process = local_rank == 0

    global_size = local_size * n_pods
    global_rank = local_rank + pod_rank * local_size

    return DistributedInfo(
        global_size=global_size,
        global_rank=global_rank,
        n_pods=n_pods,
        pod_rank=pod_rank,
        local_comm=local_comm,
        local_rank=local_rank,
        local_size=local_size,
        is_parallelised=is_parallelised,
        is_main_process=is_main_process,
    )


def adjust_logger_dist(dist_info: DistributedInfo):
    """Avoids stdout clutter by not having auxuilary processes log INFO."""
    if not dist_info.is_main_process:
        logger.setLevel(WARNING)


def get_device_mpi(dist_info: DistributedInfo):
    if not torch.cuda.is_available():
        return "cpu"
    if not dist_info.is_parallelised:
        return "cuda"

    # Pick the right gpu
    n_gpus = torch.cuda.device_count()
    if dist_info.local_size > n_gpus:
        if dist_info.is_main_process:
            warnings.warn("Starting more processes than availiable devices")

    if dist_info.is_main_process:
        logger.info(f"Distributing {dist_info.local_size} processes over {n_gpus} gpus")
    return f"cuda:{dist_info.local_rank % n_gpus}"


def check_sizes_mpi(dist_info: DistributedInfo, tensor: torch.Tensor):
    assert dist_info.is_parallelised
    sizes = dist_info.local_comm.gather(tensor.shape, root=0)
    if dist_info.is_main_process:
        assert sizes is not None
        # data[i] holds the shape of the tensor in rank i. They should all be equal shapes!
        assert (
            len(set(sizes)) == 1
        ), f"mismatched shape of tensors across processes, shapes = {sizes}"


def sum_across_processes(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    cpu_tensor = tensor.cpu()
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, cpu_tensor, op=MPI.SUM)
    return cpu_tensor.to(tensor.device)
