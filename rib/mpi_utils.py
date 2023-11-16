"""
Utilities for dealing with parallel processes
"""
import warnings
from logging import WARNING
from typing import NamedTuple

import torch
from mpi4py import MPI


class MpiInfo(NamedTuple):
    comm: MPI.Intracomm
    rank: int
    size: int
    is_parallelised: bool
    is_main_process: bool  # also true when not using mpi


def get_mpi_info() -> MpiInfo:
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    return MpiInfo(comm, rank=rank, size=size, is_parallelised=size > 1, is_main_process=rank == 0)


def adjust_logger_mpi(logger):
    """Avoids stdout clutter by not having auxuilary processes log INFO."""
    if not get_mpi_info().rank == 0:
        logger.setLevel(WARNING)


def get_device_mpi(logger):
    info = get_mpi_info()

    if not torch.cuda.is_available():
        return "cpu"
    if not info.is_parallelised:
        return "cuda"

    # pick the right gpu
    n_gpus = torch.cuda.device_count()
    if info.size > n_gpus:
        # could throw error instead, but multiple gpu on
        warnings.warn("starting more processes than availiable devices")

    logger.info(f"Distributing {info.size} processes over {n_gpus} gpus")
    return f"cuda:{info.rank % n_gpus}"


def check_sizes_mpi(tensor: torch.Tensor):
    info = get_mpi_info()
    assert info.is_parallelised
    sizes = info.comm.gather(tensor.shape, root=0)
    if info.rank == 0:
        assert sizes is not None
        # data[i] holds the shape of the tensor in rank i. They should all be equal shapes!
        assert (
            len(set(sizes)) == 1
        ), f"mismatched shape of tensors across processes, shapes = {sizes}"
