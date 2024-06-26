"""
NOTE: If adding a distributed test, manually add the test to `tests/run_distributed_tests.sh`.

Distributed tests are not run by default. To run them, use the --runmpi flag, and only run one test
at a time. The reason is that each distributed test needs to run in a separate process from other
tests.
"""

import json
import subprocess
from pathlib import Path
from typing import Optional

import pytest
import torch
import yaml
from mpi4py import MPI

from rib.edge_combiner import combine_edges
from rib.rib_builder import RibBuildResults, rib_build
from rib.settings import REPO_ROOT
from rib.utils import replace_pydantic_model
from tests.utils import (
    assert_basis_similarity,
    assert_is_close,
    get_modular_arithmetic_config,
    get_tinystories_config,
)


def get_single_edges(tmpdir: Path, dist_split_over: str, n_stochastic_sources_edges: Optional[int]):
    config = get_modular_arithmetic_config(
        {
            "exp_name": "test_single",
            "calculate_edges": True,
            "interaction_matrices_path": tmpdir / "compute_cs_rib_Cs.pt",
            "dist_split_over": dist_split_over,
            "n_stochastic_sources_edges": n_stochastic_sources_edges,
        }
    )
    return rib_build(config).edges


def run_distributed_edges(config_path: str, n_pods: int, pod_rank: int, n_processes: int):
    build_script_path = f"{REPO_ROOT}/rib_scripts/rib_build/run_rib_build.py"
    mpi_command = (
        f"mpirun -n {n_processes} python {build_script_path} {config_path} "
        f"--n_pods={n_pods} --pod_rank={pod_rank}"
    )
    return subprocess.run(mpi_command, shell=True, capture_output=True)


def get_double_edges(tmpdir: Path, dist_split_over: str, n_stochastic_sources_edges: Optional[int]):
    exp_name = "test_double"
    double_config_path = tmpdir / "double_config.yaml"
    double_outdir_path = tmpdir / "double_out"

    double_config = get_modular_arithmetic_config(
        {
            "exp_name": exp_name,
            "calculate_edges": True,
            "calculate_Cs": True,
            "interaction_matrices_path": f"{tmpdir}/compute_cs_rib_Cs.pt",
            "out_dir": double_outdir_path,
            "dist_split_over": dist_split_over,
            "n_stochastic_sources_edges": n_stochastic_sources_edges,
        }
    )
    with open(double_config_path, "w") as f:
        # yaml.dump can't convert PosixPath to str so we convert to json first
        yaml.dump(json.loads(double_config.model_dump_json()), f)

    # mpi might be initialized which causes problems for running an mpiexec subcommand.
    MPI.Finalize()
    logs = run_distributed_edges(double_config_path, n_pods=1, pod_rank=0, n_processes=2)
    combine_edges(f"{double_outdir_path}/distributed_{exp_name}")
    edges = RibBuildResults(
        **torch.load(f"{double_outdir_path}/distributed_{exp_name}/rib_graph_combined.pt")
    ).edges
    return edges


def compare_edges(dist_split_over: str, tmpdir: Path, n_stochastic_sources_edges: Optional[int]):
    # first we compute the cs. we do this separately as there are occasional reproducibility
    # issues with computing them.
    cs_config = get_modular_arithmetic_config(
        {
            "exp_name": "compute_cs",
            "out_dir": tmpdir,
            "calculate_edges": False,
        }
    )
    rib_build(cs_config)
    # not using fixtures as we need to compute Cs first
    all_single_edges = get_single_edges(
        tmpdir=tmpdir,
        dist_split_over=dist_split_over,
        n_stochastic_sources_edges=n_stochastic_sources_edges,
    )
    all_double_edges = get_double_edges(
        tmpdir=tmpdir,
        dist_split_over=dist_split_over,
        n_stochastic_sources_edges=n_stochastic_sources_edges,
    )

    for s_edges, d_edges in zip(all_single_edges, all_double_edges):
        assert_is_close(s_edges.E_hat, d_edges.E_hat, atol=1e-9, rtol=1e-0)


@pytest.mark.mpi
@pytest.mark.xfail(reason="Currently failing for unknown reason.")
def test_squared_edges_are_same_dist_split_over_dataset(tmpdir):
    compare_edges(
        dist_split_over="dataset",
        tmpdir=tmpdir,
        n_stochastic_sources_edges=None,
    )


@pytest.mark.mpi
def test_squared_edges_are_same_dist_split_over_out_dim(tmpdir):
    compare_edges(
        dist_split_over="out_dim",
        tmpdir=tmpdir,
        n_stochastic_sources_edges=None,
    )


@pytest.mark.mpi
def test_stochastic_edges_are_same_dist_split_over_out_dim(tmpdir):
    compare_edges(
        dist_split_over="out_dim",
        tmpdir=tmpdir,
        n_stochastic_sources_edges=3,
    )


@pytest.mark.mpi
def test_distributed_basis(tmp_path):
    config = get_tinystories_config(
        {
            "exp_name": "test_basis",
            "calculate_edges": False,
            "dist_split_over": "out_dim",
            "basis_formula": "jacobian",
            "dataset": {"n_samples": 4, "return_set": "train", "n_ctx": 10},
            "node_layers": ["ln1.1", "ln2.1", "ln1.2"],
            "out_dir": None,
            "batch_size": 3,
        }
    )
    single_results = rib_build(config)

    double_config = replace_pydantic_model(config, {"out_dir": tmp_path})
    double_config_path = tmp_path / "double_config.yaml"
    with open(double_config_path, "w") as f:
        yaml.dump(double_config.model_dump(), f)
    MPI.Finalize()
    subprocess.run(
        f"mpirun -n 2 python {REPO_ROOT}/rib_scripts/rib_build/run_rib_build.py "
        f"{double_config_path.absolute()} --n_pods=1 --pod_rank=0",
        shell=True,
        capture_output=True,
    )

    double_results = RibBuildResults(**torch.load(tmp_path / "test_basis_rib_Cs.pt"))

    for ir_single, ir_double in zip(
        single_results.interaction_rotations[:-1], double_results.interaction_rotations[:-1]
    ):
        assert_basis_similarity(ir_single, ir_double, error=0.001)
