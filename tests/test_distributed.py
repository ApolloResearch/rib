"""
NOTE: Distributed tests are not run by default. To run them, use the --runmpi flag.

The reason is that each distributed test needs to run in a separate process from other tests.

If adding a distributed test, manually add the test to the CI (.github/workflows/checks.yaml)
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
from tests.utils import get_modular_arithmetic_config


def get_single_edges(
    tmpdir: Path, dist_split_over: str, edge_formula: str, n_stochastic_sources: Optional[int]
):
    config = get_modular_arithmetic_config(
        {
            "exp_name": "test_single",
            "calculate_edges": True,
            "interaction_matrices_path": tmpdir / "compute_cs_rib_Cs.pt",
            "dist_split_over": dist_split_over,
            "edge_formula": edge_formula,
            "n_stochastic_sources": n_stochastic_sources,
        }
    )
    return rib_build(config).edges


def run_distributed_edges(config_path: str, n_pods: int, pod_rank: int, n_processes: int):
    build_script_path = f"{REPO_ROOT}/rib_scripts/rib_build/run_rib_build.py"
    mpi_command = (
        f"mpirun -n {n_processes} python {build_script_path} {config_path} "
        f"--n_pods={n_pods} --pod_rank={pod_rank}"
    )
    subprocess.run(mpi_command, shell=True, capture_output=True)


def get_double_edges(
    tmpdir: Path, dist_split_over: str, edge_formula: str, n_stochastic_sources: Optional[int]
):
    exp_name = "test_double"
    double_config_path = tmpdir / "double_config.yaml"
    double_outdir_path = tmpdir / "double_out"

    double_config = get_modular_arithmetic_config(
        {
            "exp_name": exp_name,
            "calculate_edges": True,
            "interaction_matrices_path": f"{tmpdir}/compute_cs_rib_Cs.pt",
            "out_dir": double_outdir_path,
            "dist_split_over": dist_split_over,
            "edge_formula": edge_formula,
            "n_stochastic_sources": n_stochastic_sources,
        }
    )
    with open(double_config_path, "w") as f:
        # yaml.dump can't convert PosixPath to str so we convert to json first
        yaml.dump(json.loads(double_config.model_dump_json()), f)

    # mpi might be initialized which causes problems for running an mpiexec subcommand.
    MPI.Finalize()
    run_distributed_edges(double_config_path, n_pods=1, pod_rank=0, n_processes=2)
    combine_edges(f"{double_outdir_path}/distributed_{exp_name}")
    edges = RibBuildResults(
        **torch.load(f"{double_outdir_path}/distributed_{exp_name}/rib_graph_combined.pt")
    ).edges
    return edges


def compare_edges(
    dist_split_over: str, tmpdir: Path, edge_formula: str, n_stochastic_sources: Optional[int]
):
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
        edge_formula=edge_formula,
        n_stochastic_sources=n_stochastic_sources,
    )
    all_double_edges = get_double_edges(
        tmpdir=tmpdir,
        dist_split_over=dist_split_over,
        edge_formula=edge_formula,
        n_stochastic_sources=n_stochastic_sources,
    )

    for s_edges, d_edges in zip(all_single_edges, all_double_edges):
        assert (
            s_edges.E_hat.shape == d_edges.E_hat.shape
        ), f"mismatching shape for {s_edges.in_node_layer}, {s_edges.shape}!={d_edges.shape}"
        assert torch.allclose(
            s_edges.E_hat, d_edges.E_hat, atol=1e-9
        ), f"on {s_edges.in_node_layer} mean error {(s_edges.E_hat-d_edges.E_hat).abs().mean().item()}"


@pytest.mark.mpi
def test_squared_edges_are_same_dist_split_over_dataset(tmpdir):
    compare_edges(
        dist_split_over="dataset", tmpdir=tmpdir, edge_formula="squared", n_stochastic_sources=None
    )


@pytest.mark.mpi
def test_squared_edges_are_same_dist_split_over_out_dim(tmpdir):
    compare_edges(
        dist_split_over="out_dim", tmpdir=tmpdir, edge_formula="squared", n_stochastic_sources=None
    )


@pytest.mark.mpi
def test_stochastic_edges_are_same_dist_split_over_out_dim(tmpdir):
    compare_edges(
        dist_split_over="out_dim", tmpdir=tmpdir, edge_formula="stochastic", n_stochastic_sources=3
    )
