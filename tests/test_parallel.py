import subprocess
from typing import Any

import pytest
import torch
import yaml
from mpi4py import MPI
from torch.utils.data import ConcatDataset, TensorDataset

from rib.edge_combiner import combine_edges
from rib.loader import get_dataset_chunk
from rib.rib_builder import RibBuildConfig, RibBuildResults, rib_build
from rib.settings import REPO_ROOT


@pytest.mark.slow
class TestDistributed:
    def make_config_dict(self, exp_name: str, **kwargs):
        config_str = f"""
        exp_name: {exp_name}
        seed: 0
        tlens_pretrained: null
        tlens_model_path: rib_scripts/train_modular_arithmetic/sample_checkpoints/lr-0.001_bs-10000_norm-None_2023-11-28_16-07-19/model_epoch_60000.pt
        interaction_matrices_path: null
        node_layers:
            - ln1.0
            - mlp_in.0
            - unembed
            - output
        dataset:
            dataset_type: modular_arithmetic
            return_set: train
            return_set_n_samples: 10
        batch_size: 6
        truncation_threshold: 1e-15
        rotate_final_node_layer: false
        last_pos_module_type: add_resid1
        n_intervals: 0
        dtype: float64
        eval_type: accuracy
        out_dir: null
        """
        config_dict: dict[str, Any] = yaml.safe_load(config_str)
        config_dict.update(kwargs)
        return config_dict

    def get_single_edges(self, tmpdir):
        single_config_dict = self.make_config_dict(
            "test_single",
            calculate_edges=True,
            interaction_matrices_path=f"{tmpdir}/compute_cs_rib_Cs.pt",
        )
        return rib_build(RibBuildConfig(**single_config_dict)).edges

    def run_distributed_edges(self, config_path: str, n_pods: int, pod_rank: int, n_processes: int):
        build_script_path = f"{REPO_ROOT}/rib_scripts/rib_build/run_rib_build.py"
        mpi_command = (
            f"mpirun -n {n_processes} python {build_script_path} {config_path} "
            f"--n_pods={n_pods} --pod_rank={pod_rank}"
        )
        subprocess.run(mpi_command, shell=True, check=True)

    def get_double_edges(self, tmpdir):
        double_config_path = f"{tmpdir}/double_config.yaml"
        double_outdir_path = f"{tmpdir}/double_out/"

        double_config = self.make_config_dict(
            "test_double",
            calculate_edges=True,
            interaction_matrices_path=f"{tmpdir}/compute_cs_rib_Cs.pt",
            out_dir=double_outdir_path,
        )
        with open(double_config_path, "w") as f:
            yaml.dump(double_config, f)

        # mpi might be initialized which causes problems for running an mpiexec subcommand.
        MPI.Finalize()
        self.run_distributed_edges(double_config_path, n_pods=1, pod_rank=0, n_processes=2)
        combine_edges(double_outdir_path)
        edges = RibBuildResults(
            **torch.load(f"{double_outdir_path}/test_double_rib_graph_combined.pt")
        ).edges
        return edges

    def test_edges_are_same(self, tmpdir):
        # first we compute the cs. we do this separately as there are occasional reproducibility
        # issues with computing them.
        cs_config = RibBuildConfig(
            **self.make_config_dict("compute_cs", out_dir=tmpdir, calculate_edges=False)
        )
        rib_build(cs_config)
        # not using fixtures as we need to compute Cs first
        all_single_edges = self.get_single_edges(tmpdir)
        all_double_edges = self.get_double_edges(tmpdir)

        for s_edges, d_edges in zip(all_single_edges, all_double_edges):
            assert (
                s_edges.E_hat.shape == d_edges.E_hat.shape
            ), f"mismatching shape for {s_edges.in_node_layer_name}, {s_edges.shape}!={d_edges.shape}"
            assert torch.allclose(
                s_edges.E_hat, d_edges.E_hat, atol=1e-9
            ), f"on {s_edges.in_node_layer_name} mean error {(s_edges.E_hat-d_edges.E_hat).abs().mean().item()}"


@pytest.mark.parametrize("num_chunks", [1, 3, 5, 7, 10])
def test_get_dataset_chunk(num_chunks):
    original = TensorDataset(torch.arange(10))

    chunks = [get_dataset_chunk(original, i, num_chunks) for i in range(num_chunks)]
    reconstructed = ConcatDataset(chunks)

    assert len(original) == len(reconstructed)
    for i in range(len(original)):
        assert original[i] == reconstructed[i]
