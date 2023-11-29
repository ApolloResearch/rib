from typing import Any

import pytest
import torch
import yaml
from mpi4py import MPI
from torch.utils.data import ConcatDataset, TensorDataset

from experiments.lm_rib_build.combine_edges import main as combine_edges
from experiments.lm_rib_build.distributed_edges import main as run_edges
from experiments.lm_rib_build.run_lm_rib_build import Config
from experiments.lm_rib_build.run_lm_rib_build import main as run_rib_build
from rib.loader import get_dataset_chunk


@pytest.mark.slow
class TestDistributed:
    def make_config_dict(self, exp_name: str, **kwargs):
        config_str = f"""
        exp_name: {exp_name}
        seed: 0
        tlens_pretrained: null
        tlens_model_path: experiments/train_modular_arithmetic/sample_checkpoints/lr-0.001_bs-10000_norm-None_2023-11-28_16-07-19/model_epoch_60000.pt
        interaction_matrices_path: null
        node_layers:
            - ln1.0
            - mlp_in.0
            - unembed
            - output
        dataset:
            source: custom
            name: modular_arithmetic
            return_set: train
        batch_size: 512
        truncation_threshold: 1e-6
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
        return run_rib_build(Config(**single_config_dict))["edges"]

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
        run_edges(double_config_path, n_pods=1, pod_rank=0, n_processes=2)
        combine_edges(double_outdir_path)
        results = torch.load(f"{double_outdir_path}/test_double_rib_graph_combined.pt")
        return results["edges"]

    def test_edges_are_same(self, tmpdir):
        # first we compute the cs. we do this separately as there are occasional reproducibility
        # issues with computing them.
        cs_config = Config(
            **self.make_config_dict("compute_cs", out_dir=tmpdir, calculate_edges=False)
        )
        run_rib_build(cs_config)
        # not using fixtures as we need to compute Cs first
        single_edges = self.get_single_edges(tmpdir)
        double_edges = self.get_double_edges(tmpdir)
        for (module, s_edges), (_, d_edges) in zip(single_edges, double_edges):
            assert (
                s_edges.shape == d_edges.shape
            ), f"mismatching shape for {module}, {s_edges.shape}!={d_edges.shape}"
            assert torch.allclose(
                s_edges, d_edges, atol=1e-9
            ), f"on {module} mean error {(s_edges-d_edges).abs().mean().item()}"


@pytest.mark.parametrize("num_chunks", [1, 3, 5, 7, 10])
def test_get_dataset_chunk(num_chunks):
    original = TensorDataset(torch.arange(10))

    chunks = [get_dataset_chunk(original, i, num_chunks) for i in range(num_chunks)]
    reconstructed = ConcatDataset(chunks)

    assert len(original) == len(reconstructed)
    for i in range(len(original)):
        assert original[i] == reconstructed[i]
