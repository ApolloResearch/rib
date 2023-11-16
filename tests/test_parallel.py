import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

from rib.log import logger


@pytest.mark.slow
def test_distributed_calc_gives_same_edges():
    def make_config(exp_name: str, temp_dir: str, rib_dir: str):
        config_str = f"""
        exp_name: {exp_name}
        seed: 0
        tlens_pretrained: null
        tlens_model_path: {rib_dir}/experiments/train_modular_arithmetic/sample_checkpoints/lr-0.001_bs-10000_norm-None_2023-09-27_18-19-33/model_epoch_60000.pt
        node_layers:
            - ln1.0
            - mlp_in.0
            - unembed
            - output
        dataset:
            source: custom
            name: modular_arithmetic
            return_set: train
        batch_size: 256
        edge_batch_size: 256
        truncation_threshold: 1e-6
        rotate_final_node_layer: false
        last_pos_module_type: add_resid1
        n_intervals: 0
        dtype: float32
        eval_type: accuracy
        out_dir: {temp_dir}
        """
        config_path = f"{temp_dir}/{exp_name}.yaml"
        with open(config_path, "w") as f:
            f.write(config_str)
        return config_path

    rib_dir = str(Path(__file__).parent.parent)
    run_file = rib_dir + "/experiments/lm_rib_build/run_lm_rib_build.py"

    with tempfile.TemporaryDirectory() as temp_dir:
        single_config_path = make_config("test_single", temp_dir=temp_dir, rib_dir=rib_dir)
        double_config_path = make_config("test_double", temp_dir=temp_dir, rib_dir=rib_dir)
        subprocess.run(["python", run_file, single_config_path], capture_output=True, check=True)
        logger.info("done with single!")
        subprocess.run(
            ["mpiexec", "--verbose", "-n", "1", "python", run_file, double_config_path],
            capture_output=True,
            check=True,
        )
        logger.info("done with double!")

        single_edges = torch.load(f"{temp_dir}/test_single_rib_graph.pt")["edges"]
        double_edges = torch.load(f"{temp_dir}/test_double_rib_graph.pt")["edges"]

        print([(m, e.shape) for m, e in single_edges])
        print([(m, e.shape) for m, e in double_edges])

        for (module, s_edges), (_, d_edges) in zip(single_edges, double_edges):
            print(s_edges.shape, d_edges.shape)
            assert torch.allclose(
                s_edges, d_edges, atol=1e-3
            ), f"on {module} mean error {(s_edges-d_edges).abs().mean().item()}"
