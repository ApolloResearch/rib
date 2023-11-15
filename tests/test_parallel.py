import subprocess
import tempfile
from pathlib import Path

import pytest
import torch


@pytest.mark.slow
def test_distributed_calc_gives_same_edges():
    rib_dir = str(Path(__file__).parent.parent)

    def make_config(name: str, temp_dir: str):
        config_str = f"""
        exp_name: {name}
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
        batch_size: 1024
        edge_batch_size: 1024
        truncation_threshold: 1e-6
        rotate_final_node_layer: false
        last_pos_module_type: add_resid1
        n_intervals: 0
        dtype: float64
        eval_type: accuracy
        out_dir: {temp_dir}
        """
        config_path = f"{temp_dir}/{name}.yaml"
        with open(config_path, "w") as f:
            f.write(config_str)
        return config_path


    run_file = rib_dir + "/experiments/lm_rib_build/run_lm_rib_build.py"

    with tempfile.TemporaryDirectory() as temp_dir:
        single_config_path = make_config("test_single", temp_dir)
        double_config_path = make_config("test_double", temp_dir)
        subprocess.run(["python", run_file, single_config_path], capture_output=True)
        print("done with single!")
        subprocess.run(
            ["mpiexec", "-n", "2", "python", run_file, double_config_path], capture_output=True
        )
        print("done with double!")

        single_edges = torch.load(f"{temp_dir}/test_single_rib_graph.pt")["edges"]
        double_edges = torch.load(f"{temp_dir}/test_double_rib_graph.pt")["edges"]

        for (module, s_edges), (_, d_edges) in zip(single_edges, double_edges):
            assert torch.allclose(s_edges, d_edges, atol=1e-10), (module, s_edges, d_edges)