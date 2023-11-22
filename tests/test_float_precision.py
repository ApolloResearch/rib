import json
import tempfile

import pytest
import torch
import yaml

from experiments.lm_ablations.run_lm_ablations import Config as AblationConfig
from experiments.lm_ablations.run_lm_ablations import main as ablation_main
from experiments.lm_rib_build.run_lm_rib_build import Config as RibConfig
from experiments.lm_rib_build.run_lm_rib_build import main as rib_main


@pytest.mark.slow
def test_pythia_floating_point_errors() -> None:
    """Test that float32 and float64 results match.

    Note that they do not match everywhere yet!"""

    rib_config_str = """
    force_overwrite_output: true
    seed: 0
    tlens_pretrained: pythia-14m
    tlens_model_path: null
    dataset:
      source: huggingface
      name: NeelNanda/pile-10k
      tokenizer_name: EleutherAI/pythia-14m
      return_set: train  # pile-10k only has train, so we take the first 90% for building and last 10% for ablations
      return_set_frac: null
      return_set_n_samples: 10
      return_set_portion: first
    node_layers:
      - ln1.0
      - mlp_out.0
      - ln2.3
      - mlp_out.3
      - ln1.5
      - mlp_out.5
      - output
    batch_size: 4  #  A100 can handle 24
    gram_batch_size: 20  #  A100 can handle 80
    truncation_threshold: 1e-6
    rotate_final_node_layer: false
    n_intervals: 10
    calculate_edges: false
    eval_type: null
    """

    ablation_config_str = """
    force_overwrite_output: true
    ablation_type: rib
    schedule:
      schedule_type: linear
      early_stopping_threshold: null
      n_points: 3
      specific_points: [128]
    dataset:
      source: huggingface
      name: NeelNanda/pile-10k
      tokenizer_name: EleutherAI/pythia-14m
      return_set: train  # pile-10k only has train, so we take the first 90% for building and last 10% for ablations
      return_set_frac: null
      return_set_n_samples: 10
      return_set_portion: first
    ablation_node_layers:
      - ln1.0
      - mlp_out.0
      - ln2.3
      - mlp_out.3
      - ln1.5
      - mlp_out.5
    batch_size: 30  # A100 can handle 60
    eval_type: ce_loss
    seed: 0
    """

    rib_config = yaml.safe_load(rib_config_str)
    ablation_config = yaml.safe_load(ablation_config_str)

    temp_dir = tempfile.TemporaryDirectory().name
    rib_config["out_dir"] = temp_dir
    ablation_config["out_dir"] = temp_dir

    rib_results = {}
    for dtype in ["float32", "float64"]:
        exp_name = f"float-precision-test-pythia-14m-{dtype}"
        rib_config["dtype"] = dtype
        rib_config["exp_name"] = exp_name
        rib_main(RibConfig(**rib_config))
        basis_matrices = torch.load(f"{temp_dir}/float-precision-test-pythia-14m-{dtype}_rib_Cs.pt")
        rib_results[dtype] = basis_matrices

    for node_layer in rib_results["float32"]["gram_matrices"].keys():
        float32_gram_matrix = rib_results["float32"]["gram_matrices"][node_layer]
        float64_gram_matrix = rib_results["float64"]["gram_matrices"][node_layer]
        assert torch.allclose(float32_gram_matrix.to(torch.float64), float64_gram_matrix, atol=1e-4)

    if torch.cuda.is_available():
        # This is pretty awful
        # And even more wrong on CPU apparently
        for node_layer_index in range(len(rib_results["float32"]["interaction_rotations"])):
            n_max = 4
            if rib_results["float32"]["interaction_rotations"][node_layer_index]["C"] is None:
                continue
            float32_C = rib_results["float32"]["interaction_rotations"][node_layer_index]["C"][
                :, :n_max
            ]
            float64_C = rib_results["float64"]["interaction_rotations"][node_layer_index]["C"][
                :, :n_max
            ]
            assert torch.allclose(
                float32_C.to(torch.float64), float64_C, rtol=0.5, atol=0.5 * float64_C.max()
            ), node_layer_index

    # Still bad but slightly better
    for node_layer_index in range(len(rib_results["float32"]["eigenvectors"])):
        n_max = 10
        if rib_results["float32"]["eigenvectors"][node_layer_index]["U"] is None:
            continue
        float_32_U = rib_results["float32"]["eigenvectors"][node_layer_index]["U"][:, :n_max]
        float_64_U = rib_results["float64"]["eigenvectors"][node_layer_index]["U"][:, :n_max]
        assert torch.allclose(
            float_32_U.to(torch.float64), float_64_U, atol=1e-3 * float_64_U.max()
        )

    ablation_results = {}
    for dtype in ["float32", "float64"]:
        exp_name = f"float-precision-test-pythia-14m-{dtype}"
        ablation_config["dtype"] = dtype
        ablation_config["exp_name"] = exp_name
        ablation_config["interaction_graph_path"] = f"{temp_dir}/{exp_name}_rib_Cs.pt"
        ablation_main(AblationConfig(**ablation_config))
        ablation_result = json.load(open(f"{temp_dir}/{exp_name}_ablation_results.json"))["results"]
        ablation_results[dtype] = ablation_result

    # ln2.3 (and others) are broken (https://github.com/ApolloResearch/rib/issues/212)
    for node_layer in ablation_results["float32"].keys():
        # ln are broken. ln1.0 seemed fine on GPU (a6000) but broken on
        if node_layer in ["ln2.3", "ln1.5", "ln1.0"]:
            continue
        for n_vecs_ablated in ablation_results["float32"][node_layer].keys():
            float32_ablation_result = ablation_results["float32"][node_layer][n_vecs_ablated]
            float64_ablation_result = ablation_results["float64"][node_layer][n_vecs_ablated]
            assert torch.allclose(
                torch.tensor(float32_ablation_result),
                torch.tensor(float64_ablation_result),
                atol=1e-3,
            ), f"Diff for {node_layer} {n_vecs_ablated} = {float32_ablation_result} != {float64_ablation_result}"
        if "mlp_out" in node_layer:
            for dtype in ["float32", "float64"]:
                # Should be identical due to residual stream size
                ablation_result_128 = ablation_results[dtype][node_layer]["128"]
                ablation_result_642 = ablation_results[dtype][node_layer]["642"]
                assert torch.allclose(
                    torch.tensor(ablation_result_128),
                    torch.tensor(ablation_result_642),
                    atol=1e-3,
                ), f"Diff for {node_layer} {dtype} = {ablation_result_128} != {ablation_result_642}"
