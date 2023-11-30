"""Test that float32 and float64 results match.
Note that they do not match everywhere yet."""

import json
import tempfile

import pytest
import torch
import yaml

from experiments.lm_ablations.run_lm_ablations import Config as AblationConfig
from experiments.lm_ablations.run_lm_ablations import main as ablation_main
from experiments.lm_rib_build.run_lm_rib_build import Config as RibConfig
from experiments.lm_rib_build.run_lm_rib_build import main as rib_main


# Skipped because slow (5 and 10 mins for RIB build and ablations, respectively)
@pytest.mark.skip_ci
@pytest.mark.slow
class TestPythiaFloatingPointErrors:
    @pytest.fixture(scope="class")
    def temp_object(self) -> tempfile.TemporaryDirectory:
        """Create a temporary directory for the RIB build and ablation results."""
        return tempfile.TemporaryDirectory()

    @pytest.fixture(scope="class")
    def rib_results(self, temp_object) -> dict:
        """Run RIB build with float32 and float64 and return the results."""
        rib_config_str = """
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
            seed: 42
            """
        rib_config = yaml.safe_load(rib_config_str)
        temp_dir = temp_object.name
        rib_config["out_dir"] = temp_dir

        rib_results = {}
        for dtype in ["float32", "float64"]:
            exp_name = f"float-precision-test-pythia-14m-{dtype}"
            rib_config["dtype"] = dtype
            rib_config["exp_name"] = exp_name
            if not torch.cuda.is_available():
                # Try to reduce memory usage for CI
                rib_config["batch_size"] = 1
                rib_config["gram_batch_size"] = 1
            print("Running RIB build with batch size", rib_config["batch_size"], "for", dtype)
            rib_main(RibConfig(**rib_config))
            basis_matrices = torch.load(
                f"{temp_dir}/float-precision-test-pythia-14m-{dtype}_rib_Cs.pt"
            )
            rib_results[dtype] = basis_matrices

        return rib_results

    def test_gram_matrices(self, rib_results: dict) -> None:
        """Test that all the gram matrices are similar between float32 and float64."""
        for node_layer in rib_results["float32"]["gram_matrices"].keys():
            float32_gram_matrix = rib_results["float32"]["gram_matrices"][node_layer]
            float64_gram_matrix = rib_results["float64"]["gram_matrices"][node_layer]
            assert torch.allclose(
                float32_gram_matrix.to(torch.float64), float64_gram_matrix, atol=1e-4
            ), f"Gram matrix difference {node_layer} between float32 and float64."

    def test_interaction_rotations(self, rib_results: dict) -> None:
        """Test that some (n_max) of the interaction rotations are identical between float32 and float64."""
        # FIXME This test is absolutely awful at the moment. We'd love to have a code that is consistent
        #       enough to run this test with tigher tolerances & all settings (GPU, batch size)
        #       Currently it fails on CPU (batch size [4, 20] but also [1, 1])
        #       and even on GPU if batch size [1, 1].
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available. This test does not work on CPU.")

        if not rib_results["float32"]["config"]["batch_size"] > 1:
            pytest.skip("This test does not work with batch size 1.")

        if rib_results["float32"]["config"]["basis_formula"] == "(1-0)*alpha":
            pytest.skip('This test does not work with the new "(1-0)*alpha" basis.')

        for node_layer_index in range(len(rib_results["float32"]["interaction_rotations"])):
            # This only tests the first column, would like to improve in the future!
            n_max = 1
            if rib_results["float32"]["interaction_rotations"][node_layer_index]["C"] is None:
                continue
            float32_C = rib_results["float32"]["interaction_rotations"][node_layer_index]["C"][
                :, :n_max
            ]
            float64_C = rib_results["float64"]["interaction_rotations"][node_layer_index]["C"][
                :, :n_max
            ]
            # This is a super weak test, would like to improve in the future!
            assert torch.allclose(
                float32_C.to(torch.float64),
                float64_C,
                atol=0.5 * float64_C.max(),
            ), f"Interaction rotation {node_layer_index} difference between float32 and float64."

    def test_eigenvectors(self, rib_results: dict) -> None:
        """Test that some (n_max) of the eigenvectors are identical between float32 and float64."""
        # This tests is pretty approximate (especially we only test the first n_max=10 columns)
        # but not absolutely awful.
        for node_layer_index in range(len(rib_results["float32"]["eigenvectors"])):
            n_max = 10
            if rib_results["float32"]["eigenvectors"][node_layer_index]["U"] is None:
                continue
            float_32_U = rib_results["float32"]["eigenvectors"][node_layer_index]["U"][:, :n_max]
            float_64_U = rib_results["float64"]["eigenvectors"][node_layer_index]["U"][:, :n_max]
            assert torch.allclose(
                float_32_U.to(torch.float64), float_64_U, atol=1e-3 * float_64_U.max()
            ), f"Eigenvector difference {node_layer_index} between float32 and float64."

    @pytest.fixture(scope="class")
    def ablation_results(self, temp_object, rib_results) -> dict:
        # rib_results is an argument to make sure this runs after the rib_results fixture but
        # it actually just uses the temp dir which has been written to by the rib_results fixture.
        ablation_config_str = """
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
        seed: 42
        """
        ablation_config = yaml.safe_load(ablation_config_str)
        temp_dir = temp_object.name
        ablation_config["out_dir"] = temp_dir

        ablation_results = {}
        for dtype in ["float32", "float64"]:
            exp_name = f"float-precision-test-pythia-14m-{dtype}"
            # Note that ablation_config["dtype"] does not matter much, but whether the file in
            # ablation_config["interaction_graph_path"] is float32 or float64 does matter a lot.
            ablation_config["dtype"] = dtype
            ablation_config["exp_name"] = exp_name
            ablation_config["interaction_graph_path"] = f"{temp_dir}/{exp_name}_rib_Cs.pt"
            if not torch.cuda.is_available():
                # Try to reduce memory usage for CI
                ablation_config["batch_size"] = 1
            print("Running ablations with batch size", ablation_config["batch_size"], "for", dtype)
            ablation_main(AblationConfig(**ablation_config))
            ablation_result = json.load(open(f"{temp_dir}/{exp_name}_ablation_results.json"))[
                "results"
            ]
            ablation_results[dtype] = ablation_result

        # After running the ablations we no longer need to keep the dir since both the RIB build
        # and ablation results are loaded in memory inside these dicts.
        temp_object.cleanup()

        return ablation_results

    # Skip because broken on CI (but works on devboxes)
    @pytest.mark.skip_ci
    def test_ablation_result_float_precision(self, ablation_results: dict) -> None:
        # ln2.3 (and others) are broken (https://github.com/ApolloResearch/rib/issues/212)
        # ln1.- are broken. ln1.0 seemed fine on GPU (a6000) but broken on CPU
        for node_layer in ablation_results["float32"].keys():
            if node_layer in ["ln2.3", "ln1.5", "ln1.0"]:
                continue
            for n_vecs_ablated in ablation_results["float32"][node_layer].keys():
                float32_ablation_result = ablation_results["float32"][node_layer][n_vecs_ablated]
                float64_ablation_result = ablation_results["float64"][node_layer][n_vecs_ablated]
                assert torch.allclose(
                    torch.tensor(float32_ablation_result),
                    torch.tensor(float64_ablation_result),
                    atol=1e-3,
                ), f"Float difference {node_layer} {n_vecs_ablated}: {float32_ablation_result} (float32) != {float64_ablation_result} (float64), full results: {ablation_results['float32'][node_layer]} (float32) != {ablation_results['float64'][node_layer]} (float64)"

    # Skip because broken on CI (but works on devboxes)
    @pytest.mark.skip_ci
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_ablation_result_flatness(self, ablation_results: dict, dtype: str) -> None:
        for node_layer in ablation_results["float32"].keys():
            if "mlp_out" in node_layer:
                # Should be identical due to residual stream size
                ablation_result_128 = ablation_results[dtype][node_layer]["128"]
                ablation_result_642 = ablation_results[dtype][node_layer]["642"]
                assert torch.allclose(
                    torch.tensor(ablation_result_128),
                    torch.tensor(ablation_result_642),
                    atol=1e-3,
                ), f"MLP non-flat ablation curve {dtype} {node_layer}: {ablation_result_128} (128) != {ablation_result_642} (642), full results: {ablation_results[dtype][node_layer]}"
