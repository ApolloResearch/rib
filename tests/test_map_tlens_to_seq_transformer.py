"""Run the modular arithmetic train script with a mock config and check that train accuracy is 100%.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

# Append the root directory to sys.path
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

from experiments.convert_tflens.run_convert import main as convert_tflens_main
from experiments.train_modular_arithmetic.run_train_modular_arithmetic import (
    evaluate_model,
    train_model,
)
from rib.data import ModularArithmeticDataset


def test_trained_model_conversion():
    """Test that the accuracy of a trained transformer_lens modular arithmetic model
    matches after conversion to a sequential_transformer.
    """

    convert_config = """
    seed: 0
    tlens_pretrained: null
    tlens_model_path: C:/Users/Avery/Projects/apollo/data/saved_models/modular_arithmetic/checkpoints/lr-0.001_bs-10000_2023-08-19_11-39-21/model_epoch_45000.pt
    node_layers: ["embed", "attn.0", "mlp_out.0"]
    dtype: float32
    """

    # Create a temporary file and write the mock config to it
    temp_config = tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False)
    temp_config.write(convert_config)
    temp_config.close()

    # defining this manually won't be a problem in most cases but it should really be based on model's config
    data = ModularArithmeticDataset(modulus=113, frac_train=0.30, seed=0, train=True)
    data_loader = DataLoader(data, batch_size=10000, shuffle=False)

    tlens_model, seq_model = convert_tflens_main(temp_config.name)

    tlens_model_accuracy = evaluate_model(tlens_model, data_loader, "cpu")
    seq_model_accuracy = evaluate_model(seq_model, data_loader, "cpu")

    assert tlens_model_accuracy == seq_model_accuracy


def test_random_model_conversion():
    """Test that the accuracy of a randomly initialized transformer_lens modular arithmetic model
    matches after conversion to a sequential_transformer.
    """

    convert_config = """
    seed: 0
    tlens_pretrained: null
    tlens_model_path: null
    node_layers: ["embed", "attn.0", "mlp_out.0"]
    dtype: float32
    """

    # Create a temporary file and write the mock config to it
    temp_config = tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False)
    temp_config.write(convert_config)
    temp_config.close()

    data = ModularArithmeticDataset(modulus=113, frac_train=0.30, seed=0, train=True)
    data_loader = DataLoader(data, batch_size=temp_config.train.batch_size, shuffle=False)

    tlens_model, seq_model = convert_tflens_main(temp_config.name)

    tlens_model_accuracy = evaluate_model(tlens_model, data_loader, "cpu")
    seq_model_accuracy = evaluate_model(seq_model, data_loader, "cpu")

    assert tlens_model_accuracy == seq_model_accuracy


def test_pretrained_gpt2_conversion():
    """Test that the outputs of a pretrained gpt2 transformer_lens model
    matches after conversion to a sequential_transformer.
    """

    convert_config = """
    seed: 0
    tlens_pretrained: gpt2
    tlens_model_path: null
    node_layers: ["embed", "attn.0", "mlp_out.0"]
    dtype: float32
    """

    # Create a temporary file and write the mock config to it
    temp_config = tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False)
    temp_config.write(convert_config)
    temp_config.close()

    tlens_model, seq_model = convert_tflens_main(temp_config.name)

    input_ids = torch.randint(0, seq_model.cfg.d_vocab, size=(10, seq_model.cfg.n_ctx))
    tlens_output = tlens_model(input_ids)
    seq_output = seq_model(input_ids)

    assert torch.allclose(tlens_output, seq_output)


@pytest.mark.slow
def test_training_model_conversion():
    """Test that the outputs of a sample of data for a pretrained transformer_lens
    modular arithmetic model matches after conversion to a sequential_transformer.
    """

    convert_config = """
    seed: 0
    tlens_pretrained: null
    tlens_model_path: null
    node_layers: ["embed", "attn.0", "mlp_out.0"]
    dtype: float32
    """

    train_config = {
        "seed": 0,
        "modulus": 113,
        "frac_train": 0.25,
        "fn_name": "add",
        "learning_rate": 0.001,
        "batch_size": 10000,
        "epochs": 1000,
        "save_dir": None,
        "save_every_n_epochs": None,
        "wandb": None,
    }

    # Create a temporary file and write the mock config to it
    temp_config = tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False)
    temp_config.write(convert_config)
    temp_config.close()

    train_data = ModularArithmeticDataset(
        train_config["modulus"], train_config["frac_train"], seed=train_config["seed"], train=True
    )
    train_data_loader = DataLoader(train_data, batch_size=train_config["batch_size"], shuffle=False)

    test_data = ModularArithmeticDataset(
        train_config["modulus"], train_config["frac_train"], seed=train_config["seed"], train=False
    )
    test_loader = DataLoader(test_data, batch_size=train_config["batch_size"], shuffle=False)

    tlens_model, seq_model = convert_tflens_main(temp_config.name)

    tlens_model = train_model(
        config=train_config,
        model=tlens_model,
        train_loader=train_data_loader,
        test_loaded=test_loader,
        device="cpu",
        run_name="test",
    )
    seq_model = train_model(
        config=train_config,
        model=seq_model,
        train_loader=train_data_loader,
        test_loaded=test_loader,
        device="cpu",
        run_name="test",
    )

    tlens_model_accuracy = evaluate_model(tlens_model, test_loader, "cpu")
    seq_model_accuracy = evaluate_model(seq_model, test_loader, "cpu")

    assert tlens_model_accuracy == seq_model_accuracy


if __name__ == "__main__":
    test_trained_model_conversion()
    test_random_model_conversion()
    test_pretrained_gpt2_conversion()
    test_training_model_conversion()
