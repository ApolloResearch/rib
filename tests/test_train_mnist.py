"""Run the mnist train script with a mock config and check that accuracy is > 95%.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Append the root directory to sys.path
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

from experiments.train_mnist.train import evaluate_model
from experiments.train_mnist.train import main as train_main

MOCK_CONFIG = """
seed: 0
model:
  hidden_sizes: [100, 100]
  activation_fn: relu
  bias: true
  fold_bias: true
train:
  learning_rate: 0.001
  batch_size: 64
  epochs: 3
  save_dir: null
  save_every_n_epochs: null
wandb: null
"""


def mock_evaluate_model(*args, **kwargs):
    # Call the original function to get the real accuracy
    accuracy = evaluate_model(*args, **kwargs)
    mock_evaluate_model.accuracy = accuracy  # Store the accuracy on the mock itself
    return accuracy


@pytest.mark.slow
def test_main_accuracy():
    # Create a temporary file and write the mock config to it
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml") as temp_config:
        temp_config.write(MOCK_CONFIG)
        temp_config.flush()

        with patch("experiments.train_mnist.train.evaluate_model", side_effect=mock_evaluate_model):
            # Call the main function
            train_main(temp_config.name)

            # Assert the accuracy from our mock function
            assert mock_evaluate_model.accuracy > 95
