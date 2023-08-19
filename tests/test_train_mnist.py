"""Run the mnist train script with a mock config and check that accuracy is > 95%.
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Append the root directory to sys.path
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

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


@pytest.mark.slow
def test_main_accuracy():
    # Create a temporary file and write the mock config to it
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml") as temp_config:
        temp_config.write(MOCK_CONFIG)
        temp_config.flush()

        accuracy = train_main(temp_config.name)
        assert accuracy > 95.0
