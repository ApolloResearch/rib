"""Run the modular arithmetic train script with a mock config and check that train accuracy is 100%.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Append the root directory to sys.path
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

from experiments.train_modular_arithmetic.run_train_modular_arithmetic import (
    main as train_main,
)

MOCK_CONFIG = """
seed: 0
model:
  n_layers: 1
  d_model: 128
  d_head: 32
  n_heads: 4
  d_mlp: 512
  d_vocab: 114
  n_ctx: 3
  act_fn: relu
  normalization_type: LN
train:
  modulus: 113
  frac_train: .25
  fn_name: add
  learning_rate: 0.001
  batch_size: 10000
  epochs: 100
  eval_every_n_epochs: 40
  save_dir: null
  save_every_n_epochs: null
wandb: null
"""


@pytest.mark.slow
def test_main_accuracy():
    """Test that the accuracy of the model is 100%.

    We don't use a context manager here because windows doesn't support opening temp files more than once.
    """
    # Create a temporary file and write the mock config to it
    temp_config = tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False)
    temp_config.write(MOCK_CONFIG)
    temp_config.close()

    train_accuracy, _ = train_main(temp_config.name)
    assert train_accuracy == 100.0

    Path(temp_config.name).unlink()
