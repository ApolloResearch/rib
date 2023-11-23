"""Run the modular arithmetic train script and check that train accuracy is >5%."""
import pytest
import yaml

from experiments.train_modular_arithmetic.run_train_modular_arithmetic import Config
from experiments.train_modular_arithmetic.run_train_modular_arithmetic import (
    main as train_main,
)

CONFIG_STR = """
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
  normalization_type: null
dataset:
  source: custom
  name: modular_arithmetic
  return_set: both
  modulus: 113
  frac_train: .30
  fn_name: add
  seed: 0
train:
  learning_rate: 0.001
  batch_size: 10000
  epochs: 50
  eval_every_n_epochs: 40
  save_dir: null
  save_every_n_epochs: null
wandb: null
"""


@pytest.mark.slow
def test_main_accuracy():
    """Test that the accuracy of the model is above 5% after 50 epochs.

    We should reach >99% after 200 epochs, but this takes too long to run in CI.
    """
    config_dict = yaml.safe_load(CONFIG_STR)
    config = Config(**config_dict)
    train_accuracy, _ = train_main(config)
    assert train_accuracy > 5
