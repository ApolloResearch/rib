"""Run the mnist train script and checks that accuracy is > 95%.
"""
import pytest
import yaml

from experiments.train_mnist.run_train_mnist import Config
from experiments.train_mnist.run_train_mnist import main as train_main

CONFIG_STR = """
seed: 0
model:
  input_size: 784
  output_size: 10
  hidden_sizes: [30, 30]
  activation_fn: relu
  bias: true
train:
  learning_rate: 0.001
  batch_size: 64
  epochs: 1
  save_dir: null
  save_every_n_epochs: null
wandb: null
"""


@pytest.mark.slow
def test_main_accuracy():
    """Test that the accuracy of the model is > 95%."""

    config_dict = yaml.safe_load(CONFIG_STR)
    config = Config(**config_dict)
    accuracy = train_main(config)
    assert accuracy > 90.0
