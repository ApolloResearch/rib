"""Run the mnist train script with a mock config and check that accuracy is > 95%.
"""
import pytest
import yaml

from experiments.train_mnist.run_train_mnist import Config
from experiments.train_mnist.run_train_mnist import main as train_main

MOCK_CONFIG = """
seed: 0
model:
  hidden_sizes: [30, 30]
  activation_fn: relu
  bias: true
  fold_bias: true
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
    """Test that the accuracy of the model is > 95%.

    We don't use a context manager here because windows doesn't support opening temp files more than once.
    """

    config_dict = yaml.safe_load(MOCK_CONFIG)
    config = Config(**config_dict)
    accuracy = train_main(config)
    assert accuracy > 90.0
