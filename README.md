# rib

Library containing methods related to Rotation into the Interaction Basis.

## Installation

From the root of the repository, run

```bash
pip install -e .
```

## Usage

### MNIST

To train an MLP on MNIST, define a config file (see `configs/mnist.yaml` for an example) and run

```bash
python scripts/train_mnist.py --config <path_to_config_file>
```

You may be asked to enter your wandb API key, which you can find it in your [wandb account settings](https://wandb.ai/settings). Alternatively, you can set the environment variable `WANDB_API_KEY` to your API key.

## Development

To install the development dependencies that includes formatters, linters, and type checkers, run

```bash
pip install -e ".[dev]"
```

Suggested extensions and settings for VSCode are provided in `.vscode/`.

### Pre-commit hooks

This repository uses [pre-commit](https://pre-commit.com/) to run a series of checks on the code before committing. These are installed with the development dependencies above.