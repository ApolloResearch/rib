# rib

Library containing methods related to Rotation into the Interaction Basis.

## Installation

From the root of the repository, run

```bash
pip install -e .
```

## Usage

### MNIST

To train an MLP on MNIST, define/select a config file (see `configs/mnist/train_mnist_*.yaml` for examples) and run

```bash
python scripts/train_mnist.py <path_to_config_file>
```

You may be asked to enter your wandb API key, which you can find it in your [wandb account settings](https://wandb.ai/settings). Alternatively, you can set the environment variable `WANDB_API_KEY` to your API key.

To evaluate the impact of ablating eigenvectors from the orthogonal basis, define/select a config (see `configs/mnist/mnist_orthogonal_ablation_*.yaml` for examples) and run

```bash
python scripts/mnist_orthogonal_rotation.py <path_to_config_file>
```

## Development

To install the development dependencies that includes formatters, linters, and type checkers, run

```bash
pip install -e ".[dev]"
```

Suggested extensions and settings for VSCode are provided in `.vscode/`.

### Pre-commit hooks

A pre-commit hook is saved in the .pre-commit file. To use this hook, copy it to the .git/hooks/ folder and make it executable (i.e. `cp .pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit`).