# rib

Library containing methods related to Rotation into the Interaction Basis.

## Installation

From the root of the repository, run

```bash
pip install -e .
```

## Usage
### MNIST
To train an MLP on MNIST, define a config file (see `rib/configs/mnist.yaml` for an example) and run

```bash
python scripts/train_mnist.py --config <path_to_config_file>
```

## Development

To install the development dependencies, run

```bash
pip install -e ".[dev]"
```

Suggested extensions and settings for VSCode are provided in `.vscode/`.

### Pre-commit hooks

To use the suggestion precommit hook copy the file `pre-commit` to `.git/hooks/` and make it executable (`chmod +x .git/hooks/pre-commit`).
