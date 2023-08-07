# rib

This repository contains the core functionality and experiments related to Rotation into the
Interaction Basis.

For a formal introduction to the method, see
[this writeup](https://www.overleaf.com/project/6437d0bde0eaf2e8c7ac3649).

## Installation

From the root of the repository, run

```bash
pip install -e .
```

## Usage

The core functionality of RIB is contained in the `rib` package. Experiments that use RIB are
contained in the experiments directory, with the directory encapsulating related files such as
configs, scripts, and outputs.

Many scripts take in a yaml config file or a json data file as cli argument. These files reside in
either the same directory as the script or a child directory. The docstrings of the
scripts describe how to use them.

You may be asked to enter your wandb API key, which you can find it in your
[wandb account settings](https://wandb.ai/settings). Alternatively, you can set the environment
variable `WANDB_API_KEY` to your API key.

### MNIST

Supported experiments:

- Training an MLP on MNIST: `experiments/train_mnist/`
- Ablating eigenvectors from the orthogonal basis: `experiments/mnist_orthog_ablation/`
- Calculating the interaction graph: `experiments/mnist_rib_build/`

## Development

To install the development dependencies that includes formatters, linters, and type checkers, run

```bash
pip install -e ".[dev]"
```

Suggested extensions and settings for VSCode are provided in `.vscode/`.

### Pre-commit hooks

A pre-commit hook is saved in the .pre-commit file. To use this hook, copy it to the `.git/hooks/`
dir and make it executable
(`cp .pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit`).
