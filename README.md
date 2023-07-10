# rib

This repository contains the core functionality and experiments related to Rotation into the Interaction Basis.

## Installation

From the root of the repository, run

```bash
pip install -e .
```

## Usage

The core functionality of RIB is contained in the `rib` package. Experiments that use RIB are
contained in the experiments directory, with the directory encapsulating related files such as
configs, scripts, and outputs.

You may be asked to enter your wandb API key, which you can find it in your [wandb account settings](https://wandb.ai/settings). Alternatively, you can set the environment variable `WANDB_API_KEY` to your API key.

### MNIST

To train an MLP on MNIST, define/select a config file (see `experiments/train_mnist/*.yaml` for examples) and run

```bash
python experiments/train_mnist/train.py <path_to_config_file>
```

To evaluate the impact of ablating eigenvectors from the orthogonal basis, define/select a config (see `experiments/mnist_orthog_ablation/*yaml` for examples) and run

```bash
python experiments/mnist_orthog_ablation/run_ablations.py <path_to_config_file>
```

and then

```bash
python experiments/mnist_orthog_ablation/plot_ablations.py <path_to_results_file>
```

to plot the results.

## Development

To install the development dependencies that includes formatters, linters, and type checkers, run

```bash
pip install -e ".[dev]"
```

Suggested extensions and settings for VSCode are provided in `.vscode/`.

### Pre-commit hooks

A pre-commit hook is saved in the .pre-commit file. To use this hook, copy it to the .git/hooks/ folder and make it executable (i.e. `cp .pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit`).
