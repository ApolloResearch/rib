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
- Ablating vectors from the interaction basis: `experiments/mnist_rib_ablation/`

### LMs

Supported experiments:

- Training a transformer_lens model on Modular Arithmetic: `experiments/train_modular_arithmetic/`
- Building the interaction graph for a LM: `experiments/lm_rib_build/`. Currently supports the
modular arithmetic and gpt2 models (loading via transformer-lens). Although any model other than
the modular arithmetic will currently fail due to severe memory issues.

As can be seen in `experiments/lm_rib_build/lm_build_rib_graph.py`, the process for building a graph
for an LM is as follows:
- Load a pretrained LM (currently only supports some transformer-lens models)
- Map the LM to a SequentialTransformer model, which allows us to analyse (e.g. take jacobians of)
arbitrary sections of the LM.
- Fold in the model's biases into the weights. This is required for the RIB decompositions to
consider all the relevant interactions.
- Run the RIB algorithm, outlined in the Code Implementation section of [this writeup](https://www.overleaf.com/project/6437d0bde0eaf2e8c7ac3649).
- Plot the graph using `experiments/lm_rib_build/plot_rib_graph.py`, passing the path to the
results file generated from `experiments/lm_rib_build/lm_build_rib_graph.py`

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
