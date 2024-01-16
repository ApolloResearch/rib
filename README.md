# rib

This repository contains the core functionality and experiments related to Rotation into the
Interaction Basis.

For a formal introduction to the method, see
[this writeup](https://www.overleaf.com/project/6516ddc99f52dd99cab58d8d).

## Installation

From the root of the repository, run

```bash
pip install -e .
```

## Usage

The core functionality of RIB is contained in the `rib` package. Experiments that use RIB are
contained in the experiments directory, with the directory encapsulating related files such as
configs, scripts, and outputs.

Most experiment scripts take in a yaml config file or a json data file as cli argument. These files reside in
either the same directory as the script or a child directory. The docstrings of the
scripts describe how to use them.

You may be asked to enter your wandb API key, which you can find it in your
[wandb account settings](https://wandb.ai/settings). Alternatively, you can set the environment
variable `WANDB_API_KEY` to your API key.

### MNIST

Supported experiments:

- Training an MLP: `experiments/train_mnist/`
- Ablating vectors from the interaction or orthogonal basis: `experiments/mnist_ablations/`
- Building an interaction graph: `experiments/mnist_rib_build/`

### LMs

Language models are implemented in the `rib.models.transformer.SequentialTransformer` class. This
is a sequential version of a transformer consisting of a series of modules. The inputs to a module
are the unmodified outputs to the previous module.

An image of the Sequential Transformer architecture is provided [here](docs/SequentialTransformer.drawio.png).

Supported experiments:

- Training a 1-layer LM on a modular arithmetic task: `experiments/train_modular_arithmetic/`
- Ablating vectors from the interaction or orthogonal basis: `experiments/lm_ablations/`
- Building an interaction graph: `experiments/lm_rib_build/`

As can be seen in `experiments/lm_rib_build/run_lm_rib_build.py`, the process for building a graph
for an LM is as follows:

- Load a pretrained LM (currently only supports some transformer-lens models and modular arithmetic).
- Map the LM to a SequentialTransformer model, which allows us to analyse (e.g. take jacobians of)
arbitrary sections of the LM.
- Fold in the model's biases into the weights. This is required for our integrated gradient formalism.
- Run the RIB algorithm, outlined in the Code Implementation section of [this writeup](https://www.overleaf.com/project/6516ddc99f52dd99cab58d8d).
- Plot the RIB graph using `experiments/lm_rib_build/plot_lm_graph.py`, passing in the path to the
results file generated from `experiments/lm_rib_build/run_lm_rib_build.py`

### Bases and attributions

There are four basis formulas and two edges formulas implemented. Sensible combinations are:
* `jacobian` basis with `squared` edges: Most up-to-date and possibly correct version
* `(1-0)*alpha` basis with `squared` edges: Used for OP report, but the Lambdas are technically
  wrong. Can and does produce stray edges.
* `(1-alpha)^2` basis with `functional` edges: Old functional-based approach. Self-consistent (and
  working Lambdas) but we know counterexampes where this method would give wrong results.

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
