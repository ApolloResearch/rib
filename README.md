# rib (LIB)

This repository contains the core functionality related to Local Interaction Basis (LIB) method.
This method was previously named RIB; but this code base will not be updated to the new name.

This code accompanies the paper TODO.

## Installation

From the root of the repository, run

```bash
pip install -e .
```

## Usage

The core functionality of RIB (LIB) is contained in the `rib` package. Experiments that use RIB are
contained in the rib_scripts directory, with the directory encapsulating related files such as
configs, scripts, and outputs.

Most scripts take in a yaml config file or a json data file as cli argument. These files reside in
the same directory as the script. The docstrings of the scripts describe how to use them.

You may be asked to enter your wandb API key, which you can find in your
[wandb account settings](https://wandb.ai/settings). Alternatively, you can set the environment
variable `WANDB_API_KEY` to your API key.

Supported rib_scripts:

- Training an MLP (e.g. on MNIST or CIFAR): `rib_scripts/train_mlp/`
- Training a transformer on modular arithmetic: `rib_scripts/train_modular_arithmetic/`
- Building a RIB graph (calculating the basis and the edges): `rib_scripts/rib_build/`
- Ablating vectors from a RIB/SVD basis, or edges from a graph: `rib_scripts/ablations/`

The ablations and rib_build scripts work for both MLPs and transformers.

Language models are implemented in the `rib.models.transformer.SequentialTransformer` class. This
is a sequential version of a transformer consisting of a series of modules. The inputs to a module
are the unmodified outputs to the previous module.

An image of the Sequential Transformer architecture is provided [here](docs/SequentialTransformer.drawio.png).

The process for building a graph is as follows. The first steps are all covered by `rib_scripts/rib_build/run_rib_build.py`
- Load a pretrained model (currently supports some transformer-lens models, modular arithmetic and MLPs).
- (If transformer) Map the LM to a SequentialTransformer model, which allows us to build the graph
around arbitrary sections of the LM.
- Fold in the model's biases into the weights. This is required for our integrated gradient formalism.
- Run the RIB algorithm, finding a basis for each layer and computing the interaction edges between them.
- Plot the RIB graph using `rib_scripts/rib_build/plot_graph.py`, passing in the path to the
results file generated from `rib_scripts/rib_build/run_rib_build.py`.

### Basis and attribution settings

There are four basis formulas and two edges formulas implemented. Sensible combinations are:
* `jacobian` basis with `squared` edges: Most up-to-date and possibly correct version
* `(1-0)*alpha` basis with `squared` edges: Lambdas are technically wrong. Can and does produce stray edges.
* `(1-alpha)^2` basis with `functional` edges: Old functional-based approach. Self-consistent (and
  working Lambdas) but we know counterexampes where this method would give bad results.

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


### Testing

Tests are written using `pytest`. By default, only "fast" tests are run. This should be very fast
on a gpu and tolerably fast on a cpu. To run all tests, use `pytest --runslow`.

There are some tests that check RIB builds can be distributed across multiple GPUs. These tests are
skipped by default, as running multiple such tests in a single pytest process causes mpi errors.
To run these, use the `tests/run_distributed_tests.sh` script.
