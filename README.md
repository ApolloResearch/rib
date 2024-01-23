# rib

This repository contains the core functionality and rib_scripts related to Rotation into the
Interaction Basis (RIB).

For a formal introduction to the method, see
[this writeup](https://www.overleaf.com/project/65534543ea5ce85765a0a6f3).

## Installation

From the root of the repository, run

```bash
pip install -e .
```

## Usage

The core functionality of RIB is contained in the `rib` package. Experiments that use RIB are
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
- Ablating vectors from the a basis (e.g. RIB or orthogonal basis): `rib_scripts/ablations/`
- Building a RIB graph (calculating the basis and the edges): `rib_scripts/rib_build/`

The ablations and rib_build scripts work for both MLPs and transformers.

Language models are implemented in the `rib.models.transformer.SequentialTransformer` class. This
is a sequential version of a transformer consisting of a series of modules. The inputs to a module
are the unmodified outputs to the previous module.

An image of the Sequential Transformer architecture is provided [here](docs/SequentialTransformer.drawio.png).

As can be seen in `rib_scripts/rib_build/run_rib_build.py`, the process for building a graph is as
follows:

- Load a pretrained model (currently supports some transformer-lens models, modular arithmetic and MLPs).
- (If transformer) Map the LM to a SequentialTransformer model, which allows us to build the graph
around arbitrary sections of the LM.
- Fold in the model's biases into the weights. This is required for our integrated gradient formalism.
- Run the RIB algorithm, outlined in the Code Implementation section of [this writeup](https://www.overleaf.com/project/65534543ea5ce85765a0a6f3).
- Plot the RIB graph using `rib_scripts/rib_build/plot_graph.py`, passing in the path to the
results file generated from `rib_scripts/rib_build/run_lm_rib_build.py`.

### Bases and attributions

There are four basis formulas and two edges formulas implemented. Sensible combinations are:
* `jacobian` basis with `squared` edges: Most up-to-date and possibly correct version
* `(1-0)*alpha` basis with `squared` edges: Used for OP report, but the Lambdas are technically
  wrong. Can and does produce stray edges.
* `(1-alpha)^2` basis with `functional` edges: Old functional-based approach. Self-consistent (and
  working Lambdas) but we know counterexampes where this method would give wrong results.

### Math equations
`jacobian` basis:

![image](https://github.com/ApolloResearch/rib/assets/148209923/931e8851-6bf3-47c6-a7b1-faef8a7d02a7)

with Lambda TODO

`(1-0)*alpha` basis:

![image](https://github.com/ApolloResearch/rib/assets/148209923/130433bf-57ce-47f0-9201-3c7009ecbfd4)

with Lambda

![image](https://github.com/ApolloResearch/rib/assets/148209923/c46243aa-b076-44d1-9d75-6662041caf8f)

`squared` attribution:

![image](https://github.com/ApolloResearch/rib/assets/148209923/d46933bf-be93-4270-a732-830c8fe446ca)

`functional` attribution and `(1-alpha)^2` basis are deprecated and not documented in the current version of the LaTeX doc.

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
