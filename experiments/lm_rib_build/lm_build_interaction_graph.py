"""This script builds a RIB graph for a language model.

We build the graph using a SequentialTransformer model, but the model weights can be loaded
from a transformer-lens model and ported to a SequentialTransformer model.

Steps to build the graph:
1. Load a model from transformerlens (either from_pretrained or via ModelConfig).
2. Fold in the biases into the weights.
3. Convert the model to a SequentialTransformer model.
4. Given a list of modules we wish to build the graph around, create a list of MultiSequential
    models that can be passed to jacrev. (Note that this means we will effectively have two models
    in memory.)
5. Collect the gram matrices at each node layer using the full model (as in mnist_rib_build).
6. Calculate Cs, but when calculating the jacobian inside interaction_edge_forward_hook_fn, we
    use the MultiSequential model instead of the module passed as the first argument to the hook_fn.


"""
import fire


def main(config_path_str: str) -> None:
    """Build the interaction graph and store it on disk."""


if __name__ == "__main__":
    fire.Fire(main)
