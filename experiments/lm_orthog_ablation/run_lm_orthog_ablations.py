"""
<This file should do for lms what `experiments/mnist_orthog_ablation/run_orthog_ablations.py` does
for mnist.>

Run an lm while rotating to and from a (truncated) orthogonal basis.

The process is as follows:
    1. Load an LM trained on a dataset and a test set of sentences.
    2. Calculate gram matrices at each node layer.
    3. Calculate a rotation matrix at each node layer representing the operation of rotating to and
        from the partial eigenbasis of the gram matrix. The partial eigenbasis is equal to the
        entire eigenbasis with the zeroed out eigenvectors corresponding to the n smallest
        eigenvalues. The values of n follow the ablation schedule given below.
    4. Run the test set through the LM, applying the rotations at each node layer, and calculate
        the resulting accuracy.
    5. Repeat steps 3 and 4 for a range of values of n, plotting the resulting accuracies.

A node layer is positioned at the input of each module_name specified in the config. In this script,
we don't create a node layer at the output of the final module, as ablating nodes in this layer
is not useful.

Usage:
    python run_lm_orthog_ablations.py <path/to/yaml_config_file>

"""

import json
from pathlib import Path
from typing import Literal, Optional

import fire
import torch
import yaml
from datasets import load_dataset
from pydantic import BaseModel, Field, field_validator
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens import HookedTransformer
from transformers import GPT2TokenizerFast

from rib.ablations import ablate_and_test
from rib.data import ModularArithmeticDataset
from rib.data_accumulator import collect_gram_matrices
from rib.hook_manager import HookedModel
from rib.linalg import eigendecompose
from rib.log import logger
from rib.models import SequentialTransformer, SequentialTransformerConfig
from rib.models.sequential_transformer.converter import convert_tlens_weights
from rib.types import TORCH_DTYPES
from rib.utils import (
    REPO_ROOT,
    calc_ablation_schedule,
    eval_model_accuracy,
    load_config,
    overwrite_output,
    set_seed,
)


class Config(BaseModel):
    exp_name: str = Field(..., description="The name of the experiment")
    tlens_pretrained: Optional[Literal["gpt2"]] = Field(
        None, description="Pretrained transformer lens model."
    )
    tlens_model_path: Optional[Path] = Field(
        None, description="Path to saved transformer lens model."
    )
    dataset: Literal["modular_arithmetic", "wikitext"] = Field(
        ...,
        description="The dataset to use to build the graph.",
    )
    batch_size: int = Field(..., description="The batch size to use when building the graph.")

    last_pos_only: bool = Field(
        False,
        description="Whether the unembed module should only be applied to the last position.",
    )
    ablate_every_vec_cutoff: Optional[int] = Field(
        None,
        description="The point at which we start ablating every individual vector. If None, always ablate every vector.",
    )
    dtype: str = Field(..., description="The dtype to use when building the graph.")
    node_layers: list[str] = Field(
        ..., description="Names of the node layers to build the graph with."
    )
    seed: int

    @field_validator("dtype")
    def dtype_validator(cls, v):
        assert v in TORCH_DTYPES, f"dtype must be one of {TORCH_DTYPES}"
        return v


@torch.inference_mode()
def evaluate_model(model, test_loader: DataLoader, device: str) -> float:
    """Evaluate the Transformer on Modular Arithmetic.

    Args:
        model: Transformer model.
        test_loader: DataLoader for the test set.
        device: Device to use for evaluation.

    Returns:
        Test accuracy.
    """

    # Test the model
    model.eval()
    correct = 0
    total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)[0]
        total += y.size(0)
        sm_argmax = nn.functional.softmax(outputs, dim=-1).argmax(dim=-1)[:, -1].detach()
        correct += (y == sm_argmax.view(-1)).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def run_ablations(
    hooked_model: HookedModel,
    data_loader: DataLoader,
    config: Config,
    graph_module_names: list[str],
    ablate_every_vec_cutoff: Optional[int],
    dtype: torch.dtype,
    device: str,
) -> dict[str, dict[int, float]]:
    """Rotate to and from orthogonal basis and compare accuracies with and without ablation.

    Args:
        hooked_model: The hooked model.
        data_loader: The data loader to use for testing.
        config: The config.
        graph_module_names: The names of the modules we want to build the graph around.
        ablate_every_vec_cutoff: The point in the ablation schedule to start ablating every vector.
        dtype: The data type to use for model computations.
        device: The device to run the model on.


    Returns:
        A dictionary mapping node lyae to accuracy results.
    """

    gram_matrices = collect_gram_matrices(
        hooked_model=hooked_model,
        module_names=graph_module_names,
        data_loader=data_loader,
        dtype=dtype,
        device=device,
        collect_output_gram=True,
        hook_names=config.node_layers,
    )
    results: dict[str, dict[int, float]] = {}
    for hook_name, module_name in zip(config.node_layers, graph_module_names):
        _, eigenvecs = eigendecompose(gram_matrices[hook_name])

        ablation_schedule = calc_ablation_schedule(
            ablate_every_vec_cutoff=ablate_every_vec_cutoff,
            n_vecs=len(eigenvecs),
        )
        module_accuracies: dict[int, float] = ablate_and_test(
            hooked_model=hooked_model,
            module_name=module_name,
            interaction_rotation=eigenvecs,
            interaction_rotation_pinv=eigenvecs.T,
            test_loader=data_loader,
            device=device,
            ablation_schedule=ablation_schedule,
            hook_name=hook_name,
        )

        results[hook_name] = module_accuracies

    return results


def load_sequential_transformer(config: Config, device: str) -> tuple[SequentialTransformer, dict]:
    """Load a SequentialTransformer model from a pretrained transformerlens model.

    Requires config to contain a pretrained model name or a path to a transformerlens model.

    First loads a HookedTransformer model, then uses its config to create an instance of
    SequentialTransformerConfig, which is then used to create a SequentialTransformer.

    Args:
        config (Config): The config, containing either `tlens_pretrained` or `tlens_model_path`.
        device (str): The device to load the model on.

    Returns:
        - SequentialTransformer: The SequentialTransformer model.
        - dict: The config used in the transformerlens model.
    """

    if config.tlens_pretrained is not None:
        tlens_model = HookedTransformer.from_pretrained(config.tlens_pretrained)
        # Create a SequentialTransformerConfig from the HookedTransformerConfig
        tlens_cfg_dict = tlens_model.cfg.to_dict()
    elif config.tlens_model_path is not None:
        with open(config.tlens_model_path.parent / "config.yaml", "r") as f:
            # The config specified in the YAML file used to train the tlens model
            provided_tlens_cfg_dict = yaml.safe_load(f)["model"]
        tlens_model = HookedTransformer(provided_tlens_cfg_dict)
        # The entire tlens config (including default values)
        tlens_cfg_dict = tlens_model.cfg.to_dict()

        # Load the weights from the tlens model
        tlens_model.load_state_dict(torch.load(config.tlens_model_path, map_location=device))

    seq_cfg = SequentialTransformerConfig(**tlens_cfg_dict)
    # Set the dtype to the one specified in the config for this script (as opposed to the one used
    # to train the tlens model)
    seq_cfg.dtype = TORCH_DTYPES[config.dtype]
    seq_model = SequentialTransformer(seq_cfg, config.node_layers, config.last_pos_only)

    # Load the transformer-lens weights into the sequential transformer model
    state_dict = convert_tlens_weights(list(seq_model.state_dict().keys()), tlens_model)
    seq_model.load_state_dict(state_dict)

    return seq_model, tlens_cfg_dict


def create_data_loader(config: Config, train: bool = False) -> DataLoader:
    """Create a DataLoader for the dataset specified in `config.dataset`.

    Args:
        config (Config): The config, containing the dataset name.

    Returns:
        DataLoader: The DataLoader to use for building the graph.
    """
    if config.dataset == "modular_arithmetic":
        # Get the dataset config from our training config
        assert config.tlens_model_path is not None, "tlens_model_path must be specified"
        with open(config.tlens_model_path.parent / "config.yaml", "r") as f:
            # The config specified in the YAML file used to train the tlens model
            train_config = yaml.safe_load(f)["train"]
        test_data = ModularArithmeticDataset(
            train_config["modulus"], train_config["frac_train"], seed=config.seed, train=train
        )
        # Note that the batch size for training typically gives 1 batch per epoch. We use a smaller
        # batch size here, mostly for verifying that our iterative code works.
        test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)
    elif config.dataset == "wikitext":
        # Step 1: Load a sample language modelling dataset
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test[:30%]")

        # Step 2: Tokenize using GPT-2 tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Remove empty data points
        dataset = dataset.filter(lambda example: len(example["text"]) > 0)
        tokenized_dataset = dataset.map(
            lambda examples: tokenizer(
                examples["text"], truncation=True, padding="max_length", max_length=1024
            ),
            batched=True,
        )
        # Create a dataloader from the Dataset
        input_ids = torch.tensor(tokenized_dataset["input_ids"], dtype=torch.long)

        # Create labels by shifting input_ids by 1
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = tokenizer.pad_token_id

        test_loader = DataLoader(
            TensorDataset(input_ids, labels), batch_size=config.batch_size, shuffle=True
        )
    return test_loader


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)

    set_seed(config.seed)

    if config.exp_name is not None:
        out_file = Path(__file__).parent / "out" / f"{config.exp_name}_orthog_ablation_results.json"
        if out_file.exists() and not overwrite_output(out_file):
            print("Exiting.")
            return
        out_file.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = TORCH_DTYPES[config.dtype]
    seq_model, _ = load_sequential_transformer(config, device)
    seq_model.eval()
    seq_model.to(device=torch.device(device), dtype=dtype)
    seq_model.fold_bias()
    hooked_model = HookedModel(seq_model)

    data_loader = create_data_loader(config, train=False)
    assert (
        config.tlens_pretrained is None
    ), "Currently can't build graphs for pretrained models due to memory limits."

    # Test model accuracy before ablation
    accuracy = eval_model_accuracy(hooked_model, data_loader, dtype=dtype, device=device)
    logger.info("Accuracy before ablation: %.2f%%", accuracy * 100)

    graph_module_names = [f"sections.{sec}" for sec in seq_model.sections if sec != "pre"]

    accuracies: dict[str, dict[int, float]] = run_ablations(
        hooked_model=hooked_model,
        data_loader=data_loader,
        config=config,
        graph_module_names=graph_module_names,
        ablate_every_vec_cutoff=config.ablate_every_vec_cutoff,
        dtype=dtype,
        device=device,
    )
    results = {
        "exp_name": config.exp_name,
        "config": json.loads(config.model_dump_json()),
        "accuracies": accuracies,
    }
    if config.exp_name is not None:
        with open(out_file, "w") as f:
            json.dump(results, f)
        logger.info("Wrote results to %s", out_file)


if __name__ == "__main__":
    fire.Fire(main)
