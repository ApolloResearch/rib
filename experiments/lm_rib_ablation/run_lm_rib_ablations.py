"""Ablate vectors from the interaction basis and calculate the difference in loss.

The process is as follows:
    1. Load a SequentialTransformer model and dataset.
    2. For each number of ablated nodes `n`, create a rotation matrix that has the effect of
    rotating to and from the interaction basis with `n` fewer basis vectors.
    3. Run the test set through the model, applying the above rotations at each node layer, and
    calculate the resulting accuracy.
    5. Repeat steps 3 and 4 for a range of values of n, plotting the resulting accuracies.

A node layer is positioned at the input of each module_name specified in the config. In this script,
we don't create a node layer at the output of the final module, as ablating nodes in this layer
is not useful.

Usage:
    python run_lm_rib_ablations.py <path/to/yaml_config_file>

"""

import json
from pathlib import Path
from typing import Any, Optional

import fire
import torch
import yaml
from datasets import load_dataset
from jaxtyping import Float
from pydantic import BaseModel, Field, field_validator
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens import HookedTransformer
from transformers import GPT2TokenizerFast

from rib.ablations import ablate_and_test
from rib.data import ModularArithmeticDataset
from rib.hook_manager import HookedModel
from rib.log import logger
from rib.models import SequentialTransformer, SequentialTransformerConfig
from rib.models.sequential_transformer.converter import convert_tlens_weights
from rib.types import TORCH_DTYPES
from rib.utils import (
    calc_ablation_schedule,
    eval_model_accuracy,
    load_config,
    overwrite_output,
    set_seed,
)


class Config(BaseModel):
    exp_name: Optional[str]
    interaction_graph_path: Path
    ablate_every_vec_cutoff: Optional[int] = Field(
        None,
        description="The point at which we start ablating every individual vector. If None, always ablate every vector.",
    )
    node_layers: list[str]
    batch_size: int
    dtype: str
    seed: int

    @field_validator("dtype")
    def dtype_validator(cls, v):
        assert v in TORCH_DTYPES, f"dtype must be one of {TORCH_DTYPES}"
        return v


def load_sequential_transformer(
    ablation_config: Config, graph_config_dict: dict, device: str
) -> SequentialTransformer:
    """Load a SequentialTransformer model from a pretrained transformerlens model.

    Requires config to contain a pretrained model name or a path to a transformerlens model.

    First loads a HookedTransformer model, then uses its config to create an instance of
    SequentialTransformerConfig, which is then used to create a SequentialTransformer.

    Args:
        ablation_config: The config for the ablation experiment.
        graph_config_dict: The config dictionary used to build the interaction graph.
        device: The device to run the model on.

    Returns:
        - SequentialTransformer: The SequentialTransformer model.
        - dict: The config used in the transformerlens model.
    """

    if graph_config_dict["tlens_pretrained"] is not None:
        tlens_model = HookedTransformer.from_pretrained(graph_config_dict["tlens_pretrained"])
        # Create a SequentialTransformerConfig from the HookedTransformerConfig
        tlens_cfg_dict = tlens_model.cfg.to_dict()
    elif graph_config_dict["tlens_model_path"] is not None:
        with open(Path(graph_config_dict["tlens_model_path"]).parent / "config.yaml", "r") as f:
            # The config specified in the YAML file used to train the tlens model
            provided_tlens_cfg_dict = yaml.safe_load(f)["model"]
        tlens_model = HookedTransformer(provided_tlens_cfg_dict)
        # The entire tlens config (including default values)
        tlens_cfg_dict = tlens_model.cfg.to_dict()
        # Load the weights from the tlens model
        tlens_model.load_state_dict(
            torch.load(graph_config_dict["tlens_model_path"], map_location=device)
        )

    seq_cfg = SequentialTransformerConfig(**tlens_cfg_dict)
    # Set the dtype to the one specified in the config for this script (as opposed to the one used
    # to train the tlens model)
    seq_cfg.dtype = TORCH_DTYPES[ablation_config.dtype]
    seq_model = SequentialTransformer(
        seq_cfg, ablation_config.node_layers, graph_config_dict["last_pos_only"]
    )

    # Load the transformer-lens weights into the sequential transformer model
    state_dict = convert_tlens_weights(list(seq_model.state_dict().keys()), tlens_model)
    seq_model.load_state_dict(state_dict)

    return seq_model


def create_data_loader(
    ablation_config: Config, graph_config_dict: dict, train: bool = False
) -> DataLoader:
    """Create a DataLoader for the dataset specified in `config.dataset`.

    Args:
        config (Config): The config, containing the dataset name.

    Returns:
        DataLoader: The DataLoader to use for building the graph.
    """
    if graph_config_dict["dataset"] == "modular_arithmetic":
        # Get the dataset config from our training config
        assert (
            graph_config_dict["tlens_model_path"] is not None
        ), "tlens_model_path must be specified"
        with open(Path(graph_config_dict["tlens_model_path"]).parent / "config.yaml", "r") as f:
            # The config specified in the YAML file used to train the tlens model
            train_config = yaml.safe_load(f)["train"]
        test_data = ModularArithmeticDataset(
            train_config["modulus"],
            train_config["frac_train"],
            seed=ablation_config.seed,
            train=train,
        )
        # Note that the batch size for training typically gives 1 batch per epoch. We use a smaller
        # batch size here, mostly for verifying that our iterative code works.
        test_loader = DataLoader(test_data, batch_size=ablation_config.batch_size, shuffle=False)
    elif graph_config_dict["dataset"] == "wikitext":
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
            TensorDataset(input_ids, labels), batch_size=ablation_config.batch_size, shuffle=True
        )
    return test_loader


def run_ablations(
    ablation_config: Config,
    graph_config_dict: dict[str, Any],
    interaction_matrices: list[
        tuple[Float[Tensor, "d_hidden d_hidden_trunc"], Float[Tensor, "d_hidden_trunc d_hidden"]]
    ],
) -> dict[str, dict[int, float]]:
    """Rotate to and from orthogonal basis and compare accuracies with and without ablation.

    Args:
        ablation_config: The config for the ablation experiment.
        graph_config_dict: The config dictionary used to build the interaction graph.
        interaction_matrices: The interaction rotation matrix and its pseudoinverse.

    Returns:
        A dictionary mapping node layers to accuracy results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = TORCH_DTYPES[ablation_config.dtype]
    seq_model = load_sequential_transformer(ablation_config, graph_config_dict, device)
    seq_model.eval()
    seq_model.to(device=torch.device(device), dtype=dtype)
    seq_model.fold_bias()
    hooked_model = HookedModel(seq_model)

    test_loader = create_data_loader(ablation_config, graph_config_dict, train=False)

    assert (
        graph_config_dict["tlens_pretrained"] is None
    ), "Currently can't build graphs for pretrained models due to memory limits."

    accuracy = eval_model_accuracy(hooked_model, test_loader, dtype=dtype, device=device)
    logger.info("Accuracy of model without any ablation: %s", accuracy)

    # Don't build the graph for the section of the model before the first node layer
    graph_module_names = [f"sections.{sec}" for sec in seq_model.sections if sec != "pre"]

    results: dict[str, dict[int, float]] = {}
    for hook_name, module_name, (C, C_pinv) in zip(
        ablation_config.node_layers, graph_module_names, interaction_matrices
    ):
        ablation_schedule = calc_ablation_schedule(
            ablate_every_vec_cutoff=ablation_config.ablate_every_vec_cutoff,
            n_vecs=C.shape[1],
        )
        module_accuracies: dict[int, float] = ablate_and_test(
            hooked_model=hooked_model,
            module_name=module_name,
            interaction_rotation=C,
            interaction_rotation_pinv=C_pinv,
            test_loader=test_loader,
            device=device,
            ablation_schedule=ablation_schedule,
            hook_name=hook_name,
        )
        results[hook_name] = module_accuracies

    return results


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)

    set_seed(config.seed)
    interaction_graph_info = torch.load(config.interaction_graph_path)

    # Get the interaction rotations and their pseudoinverses (C and C_pinv)
    interaction_matrices: list[
        tuple[Float[Tensor, "d_hidden d_hidden_trunc"], Float[Tensor, "d_hidden_trunc d_hidden"]]
    ] = []
    for module_name in config.node_layers:
        for rotation_info in interaction_graph_info["interaction_rotations"]:
            if rotation_info["node_layer_name"] == module_name:
                interaction_matrices.append((rotation_info["C"], rotation_info["C_pinv"]))
                break
    assert len(interaction_matrices) == len(
        config.node_layers
    ), f"Could not find all modules in the interaction graph config. "

    out_file = Path(__file__).parent / "out" / f"{config.exp_name}_ablation_results.json"
    if out_file.exists() and not overwrite_output(out_file):
        print("Exiting.")
        return

    out_file.parent.mkdir(parents=True, exist_ok=True)

    accuracies: dict[str, dict[int, float]] = run_ablations(
        ablation_config=config,
        graph_config_dict=interaction_graph_info["config"],
        interaction_matrices=interaction_matrices,
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
