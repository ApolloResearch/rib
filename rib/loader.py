"""Utilities for loading models and data."""

from pathlib import Path
from typing import Optional

import torch
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens import HookedTransformer
from transformers import GPT2TokenizerFast

from rib.data import ModularArithmeticDataset
from rib.models import SequentialTransformer, SequentialTransformerConfig
from rib.models.sequential_transformer.converter import convert_tlens_weights


def load_sequential_transformer(
    node_layers: list[str],
    last_pos_only: bool,
    tlens_pretrained: Optional[str],
    tlens_model_path: Optional[Path],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> tuple[SequentialTransformer, dict]:
    """Load a SequentialTransformer model from a pretrained transformerlens model.

    Requires config to contain a pretrained model name or a path to a transformerlens model.

    First loads a HookedTransformer model, then uses its config to create an instance of
    SequentialTransformerConfig, which is then used to create a SequentialTransformer.

    Args:
        node_layers (list[str]): The module names defining the graph sections.
        last_pos_only (bool): Whether to only use the last position index in the unembed module.
        tlens_pretrained (Optional[str]): The name of a pretrained transformerlens model.
        tlens_model_path (Optional[Path]): The path to a transformerlens model.
        dtype (Optional[torch.dtype]): The dtype to use for the model.
        device (Optional[str]): The device to use for the model.

    Returns:
        - SequentialTransformer: The SequentialTransformer model.
        - dict: The config used in the transformerlens model.
    """
    assert (
        tlens_pretrained is not None or tlens_model_path is not None
    ), "Either `tlens_pretrained` or `tlens_model_path` must be specified."
    if tlens_pretrained is not None:
        tlens_model = HookedTransformer.from_pretrained(tlens_pretrained)
        # Create a SequentialTransformerConfig from the HookedTransformerConfig
        tlens_cfg_dict = tlens_model.cfg.to_dict()
    elif tlens_model_path is not None:
        tlens_model_path = (
            Path(tlens_model_path) if isinstance(tlens_model_path, str) else tlens_model_path
        )
        with open(tlens_model_path.parent / "config.yaml", "r") as f:
            # The config specified in the YAML file used to train the tlens model
            provided_tlens_cfg_dict = yaml.safe_load(f)["model"]
        tlens_model = HookedTransformer(provided_tlens_cfg_dict)
        # The entire tlens config (including default values)
        tlens_cfg_dict = tlens_model.cfg.to_dict()

        # Load the weights from the tlens model
        tlens_model.load_state_dict(torch.load(tlens_model_path, map_location=device))

    seq_cfg = SequentialTransformerConfig(**tlens_cfg_dict)
    # Set the dtype to the one specified in the config for this script (as opposed to the one used
    # to train the tlens model)
    seq_cfg.dtype = dtype
    seq_model = SequentialTransformer(seq_cfg, node_layers, last_pos_only)

    # Load the transformer-lens weights into the sequential transformer model
    state_dict = convert_tlens_weights(list(seq_model.state_dict().keys()), tlens_model)
    seq_model.load_state_dict(state_dict)

    return seq_model, tlens_cfg_dict


def create_data_loader(
    dataset_name: str,
    tlens_model_path: Path,
    seed: int,
    batch_size: int,
    frac_train: Optional[float] = None,
    train: bool = False,
) -> DataLoader:
    """Create a DataLoader for the dataset specified in `config.dataset`.

    Args:
        config (Config): The config, containing the dataset name.

    Returns:
        DataLoader: The DataLoader to use for building the graph.
    """
    if dataset_name == "modular_arithmetic":
        # Get the dataset config from our training config
        assert tlens_model_path is not None, "tlens_model_path must be specified"
        with open(tlens_model_path.parent / "config.yaml", "r") as f:
            # The config specified in the YAML file used to train the tlens model
            train_config = yaml.safe_load(f)["train"]
        frac_train = frac_train or train_config["frac_train"]
        test_data = ModularArithmeticDataset(
            train_config["modulus"], train_config["frac_train"], seed=seed, train=train
        )
        # Note that the batch size for training typically gives 1 batch per epoch. We use a smaller
        # batch size here, mostly for verifying that our iterative code works.
        data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    elif dataset_name == "wikitext":
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

        data_loader = DataLoader(
            TensorDataset(input_ids, labels), batch_size=batch_size, shuffle=True
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return data_loader
