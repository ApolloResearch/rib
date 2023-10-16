"""Utilities for loading models and data."""

from pathlib import Path
from typing import Literal, Optional, Union, cast, overload

import torch
import yaml
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

from rib.data import ModularArithmeticDataset
from rib.models import SequentialTransformer, SequentialTransformerConfig
from rib.models.sequential_transformer.converter import convert_tlens_weights
from rib.types import DATASET_TYPES
from rib.utils import train_test_split


def load_sequential_transformer(
    node_layers: list[str],
    last_pos_module_type: Optional[Literal["add_resid1", "unembed"]],
    tlens_pretrained: Optional[str],
    tlens_model_path: Optional[Path],
    eps: Optional[float],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> tuple[SequentialTransformer, dict]:
    """Load a SequentialTransformer model from a pretrained transformerlens model.

    Requires config to contain a pretrained model name or a path to a transformerlens model.

    First loads a HookedTransformer model, then uses its config to create an instance of
    SequentialTransformerConfig, which is then used to create a SequentialTransformer.

    Args:
        node_layers (list[str]): The module names defining the graph sections.
        last_pos_module_type (Optional[Literal["add_resid1", "unembed"]]): Module in which to only
            output the last position index.
        tlens_pretrained (Optional[str]): The name of a pretrained transformerlens model.
        tlens_model_path (Optional[Path]): The path to a transformerlens model.
        eps (Optional[float]): The epsilon value to use for the layernorms in the model.
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

    # Set the dtype and layernorm epsilon to the one specified in the config for this script (as
    # opposed to the one used to train the tlens model)
    seq_cfg.dtype = dtype
    if eps is not None:
        seq_cfg.eps = eps

    seq_model = SequentialTransformer(
        seq_cfg, node_layers, last_pos_module_type=last_pos_module_type
    )

    # Load the transformer-lens weights into the sequential transformer model
    state_dict = convert_tlens_weights(
        seq_param_names=list(seq_model.state_dict().keys()),
        tlens_model=tlens_model,
        positional_embedding_type=seq_cfg.positional_embedding_type,
    )
    seq_model.load_state_dict(state_dict)

    return seq_model, tlens_cfg_dict


def create_modular_arithmetic_dataset(
    return_set: Union[Literal["train", "test", "all"], Literal["both"]],
    tlens_model_path: Optional[Path] = None,
    fn_name: Optional[Literal["add", "subtract", "x2xyy2"]] = None,
    modulus: Optional[int] = None,
    seed: Optional[int] = None,
    frac_train: Optional[float] = None,
    **_kwargs,
) -> Union[Dataset, tuple[Dataset, Dataset]]:
    """Create a ModularArithmeticDataset from the provided arguments.

    Either collects the arguments from the provided tlens model path or uses the provided arguments,
    which override the arguments in the trained model's config.

    Args:
        return_set (str): What part of the dataset to return.
        tlens_model_path (Optional[Path]): The path to the tlens model (if applicable).
        fn_name (Optional[Literal["add", "subtract", "x2xyy2"]]): The name of the function to use.
        modulus (Optional[int]): The modulus to use.
        seed (Optional[int]): The seed to use.
        frac_train (Optional[float]): The fraction of the dataset to use for training.
    """
    if tlens_model_path:
        with open(tlens_model_path.parent / "config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        modulus = modulus or cfg["train"]["modulus"]
        fn_name = fn_name or cfg["train"]["fn_name"]
        frac_train = frac_train or cfg["train"]["frac_train"]
        seed = seed or cfg["seed"]

    assert modulus is not None, "Modulus not provided and not found in tlens model config."
    assert fn_name is not None, "Function name not provided and not found in tlens model config."
    assert frac_train is not None, "frac_train not provided and not found in tlens model config."
    assert seed is not None, "Seed not provided and not found in tlens model config."

    raw_dataset = ModularArithmeticDataset(modulus=modulus, fn_name=fn_name)

    if return_set == "all":
        return raw_dataset
    else:
        train_dataset, test_dataset = train_test_split(
            raw_dataset, frac_train=frac_train, seed=seed
        )
        if return_set == "train":
            return train_dataset
        elif return_set == "test":
            return test_dataset
        else:
            assert return_set == "both"
            return train_dataset, test_dataset


def load_wikitext(
    return_set: Literal["train", "test", "all", "both"],
    return_set_frac: Optional[float] = None,
    return_set_n_samples: Optional[int] = None,
    model_str: str = "pythia-14m",
    **_kwargs,
):
    """Load the wikitext dataset.

    Args:
        return_set (str): What part of the dataset to return.
        return_set_frac (Optional[float]): The fraction of the dataset to return. If specified,
            `return_set_n_samples` should not be specified.
        return_set_n_samples (Optional[int]): The number of raw dataset samples to return. If
            specified, `return_set_frac` should not be specified.
        model_str (str): The name of the model to used to load the correct tokenizer.
    """
    assert return_set in ["train", "test"], "Only train and test sets are supported for now"
    assert "pythia" in model_str, "Only pythia models are supported for now"
    n_ctx = 1024 if "gpt2" in model_str else 2048

    assert not (
        return_set_frac and return_set_n_samples
    ), "Only one of `return_set_frac` and `return_set_n_samples` can be specified."

    if return_set_frac:
        data_split = f"{return_set}[:{int(return_set_frac * 100)}%]"
    elif return_set_n_samples:
        data_split = f"{return_set}[:{return_set_n_samples}]"

    dataset = hf_load_dataset("wikitext", "wikitext-103-raw-v1", split=data_split)

    tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{model_str}")
    tokenizer.pad_token = tokenizer.eos_token

    # Remove empty data points
    dataset = dataset.filter(lambda example: len(example["text"]) > 0)

    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=n_ctx
        ),
        batched=True,
    )

    input_ids = torch.tensor(tokenized_dataset["input_ids"], dtype=torch.long)

    # Create labels by shifting input_ids by 1
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = tokenizer.pad_token_id
    return TensorDataset(input_ids, labels)


@overload
def load_dataset(
    dataset_type: DATASET_TYPES,
    return_set: Literal["train", "test", "all"],
    tlens_model_path: Optional[Path] = None,
    **kwargs,
) -> Dataset:
    ...


@overload
def load_dataset(
    dataset_type: DATASET_TYPES,
    return_set: Literal["both"],
    tlens_model_path: Optional[Path] = None,
    **kwargs,
) -> tuple[Dataset, Dataset]:
    ...


def load_dataset(
    dataset_type: DATASET_TYPES,
    return_set: Union[Literal["train", "test", "all"], Literal["both"]],
    tlens_model_path: Optional[Path] = None,
    **kwargs,
) -> Union[Dataset, tuple[Dataset, Dataset]]:
    """
    Load a dataset based on the provided type and arguments.

    Args:
        dataset_type (str): The type of dataset to create.
        return_set (str): What part of the dataset to return.
        tlens_model_path (Optional[Path]): The path to the tlens model (if applicable).
        **kwargs: Additional arguments needed for specific dataset types.

    Returns:
        The loaded dataset or a tuple of datasets (train and test).
    """
    if dataset_type == "modular_arithmetic":
        return create_modular_arithmetic_dataset(
            tlens_model_path=tlens_model_path, return_set=return_set, **kwargs
        )
    elif dataset_type == "wikitext":
        return load_wikitext(return_set=return_set, **kwargs)
    # Add more dataset loading logic here as needed
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


@overload
def create_data_loader(dataset: Dataset, shuffle: bool, batch_size: int) -> DataLoader:
    ...


@overload
def create_data_loader(
    dataset: tuple[Dataset, Dataset], shuffle: bool, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    ...


def create_data_loader(
    dataset: Union[Dataset, tuple[Dataset, Dataset]], shuffle: bool, batch_size: int
) -> Union[DataLoader, tuple[DataLoader, DataLoader]]:
    """
    Create a DataLoader from the provided dataset.

    Args:
        dataset (Dataset or tuple[Dataset, Dataset]): The dataset(s) to create a DataLoader for. If
            a tuple is provided, the first element is used as the training dataset and the second
            element is used as the test dataset.
        shuffle (bool): Whether to shuffle the dataset(s) each epoch.
        batch_size (int): The batch size to use.

    Returns:
        The DataLoader or a tuple of DataLoaders.
    """
    if isinstance(dataset, tuple):
        train_loader = DataLoader(dataset[0], batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(dataset[1], batch_size=batch_size, shuffle=shuffle)
        return train_loader, test_loader
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
