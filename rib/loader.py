"""Utilities for loading models and data."""

from pathlib import Path
from typing import Literal, Optional, Union, overload

import torch
import yaml
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from rib.data import ModularArithmeticDataset
from rib.models import SequentialTransformer, SequentialTransformerConfig
from rib.models.sequential_transformer.converter import convert_tlens_weights
from rib.utils import train_test_split


def load_sequential_transformer(
    node_layers: list[str],
    last_pos_module_type: Optional[Literal["add_resid1", "unembed"]],
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
        last_pos_module_type (Optional[Literal["add_resid1", "unembed"]]): Module in which to only
            output the last position index.
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


@overload
def create_modular_arithmetic_data_loader(
    shuffle: bool,
    return_set: Literal["train", "test", "all"],
    tlens_model_path: Optional[Path] = None,
    fn_name: Optional[str] = None,
    modulus: Optional[int] = None,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    frac_train: Optional[float] = None,
) -> DataLoader:
    ...


@overload
def create_modular_arithmetic_data_loader(
    shuffle: bool,
    return_set: Literal["both"],
    tlens_model_path: Optional[Path] = None,
    fn_name: Optional[str] = None,
    modulus: Optional[int] = None,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    frac_train: Optional[float] = None,
) -> tuple[DataLoader, DataLoader]:
    ...


def create_modular_arithmetic_data_loader(
    shuffle: bool,
    return_set: Union[Literal["train", "test", "all"], Literal["both"]],
    tlens_model_path: Optional[Path] = None,
    fn_name: Optional[str] = None,
    modulus: Optional[int] = None,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    frac_train: Optional[float] = None,
) -> Union[DataLoader, tuple[DataLoader, DataLoader]]:
    """Create a DataLoader for the specified dataset.

    Either loads the relevant config from the config.yaml associated with `tlens_model_path`, and/or
    uses the provided arguments.

    Args:
        shuffle (bool): Whether to shuffle the dataset(s) each epoch.
        return_set (Literal["train", "test", "both", "all"]): Whether to return the training set,
            test set, both, or just the full dataset.
        tlens_model_path (Optional[Path]): The path to the tlens model.
        fn_name (Optional[str]): The name of the function to use for the modular arithmetic dataset.
        modulus (Optional[int]): The modulus to use for the modular arithmetic dataset.
        batch_size (Optional[int]): The batch size to use.
        seed (Optional[int]): The seed to use for splitting the dataset.
        frac_train (Optional[float]): The fraction of the dataset to use for training.

    Returns:
        The DataLoader or tuple of DataLoaders (in the case where `return_set` is "both")
    """
    assert not (return_set == "all" and frac_train is not None), (
        "If `return_set` is 'all' the whole dataset will be returned, so `frac_train` should be "
        "None."
    )
    if tlens_model_path is not None:
        with open(tlens_model_path.parent / "config.yaml", "r") as f:
            # The config specified in the YAML file used to train the tlens model
            cfg = yaml.safe_load(f)

        modulus = modulus if modulus is not None else cfg["train"]["modulus"]
        fn_name = fn_name if fn_name is not None else cfg["train"]["fn_name"]
        batch_size = batch_size if batch_size is not None else cfg["train"]["batch_size"]
        frac_train = frac_train if frac_train is not None else cfg["train"]["frac_train"]
        seed = seed if seed is not None else cfg["seed"]
    else:
        assert (
            modulus is not None
            and fn_name is not None
            and batch_size is not None
            and frac_train is not None
            and seed is not None
        ), (
            "If `tlens_model_path` is not specified, then `modulus`, `fn_name`, `batch_size`, "
            "`frac_train`, and `seed` must be specified."
        )
    raw_dataset = ModularArithmeticDataset(modulus=modulus, fn_name=fn_name)

    if return_set == "all":
        return DataLoader(raw_dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        train_dataset, test_dataset = train_test_split(
            raw_dataset, frac_train=frac_train, seed=seed
        )
        if return_set == "train":
            return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        elif return_set == "test":
            return DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        else:
            assert return_set == "both"
            return (
                DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
                DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle),
            )
