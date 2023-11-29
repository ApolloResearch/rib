"""Utilities for loading models and data."""

from pathlib import Path
from typing import Literal, Optional, Union, overload

import torch
import yaml
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

from rib.data import (
    HFDatasetConfig,
    ModularArithmeticDataset,
    ModularArithmeticDatasetConfig,
)
from rib.models import SequentialTransformer, SequentialTransformerConfig
from rib.models.mlp import MLP
from rib.models.sequential_transformer.converter import convert_tlens_weights
from rib.utils import set_seed, train_test_split


def load_sequential_transformer(
    node_layers: list[str],
    last_pos_module_type: Optional[Literal["add_resid1", "unembed"]],
    tlens_pretrained: Optional[str],
    tlens_model_path: Optional[Path],
    eps: Optional[float] = None,
    fold_bias: bool = True,
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
        eps (Optional[float]): The epsilon value to use for the layernorms in the model. If None,
            the value from the tlens model is used. Defaults to None.
        fold_bias (bool): Whether to fold the bias into the weights. Defaults to True.
        dtype (torch.dtype): The dtype to use for the model. Defaults to float32.
        device (str): The device to use for the model. Defaults to "cpu".

    Returns:
        - SequentialTransformer: The SequentialTransformer model.
        - dict: The config used in the transformerlens model.
    """
    assert (
        tlens_pretrained is not None or tlens_model_path is not None
    ), "Either `tlens_pretrained` or `tlens_model_path` must be specified."
    if tlens_pretrained is not None:
        tlens_model = HookedTransformer.from_pretrained(
            tlens_pretrained, device="cpu", torch_dtype=dtype
        )
        # Create a SequentialTransformerConfig from the HookedTransformerConfig
        tlens_cfg_dict = tlens_model.cfg.to_dict()

    elif tlens_model_path is not None:
        tlens_model_path = (
            Path(tlens_model_path) if isinstance(tlens_model_path, str) else tlens_model_path
        )
        with open(tlens_model_path.parent / "config.yaml", "r") as f:
            # The config specified in the YAML file used to train the tlens model
            provided_tlens_cfg_dict = yaml.safe_load(f)["model"]

        # Set the dtype to the one specified in the config for this script
        provided_tlens_cfg_dict["dtype"] = dtype

        tlens_model = HookedTransformer(provided_tlens_cfg_dict, move_to_device=False)
        # The entire tlens config (including default values)
        tlens_cfg_dict = tlens_model.cfg.to_dict()

        # Load the weights from the tlens model
        tlens_model.load_state_dict(torch.load(tlens_model_path, map_location="cpu"))

    seq_cfg = SequentialTransformerConfig(**tlens_cfg_dict)

    # Set the layernorm epsilon to the one specified in the config for this script (not the one
    # used to train the model)
    if eps is not None:
        seq_cfg.eps = eps

    seq_model = SequentialTransformer(
        seq_cfg, node_layers, last_pos_module_type=last_pos_module_type
    )

    # Load the transformer-lens weights into the sequential transformer model
    state_dict = convert_tlens_weights(
        seq_model=seq_model,
        tlens_model=tlens_model,
        positional_embedding_type=seq_cfg.positional_embedding_type,
    )
    seq_model.load_state_dict(state_dict)

    if fold_bias:
        seq_model.fold_bias()

    # Ensure that our model has the correct dtype (by checking the dtype of the first parameter)
    assert next(seq_model.parameters()).dtype == dtype, (
        f"Model dtype ({next(seq_model.parameters()).dtype}) does not match specified dtype "
        f"({dtype})."
    )
    return seq_model.to(device), tlens_cfg_dict


def load_mlp(config_dict: dict, mlp_path: Path, device: str, fold_bias: bool = True) -> MLP:
    mlp = MLP(
        hidden_sizes=config_dict["model"]["hidden_sizes"],
        input_size=784,
        output_size=10,
        activation_fn=config_dict["model"]["activation_fn"],
        bias=config_dict["model"]["bias"],
        fold_bias=False,
    )
    mlp.load_state_dict(torch.load(mlp_path, map_location=torch.device(device)))
    if fold_bias:
        mlp.fold_bias()
    return mlp


def create_modular_arithmetic_dataset(
    dataset_config: ModularArithmeticDatasetConfig,
    return_set: Union[Literal["train", "test", "all"], Literal["both"]],
    tlens_model_path: Optional[Path] = None,
) -> Union[Dataset, tuple[Dataset, Dataset]]:
    """Create a ModularArithmeticDataset from the provided arguments.

    Either collects the arguments from the provided tlens model path or uses the provided arguments,
    which override the arguments in the trained model's config.

    Args:
        dataset_config(ModularArithmeticDatasetConfig): The dataset config (overrides the config
            loaded from the tlens model).
        tlens_model_path (Optional[Path]): The path to the tlens model (if applicable).
    """
    modulus, fn_name, frac_train, seed = None, None, None, None
    if tlens_model_path:
        with open(tlens_model_path.parent / "config.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        modulus = cfg["dataset"]["modulus"]
        fn_name = cfg["dataset"]["fn_name"]
        frac_train = cfg["dataset"]["frac_train"]
        seed = cfg["seed"]

    modulus = dataset_config.modulus or modulus
    fn_name = dataset_config.fn_name or fn_name
    frac_train = dataset_config.frac_train if dataset_config.frac_train is not None else frac_train
    seed = dataset_config.seed if dataset_config.seed is not None else seed

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


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    n_ctx: int,
) -> TensorDataset:
    """Tokenize a dataset using the provided tokenizer.

    Tokenizes the dataset and splits it into chunks that fit the context length. The labels are
    the input_ids shifted by one position.

    Args:
        raw_dataset (Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]): The raw
            dataset to tokenize. Created from `hf_load_dataset`.
        tokenizer (AutoTokenizer): The tokenizer to use.
        n_ctx (int): The context length to use.

    Returns:
        TensorDataset: The tokenized dataset.
    """
    # Tokenize all samples and merge them together
    all_tokens = []
    for example in dataset:  # type: ignore
        tokens = tokenizer(example["text"])["input_ids"]
        all_tokens.extend(tokens)

    # Split the merged tokens into chunks that fit the context length
    chunks = [all_tokens[i : i + n_ctx] for i in range(0, len(all_tokens), n_ctx)]

    # Convert chunks to input_ids and labels
    input_ids_list = []
    labels_list = []
    for chunk in chunks:
        input_id = chunk
        label = input_id[1:] + [tokenizer.pad_token_id]

        input_ids_list.append(input_id)
        labels_list.append(label)

    # Ignore the last chunk if it's shorter than the context length
    if len(input_ids_list[-1]) < n_ctx:
        input_ids_list = input_ids_list[:-1]
        labels_list = labels_list[:-1]

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    labels = torch.tensor(labels_list, dtype=torch.long)

    return TensorDataset(input_ids, labels)


@overload
def load_dataset(
    dataset_config: Union[ModularArithmeticDatasetConfig, HFDatasetConfig],
    return_set: Literal["train", "test", "all"],
    model_n_ctx: int,
    tlens_model_path: Optional[Path] = None,
) -> Dataset:
    ...


@overload
def load_dataset(
    dataset_config: Union[ModularArithmeticDatasetConfig, HFDatasetConfig],
    return_set: Literal["both"],
    model_n_ctx: int,
    tlens_model_path: Optional[Path] = None,
) -> tuple[Dataset, Dataset]:
    ...


def load_dataset(
    dataset_config: Union[ModularArithmeticDatasetConfig, HFDatasetConfig],
    return_set: Union[Literal["train", "test", "all"], Literal["both"]],
    model_n_ctx: int,
    tlens_model_path: Optional[Path] = None,
) -> Union[Dataset, tuple[Dataset, Dataset]]:
    """
    Load a dataset based on the provided type and arguments.

    Useful hugginface datasets (credit to TransformerLens authors for creating/collecting these):
        - stas/openwebtext-10k (approx the GPT-2 training data
            https://huggingface.co/datasets/openwebtext)
        - NeelNanda/pile-10k (The Pile, a big mess of tons of diverse data
            https://pile.eleuther.ai/)
        - NeelNanda/c4-10k (Colossal, Cleaned, Common Crawl - basically openwebtext but bigger
            https://huggingface.co/datasets/c4)
        - NeelNanda/code-10k (Codeparrot Clean, a Python code dataset
            https://huggingface.co/datasets/codeparrot/codeparrot-clean)
        - NeelNanda/c4-code-20k (c4 + code - the 20K data points from c4-10k and code-10k. This is
            the mix of datasets used to train my interpretability-friendly models, though note that
            they are *not* in the correct ratio! There's 10K texts for each, but about 22M tokens of
            code and 5M tokens of C4)
        - NeelNanda/wiki-10k (Wikipedia, generated from the 20220301.en split of
            https://huggingface.co/datasets/wikipedia)

    Args:
        dataset_config (Union[ModularArithmeticDatasetConfig, HFDatasetConfig]): The dataset config.
        return_set (Union[Literal["train", "test", "all"], Literal["both"]]): The dataset to return.
        model_n_ctx (int): The max context length of the model. Data sequences are packed to
            dataset_config.n_ctx if it is not None and is <= model_n_ctx, otherwise to model_n_ctx.
        tlens_model_path (Optional[Path]): The path to the tlens model. Used for collecting config
            for the modular arithmetic dataset used to train the model.

    Returns:
        The loaded dataset or a tuple of datasets (train and test).

    """

    if isinstance(dataset_config, ModularArithmeticDatasetConfig):
        return create_modular_arithmetic_dataset(
            dataset_config=dataset_config, return_set=return_set, tlens_model_path=tlens_model_path
        )
    else:
        n_ctx = dataset_config.n_ctx or model_n_ctx
        assert n_ctx <= model_n_ctx, (
            f"Dataset context length ({dataset_config.n_ctx}) must be <= model context length "
            f"({model_n_ctx})."
        )

        # Load dataset from huggingface
        assert return_set in ["train", "test"], "Can only load train or test sets from HF"
        assert not (
            dataset_config.return_set_frac and dataset_config.return_set_n_samples
        ), "Only one of `return_set_frac` and `return_set_n_samples` can be specified."

        if dataset_config.return_set_frac:
            percent = int(dataset_config.return_set_frac * 100)
            if dataset_config.return_set_portion == "first":
                data_split = f"{return_set}[:{percent}%]"
            elif dataset_config.return_set_portion == "last":
                data_split = f"{return_set}[-{percent}%:]"
        elif dataset_config.return_set_n_samples:
            if dataset_config.return_set_portion == "first":
                data_split = f"{return_set}[:{dataset_config.return_set_n_samples}]"
            elif dataset_config.return_set_portion == "last":
                data_split = f"{return_set}[-{dataset_config.return_set_n_samples}:]"

        raw_dataset = hf_load_dataset(dataset_config.name, split=data_split)

        tokenizer = AutoTokenizer.from_pretrained(dataset_config.tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token
        dataset = tokenize_dataset(dataset=raw_dataset, tokenizer=tokenizer, n_ctx=n_ctx)
        return dataset


@overload
def create_data_loader(dataset: Dataset, shuffle: bool, batch_size: int, seed: int) -> DataLoader:
    ...


@overload
def create_data_loader(
    dataset: tuple[Dataset, Dataset], shuffle: bool, batch_size: int, seed: int
) -> tuple[DataLoader, DataLoader]:
    ...


def create_data_loader(
    dataset: Union[Dataset, tuple[Dataset, Dataset]],
    shuffle: bool,
    batch_size: int,
    seed: Optional[int] = None,
) -> Union[DataLoader, tuple[DataLoader, DataLoader]]:
    """
    Create a DataLoader from the provided dataset.

    Args:
        dataset (Dataset or tuple[Dataset, Dataset]): The dataset(s) to create a DataLoader for. If
            a tuple is provided, the first element is used as the training dataset and the second
            element is used as the test dataset.
        shuffle (bool): Whether to shuffle the dataset(s) each epoch.
        batch_size (int): The batch size to use.
        seed (Optional[int]): The seed to use for the DataLoader.

    Returns:
        The DataLoader or a tuple of DataLoaders.
    """
    if seed is not None:
        set_seed(seed)
    if isinstance(dataset, tuple):
        train_loader = DataLoader(dataset[0], batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(dataset[1], batch_size=batch_size, shuffle=shuffle)
        return train_loader, test_loader
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_dataset_chunk(dataset: Dataset, chunk_idx: int, total_chunks: int) -> Dataset:
    """
    Returns a subset of the dataset, determined by the `chunk_idx` and `total_chunks`.

    Useful for dataparellism, if we want each process to use a different dataset chunk.

    Args:
        dataset (Dataset): The dataset to use. Must be a Map-style dataset (implements `__len__`
            and `__get_item__`).
        chunk_idx (int): The id
        total_chunks (int): Total number of chunks. If this is exactly 1, we return all of `dataset`.

    Returns:
        The DataLoader or a tuple of DataLoaders.
    """
    assert chunk_idx < total_chunks, f"chunk_idx {chunk_idx} >= total # of chunks {total_chunks}"
    if total_chunks == 1:
        return dataset
    dataset_len = len(dataset)  # type: ignore
    assert (
        total_chunks <= dataset_len
    ), f"more chunks than elements of the dataset ({total_chunks} > {dataset_len})"
    dataset_idx_start = dataset_len * chunk_idx // total_chunks
    dataset_idx_end = dataset_len * (chunk_idx + 1) // total_chunks
    return Subset(dataset, range(dataset_idx_start, dataset_idx_end))
