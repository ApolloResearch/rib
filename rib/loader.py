"""Utilities for loading models and data."""

from pathlib import Path
from typing import Literal, Optional, Union

import torch
import torchvision
import yaml
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import Dataset, Subset, TensorDataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

from rib.data import (
    BlockVectorDataset,
    BlockVectorDatasetConfig,
    DatasetConfig,
    HFDatasetConfig,
    ModularArithmeticDataset,
    ModularArithmeticDatasetConfig,
    VisionDatasetConfig,
)
from rib.models import SequentialTransformer, SequentialTransformerConfig
from rib.models.mlp import MLP, MLPConfig
from rib.models.modular_mlp import ModularMLP, ModularMLPConfig
from rib.models.sequential_transformer.converter import convert_tlens_weights
from rib.types import RibBuildResults
from rib.utils import REPO_ROOT, get_data_subset, to_root_path, train_test_split


def load_sequential_transformer(
    node_layers: list[str],
    last_pos_module_type: Optional[Literal["add_resid1", "unembed"]],
    tlens_pretrained: Optional[str],
    tlens_model_path: Optional[Path],
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
        # tlens cfg seems to be wrong here, looking at loss by position
        # the model was never trained at 512 position or after.
        if tlens_pretrained.startswith("tiny-stories"):
            tlens_cfg_dict["n_ctx"] = 511

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


def load_mlp(
    config: Union[MLPConfig, ModularMLPConfig],
    mlp_path: Optional[Path],
    device: str,
    fold_bias: bool = True,
    seed: Optional[int] = None,
) -> Union[MLP, ModularMLP]:
    mlp: Union[MLP, ModularMLP]
    if isinstance(config, ModularMLPConfig):
        mlp = ModularMLP(config, seed=seed)
        mlp.to(device)
    else:
        assert isinstance(config, MLPConfig)
        assert mlp_path is not None, "mlp_path must be provided for MLPConfig"
        mlp = MLP(config)
        mlp_path = Path(str(mlp_path).replace("ssd-apollo", "ssd-interp"))
        mlp.load_state_dict(torch.load(mlp_path, map_location=torch.device(device)))
    if fold_bias:
        mlp.fold_bias()
    return mlp


def create_modular_arithmetic_dataset(
    dataset_config: ModularArithmeticDatasetConfig,
    return_set: Literal["train", "test", "all"],
    tlens_model_path: Optional[Path] = None,
) -> Dataset:
    """Create a ModularArithmeticDataset from the provided arguments.

    Either collects the arguments from the provided tlens model path or uses the provided arguments,
    which override the arguments in the trained model's config.

    Args:
        dataset_config(ModularArithmeticDatasetConfig): The dataset config (overrides the config
            loaded from the tlens model).
        tlens_model_path (Optional[Path]): The path to the tlens model (if applicable).

    Returns:
        The dataset.
    """
    modulus, fn_name, frac_train, seed = (
        dataset_config.modulus,
        dataset_config.fn_name,
        dataset_config.frac_train,
        dataset_config.seed,
    )
    if tlens_model_path:
        with open(tlens_model_path.parent / "config.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        modulus = modulus or cfg["dataset"]["modulus"]
        fn_name = fn_name or cfg["dataset"]["fn_name"]
        if frac_train is not None:
            assert frac_train == cfg["dataset"]["frac_train"], (
                f"frac_train provided ({frac_train}) does not match frac_train in tlens model "
                f"config ({cfg['dataset']['frac_train']})"
            )
        else:
            frac_train = cfg["dataset"]["frac_train"]
        seed = seed if seed is not None else cfg["dataset"]["seed"]

    assert modulus is not None, "Modulus not provided and not found in tlens model config."
    assert fn_name is not None, "Function name not provided and not found in tlens model config."
    assert frac_train is not None, "frac_train not provided and not found in tlens model config."

    raw_dataset: Dataset = ModularArithmeticDataset(modulus=modulus, fn_name=fn_name)

    if return_set == "all":
        dataset = raw_dataset
    else:
        assert return_set in ["train", "test"], f"Unsuppored return_set value: {return_set}"
        train_dataset, test_dataset = train_test_split(
            raw_dataset, frac_train=frac_train, seed=seed
        )
        dataset = train_dataset if return_set == "train" else test_dataset

    dataset_subset = get_data_subset(
        dataset,
        frac=dataset_config.return_set_frac,
        n_samples=dataset_config.return_set_n_samples,
        seed=seed,
    )
    return dataset_subset


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
    # we ignore the final chunk, as it contains a token we don't have a label for
    # and is also probably too short and we don't want to pad.
    input_ids_list = []
    labels_list = []
    for i, chunk in enumerate(chunks[:-1]):
        input_id = chunk
        label = input_id[1:] + [chunks[i + 1][0]]  # with first token from next chunk

        input_ids_list.append(input_id)
        labels_list.append(label)

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    labels = torch.tensor(labels_list, dtype=torch.long)

    return TensorDataset(input_ids, labels)


def create_hf_dataset(
    dataset_config: HFDatasetConfig,
    return_set: Literal["train", "test", "all"],
    model_n_ctx: Optional[int] = None,
) -> Dataset:
    """Create a HuggingFace dataset from the provided arguments.

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
        dataset_config (HFDatasetConfig): The dataset config.
        return_set (Literal["train", "test", "all"]): The dataset to return.
        model_n_ctx (int): The max context length of the model. Used for HFDatasetConfigs. Data
            sequences are packed to dataset_config.n_ctx if it is not None and is <= model_n_ctx,
            otherwise to model_n_ctx.

    Returns:
        The dataset.
    """
    assert model_n_ctx is not None
    n_ctx = dataset_config.n_ctx or model_n_ctx
    assert n_ctx <= model_n_ctx, (
        f"Dataset context length ({dataset_config.n_ctx}) must be <= model context length "
        f"({model_n_ctx})."
    )

    assert return_set in ["train", "test"], "Can only load train or test sets from HF"

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
    return tokenize_dataset(dataset=raw_dataset, tokenizer=tokenizer, n_ctx=n_ctx)


def create_vision_dataset(
    dataset_config: VisionDatasetConfig,
    return_set: Literal["train", "test", "all"],
) -> Dataset:
    dataset_fn = getattr(torchvision.datasets, dataset_config.name)
    assert return_set != "all", "Cannot return 'all' for vision datasets."
    raw_dataset = dataset_fn(
        root=REPO_ROOT / ".data",
        train=return_set == "train",
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    dataset = get_data_subset(
        raw_dataset,
        frac=dataset_config.return_set_frac,
        n_samples=dataset_config.return_set_n_samples,
        seed=dataset_config.seed,
    )
    return dataset


def create_block_vector_dataset(
    dataset_config: BlockVectorDatasetConfig,
) -> Dataset:
    raw_dataset = BlockVectorDataset(dataset_config=dataset_config)

    dataset = get_data_subset(
        raw_dataset,
        frac=dataset_config.return_set_frac,
        n_samples=dataset_config.return_set_n_samples,
        seed=dataset_config.seed,
    )
    return dataset


def load_dataset(
    dataset_config: DatasetConfig,
    return_set: Literal["train", "test", "all"],
    model_n_ctx: Optional[int] = None,
    tlens_model_path: Optional[Path] = None,
) -> Dataset:
    """
    Load a dataset based on the provided type and arguments.

    Args:
        dataset_config (DatasetConfig): The dataset config.
        return_set (Literal["train", "test", "all"]): The dataset to return.
        model_n_ctx (int): The max context length of the model, used for HFDatasetConfigs. Data
            sequences are packed to dataset_config.n_ctx if it is not None and is <= model_n_ctx,
            otherwise to model_n_ctx.
        tlens_model_path (Optional[Path]): The path to the tlens model, used for modular arithmetic
            to collect config info used to train the model.

    Returns:
        The dataset.
    """

    if isinstance(dataset_config, ModularArithmeticDatasetConfig):
        return create_modular_arithmetic_dataset(
            dataset_config=dataset_config, return_set=return_set, tlens_model_path=tlens_model_path
        )
    elif isinstance(dataset_config, HFDatasetConfig):
        return create_hf_dataset(
            dataset_config=dataset_config, return_set=return_set, model_n_ctx=model_n_ctx
        )
    elif isinstance(dataset_config, VisionDatasetConfig):
        return create_vision_dataset(dataset_config=dataset_config, return_set=return_set)
    else:
        assert isinstance(dataset_config, BlockVectorDatasetConfig)
        return create_block_vector_dataset(dataset_config=dataset_config)


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


def load_model_and_dataset_from_rib_results(
    results: RibBuildResults, device: str, dtype: torch.dtype
) -> tuple[Union[SequentialTransformer, MLP], Dataset]:
    """Loads the model and dataset used for a rib build from the results dictionary.

    Combines both model and dataset loading in one function as the dataset conditionally needs
    extra arguments depending on the dataset type."""
    data_config: DatasetConfig
    model: Union[SequentialTransformer, MLP]

    if (
        "tlens_model_path" in results["config"]
        and results["config"]["tlens_model_path"] is not None
    ):
        tlens_model_path = to_root_path(Path(results["config"]["tlens_model_path"]))
    else:
        tlens_model_path = None

    if "n_heads" in results["model_config_dict"]:  # sequential transformer
        model, _ = load_sequential_transformer(
            node_layers=results["config"]["node_layers"],
            last_pos_module_type=results["config"]["last_pos_module_type"],
            tlens_pretrained=results["config"]["tlens_pretrained"],
            tlens_model_path=tlens_model_path,
            fold_bias=True,
            dtype=dtype,
            device=device,
        )
        if results["config"]["dataset"]["source"] == "huggingface":
            data_config = HFDatasetConfig(**results["config"]["dataset"])
        else:
            data_config = ModularArithmeticDatasetConfig(**results["config"]["dataset"])
        model_n_ctx = model.cfg.n_ctx

    else:  # mlp
        mlp_config = MLPConfig(**results["model_config_dict"])
        model = load_mlp(
            config=mlp_config,
            mlp_path=Path(results["config"]["mlp_path"]),
            fold_bias=True,
            device=device,
        )
        model.to(device=torch.device(device), dtype=dtype)
        data_config = VisionDatasetConfig(**results["config"]["dataset"])
        model_n_ctx, tlens_model_path = None, None

    dataset = load_dataset(
        data_config, return_set="train", model_n_ctx=model_n_ctx, tlens_model_path=tlens_model_path
    )
    return model, dataset
