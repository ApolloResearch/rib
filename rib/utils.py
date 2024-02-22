import random
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Type, TypeVar, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from jaxtyping import Float, Int
from pydantic import BaseModel
from pydantic.v1.utils import deep_update
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from rib.log import logger
from rib.settings import REPO_ROOT

if TYPE_CHECKING:  # Prevent circular import to import type annotations
    from rib.hook_manager import Hook, HookedModel

T = TypeVar("T", bound=BaseModel)


def to_root_path(path: Union[str, Path]):
    """Converts relative paths to absolute ones, assuming they are relative to the rib root."""
    return Path(path) if Path(path).is_absolute() else Path(REPO_ROOT / path)


@torch.inference_mode()
def eval_model_accuracy(
    hooked_model: "HookedModel",
    dataloader: DataLoader,
    hooks: Optional[list["Hook"]] = None,
    dtype: Optional[torch.dtype] = None,
    device: str = "cuda",
) -> float:
    """Run the model on the dataset and return the accuracy.

    In the case where the output has a position dimension, we take the last position as the output.

    Args:
        hooked_model: The model to evaluate.
        dataloader: The dataloader for the dataset.
        hooks: The hooks to use.
        dtype: The data type to cast the inputs to. Ignored if int32 or int64.
        device: The device to run the model on.

    Returns:
        The accuracy of the model on the dataset.
    """

    correct_predictions: int = 0
    total_predictions: int = 0

    for batch in dataloader:
        data, labels = batch
        data, labels = data.to(device=device), labels.to(device)
        # Change the dtype unless the inputs are integers (e.g. like they are for LMs)
        if data.dtype not in [torch.int64, torch.int32] and dtype is not None:
            data = data.to(dtype=dtype)
        raw_output: Union[
            Float[Tensor, "batch d_vocab"], tuple[Float[Tensor, "batch pos d_vocab"]]
        ] = hooked_model(data, hooks=hooks)
        if isinstance(raw_output, tuple):
            assert len(raw_output) == 1, "Only one output is supported."
            # Check if the pos is 1, if so, squeeze it out. (This is the case for modular addition)
            output: Float[Tensor, "... d_vocab"] = raw_output[0]
            if output.ndim == 3 and output.shape[1] == 1:
                output = output[:, -1, :]
        else:
            output = raw_output

        # Assuming output is raw logits and labels are class indices.
        predicted_labels: Union[Int[Tensor, "batch"], Int[Tensor, "batch pos"]] = output.argmax(
            dim=-1
        )
        correct_predictions += (predicted_labels == labels).sum().item()
        total_predictions += (
            labels.shape[0] * labels.shape[1] if labels.ndim == 2 else labels.shape[0]
        )

    accuracy: float = correct_predictions / total_predictions
    return accuracy


@torch.inference_mode()
def eval_cross_entropy_loss(
    hooked_model: "HookedModel",
    dataloader: DataLoader,
    hooks: Optional[list["Hook"]] = None,
    dtype: Optional[torch.dtype] = None,
    device: str = "cuda",
) -> float:
    """Run the model on the dataset and return the per-token cross entropy loss.

    Assumes that we have a regular language model with outputs for each position dimension.

    Args:
        hooked_model: The model to evaluate.
        dataloader: The dataloader for the dataset.
        hooks: The hooks to use.
        dtype: The data type to cast the inputs to. Ignored if int32 or int64.
        device: The device to run the model on.

    Returns:
        The cross entropy loss of the model on the dataset.
    """
    n_batches = len(dataloader)
    loss: float = 0.0

    for batch in dataloader:
        data, labels = batch
        data, labels = data.to(device=device), labels.to(device)
        # Change the dtype unless the inputs are integers (e.g. like they are for LMs)
        if data.dtype not in [torch.int64, torch.int32] and dtype is not None:
            data = data.to(dtype=dtype)
        output: Float[Tensor, "batch pos d_vocab"] = hooked_model(data, hooks=hooks)[0]
        assert output.ndim == 3, "Output must have a position dimension."
        assert output.shape[1] == labels.shape[1], "Output and labels must have the same length."
        n_tokens = output.shape[0] * output.shape[1]
        # Reshape the output and labels to be 2D.
        output = output.reshape(n_tokens, -1)
        labels = labels.reshape(-1)
        # Assuming output is raw logits and labels are class indices.
        batch_loss = torch.nn.functional.cross_entropy(output, labels, reduction="sum").item()
        # Update the per-token loss
        loss += batch_loss / n_tokens

    return loss / n_batches


def load_config(config_path_or_obj: Union[Path, str, T], config_model: Type[T]) -> T:
    """Load the config of class `config_model`, either from YAML file or existing config object.

    Args:
        config_path_or_obj (Union[Path, str, `config_model`]): if config object, must be instance
            of `config_model`. If str or Path, this must be the path to a .yaml.
        config_model: the class of the config that we are loading
    """
    if isinstance(config_path_or_obj, config_model):
        return config_path_or_obj

    if isinstance(config_path_or_obj, str):
        config_path_or_obj = Path(config_path_or_obj)

    assert isinstance(
        config_path_or_obj, Path
    ), f"passed config is of invalid type {type(config_path_or_obj)}"
    assert (
        config_path_or_obj.suffix == ".yaml"
    ), f"Config file {config_path_or_obj} must be a YAML file."
    assert Path(config_path_or_obj).exists(), f"Config file {config_path_or_obj} does not exist."
    with open(config_path_or_obj, "r") as f:
        config_dict = yaml.safe_load(f)
    return config_model(**config_dict)


def check_outfile_overwrite(out_file: Path, force: bool = False) -> bool:
    """Check if out_file exists and whether it should be overwritten."""

    def _response():
        response = input(f"Output file {out_file} already exists. Overwrite? (y/n) ")
        return response.lower() == "y"

    if out_file.exists():
        if force:
            logger.info(f"Overwriting {out_file} (reason: config or cmdline)")
            return True
        elif _response():
            logger.info(f"Overwriting {out_file} (based on user prompt)")
            return True
        else:
            logger.info("Exiting.")
            return False
    else:
        return True


def set_seed(seed: Optional[int]) -> None:
    """Set the random seed for random, PyTorch and NumPy"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def find_root(
    func: Callable,
    xmin: Union[Float[Tensor, "1"], float] = torch.tensor(-1.0),
    xmax: Union[Float[Tensor, "1"], float] = torch.tensor(1.0),
    tol: float = 1e-6,
    max_iter: int = 100,
):
    """Find the root of a function using bisection."""
    # gelu requires higher precision to converge, and operates on tensors.
    if isinstance(xmin, torch.Tensor):
        xmin = xmin.double()
    if isinstance(xmax, torch.Tensor):
        xmax = xmax.double()
    # check that func(xmin) and func(xmax) have opposite signs
    assert func(xmin) * func(xmax) < 0, "func(xmin) and func(xmax) must have opposite signs"

    for i in range(max_iter):
        xmid = (xmin + xmax) / 2
        if func(xmid) * func(xmin) < 0:
            xmax = xmid
        else:
            xmin = xmid
        if abs(func(xmid)) < tol:
            return xmid
    raise ValueError(f"Finding the root of {func} via bisection failed to converge")


def train_test_split(
    dataset: Dataset, frac_train: float, seed: Optional[int]
) -> tuple[Dataset, Dataset]:
    """Split a dataset into a training and test set.

    Use a fixed seed for the split so that we can reproduce the results. If seed is None, use a
    random seed.

    Args:
        dataset: The dataset to split.
        frac_train: The fraction of the dataset to use for training.
        seed: The random seed to use for the split. If None, a random seed is used.

    Returns:
        The training and test sets.
    """
    assert 0 <= frac_train <= 1, "frac_train must be between 0 and 1."
    train_size = int(len(dataset) * frac_train)  # type: ignore
    test_size = len(dataset) - train_size  # type: ignore

    if seed is None:
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    else:
        generator = torch.Generator().manual_seed(seed)
        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size], generator=generator
        )
    return train_dataset, test_dataset


def get_data_subset(
    dataset: Dataset, frac: Optional[float], n_samples: Optional[int], seed: Optional[int] = None
) -> Dataset:
    """Get a random subset of the dataset.

    If frac is not None, returns a random sample of the dataset of size frac * len(dataset).
    If n_samples is not None, returns a random sample of the dataset of size n_samples.

    Args:
        dataset (Dataset): The dataset to return a subset of.
        frac (Optional[float]): The fraction of the dataset to return.
        n_samples (Optional[int]): The number of samples to return.
        seed (Optional[int]): The seed to use for the random number generator.

    Returns:
        Dataset: The subset of the dataset.
    """
    assert frac is None or n_samples is None, "Only one of `frac` and `n_samples` can be specified."
    len_dataset = len(dataset)  # type: ignore
    indices = list(range(len_dataset))

    if seed is not None:
        random.seed(seed)

    if frac is not None:
        selected_indices = sorted(random.sample(indices, int(len_dataset * frac)))
        return Subset(dataset, selected_indices)
    elif n_samples is not None:
        assert (
            n_samples <= len_dataset
        ), f"n_samples ({n_samples}) must be <= len_dataset ({len_dataset})."
        selected_indices = sorted(random.sample(indices, n_samples))
        return Subset(dataset, selected_indices)
    else:
        return dataset


def check_device_is_cpu(X: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if X is not None:
        assert X.device == torch.device("cpu"), "X must be on the CPU."
    return X


BaseModelType = TypeVar("BaseModelType", bound=BaseModel)


def replace_pydantic_model(model: BaseModelType, *updates: dict) -> BaseModelType:
    """Create a new model with (potentially nested) updates in the form of dictionaries.

    Args:
        model: The model to update.
        updates: The zero or more dictionaries of updates that will be applied sequentially.

    Returns:
        A replica of the model with the updates applied.

    Examples:
        >>> class Foo(BaseModel):
        ...     a: int
        ...     b: int
        >>> foo = Foo(a=1, b=2)
        >>> foo2 = replace_pydantic_model(foo, {"a": 3})
        >>> foo2
        Foo(a=3, b=2)
        >>> class Bar(BaseModel):
        ...     foo: Foo
        >>> bar = Bar(foo={"a": 1, "b": 2})
        >>> bar2 = replace_pydantic_model(bar, {"foo": {"a": 3}})
        >>> bar2
        Bar(foo=Foo(a=3, b=2))
    """
    return model.__class__(**deep_update(model.model_dump(), *updates))


def get_chunk_indices(data_size: int, chunk_idx: int, n_chunks: int) -> tuple[int, int]:
    """
    Returns the start and end indices of a chunk, determined by the `chunk_idx` and `n_chunks`,
    for a given size.

    Useful for data parallelism, if we want each process to work on a different chunk of data.

    Args:
        data_size (int): The total size of the data, equivalent to len(dataset).
        chunk_idx (int): The index of the chunk to be returned. Must be less than `n_chunks`.
        n_chunks (int): Total number of chunks. If this is exactly 1, the whole range is returned.

    Returns:
        A tuple containing the start and end indices of the chunk.
    """
    assert chunk_idx < n_chunks, f"chunk_idx {chunk_idx} >= total # of chunks {n_chunks}"
    if n_chunks == 1:
        return 0, data_size  # Return the whole range as a single chunk

    assert n_chunks <= data_size, f"more chunks than data size ({n_chunks} > {data_size})"

    chunk_start = data_size * chunk_idx // n_chunks
    chunk_end = data_size * (chunk_idx + 1) // n_chunks

    return chunk_start, chunk_end


def lm_cross_entropy_loss(
    logits: Float[torch.Tensor, "batch pos d_vocab"],
    tokens: Int[torch.Tensor, "batch pos"],
    per_token: bool = False,
) -> Union[Float[torch.Tensor, ""], Float[torch.Tensor, "batch pos"]]:
    """Cross entropy loss for the language model, gives the loss for predicting the NEXT token.

    Args:
        logits (torch.Tensor): Logits. Shape [batch, pos, d_vocab]
        tokens (torch.Tensor[int64]): Input tokens. Shape [batch, pos]
        per_token (bool, optional): Whether to return the log probs predicted for the correct token,
        or the loss (ie mean of the predicted log probs). Note that the returned array has shape
        [batch, seq-1] as we cannot predict the first token (alternately, we ignore the final
        logit). Defaults to False.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # Use torch.gather to find the log probs of the correct tokens
    # Offsets needed because we're predicting the NEXT token (this means the final logit is meaningless)
    # None and [..., 0] needed because the tensor used in gather must have the same rank.
    predicted_log_probs = log_probs.gather(dim=-1, index=tokens[..., None])[..., 0]
    if per_token:
        return -predicted_log_probs
    else:
        return -predicted_log_probs.mean()
