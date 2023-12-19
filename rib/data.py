"""Define custom datasets."""
from typing import Literal, Optional

import torch
from jaxtyping import Float, Int
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from torch import Tensor
from torch.utils.data import Dataset

from rib.types import TORCH_DTYPES, StrDtype


class DatasetConfig(BaseModel):
    """Base class for dataset configs."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    return_set_frac: Optional[float] = Field(
        None,
        description="The fraction of the returned dataset (train/test/all) to use. Cannot be"
        "used with return_set_n_samples.",
    )
    return_set_n_samples: Optional[int] = Field(
        None,
        description="The number of raw samples to return from the dataset (train/test/all). "
        "Cannot be used with return_set_frac.",
    )

    @model_validator(mode="after")
    def verify_return_set_frac_and_n_samples(self) -> "DatasetConfig":
        """Verify not both return_set_frac and return_set_n_samples are set and check values."""
        frac = self.return_set_frac

        if frac is not None:
            if self.return_set_n_samples is not None:
                raise ValueError(
                    "Cannot have both return_set_frac and return_set_n_samples be non-None."
                )
            if isinstance(self, HFDatasetConfig) and (frac < 0.01 or frac > 1):
                raise ValueError(
                    f"return_set_frac must be > 0.01 and < 1 since huggingface dataset `split` "
                    f"method does not correctly convert other values to perecentages."
                )
            if frac <= 0 or frac > 1:
                raise ValueError(f"return_set_frac must be > 0 and <= 1.")
        return self


class HFDatasetConfig(DatasetConfig):
    """Config for the HuggingFace datasets library."""

    dataset_type: Literal["huggingface"] = "huggingface"
    name: str = Field(
        ..., description="The name of the dataset to load from the HuggingFace datasets library."
    )
    tokenizer_name: str = Field(
        ...,
        description="The HuggingFace name for the tokenizer. Please check whether the tokenizer is "
        "compatible with the model you are using.",
    )
    return_set: Literal["train", "test"] = Field(
        ..., description="The dataset split to return from HuggingFace."
    )
    return_set_portion: Literal["first", "last"] = Field(
        "first", description="Whether to load the first or last portion of the return_set."
    )
    n_ctx: Optional[int] = Field(
        None,
        description="Dataset will be packed to sequences of this length. Should be <1024 for gpt2."
        "<2048 for most other models.",
    )


class ModularArithmeticDatasetConfig(DatasetConfig):
    """Config for the modular arithmetic dataset.

    We set fields to optional so that we have the option of loading them in from a pre-saved config
    file (see `rib/loader.create_modular_arithmetic_dataset`)
    """

    dataset_type: Literal["modular_arithmetic"] = "modular_arithmetic"
    return_set: Literal["train", "test", "all"] = Field(
        "train",
        description="The dataset to return. If 'all', returns the combined train and test datasets.",
    )
    modulus: Optional[int] = Field(None, description="The modulus to use for the dataset.")
    fn_name: Optional[Literal["add", "subtract", "x2xyy2"]] = Field(
        None,
        description="The function to use for the dataset. One of 'add', 'subtract', or 'x2xyy2'.",
    )
    frac_train: Optional[float] = Field(
        None, description="Fraction of the dataset to use for training."
    )
    seed: Optional[int] = Field(0, description="The random seed value for reproducibility.")


class ModularArithmeticDataset(Dataset):
    """Defines the dataset used for Neel Nanda's modular arithmetic task."""

    def __init__(
        self,
        modulus: int,
        fn_name: Literal["add", "subtract", "x2xyy2"] = "add",
    ):
        self.modulus = modulus
        self.fn_name = fn_name

        self.fns_dict = {
            "add": lambda x, y: (x + y) % self.modulus,
            "subtract": lambda x, y: (x - y) % self.modulus,
            "x2xyy2": lambda x, y: (x**2 + x * y + y**2) % self.modulus,
        }
        self.fn = self.fns_dict[fn_name]

        self.data = torch.tensor([(i, j, modulus) for i in range(modulus) for j in range(modulus)])
        self.labels = torch.tensor([self.fn(i, j) for i, j, _ in self.data])

    def __getitem__(self, idx: int) -> tuple[Int[Tensor, "seq"], Int[Tensor, "1"]]:
        return self.data[idx], self.labels[idx]

    def __len__(self) -> int:
        return len(self.data)


class VisionDatasetConfig(DatasetConfig):
    dataset_type: Literal["torchvision"] = "torchvision"
    name: Literal["CIFAR10", "MNIST"] = "MNIST"
    seed: Optional[int] = 0
    return_set: Literal["train", "test"] = "train"
    return_set_frac: Optional[float] = None  # Needed for some reason to avoid mypy errors
    return_set_n_samples: Optional[int] = None  # Needed for some reason to avoid mypy errors


class BlockVectorDatasetConfig(DatasetConfig):
    dataset_type: Literal["block_vector"] = "block_vector"
    size: int = Field(
        1000,
        description="Number of samples in the dataset.",
    )
    length: int = Field(
        4,
        description="Length of each vector.",
    )
    first_block_length: Optional[int] = Field(
        None,
        description="Length of the first block. If None, defaults to length // 2.",
        validate_default=True,
    )
    data_variances: list[float] = Field(
        [1.0, 1.0],
        description="Variance of the two blocks of the vectors.",
    )
    data_perfect_correlation: bool = Field(
        False,
        description="Whether to make the data within each block perfectly correlated.",
    )
    dtype: StrDtype = "float64"
    seed: Optional[int] = 0

    @field_validator("first_block_length", mode="after")
    @classmethod
    def set_first_block_length(cls, v: Optional[int], info: ValidationInfo) -> int:
        if v is None:
            return info.data["length"] // 2
        return v


class BlockVectorDataset(Dataset):
    def __init__(
        self,
        dataset_config: BlockVectorDatasetConfig,
    ):
        """Generate a dataset of random normal vectors.

        The components in `[:first_block_length]` have variance `data_variances[0]`, while the
        components in `[first_block_length:length]` have variance `data_variances[1]`.
        If `data_perfect_correlation` is true, the entries in each block are identical. Otherwise
        they have no correlation.
        """
        self.cfg = dataset_config
        self.data = self.generate_data()
        # Not needed, just here for Dataset class
        self.labels = torch.nan * torch.ones(self.cfg.size)

    def __len__(self):
        return self.cfg.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def generate_data(self) -> Float[Tensor, "size length"]:
        """Generate a dataset of vectors with two blocks of variance.

        Warning, changing the structure of this function may break reproducibility.

        Returns:
            A dataset of vectors with two blocks of variance.
        """
        dtype = TORCH_DTYPES[self.cfg.dtype]
        size = self.cfg.size
        length = self.cfg.length
        first_block_length = self.cfg.first_block_length
        data_variances = self.cfg.data_variances
        data_perfect_correlation = self.cfg.data_perfect_correlation

        first_block_length = first_block_length or length // 2
        second_block_length = length - first_block_length
        data = torch.empty((size, length), dtype=dtype)

        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)

        if not data_perfect_correlation:
            data[:, 0:first_block_length] = data_variances[0] * torch.randn(
                size, first_block_length, dtype=dtype
            )
            data[:, first_block_length:] = data_variances[1] * torch.randn(
                size, second_block_length, dtype=dtype
            )
        else:
            data[:, 0:first_block_length] = data_variances[0] * torch.randn(
                size, 1, dtype=dtype
            ).repeat(1, first_block_length)
            data[:, first_block_length:] = data_variances[1] * torch.randn(
                size, 1, dtype=dtype
            ).repeat(1, second_block_length)

        return data
