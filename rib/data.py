"""Define custom datasets."""
from typing import Literal, Optional

import torch
from jaxtyping import Int
from pydantic import BaseModel, Field, field_validator
from torch import Tensor
from torch.utils.data import Dataset


class HFDatasetConfig(BaseModel):
    """Config for the HuggingFace datasets library."""

    source: Literal["huggingface"]
    name: str = Field(
        ..., description="The name of the dataset to load from the HuggingFace datasets library."
    )
    tokenizer_name: str = Field(
        ..., description="The name of the model for fetching the tokenizer."
    )
    return_set: Literal["train", "test"] = Field(..., description="The dataset to return.")
    return_set_frac: Optional[float] = Field(
        None,
        description="The fraction of the returned dataset (train/test/all/both) to use. Cannot be"
        "used with return_set_n_samples.",
    )
    return_set_n_samples: Optional[int] = Field(
        None,
        description="The number of raw samples to return from the dataset (train/test/all/both). "
        "Cannot be used with return_set_frac.",
    )
    return_set_portion: Literal["first", "last"] = Field(
        "first", description="Whether to load the first or last portion of the return_set."
    )

    @field_validator("return_set_frac")
    def check_return_set_frac(cls, v):
        # Check that 0.01 <= v <= 1
        if v is not None and (v < 0.01 or v > 1):
            raise ValueError(
                f"return_set_frac must be > 0.01 and < 1 since huggingface dataset `split` "
                f"method does not correctly convert other values to perecentages."
            )
        return v


class ModularArithmeticDatasetConfig(BaseModel):
    """Config for the modular arithmetic dataset.

    We set fields to optional so that we have the option of loading them in from a pre-saved config
    file (see `rib/loader.create_modular_arithmetic_dataset`)
    """

    source: Literal["custom"]
    name: Literal["modular_arithmetic"]
    return_set: Literal["train", "test", "all", "both"] = Field(
        ...,
        description="The dataset to return. If 'both', returns both the train and test datasets."
        "If 'all', returns the combined train and test datasets.",
    )
    modulus: Optional[int] = Field(None, description="The modulus to use for the dataset.")
    fn_name: Optional[Literal["add", "subtract", "x2xyy2"]] = Field(
        None,
        description="The function to use for the dataset. One of 'add', 'subtract', or 'x2xyy2'.",
    )
    frac_train: Optional[float] = Field(
        None, description="Fraction of the dataset to use for training."
    )
    seed: Optional[int] = Field(None, description="The random seed value for reproducibility.")


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
