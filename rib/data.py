"""Defines various custom datasets used."""

import random

import torch
from jaxtyping import Int
from torch import Tensor
from torch.utils.data import Dataset


class ModularArithmeticDataset(Dataset):
    """Defines the dataset used for Neel Nanda's modular arithmetic task."""

    def __init__(
        self,
        modulus: int,
        frac_train: float = 0.3,
        fn_name: str = "add",
        device: str = "cpu",
        seed: int = 0,
        train: bool = True,
    ):
        self.modulus = modulus
        self.frac_train = frac_train
        self.fn_name = fn_name
        self.device = device
        self.seed = seed
        self.train = train

        self.fns_dict = {
            "add": lambda x, y: (x + y) % self.modulus,
            "subtract": lambda x, y: (x - y) % self.modulus,
            "x2xyy2": lambda x, y: (x**2 + x * y + y**2) % self.modulus,
        }
        self.fn = self.fns_dict[fn_name]

        self.x, self.labels = self.construct_dataset()
        (
            self.train_x,
            self.train_labels,
            self.test_x,
            self.test_labels,
        ) = self.split_dataset()

    def __getitem__(self, index: int) -> tuple[Int[Tensor, "seq"], Int[Tensor, "1"]]:
        if self.train:
            return self.train_x[index], self.train_labels[index]
        else:
            return self.test_x[index], self.test_labels[index]

    def __len__(self) -> int:
        if self.train:
            return len(self.train_x)
        else:
            return len(self.test_x)

    def construct_dataset(
        self,
    ) -> tuple[Int[Tensor, "modulus_squared seq"], Int[Tensor, "modulus_squared"]]:
        x = torch.tensor(
            [(i, j, self.modulus) for i in range(self.modulus) for j in range(self.modulus)],
            device=self.device,
        )
        y = torch.tensor([self.fn(i, j) for i, j, _ in x], device=self.device)
        return x, y

    def split_dataset(
        self,
    ) -> tuple[
        Int[Tensor, "modulus_squared_train seq"],
        Int[Tensor, "modulus_squared_train"],
        Int[Tensor, "modulus_squared_test seq"],
        Int[Tensor, "modulus_squared_test"],
    ]:
        random.seed(self.seed)
        indices = list(range(len(self.x)))
        random.shuffle(indices)
        div = int(self.frac_train * len(indices))
        train_indices, test_indices = indices[:div], indices[div:]
        train_x, train_labels = self.x[train_indices], self.labels[train_indices]
        test_x, test_labels = self.x[test_indices], self.labels[test_indices]
        return train_x, train_labels, test_x, test_labels
