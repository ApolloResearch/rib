"""Define custom datasets."""

import torch
from jaxtyping import Int
from torch import Tensor
from torch.utils.data import Dataset


class ModularArithmeticDataset(Dataset):
    """Defines the dataset used for Neel Nanda's modular arithmetic task."""

    def __init__(
        self,
        modulus: int,
        fn_name: str = "add",
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
