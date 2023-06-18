import os
from typing import Callable

import torch


class MLP(torch.nn.Module):
    def __init__(self) -> None:
        super(MLP, self).__init__()
        # Cast linear as a function from torch.Tensor to torch.Tensor
        self.linear: Callable[[torch.Tensor], torch.Tensor] = torch.nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    model = MLP()
    print(model)
    print(model(torch.zeros(10)))
    print("Success!")
