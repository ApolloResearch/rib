import torch


class MLP(torch.nn.Module):
    def __init__(self) -> None:
        super(MLP, self).__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


if __name__ == "__main__":
    model = MLP()
    print(model)
    print(model(torch.zeros(10)))
