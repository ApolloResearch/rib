import torch


# Define an MLP
class MLP(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Create an MLP
    mlp = MLP(10, 20, 1)
    # Save the model
    torch.save(mlp.state_dict(), "mlp.pt")
    # Load the model
    mlp = MLP(10, 20, 1)
    mlp.load_state_dict(torch.load("mlp.pt"))
    # Print the model
    print(mlp)
