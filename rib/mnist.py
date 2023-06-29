"""
This module is for training a simple MLP on MNIST.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import torch
import wandb
import yaml
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from rib.log import logger


class ModelConfig(BaseModel):
    hidden_sizes: Optional[List[int]]


class TrainConfig(BaseModel):
    learning_rate: float
    batch_size: int
    epochs: int
    save_dir: Path
    save_every_n_epochs: Optional[int]


class WandbConfig(BaseModel):
    project: str
    entity: str


class Config(BaseModel):
    seed: int
    log_filename: str
    model: ModelConfig
    train: TrainConfig
    wandb: Optional[WandbConfig]


def load_config(config_path: Path) -> Config:
    """Load the config from a YAML file into a Pydantic model."""
    assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)
    return config


def save_model(config: Config, save_dir: Path, model: nn.Module, epoch: int) -> None:
    # If the save_dir doesn't exist, create it and save the config
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
        logger.info("Saving config to %s", save_dir)
        with open(save_dir / "config.yaml", "w") as f:
            yaml.dump(json.loads(config.json()), f)
    logger.info("Saving model to %s", save_dir)
    torch.save(model.state_dict(), save_dir / f"model_epoch_{epoch + 1}.pt")


class MLP(nn.Module):
    def __init__(
        self, hidden_sizes: Optional[List[int]], input_size: int = 784, output_size: int = 10
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = []

        sizes = [input_size] + hidden_sizes + [output_size]
        layers: List[Union[nn.Linear, nn.ReLU]] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            # Don't add ReLU to the last layer
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


def train(config: Config) -> None:
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Load the MNIST dataset
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(
        root=Path(__file__).parent.parent / ".data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_data, batch_size=config.train.batch_size, shuffle=True)

    # Initialize the MLP model
    model = MLP(config.model.hidden_sizes)
    model = model.to(device)

    # Define the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)

    if config.wandb:
        run_name = f"lr-{config.train.learning_rate}_bs-{config.train.batch_size}"
        wandb.init(
            name=run_name,
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=config.dict(),
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = config.train.save_dir / f"{run_name}_{timestamp}"

    samples = 0
    # Training loop
    for epoch in tqdm(range(config.train.epochs), total=config.train.epochs, desc="Epochs"):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            samples += images.shape[0]

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                logger.info(
                    "Epoch [%d/%d], Step [%d/%d], Loss: %f",
                    epoch + 1,
                    config.train.epochs,
                    i + 1,
                    len(train_loader),
                    loss.item(),
                )

                if config.wandb:
                    wandb.log({"train/loss": loss.item(), "train/samples": samples})

        if config.train.save_every_n_epochs and (epoch + 1) % config.train.save_every_n_epochs == 0:
            save_model(config, save_dir, model, epoch)

    if not (save_dir / f"model_epoch_{epoch + 1}.pt").exists():
        save_model(config, save_dir, model, epoch)


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path)
    train(config)
