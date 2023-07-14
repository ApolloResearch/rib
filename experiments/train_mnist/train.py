"""Train a model on MNIST.

Usage:
    python scripts/train_mnist.py <path/to/config.yaml>
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import fire
import torch
import wandb
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from rib.log import logger
from rib.models import MLP
from rib.models.utils import save_model
from rib.utils import REPO_ROOT, load_config


class ModelConfig(BaseModel):
    hidden_sizes: Optional[list[int]]
    activation_fn: str = "relu"
    bias: bool = True
    fold_bias: bool = True


class TrainConfig(BaseModel):
    learning_rate: float
    batch_size: int
    epochs: int
    save_dir: Optional[Path]
    save_every_n_epochs: Optional[int]


class WandbConfig(BaseModel):
    project: str
    entity: str


class Config(BaseModel):
    seed: int
    model: ModelConfig
    train: TrainConfig
    wandb: Optional[WandbConfig]


def train(config: Config) -> None:
    """Train the MLP on MNIST.
    https://pylint.readthedocs.io/en/latest/user_guide/messages/warning/logging-fstring-interpolation.html
        If config.wandb is not None, log the results to Weights & Biases.
    """
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    if not config.train.save_dir:
        config.train.save_dir = Path(__file__).parent / ".checkpoints" / "mnist"

    # Load the MNIST dataset
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(
        root=REPO_ROOT / ".data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_data, batch_size=config.train.batch_size, shuffle=True)

    # Initialize the MLP model
    model = MLP(
        config.model.hidden_sizes,
        input_size=784,
        output_size=10,
        activation_fn=config.model.activation_fn,
        bias=config.model.bias,
        fold_bias=config.model.fold_bias,
    )
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
            config=config.model_dump(),
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
                    wandb.log({"train/loss": loss.item(), "train/samples": samples}, step=samples)

        if config.train.save_every_n_epochs and (epoch + 1) % config.train.save_every_n_epochs == 0:
            save_model(json.loads(config.model_dump_json()), save_dir, model, epoch)

    if not (save_dir / f"model_epoch_{epoch + 1}.pt").exists():
        save_model(json.loads(config.model_dump_json()), save_dir, model, epoch)


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)
    train(config)


if __name__ == "__main__":
    fire.Fire(main)
