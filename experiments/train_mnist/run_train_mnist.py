"""Train a model on MNIST.

Usage:
    python train.py <path/to/config.yaml>
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import fire
import torch
import wandb
from pydantic import BaseModel, Field
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from rib.log import logger
from rib.models import MLP
from rib.models.utils import save_model
from rib.types import RootPath
from rib.utils import REPO_ROOT, load_config, set_seed


class ModelConfig(BaseModel):
    hidden_sizes: Optional[list[int]]
    activation_fn: str = "relu"
    bias: bool = True


class TrainConfig(BaseModel):
    learning_rate: float
    batch_size: int
    epochs: int
    save_dir: Optional[RootPath] = Field(
        Path(__file__).parent / ".checkpoints" / "mnist",
        description="Directory for the output files. Defaults to `./.checkpoints/modular_arthitmatic`. If None, no output is written.",
    )
    save_every_n_epochs: Optional[int]


class WandbConfig(BaseModel):
    project: str
    entity: Optional[str]


class Config(BaseModel):
    seed: int
    model: ModelConfig
    train: TrainConfig
    wandb: Optional[WandbConfig]


@logging_redirect_tqdm()
def train_model(
    config: Config, model: MLP, train_loader: DataLoader, device: str, run_name: str
) -> MLP:
    """Train the MLP on MNIST.

    If config.wandb is not None, log the results to Weights & Biases.

    Args:
        config: Config for the experiment.
        model: MLP model.
        train_loader: DataLoader for the training set.
        device: Device to use for training.
        run_name: Name of the run.

    Returns:
        Trained MLP model.
    """

    model.train()
    # Define the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = config.train.save_dir / f"{run_name}_{timestamp}" if config.train.save_dir else None

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

        if (
            save_dir
            and config.train.save_every_n_epochs
            and (epoch + 1) % config.train.save_every_n_epochs == 0
        ):
            save_model(json.loads(config.model_dump_json()), save_dir, model, epoch)

    if save_dir and not (save_dir / f"model_epoch_{epoch + 1}.pt").exists():
        save_model(json.loads(config.model_dump_json()), save_dir, model, epoch)

    return model


@torch.inference_mode()
def evaluate_model(model: MLP, test_loader: DataLoader, device: str) -> float:
    """Evaluate the MLP on MNIST.

    Args:
        model: MLP model.
        test_loader: DataLoader for the test set.
        device: Device to use for evaluation.

    Returns:
        Test accuracy.
    """

    # Test the model
    model.eval()
    correct = 0
    total = 0
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def main(config_path_or_obj: Union[str, Config]) -> float:
    config = load_config(config_path_or_obj, config_model=Config)

    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Load the MNIST train dataset
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
        fold_bias=False,  # false even if config.model.fold_bias is true; we fold after training
    )
    model = model.to(device)

    run_name = f"lr-{config.train.learning_rate}_bs-{config.train.batch_size}"
    if config.wandb:
        wandb.init(
            name=run_name,
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=config.model_dump(),
        )

    trained_model = train_model(config, model, train_loader, device, run_name)

    # Evaluate the model on the test set
    test_data = datasets.MNIST(
        root=REPO_ROOT / ".data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=config.train.batch_size, shuffle=False)
    accuracy = evaluate_model(trained_model, test_loader, device)
    logger.info("Accuracy of the network on the 10000 test images: %d %%", accuracy)
    if config.wandb:
        wandb.log({"test/accuracy": accuracy})
    return accuracy


if __name__ == "__main__":
    fire.Fire(main)
