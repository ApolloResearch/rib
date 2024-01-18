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
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from rib.data import VisionDatasetConfig
from rib.loader import load_dataset
from rib.log import logger
from rib.models import MLP
from rib.models.mlp import MLPConfig
from rib.models.utils import save_model
from rib.types import RootPath
from rib.utils import load_config, replace_pydantic_model, set_seed


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    learning_rate: float
    batch_size: int
    epochs: int
    save_dir: Optional[RootPath] = Field(
        Path(__file__).parent / ".checkpoints" / "mnist",
        description="Directory for the output files. Defaults to `./.checkpoints/mnist`. If None, "
        "no output is written. If a relative path, it is relative to the root of the rib repo.",
    )
    save_every_n_epochs: Optional[int]


class WandbConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    project: str
    entity: Optional[str]


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: Optional[int] = 0
    model: MLPConfig
    train: TrainConfig
    wandb: Optional[WandbConfig]
    dataset: VisionDatasetConfig = VisionDatasetConfig()


@logging_redirect_tqdm()
def train_model(
    config: Config, model: MLP, train_loader: DataLoader, device: str, run_name: str
) -> MLP:
    """Train the MLP.

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

    assert config.dataset.return_set == "train", "currently only supports training on the train set"
    train_dataset = load_dataset(config.dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)

    # Get the input size from the flattened first input
    data_in_dim = len(train_dataset[0][0].flatten())
    assert (
        config.model.input_size == data_in_dim
    ), f"mismatch between data size {data_in_dim} and config in_dim {config.model.input_size}"
    # Initialize the MLP model
    model = MLP(config.model)
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
    test_dataset = load_dataset(replace_pydantic_model(config.dataset, {"return_set": "test"}))
    test_loader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=True)
    accuracy = evaluate_model(trained_model, test_loader, device)
    logger.info(f"Accuracy of the network on the {len(test_dataset)} test images: %d %%", accuracy)  # type: ignore
    if config.wandb:
        wandb.log({"test/accuracy": accuracy})
    return accuracy


if __name__ == "__main__":
    fire.Fire(main)
