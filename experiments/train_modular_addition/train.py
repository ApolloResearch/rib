"""Train a model on Modular Arithmetic.

Usage:
    python train.py <path/to/config.yaml>
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
from tqdm.contrib.logging import logging_redirect_tqdm

from rib.log import logger
from rib.models import MLP
from rib.models.utils import save_model
from rib.utils import REPO_ROOT, load_config


class ModelConfig(BaseModel):
    hidden_sizes: Optional[list[int]]
    num_layers: int
    residual_dim: int
    num_heads: int
    num_blocks: int
    vocab_dim: int
    token_len: int
    activation_fn: str = "relu"
    bias: bool = True
    fold_bias: bool = True


class TrainConfig(BaseModel):
    modulus: int
    frac_train: float
    fn_name: str
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


@logging_redirect_tqdm()
def train_model(
    config: Config, model: Transformer, train_loader: DataLoader, device: str, run_name: str
) -> Transformer:
    """Train the Transformer on Modular Arithmetic.

    If config.wandb is not None, log the results to Weights & Biases.

    Args:
        config: Config for the experiment.
        model: Transformer model.
        train_loader: DataLoader for the training set.
        device: Device to use for training.
        run_name: Name of the run.

    Returns:
        Trained Transformer model.
    """

    model.train()
    # Define the loss and optimizer
    criterion = loss_of_final_number()  # TODO custom function that uses negative log likelihood
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = config.train.save_dir / f"{run_name}_{timestamp}" if config.train.save_dir else None

    samples = 0
    # Training loop
    for epoch in tqdm(range(config.train.epochs), total=config.train.epochs, desc="Epochs"):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            samples += x.shape[0]

            outputs = model(x)
            loss = criterion(outputs, y)

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
def evaluate_model(model: Transformer, test_loader: DataLoader, device: str) -> float:
    """Evaluate the Transformer on Modular Arithmetic.

    Args:
        model: Transformer model.
        test_loader: DataLoader for the test set.
        device: Device to use for evaluation.

    Returns:
        Test accuracy.
    """

    # Test the model TODO change for transformer
    model.eval()
    correct = 0
    total = 0
    for x, y in tqdm(test_loader, desc="Evaluating"):
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)

    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    if not config.train.save_dir:
        config.train.save_dir = Path(__file__).parent / ".checkpoints" / "modular_arithmetic"

    # Load the MNIST train dataset
    transform = transforms.ToTensor()
    train_data = datasets.modular_arithmetic(
        root=REPO_ROOT / ".data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_data, batch_size=config.train.batch_size, shuffle=True)

    # Initialize the Transformer model  TODO change for transformer
    model = Transformer(
        config.model.hidden_sizes,
        input_size=784,
        output_size=10,
        activation_fn=config.model.activation_fn,
        bias=config.model.bias,
        fold_bias=config.model.fold_bias,
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
    test_data = datasets.modular_arithmetic(
        root=REPO_ROOT / ".data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=config.train.batch_size, shuffle=False)
    accuracy = evaluate_model(trained_model, test_loader, device)
    logger.info("Accuracy of the network on the x test samples: %d %%", accuracy)  # TODO fill in x
    if config.wandb:
        wandb.log({"test/accuracy": accuracy})


if __name__ == "__main__":
    fire.Fire(main)
