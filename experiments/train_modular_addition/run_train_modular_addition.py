"""Train a model on Modular Arithmetic.

Usage:
    python train.py <path/to/config.yaml>
"""
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import fire
import torch
import torch.optim as optim
import wandb
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from rib.data import ModularArithmeticDataset
from rib.log import logger
from rib.models import TransformerLensHooked
from rib.models.utils import save_model
from rib.utils import REPO_ROOT, load_config


class ModelConfig(BaseModel):
    n_layers: int
    d_model: int
    d_head: int
    n_heads: int
    d_mlp: int
    d_vocab: int
    n_ctx: int
    act_fn: str
    normalization_type: str


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
    entity: Optional[str]


class Config(BaseModel):
    seed: int
    model: ModelConfig
    train: TrainConfig
    wandb: Optional[WandbConfig]


@logging_redirect_tqdm()
def train_model(
    config: Config,
    model,
    train_loader: DataLoader,
    device: str,
    run_name: str,
    test_loader: DataLoader,
) -> HookedTransformer:
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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.learning_rate, weight_decay=1, betas=(0.9, 0.98)
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 10, 1))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = config.train.save_dir / f"{run_name}_{timestamp}" if config.train.save_dir else None

    samples = 0
    # Training loop
    for epoch in tqdm(range(config.train.epochs), total=config.train.epochs, desc="Epochs"):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            samples += x.shape[0]

            optimizer.zero_grad()
            outputs = model(x)

            # Only need the logit for the last sequence element
            loss = nn.functional.cross_entropy(outputs[:, -1], y)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 100 == 0:
                train_accuracy = evaluate_model(model, train_loader, device)
                test_accuracy = evaluate_model(model, test_loader, device)

                logger.info(
                    "Epoch [%d/%d], Step [%d/%d], Train Accuracy: %f, Test Accuracy: %f",
                    epoch + 1,
                    config.train.epochs,
                    i + 1,
                    len(train_loader),
                    train_accuracy,
                    test_accuracy,
                )

                if config.wandb:
                    wandb.log(
                        {
                            "loss": loss.item(),
                            "train_accuracy": train_accuracy,
                            "test_accuracy": test_accuracy,
                        },
                        step=samples,
                    )

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
def evaluate_model(model, test_loader: DataLoader, device: str) -> float:
    """Evaluate the Transformer on Modular Arithmetic.

    Args:
        model: Transformer model.
        test_loader: DataLoader for the test set.
        device: Device to use for evaluation.

    Returns:
        Test accuracy.
    """

    # Test the model
    model.eval()
    correct = 0
    total = 0
    for x, y in tqdm(test_loader, desc="Evaluating"):
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        total += y.size(0)
        sm_argmax = nn.functional.softmax(outputs, dim=-1).argmax(dim=-1)[:, -1].detach()
        correct += (y == sm_argmax.view(-1)).sum().item()

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

    # Load the Modular Arithmetic train dataset
    train_data = ModularArithmeticDataset(
        config.train.modulus, config.train.frac_train, device=device, seed=config.seed, train=True
    )
    test_data = ModularArithmeticDataset(
        config.train.modulus, config.train.frac_train, device=device, seed=config.seed, train=False
    )
    train_loader = DataLoader(train_data, batch_size=config.train.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.train.batch_size, shuffle=False)

    # Initialize the Transformer model
    transformer_lens_config = HookedTransformerConfig(**config.model.model_dump())
    model = HookedTransformer(transformer_lens_config)
    model = model.to(device)

    run_name = f"lr-{config.train.learning_rate}_bs-{config.train.batch_size}"
    if config.wandb:
        wandb.init(
            name=run_name,
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=config.model_dump(),
        )

    trained_model = train_model(config, model, train_loader, device, run_name, test_loader)

    # Evaluate the model on the test set
    accuracy = evaluate_model(trained_model, test_loader, device)
    logger.info(
        f"Accuracy of the network on the test samples: %d %%",
        accuracy,
    )
    if config.wandb:
        wandb.log({"test/accuracy": accuracy})


if __name__ == "__main__":
    fire.Fire(main)
