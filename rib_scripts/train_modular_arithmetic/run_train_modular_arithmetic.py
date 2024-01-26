"""Train a model on Modular Arithmetic.

Will take 10 minutes for 60000 epochs (groks by ~30000 epochs) on GPU.

Usage:
    python run_train_modular_arithmetic.py <path/to/config.yaml>
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import fire
import torch
import torch.optim as optim
import wandb
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig

from rib.data import ModularArithmeticDatasetConfig
from rib.loader import load_dataset
from rib.log import logger
from rib.models.utils import save_model
from rib.types import RootPath
from rib.utils import load_config, replace_pydantic_model, set_seed


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    n_layers: int
    d_model: int
    d_head: int
    n_heads: int
    d_mlp: int
    d_vocab: int
    n_ctx: int
    act_fn: str
    normalization_type: Optional[str]


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    learning_rate: float
    batch_size: int  # Set to max(batch_size, <number of samples in dataset>)
    epochs: int
    eval_every_n_epochs: int
    save_dir: Optional[RootPath] = Field(
        Path(__file__).parent / ".checkpoints" / "modular_arithmetic",
        description="Directory for the output files. Defaults to `./.checkpoints/modular_arithmetic`."
        "If None, no output is written. If a relative path, it is relative to the root of the rib repo.",
    )
    save_every_n_epochs: Optional[int] = None


class WandbConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    project: str
    entity: Optional[str]


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: Optional[int] = 0
    model: ModelConfig
    train: TrainConfig
    dataset: ModularArithmeticDatasetConfig
    wandb: Optional[WandbConfig]


@logging_redirect_tqdm()
def train_model(
    config: Config,
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    run_name: str,
) -> HookedTransformer:
    """Train the Transformer on Modular Arithmetic.

    If config.wandb is not None, log the results to Weights & Biases.

    Args:
        config: Config for the experiment.
        model: Transformer model.
        train_loader: DataLoader for the training set.
        test_loader: DataLoader for the test set.
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
    save_dir: Optional[Path] = (
        config.train.save_dir / f"{run_name}_{timestamp}" if config.train.save_dir else None
    )

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

            if epoch % config.train.eval_every_n_epochs == 0:
                train_accuracy = evaluate_model(model, train_loader, device)
                test_accuracy = evaluate_model(model, test_loader, device)
                model.train()

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
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        total += y.size(0)
        sm_argmax = nn.functional.softmax(outputs, dim=-1).argmax(dim=-1)[:, -1].detach()
        correct += (y == sm_argmax.view(-1)).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def main(config_path_or_obj: Union[str, Config]) -> tuple[float, float]:
    config = load_config(config_path_or_obj, config_model=Config)

    assert config.dataset.return_set == "train", "Must use the train set for training."

    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Initialize the Transformer model
    transformer_lens_config = HookedTransformerConfig(**config.model.model_dump())
    model = HookedTransformer(transformer_lens_config)
    model = model.to(device)

    assert config.dataset.return_set == "train", "currently only supports training on the train set"
    train_dataset = load_dataset(
        dataset_config=config.dataset, model_n_ctx=transformer_lens_config.n_ctx
    )
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)

    test_dataset = load_dataset(
        dataset_config=replace_pydantic_model(config.dataset, {"return_set": "test"}),
        model_n_ctx=transformer_lens_config.n_ctx,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=True)

    run_name = f"lr-{config.train.learning_rate}_bs-{config.train.batch_size}_norm-{config.model.normalization_type}"
    if config.wandb:
        wandb.init(
            name=run_name,
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=config.model_dump(),
        )

    trained_model = train_model(
        config=config,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        run_name=run_name,
    )

    # Evaluate the model on the test set
    train_accuracy = evaluate_model(trained_model, train_loader, device)
    test_accuracy = evaluate_model(trained_model, test_loader, device)
    logger.info(
        f"Accuracy of the network on the test samples: %d %%",
        test_accuracy,
    )
    if config.wandb:
        wandb.log({"test/accuracy": test_accuracy})

    return train_accuracy, test_accuracy


if __name__ == "__main__":
    fire.Fire(main)
