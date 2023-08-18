"""Train a model on Modular Arithmetic.

Usage:
    python train.py <path/to/config.yaml>
"""
import json
from datetime import datetime
from pathlib import Path
import random
from typing import Optional

import fire
import torch
import wandb
from pydantic import BaseModel
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from rib.log import logger
from rib.models import TransformerLensHooked
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


class ModularArithmeticDataset(Dataset):
    """Defines the dataset used for Neel Nanda's modular arithmetic task."""

    def __init__(self, modulus: int, frac_train: float = 0.3, fn_name: str = "add", device: str = "cpu", seed: int = 0, train: bool = True):
        self.modulus = modulus
        self.frac_train = frac_train
        self.fn_name = fn_name
        self.device = device
        self.seed = seed
        self.train = train

        self.fns_dict = {'add': lambda x, y: (x + y) % self.modulus, 'subtract': lambda x, y: (x - y) % self.modulus,
                         'x2xyy2': lambda x, y: (x ** 2 + x * y + y ** 2) % self.modulus}
        self.fn = self.fns_dict[fn_name]

        self.x, self.labels = self.construct_dataset()
        self.train_x, self.train_labels, self.test_x, self.test_labels = self.split_dataset()

    def __getitem__(self, index):
        if self.train:
            return self.train_x[index], self.train_labels[index]
        else:
            return self.test_x[index], self.test_labels[index]

    def __len__(self):
        if self.train:
            return len(self.train_x)
        else:
            return len(self.test_x)

    def construct_dataset(self):
        x = torch.tensor([(i, j, self.modulus) for i in range(self.modulus) for j in range(self.modulus)]).to(self.device)
        y = torch.tensor([self.fn(i, j) for i, j, _ in x]).to(self.device)
        return x, y

    def split_dataset(self):
        random.seed(self.seed)
        indices = list(range(len(self.x)))
        random.shuffle(indices)
        div = int(self.frac_train * len(indices))
        train_indices, test_indices = indices[:div], indices[div:]
        train_x, train_labels = self.x[train_indices], self.labels[train_indices]
        test_x, test_labels = self.x[test_indices], self.labels[test_indices]
        return train_x, train_labels, test_x, test_labels


def cross_entropy_high_precision(logits, labels):
    #only look at predictions of last numbers
    #print("logits: ", logits.size())
    logits = logits[:,-1]
    # compute individual and summed losses for final number
    logprobs = nn.functional.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels.unsqueeze(1), dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss


@logging_redirect_tqdm()
def train_model(
    config: Config, model: TransformerLensHooked, train_loader: DataLoader, device: str, run_name: str, test_loader: DataLoader
) -> TransformerLensHooked:
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
    criterion = cross_entropy_high_precision
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=1, betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 10, 1))

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
        if (epoch) % 1000 == 0:
            print("Loss: ", loss.item())

                # if config.wandb:
                #     wandb.log({"train/loss": loss.item(), "train/samples": samples}, step=samples)

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
def evaluate_model(model: TransformerLensHooked, test_loader: DataLoader, device: str) -> float:
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
    train_data = ModularArithmeticDataset(config.train.modulus, config.train.frac_train, device=device, seed=config.seed, train=True)
    test_data = ModularArithmeticDataset(config.train.modulus, config.train.frac_train, device=device, seed=config.seed, train=False)
    train_loader = DataLoader(train_data, batch_size=config.train.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.train.batch_size, shuffle=False)

    # Initialize the Transformer model
    wrapped_model = TransformerLensHooked(
        n_layers=config.model.num_layers,
        d_model=config.model.residual_dim,
        n_heads=config.model.num_heads,
        d_mlp=512,
        d_vocab=config.model.vocab_dim,
        n_ctx=config.model.token_len,
    )
    model = wrapped_model.hooked_transformer
    model = model.to(device)

    run_name = f"lr-{config.train.learning_rate}_bs-{config.train.batch_size}"
    # if config.wandb:
    #     wandb.init(
    #         name=run_name,
    #         project=config.wandb.project,
    #         entity=config.wandb.entity,
    #         config=config.model_dump(),
    #     )

    trained_model = train_model(config, model, train_loader, device, run_name, test_loader)

    # Evaluate the model on the test set
    accuracy = evaluate_model(trained_model, test_loader, device)
    logger.info(f"Accuracy of the network on the {len(test_loader.dataset)} test samples: %d %%", accuracy)
    # if config.wandb:
    #     wandb.log({"test/accuracy": accuracy})


if __name__ == "__main__":
    fire.Fire(main)
