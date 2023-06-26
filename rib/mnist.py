"""
This module contains the code for training a simple 3-layer MLP on MNIST.
"""
from pathlib import Path

import torch
import torch.nn as nn
import wandb
import yaml
from pydantic import BaseModel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rib.log import logger


class ModelConfig(BaseModel):
    input_size: int
    hidden_size: int
    output_size: int

class TrainingConfig(BaseModel):
    learning_rate: float
    batch_size: int
    epochs: int

class WandbConfig(BaseModel):
    project: str
    entity: str

class Config(BaseModel):
    seed: int
    log_filename: str
    model: ModelConfig
    training: TrainingConfig
    wandb: WandbConfig

def load_config(config_path: str) -> Config:
    assert config_path.endswith('.yaml'), 'Config file must be a YAML file.'
    assert Path(config_path).exists(), f'Config file {config_path} does not exist.'
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    config = Config(**config_dict)
    return config


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

def train(config: Config):
    torch.manual_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Load the MNIST dataset
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=config.training.batch_size, shuffle=True)

    # Initialize the MLP model
    model = MLP(config.model.input_size, config.model.hidden_size, config.model.output_size)
    model = model.to(device)

    # Define the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # Initialize wandb
    wandb.init(project=config.wandb.project, entity=config.wandb.entity)
    wandb.watch(model, log='all')

    # Training loop
    for epoch in range(config.training.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log the loss to wandb
            if (i+1) % 100 == 0:
                logger.info(f'Epoch [{epoch+1}/{config.training.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')
                wandb.log({"loss": loss.item()})

def train(config_path: str):
    config = load_config(config_path)
    train(config)