"""Train a model on MNIST.

Usage:
    train_mnist.py <path/to/config.yaml>
"""
import fire

from rib.mnist import train

if __name__ == '__main__':
    fire.Fire(train)