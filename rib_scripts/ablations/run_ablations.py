"""Run ablation experiments.

Usage:
    python run_ablations.py <path/to/yaml_config_file> [--force]
"""
# Huggingface cache, needs to be set before importing huggingface things
from os import environ

environ["HF_HOME"] = "/mnt/ssd-interp/huggingface_cache/"

import builtins
from functools import wraps
from pathlib import Path

original_open = builtins.open  # Save the original open function

def patch_path(path):
    """Function to replace '/mnt' with '/homt/mnt' in the file path."""
    if isinstance(path, Path):
        path = str(path)
    path = path.replace('/mnt/ssd-interp/checkpoints/', '/data/stefan_heimersheim/s3/checkpoints/')
    return path

@wraps(original_open)
def open_patched(file, *args, **kwargs):
    """Patched open function that modifies the file path."""
    new_file = patch_path(file)
    return original_open(new_file, *args, **kwargs)

# Patch the built-in open with our custom function
builtins.open = open_patched


import fire

from rib.ablations import load_bases_and_ablate

if __name__ == "__main__":
    fire.Fire(load_bases_and_ablate, serialize=lambda _: "")
