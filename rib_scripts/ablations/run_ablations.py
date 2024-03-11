"""Run ablation experiments.

Usage:
    python run_ablations.py <path/to/yaml_config_file> [--force]
"""
# Huggingface cache, needs to be set before importing huggingface things
from os import environ

environ["HF_HOME"] = "/mnt/ssd-interp/huggingface_cache/"

import fire

from rib.ablations import load_bases_and_ablate

if __name__ == "__main__":
    fire.Fire(load_bases_and_ablate, serialize=lambda _: "")
