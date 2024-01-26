"""Run ablation experiments.

Usage:
    python run_ablations.py <path/to/yaml_config_file> [--force]
"""

import fire

from rib.ablations import load_bases_and_ablate

if __name__ == "__main__":
    fire.Fire(load_bases_and_ablate, serialize=lambda _: "")
