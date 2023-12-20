"""Script for building a rib graph.

Usage:
    python run_rib_build.py <path/to/config.yaml> [--force] [--n_pods <int>] [--pod_rank <int>]
"""
import fire

from rib.rib_builder import rib_build

if __name__ == "__main__":
    fire.Fire(rib_build, serialize=lambda _: "")
