"""Script for combining edges from multiple ranks into a single file.

Example usage with files:
    python run_combined_edges.py out/pythia-14m_rib_graph_global_rank0.pt
        out/pythia-14m_rib_global_rank1.pt

Example usage with a directory:
    python run_combined_edges.py out/pythia-14m
"""
import fire

from rib.edge_combiner import combine_edges

if __name__ == "__main__":
    fire.Fire(combine_edges)
