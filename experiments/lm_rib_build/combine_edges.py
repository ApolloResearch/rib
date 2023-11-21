"""This script reads in results from different lm_rib_build runs and adds together the edges.

It takes in as a command line argument a list of .pt files, each of which contains a list of edges.
After checking that the configs are the same, it adds together the edges and saves the result to a
new file (named the same as the first file without a `global_rankN` suffix and with `_combined`
appended).

Example usage:
    python combined_edges.py out/pythia-14m_rib_graph_global_rank0.pt out/pythia-14m_rib_global_rank1.pt
"""
from pathlib import Path

import fire
import torch


def main(*results_files: str) -> None:
    """Combine the edges from the given results files and save the result to a new file."""
    result_file_paths = [Path(results_file) for results_file in results_files]

    # Save the combined edges to a new file
    out_file = (
        result_file_paths[0].parent
        / f"{result_file_paths[0].stem.split('_global_rank')[0]}_combined.pt"
    )
    # Check that the output file doesn't already exist
    assert not out_file.exists(), f"Output file {out_file} already exists."

    # Read in the results
    edges: list[list[tuple[str, torch.Tensor]]] = []
    configs: list[dict] = []
    global_ranks: list[int] = []
    global_sizes: list[int] = []
    for results_file in result_file_paths:
        results = torch.load(results_file)
        if not results["edges"]:
            raise ValueError(f"Results file {results_file} has no edges.")
        edges.append(results["edges"])
        configs.append(results["config"])
        global_ranks.append(results["dist_info"]["global_rank"])
        global_sizes.append(results["dist_info"]["global_size"])

    # Check that all results have the same global_size
    assert len(set(global_sizes)) == 1, f"global_sizes are not all the same: {global_sizes}"

    # Check that the global_ranks are all different.
    assert len(set(global_ranks)) == len(
        global_ranks
    ), f"global_ranks are not all different: {global_ranks}"

    # Check that the largest global_rank is one less than the global_size
    assert (
        max(global_ranks) + 1 == global_sizes[0]
    ), f"global_ranks do not match global_size: {global_ranks}, {global_sizes[0]}"

    # Check that all configs are identical
    assert len(set(map(str, configs))) == 1, f"configs are not all the same: {configs}"

    # Add together the edges
    combined_edges = []
    for edge in zip(*edges):
        edge_names = [e[0] for e in edge]
        # Check that all edge names are the same
        assert len(set(edge_names)) == 1, f"edge names are not all the same: {edge_names}"
        edge_weights = torch.stack([e[1] for e in edge], dim=0).sum(dim=0)
        combined_edges.append((edge_names[0], edge_weights))

    # Make a deep copy of the first results file
    out_results = torch.load(result_file_paths[0])
    out_results["edges"] = combined_edges

    # Set "global_size" and remove all other dist_info that isn't valid for the combined edges
    out_results["global_size"] = global_sizes[0]
    del out_results["dist_info"]

    torch.save(out_results, out_file)


if __name__ == "__main__":
    fire.Fire(main)
