from pathlib import Path

import torch

from rib.data_accumulator import Edges
from rib.log import logger
from rib.rib_builder import RibBuildConfig, RibBuildResults


def combine_edges(*inputs: str) -> None:
    """Combine the edges from a directory or list of files and save the result to a new file.

    After checking that the configs are the same, it adds together the edges and saves the result
    to a new file .

    The output file is named the same as the first file in the input without a `global_rankN` suffix
    and with `_combined` appended.

    Args:
        inputs: A list of .pt files or a directory containing .pt files. Each filename should have
        the suffix `global_rankN`, and contain a list of edges.
    """
    result_file_paths: list[Path] = []

    for input in inputs:
        path = Path(input)
        if path.is_dir():
            # Add all .pt files in the directory to result_file_paths
            result_file_paths.extend(path.glob("*.pt"))
        elif path.is_file() and path.suffix == ".pt":
            # Add the file to result_file_paths
            result_file_paths.append(path)
        else:
            raise ValueError(f"Invalid input: {input} is neither a .pt file nor a directory")

    # Check if result_file_paths is empty
    if not result_file_paths:
        raise ValueError("No .pt files found in the provided inputs")

    # Save the combined edges to a new file
    out_file = (
        result_file_paths[0].parent
        / f"{result_file_paths[0].stem.split('_global_rank')[0]}_combined.pt"
    )
    # Check that the output file doesn't already exist
    assert not out_file.exists(), f"Output file {out_file} already exists."

    # Read in the results
    all_edges: list[list[Edges]] = []
    configs: list[RibBuildConfig] = []
    global_ranks: list[int] = []
    global_sizes: list[int] = []
    for results_file in result_file_paths:
        results = RibBuildResults(**torch.load(results_file))
        assert results.contains_all_edges is False, (
            f"Results file {results_file} contains all edges. Either they've already been combined "
            "or the run didn't use parallelisation."
        )
        if not results.edges:
            raise ValueError(f"Results file {results_file} has no edges.")
        all_edges.append(results.edges)
        configs.append(results.config)
        global_ranks.append(results.dist_info.global_rank)
        global_sizes.append(results.dist_info.global_size)

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

    # Concat edges together
    combined_edges = []
    for node_layer_edges in zip(*all_edges):
        assert all(
            isinstance(edges, Edges) for edges in node_layer_edges
        ), f"node_layer_edges is not a tuple of Edges: {node_layer_edges}"
        assert (
            len(set(edges.in_node_layer_name for edges in node_layer_edges)) == 1
        ), f"in_node_layer_names are not all the same across edges: {node_layer_edges}"

        assert (
            len(set(edges.out_node_layer_name for edges in node_layer_edges)) == 1
        ), f"out_node_layer_names are not all the same across edges: {node_layer_edges}"

        combined_edges.append(
            Edges(
                in_node_layer_name=node_layer_edges[0].in_node_layer_name,
                out_node_layer_name=node_layer_edges[0].out_node_layer_name,
                E_hat=torch.stack([e.E_hat for e in node_layer_edges], dim=0).sum(dim=0),
            )
        )

    # Copy and alter the first results file
    out_results = RibBuildResults(**torch.load(result_file_paths[0]))
    out_results.edges = combined_edges
    out_results.contains_all_edges = True

    torch.save(out_results.model_dump(), out_file)
    result_file_strs = "\n".join(map(str, result_file_paths))
    logger.info(f"Combined edges in files\n{result_file_strs}\nand saved to {out_file}")
