"""
A script to run a first pass of modularity analysis on a RIB build and save various helpful related
plots. Will likely require some customization for particular use cases -- you are encouraged to
copy the code and modify it to your needs.

Example usage:

```bash
python run_modularity.py /path/to/rib_build.pt [--gamma 1.0]
   [--plot_piano] [--plot_graph] [--log_norm] [--ablation_path /path/to/ablation_results.json]
"""
import csv
from pathlib import Path
from typing import Optional, Union

import fire
import matplotlib.pyplot as plt
import torch

from rib.log import logger
from rib.modularity import (
    ByLayerLogEdgeNorm,
    ClusterListLike,
    EdgeNorm,
    GraphClustering,
    SqrtNorm,
)
from rib.rib_builder import RibBuildResults
from rib_scripts.rib_build.plot_graph import plot_graph_by_layer, plot_rib_graph


def plot_modular_graph(
    graph: GraphClustering,
    clusters: ClusterListLike = "nonsingleton",
    out_file=None,
    by_layer=False,
    line_width_factor=None,
    node_labels=None,
):
    clusters_list = graph._get_clusterlist(clusters)
    arr = graph._cluster_array(clusters_list)
    clusters_for_plotting_fn = [
        layer_clusters[: graph.nodes_per_layer[nl]]
        for nl, layer_clusters in zip(graph.node_layers, arr.tolist(), strict=True)
    ]
    if by_layer:
        plot_graph_by_layer(
            graph.results.edges,
            edge_norm=graph.edge_norm,
            line_width_factor=line_width_factor,
            clusters=clusters_for_plotting_fn,
            out_file=out_file,
            nodes_per_layer=100,  # max(self.nodes_per_layer.values()),
        )
    else:
        plot_rib_graph(
            graph.results.edges,
            edge_norm=None,
            line_width_factor=0.001,
            cluster_list=clusters_for_plotting_fn,
            out_file=out_file,
            node_labels=node_labels,
            nodes_per_layer=30,
        )


def run_modularity(
    results_path: Union[str, Path],
    gamma: float = 30,
    plot_piano: bool = True,
    plot_graph: bool = True,
    ablation_path: Optional[Union[str, Path]] = None,
    line_width_factor: Optional[float] = None,
    labels_file: Optional[Union[str, Path]] = None,
):
    # Add labels if provided
    if labels_file is not None:
        with open(labels_file, "r", newline="") as file:
            reader = csv.reader(file)
            node_labels = list(reader)
    else:
        node_labels = None
    """
    This function runs modularity analysis on a RIB build and saves various helpful related plots.

    Args:
        results_path: The path to the RIB build results file.
        threshold: The loss increase threshold for the bisect ablation.
        gamma: The resolution parameter for the modularity analysis. Higher gamma = more clusters.
        plot_piano: Whether to plot a piano plot, giving an overview of modular structure found.
        plot_graph: Whether to plot the full RIB graph with nodes colored and sorted by module.
        ablation_path: The path to the bisect ablation results file. If provided will use this for
            ByLayerLogEdgeNorm. If not provided, will fall back to SqrtNorm.
    """
    results_path = Path(results_path)
    ablation_path = Path(ablation_path) if ablation_path is not None else None
    results = RibBuildResults(**torch.load(results_path))
    edge_norm: EdgeNorm
    if ablation_path is None:
        logger.warning("No ablation path provided, will fall back to SqrtNorm")
        edge_norm = SqrtNorm()
        threshold = 0.0
    elif ablation_path.exists():
        threshold, edge_norm = ByLayerLogEdgeNorm.from_bisect_results(ablation_path, results)
    else:
        raise FileNotFoundError(f"Could not find ablation file at {ablation_path}")
    threshold_str = f"delta{threshold}" if threshold > 0 else "noablation"
    name_prefix = f"{results.exp_name}-{threshold_str}"

    logger.info(f"Making RIB graph in networkit & running clustering...")
    graph = GraphClustering(results, edge_norm, gamma=gamma)
    logger.info(f"Finished clustering.")

    if plot_piano:
        graph.paino_plot()
        paino_path = results_path.parent / f"{name_prefix}-gamma{gamma}-paino.png"
        plt.suptitle(
            f"{results.exp_name}\nRIB cluster assignment for threshold={threshold}, gamma={gamma}"
        )
        plt.savefig(paino_path)
        logger.info(f"Saved paino plot to {paino_path.absolute()}")
        plt.clf()

    if plot_graph:
        rib_graph_path = results_path.parent / f"{name_prefix}-gamma{gamma}-graph.png"
        plot_modular_graph(
            graph=graph,
            out_file=rib_graph_path,
            line_width_factor=line_width_factor,
            node_labels=node_labels,
        )


if __name__ == "__main__":
    fire.Fire(run_modularity)
