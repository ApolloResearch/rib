"""Plot a graph given a results file contain the graph edges.

Usage:
    python plot_graph.py <path/to/results.pt> [--nodes_per_layer <int>]
        [--labels_file <path/to/labels.csv>] [--out_file <path/to/out.png>]
        [--force]

    The results.pt should be the output of the run_rib_build.py script.
"""

from pathlib import Path
from typing import Optional

import fire
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from rib.log import logger
from rib.rib_builder import RibBuildResults


def num_nodes_needed(edge_masks: list[torch.Tensor]) -> list[int]:
    """
    Returns the nubmer of nodes in each layer to connect all of the nodes included in the list of
    edge masks
    """

    def input_nodes_needed(edge_mask: torch.Tensor) -> int:
        return edge_mask.any(dim=0).nonzero().max().item() + 1  # type: ignore

    def output_nodes_needed(edge_mask: torch.Tensor) -> int:
        return edge_mask.any(dim=1).nonzero().max().item() + 1  # type: ignore

    num_needed = []
    num_needed.append(input_nodes_needed(edge_masks[0]))
    for pre_mask, post_mask in zip(edge_masks[:-1], edge_masks[1:]):
        num_needed.append(max(input_nodes_needed(post_mask), output_nodes_needed(pre_mask)))
    num_needed.append(output_nodes_needed(edge_masks[-1]))
    return num_needed


def get_nx_graph(rib_results: RibBuildResults, edge_masks: Optional[list[torch.Tensor]] = None):
    if edge_masks is None:
        edge_masks = [torch.ones_like(edge.E_hat) for edge in rib_results.edges]

    num_nodes_by_layer = num_nodes_needed(edge_masks)
    logger.info(f"Number of nodes by layer: {num_nodes_by_layer}")
    num_layers = len(num_nodes_by_layer)

    G = nx.DiGraph()
    for l in range(num_layers):
        G.add_nodes_from([f"{l}.{i}" for i in range(num_nodes_by_layer[l])], layer=l)

    for l in range(num_layers - 1):
        E_hat = rib_results.edges[l].E_hat
        for out_i, in_i in zip(*edge_masks[l].nonzero().T):
            weight = E_hat[out_i, in_i].item() / E_hat.sum()
            G.add_edge(f"{l}.{in_i}", f"{l + 1}.{out_i}", weight=weight)

    def is_const_node(n: str) -> bool:
        layer, i = n.split(".")
        if i != "0":
            return False
        if rib_results.config.node_layers[-1] == "output" and layer == str(num_layers - 1):
            return False
        return True

    if rib_results.config.center:
        G = G.subgraph([n for n in G.nodes() if not is_const_node(n)])

    return G


def multipartite_layout_by_partition(G: nx.Graph) -> dict[str, np.ndarray]:
    """
    Return x,y coordinates for every node in G.

    The x coordinate depends on the layer of the node (stored in `layer` data key).
    Within a layer the nodes are sorted vertically by which partiton the node belongs to.
    The partition is stored in the `partition` data key.
    """
    layout = {}
    num_layers = max(G.nodes[n]["layer"] for n in G.nodes()) + 1
    for layer in range(num_layers):
        layer_nodes = [n for n in G.nodes() if G.nodes[n]["layer"] == layer]
        sorted_layer_nodes = sorted(layer_nodes, key=lambda n: G.nodes[n]["partition"])

        for i, n in enumerate(sorted_layer_nodes):
            y = i - len(layer_nodes) / 2  # center vertically
            layout[n] = np.array([layer, y])
    return layout


def plot_modular_graph(G: nx.Graph, weighted=True, weight_scale=10, ax=None):
    partition = nx.community.louvain_communities(G, weight="weight")

    for i, part in enumerate(partition):
        for n in part:
            G.nodes[n]["partition"] = i

    layout = multipartite_layout_by_partition(G)
    partitions_of_nodes = [partition for n, partition in G.nodes.data("partition")]
    if weighted:
        edge_weights = [weight * weight_scale for u, v, weight in G.edges.data("weight")]
    else:
        edge_weights = None

    nx.draw(
        G,
        with_labels=True,
        width=edge_weights,
        pos=layout,
        node_color=partitions_of_nodes,
        cmap="tab10",
        ax=ax,
    )


def main(results_file: str) -> None:
    rib_results = RibBuildResults(**torch.load(results_file))
    out_dir = Path(__file__).parent / "out"
    out_file_path = out_dir / f"{rib_results.exp_name}_modularity_graph.png"

    G = get_nx_graph(rib_results, edge_masks=None)
    plot_modular_graph(G)
    plt.savefig(out_file_path)
    logger.info(f"Saved graph to {out_file_path}")


if __name__ == "__main__":
    fire.Fire(main)
