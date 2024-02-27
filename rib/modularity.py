# %%
import json
from collections import Counter
from math import log10
from random import triangular
from typing import Literal, NamedTuple, Optional, Union, cast

import colorcet
import matplotlib.pyplot as plt
import networkit as nk
import numpy as np
import torch
import tqdm.notebook as tqdm
from jaxtyping import Bool, Float

from rib.rib_builder import RibBuildResults


def get_prefix_colors():
    cmap = plt.get_cmap("Paired")
    return {
        "ln1": cmap.colors[1],
        "ln1_out": cmap.colors[0],
        "attn_in": cmap.colors[3],
        "ln2": cmap.colors[9],
        "ln2_out": cmap.colors[8],
        "mlp_in": cmap.colors[5],
    }


EdgeTensor = Float[torch.Tensor, "rib_out rib_in"]


class EdgeNorm:
    """Normalizes edge tensors, flexibly by layer name, for clustering preprocessing"""

    def __call__(self, E: EdgeTensor, node_layer: str) -> EdgeTensor:
        raise NotImplementedError


class IdentityEdgeNorm(EdgeNorm):
    def __call__(self, E: EdgeTensor, node_layer: str) -> EdgeTensor:
        return E


class LogEdgeNorm(EdgeNorm):
    def __init__(self, eps):
        self.eps = eps

    def __call__(self, E: EdgeTensor, node_layer: str) -> EdgeTensor:
        return E.clip(min=self.eps).log10() - log10(self.eps)


class AdaptiveEdgeNorm(EdgeNorm):
    def __init__(self, eps_by_layer: dict[str, float]):
        self.eps_by_layer = eps_by_layer

    @staticmethod
    def _get_minimum_edge(E: EdgeTensor, mask: Bool[torch.Tensor, "rib_out rib_in"]) -> float:
        return E[1:, 1:][mask[1:, 1:]].min().item()

    @classmethod
    def from_bisect_results(
        cls, ablation_result_path, results: RibBuildResults
    ) -> "AdaptiveEdgeNorm":
        with open(ablation_result_path) as f:
            abl_result = json.load(f)
        edge_masks = {
            nl: torch.tensor(abl_result["edge_masks"][nl][str(n_needed)])
            for nl, n_needed in abl_result["n_edges_required"].items()
        }
        edges_by_layer = {edge.in_node_layer: edge for edge in results.edges}
        eps_by_layer = {
            nl: AdaptiveEdgeNorm._get_minimum_edge(edge.E_hat, edge_masks[nl])
            for nl, edge in edges_by_layer.items()
        }
        return cls(eps_by_layer)

    def __call__(self, E: EdgeTensor, node_layer: str) -> EdgeTensor:
        eps = self.eps_by_layer[node_layer]
        return E.clip(min=eps).log10() - log10(eps)


class Node(NamedTuple):
    """A node in RIBGraph representing a single RIB direction.

    `layer` is the human readable node-layer, e.g. `ln3.5`.
    """

    layer: str
    idx: int

    def __repr__(self):
        return f"{self.layer}.{self.idx}"


class NodeCluster(NamedTuple):
    """A single cluster found by a clustering algorithm.

    `cluster_id` is the numeric id of the cluster, used by networkit.
    `nodes` is a list of `Node` objects in the cluster.
    `size` is the number of nodes in the cluster.
    `span` is the number of unique node-layers in the cluster.
    """

    cluster_id: int
    nodes: list[Node]

    @property
    def size(self) -> int:
        return len(self.nodes)

    @property
    def span(self) -> int:
        return len(set(n.layer for n in self.nodes))

    def layers_counter(self):
        """Get a python Counter of the number of nodes in each node layer."""
        return Counter(n.layer for n in self.nodes)


class RIBGraph:
    """
    Wrapper around a weighted networkit graph represtenting a RIB build.

    networkit graphs index nodes by numeric ids. `self.nodes[i]` is the node with id `i`.
    To get the id of a particualr node, use `self.node_to_idx[node]`.

    Edges can be optionally normalized when making the graph.
    """

    G: nk.Graph
    nodes: list[Node]
    node_to_idx: dict[Node, int]
    clusters: Optional[list[NodeCluster]] = None
    _nk_partition: Optional[nk.structures.Partition] = None

    def __init__(
        self,
        results: RibBuildResults,
        edge_norm: Optional[EdgeNorm] = None,
        node_layers: Optional[list[str]] = None,
    ):
        self.results = results
        self.edge_norm = edge_norm or IdentityEdgeNorm()
        self.node_layers = node_layers or results.config.node_layers

        self.nodes_per_layer = {
            ir.node_layer: ir.C.shape[1] for ir in results.interaction_rotations if ir.C is not None
        }
        self.nodes = [
            Node(nl, i) for nl in self.node_layers for i in range(self.nodes_per_layer[nl])
        ]
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self._make_graph()

    def _add_layer_edges(self, edge, start_idx: int):
        E = self.edge_norm(edge.E_hat, edge.in_node_layer)
        out_idxs, in_idxs = torch.nonzero(E[start_idx:, start_idx:], as_tuple=True)
        in_ids = in_idxs + self.node_to_idx[Node(edge.in_node_layer, start_idx)]
        out_ids = out_idxs + self.node_to_idx[Node(edge.out_node_layer, start_idx)]
        weights = E[out_idxs, in_idxs].flatten().cpu()
        for i, j, w in zip(in_ids.tolist(), out_ids.tolist(), weights.tolist(), strict=True):
            self.G.addEdge(i, j, w=w)

    def _make_graph(self):
        self.G = nk.Graph(n=len(self.nodes), weighted=True)
        for edge in tqdm.tqdm(self.results.edges, desc="Adding edges:"):
            if edge.in_node_layer in self.node_layers:
                self._add_layer_edges(edge, 1)

        self.G.checkConsistency()

    def random_edges(self, k=10):
        """Get k random edges from the graph."""
        return [
            (self.nodes[u], self.nodes[v], self.G.weight(u, v))
            for u, v in nk.graphtools.randomEdges(graph.G, k)
        ]

    def layer_idx_of(self, node: Node) -> int:
        """Get the numeric node-layer index of the node-layer of a particular node."""
        return self.node_layers.index(node.layer)

    def run_leiden(self, gamma=1, iterations=10):
        algo = nk.community.ParallelLeiden(self.G, gamma=gamma, iterations=iterations)
        self._nk_partition = nk.community.detectCommunities(graph.G, algo=algo)
        self._make_clusters()

    def _make_clusters(self):
        assert self._nk_partition is not None
        for cluster_id in self._nk_partition.subsetSizeMap().keys():
            member_ids = self._nk_partition.getMembers(cluster_id)
            nodes = [self.nodes[i] for i in member_ids]
            clusters.append(NodeCluster(cluster_id, nodes))

        self.clusters = sorted(clusters, key=lambda c: c.size, reverse=True)

    def plot_cluster_spans_and_widths(self, ax=None, ms=1, alpha=1):
        assert self.clusters is not None
        ax = ax or plt.gca()
        jitter = lambda xs, w=0.2: [x + triangular(-w, w) for x in xs]
        spans = np.array([c.span for c in self.clusters])
        sizes = np.array([c.size for c in self.clusters])
        widths = sizes / spans
        ax.plot(jitter(spans), jitter(widths), "ok", ms=ms, alpha=alpha)
        ax.set_xlabel("Span of node layers (6=full block)")
        ax.set_ylabel("Size of cluster")

    def print_top_k_clusters(self, k=10):
        assert self.clusters is not None
        for c in self.clusters[:k]:
            print(f"id={c.cluster_id}\t({c.size})")
            counter = c.layers_counter()
            for l in self.node_layers:
                if counter[l]:
                    print(f"   {l}:".ljust(14), counter[l])

    def is_edge_forward(self, u: Node, v: Node):
        """Check if the edge between u and v is forward in the RIB."""
        return self.layer_idx_of(u) < self.layer_idx_of(v)

    def is_in_same_cluster(self, u: Node, v: Node):
        """Check if u and v are in the same cluster."""
        assert self._nk_partition is not None
        return self._nk_partition.inSameSubset(self.node_to_idx[u], self.node_to_idx[v])

    ###
    def frac_edges_kept(self, layer, weighted=False, absolute=False):
        """Get the fraction of edges that start at `layer` that are kept by this clustering."""
        nodes_ids_in_layer = [i for i, n in enumerate(self.nodes) if n.layer == layer]
        tot_edges = 0
        kept_edges = 0
        assert self._nk_partition is not None

        def e_func(u, v, w, e_id):
            # internal graph representation is undirected, so both (u, v) and (v, u) are edges in
            # the graph. We only want to count edges from earlier layers to later layers here.
            u_node = self.nodes[u]
            v_node = self.nodes[v]
            if self.layer_idx_of(u_node) > self.layer_idx_of(v_node):
                return

            nonlocal tot_edges, kept_edges
            tot_edges += w if weighted else 1
            if self._nk_partition.inSameSubset(u, v):
                kept_edges += w if weighted else 1

        for n_id in nodes_ids_in_layer:
            self.G.forEdgesOf(n_id, e_func)

        return (tot_edges - kept_edges) if absolute else kept_edges / tot_edges


def paino_plot(
    graph: RIBGraph,
    clusters: Union[NodeCluster, list[NodeCluster], Literal["all"]] = "all",
    ax=None,
):
    def _fill_array(arr: np.ndarray, nodes: list[Node], val: float):
        for n in nodes:
            arr[graph.node_layers.index(n.layer), n.idx] = val

    assert graph.clusters is not None
    if clusters == "all":  # all non-singletons
        clusters = [c for c in graph.clusters if c.size > 1]
    elif isinstance(clusters, NodeCluster):
        clusters = [clusters]
    clusters = cast(list[NodeCluster], clusters)

    null_color = [0.2, 0.2, 0.2, 1]
    singleton_color = [1, 1, 1, 1]
    cluster_colors = [colorcet.glasbey[i % 256] for i in range(len(clusters))]
    cmap = plt.matplotlib.colors.ListedColormap([null_color, singleton_color, *cluster_colors])
    norm = plt.matplotlib.colors.Normalize(-1, len(clusters))
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # make base array
    max_width = max(graph.nodes_per_layer.values())
    arr = np.full((len(graph.node_layers), max_width), fill_value=-1)
    _fill_array(arr, graph.nodes, 0)
    for i, c in enumerate(clusters):
        _fill_array(arr, c.nodes, i + 1)

    ax.matshow(arr[:, 1:].T, cmap=cmap, origin="lower", norm=norm, aspect="auto")
    ax.set_xlabel("Node layer")
    ax.set_ylabel("RIB index")


def edge_distribution(
    results,
    layout: tuple[int, int] = (2, 4),
    xlim=(None, None),
    ylim=(0, 1),
    vlines: Optional[dict[str, float]] = None,
):
    """Plots culmulative distribution functions of the edge values.

    Helpful for undetstanding the epsilon cutoffs used for edge normalization, especially with
    `edge_distribution(..., vlines=AdaptiveEdgeNorm.eps_by_layer)`.

    Args:
        results: The results from a RIB build.
        layout: The number of rows and columns of the plot.
        xlim: The x-axis limits.
        ylim: The y-axis limits.
        vlines: A dictionary of node_layer -> float, drawn as vertical lines on the subplots.
    """
    ps = torch.cat([torch.linspace(0.02, 0.9, 70), torch.linspace(0.9, 1, 150)])
    figsize = (layout[1] * 3, layout[0] * 3)
    fig, axs = plt.subplots(*layout, figsize=figsize, sharex=True, sharey=True)

    for edge in results.edges:
        in_nl = edge.in_node_layer
        if in_nl in ["ln_final", "ln_final_out"]:
            continue
        prefix, layer = in_nl.split(".")
        E = edge.E_hat.to(torch.float32)
        xs = torch.quantile(E, ps).cpu().numpy()
        ax = axs.flat[int(layer)]
        color = get_prefix_colors()[prefix]
        ax.plot(xs, ps, label=in_nl, color=color)
        ax.set_title(f"layer {layer}")

        if vlines is not None:
            ax.axvline(vlines[in_nl], color=color, linestyle="--")

    plt.xscale("log")
    plt.ylim(*ylim)
    plt.xlim(*xlim)
    axs.flat[-1].legend()
    plt.tight_layout()
