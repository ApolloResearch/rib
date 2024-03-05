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
import tqdm
from jaxtyping import Bool, Float

from rib.rib_builder import RibBuildResults

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


class SqrtNorm(EdgeNorm):
    def __call__(self, E: torch.Tensor, node_layer: str) -> torch.Tensor:
        return E.sqrt()


class AdaptiveEdgeNorm(EdgeNorm):
    def __init__(self, eps_by_layer: dict[str, float]):
        self.eps_by_layer = eps_by_layer

    @staticmethod
    def _get_minimum_edge(
        E: EdgeTensor, mask: Bool[torch.Tensor, "rib_out rib_in"], ignore0: bool = True
    ) -> float:
        if ignore0:
            E = E[1:, 1:]
            mask = mask[1:, 1:]
        return E[mask].min().item()

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
        ignore0 = results.config.center
        eps_by_layer = {
            nl: AdaptiveEdgeNorm._get_minimum_edge(edge.E_hat, edge_masks[nl], ignore0=ignore0)
            for nl, edge in edges_by_layer.items()
        }
        return cls(eps_by_layer)

    def __call__(self, E: EdgeTensor, node_layer: str) -> EdgeTensor:
        eps = self.eps_by_layer[node_layer]
        return E.clip(min=eps).log10() - log10(eps)


class NodeNormalizedAdaptiveEdgeNorm(AdaptiveEdgeNorm):
    r"""Lognormalizes the edges, but also normalizes each edge by (E_ij / \sum_j E_ij)"""

    def __call__(self, E: EdgeTensor, node_layer: str) -> EdgeTensor:
        logE = super().__call__(E, node_layer)
        return logE * (E.sqrt() / E.sqrt().sum(dim=0, keepdim=True))


class Node(NamedTuple):
    """A node in RIBGraph representing a single RIB direction.

    `layer` is the human readable node-layer, e.g. `ln3.5`.
    `idx` is the numeric index of the node in the layer (sorted by Lambda, as normal).
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


ClusterListLike = Union[NodeCluster, list[NodeCluster], Literal["all", "nonsingleton"]]


class GraphClustering:
    """
    RIB results, along with the results to running Leiden on the wieghted RIB graph.

    In particular, wraps around a networkit graph `G`, and a networkit partition `nk_partition`.
    networkit graphs index nodes by numeric ids. `self.nodes[i]` is the node with id `i`.
    To get the id of a particualr node, use `self.node_to_idx[node]`.

    Edges can be optionally normalized when making the graph.
    """

    G: nk.Graph
    nodes: list[Node]
    node_to_idx: dict[Node, int]
    clusters: list[NodeCluster]
    _nk_partition: nk.structures.Partition

    def __init__(
        self,
        results: RibBuildResults,
        edge_norm: Optional[EdgeNorm] = None,
        node_layers: Optional[list[str]] = None,
        gamma: float = 1.0,
    ):
        """
        Create a GraphClustering object from a RibBuildResults object.

        Args:
            results: The RIB build results.
            edge_norm: An optional edge normalization instance which normalizes each edge tensor
                before making the underlying graph. If None, will not normalize.
            node_layers: The node layers to use in the Graph. If None, uses the node_layers from the
                results. Otherwise should be a subsequence of the node_layers from the results.
            gamma: The resolution parameter for the Leiden clustering algorithm. Higher values lead
                to more smaller clusters.
        """
        self.results = results
        self.edge_norm = edge_norm or IdentityEdgeNorm()
        self.node_layers = node_layers or results.config.node_layers
        self.gamma = gamma

        self.nodes_per_layer = {
            ir.node_layer: ir.C.shape[1] for ir in results.interaction_rotations if ir.C is not None
        }
        self.nodes = [
            Node(nl, i) for nl in self.node_layers for i in range(self.nodes_per_layer[nl])
        ]
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self._make_graph()
        self._run_leiden()
        self._make_clusters()

    def update_gamma(self, gamma: float):
        """Update the resolution parameter and re-run Leiden. Faster than creating a new Graph"""
        self.gamma = gamma
        self._run_leiden()
        self._make_clusters()

    def _make_graph(self):
        start_idx = 1 if self.results.config.center else 0
        self.G = nk.Graph(n=len(self.nodes), weighted=True)
        for edges in tqdm.tqdm(self.results.edges, desc="Making RIB graph"):
            if edges.in_node_layer in self.node_layers:
                E = self.edge_norm(edges.E_hat, edges.in_node_layer)[start_idx:, start_idx:]
                out_idxs, in_idxs = torch.nonzero(E, as_tuple=True)
                in_ids = in_idxs + self.node_to_idx[Node(edges.in_node_layer, start_idx)]
                out_ids = out_idxs + self.node_to_idx[Node(edges.out_node_layer, start_idx)]
                weights = E[out_idxs, in_idxs].flatten().cpu()
                for i, j, w in zip(
                    in_ids.tolist(), out_ids.tolist(), weights.tolist(), strict=True
                ):
                    self.G.addEdge(i, j, w=w)

        self.G.checkConsistency()

    def _run_leiden(self):
        iterations = 10
        algo = nk.community.ParallelLeiden(self.G, gamma=self.gamma, iterations=iterations)
        algo.run()
        self._nk_partition = algo.getPartition()

    def _make_clusters(self):
        assert self._nk_partition is not None
        clusters = []
        for cluster_id in self._nk_partition.subsetSizeMap().keys():
            member_ids = self._nk_partition.getMembers(cluster_id)
            nodes = [self.nodes[i] for i in member_ids]
            clusters.append(NodeCluster(cluster_id, nodes))

        self.clusters = sorted(clusters, key=lambda c: c.size, reverse=True)

    def layer_idx_of(self, node: Node) -> int:
        """Get the numeric node-layer index of the node-layer of a particular node."""
        return self.node_layers.index(node.layer)

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
        """Check if u is in an earlier layer than v."""
        return self.layer_idx_of(u) < self.layer_idx_of(v)

    def is_in_same_cluster(self, u: Node, v: Node):
        """Check if u and v are in the same cluster."""
        assert self._nk_partition is not None
        return self._nk_partition.inSameSubset(self.node_to_idx[u], self.node_to_idx[v])

    ###

    def num_edges_kept(self, layer, weighted=False, fraction_kept=False):
        """Get the fraction of edges that start at `layer` that are kept by this clustering.

        Args:
            layer: The node layer the edges start at.
            weighted: If False, counts the number of edges, if True, counts the total weight of the
                edges.
            fraction_kept: If True, returns the fraction of edges kept, if False, returns the
                absolute number of edges kept.
        """
        assert layer in self.node_layers
        assert layer != self.node_layers[-1]
        next_layer = self.node_layers[self.layer_idx_of(layer) + 1]
        layer_node_ids = [i for i, n in enumerate(self.nodes) if n.layer == layer]
        next_layer_node_ids = [i for i, n in enumerate(self.nodes) if n.layer == next_layer]

        tot_edges = 0
        kept_edges = 0
        for u_id in layer_node_ids:
            for v_id in next_layer_node_ids:
                if self.G.hasEdge(u_id, v_id):
                    w = self.G.weight(u_id, v_id) if weighted else 1
                    tot_edges += w
                    if self._nk_partition.inSameSubset(u_id, v_id):
                        kept_edges += w

        return kept_edges / tot_edges if fraction_kept else kept_edges

    def _get_clusterlist(self, cluster_list_like: ClusterListLike) -> list[NodeCluster]:
        """
        Several functions take a `cluster_list_like` argument, which can be a single cluster, a list
        of clusters, or a special string like "all" or "nonsingleton". This function converts the
        input to a list of clusters.
        """
        assert self.clusters is not None
        if cluster_list_like == "all":  # all non-singletons
            return self.clusters
        elif cluster_list_like == "nonsingleton":
            return [c for c in self.clusters if c.size > 1]
        elif isinstance(cluster_list_like, NodeCluster):
            return [cluster_list_like]
        else:
            assert isinstance(cluster_list_like, list)
            return cast(list[NodeCluster], cluster_list_like)

    def _cluster_array(self, cluster_list: list[NodeCluster]) -> np.ndarray:
        """
        Helper for paino plots.
        Makes an array of size [len(self.node_layers), max_width] where each entry is:
          * -1 if that RIB direction doesn't exist (the 'paino keys', representing ln1 in pythia)
          * 0 if the RIB direction exists, but isn't in any of the designated clusters
          * 1...n representing the different clusters
        """

        def _fill_array(arr: np.ndarray, nodes: list[Node], val: float):
            for n in nodes:
                arr[self.node_layers.index(n.layer), n.idx] = val

        # make base array
        max_width = max(self.nodes_per_layer.values())
        arr = np.full((len(self.node_layers), max_width), fill_value=-1)
        _fill_array(arr, self.nodes, 0)
        # fill array with clusters
        for i, c in enumerate(cluster_list):
            _fill_array(arr, c.nodes, i + 1)

        return arr

    def paino_plot(self, clusters: ClusterListLike = "nonsingleton", ax=None):
        """
        Makes a 'paino plot' vizualizing the clusters in the RIB graph.

        This is a compact representation where each column of the plot represents a RIB layer,
        and each pixel represents a RIB direction. The color of the pixel represents the cluster
        that RIB direction is in.

        White pixels are RIB directions that are not in the clusters plotted, and dark grey pixels
        are for RIB directions that don't exist (usually because the input embedding was of smaller
        dimension).

        Args:
            clusters: The clusters to plot. Can be a single cluster, a list of clusters, or either
                "all" or "nonsingleton". If "all", plots all clusters. If "nonsingleton", plots all
                clusters with more than one node.
            ax: The matplotlib axis to plot on. If None, creates a new figure.
        """
        cluster_list = self._get_clusterlist(clusters)
        arr = self._cluster_array(cluster_list)

        # make colormap
        null_color = [0.2, 0.2, 0.2, 1]
        singleton_color = [1, 1, 1, 1]
        cluster_colors = [colorcet.glasbey[i % 256] for i in range(len(cluster_list))]
        cmap = plt.matplotlib.colors.ListedColormap([null_color, singleton_color, *cluster_colors])

        # plot
        norm = plt.matplotlib.colors.Normalize(-1, len(cluster_list))
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        ax.matshow(arr[:, 1:].T, cmap=cmap, origin="lower", norm=norm, aspect="auto")
        ax.set_xlabel("Node layer")
        ax.set_ylabel("RIB index")
