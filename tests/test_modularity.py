from math import log10

import networkit as nk
import pytest

from rib.modularity import ByLayerLogEdgeNorm, GraphClustering, Node
from rib.rib_builder import RibBuildResults, rib_build
from tests.utils import assert_is_close, get_tinystories_config


@pytest.fixture(scope="module")
def results() -> RibBuildResults:
    return rib_build(get_tinystories_config())


def _get_random_edge(graph: GraphClustering) -> tuple[Node, Node, float]:
    """Generator of k random edges, each edge is tuple of (inNode, outNode, weight)."""
    u, v = nk.graphtools.randomEdge(graph.G)
    u_node, v_node = graph.nodes[u], graph.nodes[v]
    w = graph.G.weight(u, v)
    if graph.is_edge_forward(u_node, v_node):
        return u_node, v_node, w
    else:
        return v_node, u_node, w


@pytest.mark.slow
def test_clustering(results: RibBuildResults):
    graph = GraphClustering(results)
    assert graph.clusters is not None

    # test clustering contains all nodes, and there's no overlap between clusters
    all_cluster_nodes = [n for c in graph.clusters for n in c.nodes]
    assert set(graph.nodes) == set(all_cluster_nodes)
    assert len(all_cluster_nodes) == len(set(all_cluster_nodes))

    assert len(graph.nodes) == sum(c for c in graph.nodes_per_layer.values())

    # test the underlying edges
    for _ in range(50):
        u, v, w = _get_random_edge(graph)
        # they are adjacent layers
        assert graph.layer_idx_of(u) + 1 == graph.layer_idx_of(v)
        # the weight matches the original graph
        edge = results.edges[graph.layer_idx_of(u)]
        assert edge.in_node_layer == u.layer
        orig_weight = edge.E_hat[v.idx, u.idx].item()
        assert w == orig_weight

    # test plotting functions run
    graph.plot_cluster_spans_and_widths()
    graph.print_top_k_clusters()
    graph.paino_plot()


@pytest.mark.slow
def test_edge_norm(results: RibBuildResults):
    # should keep half of edges in each layer
    get_median = lambda edge: edge.E_hat.flatten().quantile(0.5).item()
    eps_by_layer = {edge.in_node_layer: get_median(edge) for edge in results.edges}
    edge_norm = ByLayerLogEdgeNorm(eps_by_layer=eps_by_layer)
    graph = GraphClustering(results, edge_norm=edge_norm)

    for _ in range(100):
        u, v, w = _get_random_edge(graph)
        orig_weight = results.edges[graph.layer_idx_of(u)].E_hat[v.idx, u.idx].item()
        eps = eps_by_layer[u.layer]
        # we don't create 0 weight edges in the graph, so must have orig_weight > eps
        assert orig_weight > eps
        assert_is_close(w, log10(orig_weight / eps), atol=1e-5, rtol=0)

    graphfull = GraphClustering(results, edge_norm=None)
    assert_is_close(graph.G.numberOfEdges(), graphfull.G.numberOfEdges() / 2, rtol=0.01, atol=0)


@pytest.mark.slow
def test_gamma(results: RibBuildResults):
    graph_small_gamma = GraphClustering(results, gamma=1e-3)
    graph_large_gamma = GraphClustering(results, gamma=10)
    assert len(graph_small_gamma.clusters) < len(graph_large_gamma.clusters)
