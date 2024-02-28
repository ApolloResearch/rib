from math import log10

import pytest

from rib.modularity import LogEdgeNorm, RIBGraph
from rib.rib_builder import RibBuildResults, rib_build
from tests.utils import assert_is_close, get_tinystories_config


@pytest.fixture(scope="module")
def results() -> RibBuildResults:
    return rib_build(get_tinystories_config())


def test_clustering(results: RibBuildResults):
    graph = RIBGraph(results)
    graph.run_leiden()
    assert graph.clusters is not None

    # test clustering contains all nodes, and there's no overlap between clusters
    all_cluster_nodes = [n for c in graph.clusters for n in c.nodes]
    assert set(graph.nodes) == set(all_cluster_nodes)
    assert len(all_cluster_nodes) == len(set(all_cluster_nodes))

    assert len(graph.nodes) == sum(c for c in graph.nodes_per_layer.values())

    # test the underlying edges
    for u, v, w in graph.random_edges():
        # they are adjacent layers
        assert graph.layer_idx_of(u) - graph.layer_idx_of(v) in (-1, 1)
        # the weight matches the original graph

        if graph.is_edge_forward(u, v):
            edge = results.edges[graph.layer_idx_of(u)]
            assert edge.in_node_layer == u.layer
            orig_weight = edge.E_hat[v.idx, u.idx].item()

        else:
            edge = results.edges[graph.layer_idx_of(v)]
            assert edge.in_node_layer == v.layer
            orig_weight = edge.E_hat[u.idx, v.idx].item()
        assert w == orig_weight

    # test plotting functions run
    graph.plot_cluster_spans_and_widths()
    graph.print_top_k_clusters()
    graph.paino_plot()


def test_edge_norm(results: RibBuildResults):
    eps = results.edges[0].E_hat.flatten().quantile(0.5)  # use median, will keep half of edges
    edge_norm = LogEdgeNorm(eps)
    graph = RIBGraph(results, edge_norm=edge_norm)

    for u, v, w in graph.random_edges(k=50):
        if graph.is_edge_forward(u, v):
            orig_weight = results.edges[graph.layer_idx_of(u)].E_hat[v.idx, u.idx].item()
        else:
            orig_weight = results.edges[graph.layer_idx_of(v)].E_hat[u.idx, v.idx].item()
        if orig_weight < eps:
            assert w == 0
        else:
            assert_is_close(w, log10(orig_weight / eps), atol=1e-5, rtol=0)


def test_gamma(results: RibBuildResults):
    graph_small_gamma = RIBGraph(results)
    graph_small_gamma.run_leiden(gamma=1e-3)

    graph_large_gamma = RIBGraph(results)
    graph_large_gamma.run_leiden(gamma=10)

    assert len(graph_small_gamma.clusters) < len(graph_large_gamma.clusters)
