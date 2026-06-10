"""Personalized PageRank golden + edge cases. Deterministic power iteration over a
scipy CSR adjacency; no RNG. The graphs are tiny and hand-reasoned."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from ragreceipts.constants import PPR_DAMPING
from ragreceipts.retrieval.graph_ppr import personalized_pagerank


def _undirected(n: int, edges: list[tuple[int, int]], weight: float = 1.0) -> csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for a, b in edges:
        rows += [a, b]
        cols += [b, a]
        data += [weight, weight]
    return csr_matrix((data, (rows, cols)), shape=(n, n))


# A 6-node "barbell-ish" graph: a path 0-1-2-3 with a leaf 4 off node 1 and an
# isolated node 5 (disconnected). Hand-reasoned: seeding node 0 should put the most
# mass on 0, then its neighbor 1, then 1's neighbors (2, 4), then 3, with 5 ~ 0.
GRAPH = _undirected(6, [(0, 1), (1, 2), (2, 3), (1, 4)])


def test_distribution_sums_to_one_and_nonneg():
    x = personalized_pagerank(GRAPH, {0: 1.0})
    assert x.shape == (6,)
    assert x.min() >= 0.0
    assert x.sum() == pytest.approx(1.0, abs=1e-6)


def test_seed_node_holds_the_most_mass():
    x = personalized_pagerank(GRAPH, {0: 1.0})
    assert x.argmax() == 0  # the personalized seed wins
    assert x[1] > x[2] > x[3]  # mass decays with graph distance from the seed
    assert x[1] > x[4]  # 1 is the seed's direct neighbor; 4 is one hop further


def test_disconnected_node_gets_only_teleport_mass():
    # node 5 is isolated; with seed on 0 it only receives the (1-damping) teleport share
    # spread by the personalization vector, which puts ZERO direct mass on 5.
    x = personalized_pagerank(GRAPH, {0: 1.0})
    assert x[5] == pytest.approx(0.0, abs=1e-6)


def test_symmetric_seed_swap_is_mirror():
    # A 4-node path 0-1-2-3 has the automorphism (0 3)(1 2): the two endpoints are a
    # genuine symmetric pair. Seeding endpoint 0 vs endpoint 3 must mirror the whole
    # distribution. (The 6-node GRAPH above is asymmetric at nodes 2/3 — node 2 has a
    # second neighbor (1), node 3 does not — so it is the wrong graph for this property.)
    path = _undirected(4, [(0, 1), (1, 2), (2, 3)])
    x0 = personalized_pagerank(path, {0: 1.0})
    x3 = personalized_pagerank(path, {3: 1.0})
    assert x0[0] == pytest.approx(x3[3], abs=1e-6)
    assert x0[3] == pytest.approx(x3[0], abs=1e-6)
    assert x0[1] == pytest.approx(x3[2], abs=1e-6)


def test_multiple_seeds_normalized_internally():
    single = personalized_pagerank(GRAPH, {0: 5.0})  # unnormalized mass
    norm = personalized_pagerank(GRAPH, {0: 1.0})  # already a unit seed
    assert np.allclose(single, norm, atol=1e-9)  # seeds are L1-normalized inside


def test_empty_seeds_is_uniform_personalization():
    x = personalized_pagerank(GRAPH, {})
    # with a uniform personalization vector the connected component masses are equal
    # among symmetric nodes; at minimum the vector is a valid distribution.
    assert x.sum() == pytest.approx(1.0, abs=1e-6)
    assert x.min() >= 0.0


def test_dangling_node_no_nan():
    # A graph with a truly dangling column (node 2 has no edges) must not divide by zero.
    dangling = _undirected(3, [(0, 1)])  # node 2 isolated -> zero column
    x = personalized_pagerank(dangling, {0: 1.0})
    assert not np.isnan(x).any()
    assert x.sum() == pytest.approx(1.0, abs=1e-6)


def test_damping_constant_is_pinned():
    assert PPR_DAMPING == 0.5


def test_deterministic_repeat():
    a = personalized_pagerank(GRAPH, {0: 1.0})
    b = personalized_pagerank(GRAPH, {0: 1.0})
    assert np.array_equal(a, b)
