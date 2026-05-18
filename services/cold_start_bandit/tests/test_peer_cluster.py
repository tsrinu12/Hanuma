"""Unit tests for the peer-cluster index."""

import numpy as np
import pytest

from app.peer_cluster import PeerClusterIndex


def test_add_and_search_round_trip():
    idx = PeerClusterIndex(dim=4, n_clusters=2)
    e1 = np.array([1.0, 0.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0, 0.0])
    idx.add("a", e1)
    idx.add("b", e2)
    ids, _ = idx.search(e1, k=2)
    assert ids[0] == "a"
    assert ids[1] == "b"


def test_dim_validation():
    idx = PeerClusterIndex(dim=4)
    with pytest.raises(ValueError):
        idx.add("x", np.array([1.0, 2.0]))


def test_recluster_assigns_close_videos_to_same_cluster():
    rng = np.random.default_rng(0)
    idx = PeerClusterIndex(dim=8, n_clusters=2)
    # Two well-separated clouds in 8-d
    c1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    c2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i in range(20):
        idx.add(f"a{i}", c1 + 0.05 * rng.standard_normal(8))
        idx.add(f"b{i}", c2 + 0.05 * rng.standard_normal(8))
    k = idx.recluster(n_clusters=2, n_iter=30, seed=0)
    assert k == 2
    a_clusters = {idx.cluster_for(f"a{i}") for i in range(20)}
    b_clusters = {idx.cluster_for(f"b{i}") for i in range(20)}
    assert len(a_clusters) == 1
    assert len(b_clusters) == 1
    assert a_clusters != b_clusters


def test_search_empty_returns_empty():
    idx = PeerClusterIndex(dim=3)
    ids, dists = idx.search(np.array([1.0, 0.0, 0.0]), k=5)
    assert ids == []
    assert dists == []


def test_cluster_for_embedding_before_recluster():
    """Before recluster: all embeddings map to cluster 0."""
    idx = PeerClusterIndex(dim=3, n_clusters=4)
    assert idx.cluster_for_embedding(np.array([1.0, 0.0, 0.0])) == 0
    idx.add("a", np.array([1.0, 0.0, 0.0]))
    assert idx.cluster_for("a") == 0
