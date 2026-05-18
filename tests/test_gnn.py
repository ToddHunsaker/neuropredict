"""Unit tests for the GNN module.

These tests use tiny synthetic data so they run fast in CI and don't require
the real ABIDE dataset. They check shapes, basic behavior, and that the
training loop runs end-to-end without errors — not that performance is good.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from neuropredict.gnn import (
    BrainGCN,
    connectivity_to_graph,
    leave_one_site_out_gnn_cv,
)


def _random_symmetric_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random symmetric matrix with zero diagonal."""
    raw = rng.standard_normal((n, n))
    sym = (raw + raw.T) / 2
    np.fill_diagonal(sym, 0)
    return sym


def test_connectivity_to_graph_basic_shape():
    """Output should be a Data object with the expected shapes and dtypes."""
    rng = np.random.default_rng(0)
    n_regions = 10
    mat = _random_symmetric_matrix(n_regions, rng)

    graph = connectivity_to_graph(mat, label=1)

    assert isinstance(graph, Data)
    assert graph.x.shape == (n_regions, n_regions)
    assert graph.x.dtype == torch.float
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.dtype == torch.long
    assert graph.y.shape == (1,)
    assert int(graph.y.item()) == 1


def test_connectivity_to_graph_no_self_loops():
    """The thresholding should never produce self-loops."""
    rng = np.random.default_rng(1)
    mat = _random_symmetric_matrix(8, rng)
    graph = connectivity_to_graph(mat, label=0, edge_threshold_percentile=50.0)

    src, dst = graph.edge_index
    assert not torch.any(src == dst), "Self-loops detected in edge_index"


def test_connectivity_to_graph_threshold_changes_edge_count():
    """A higher percentile threshold should produce fewer edges."""
    rng = np.random.default_rng(2)
    mat = _random_symmetric_matrix(20, rng)

    sparse = connectivity_to_graph(mat, label=0, edge_threshold_percentile=95.0)
    dense = connectivity_to_graph(mat, label=0, edge_threshold_percentile=50.0)

    assert sparse.edge_index.shape[1] < dense.edge_index.shape[1]


def test_brain_gcn_forward_pass_shape():
    """Model output should be (batch_size, 2) for binary classification logits."""
    rng = np.random.default_rng(3)
    n_subjects, n_regions = 4, 20

    graphs = [
        connectivity_to_graph(_random_symmetric_matrix(n_regions, rng), label=i % 2)
        for i in range(n_subjects)
    ]

    model = BrainGCN(in_features=n_regions, hidden_features=16, out_features=8)
    loader = DataLoader(graphs, batch_size=n_subjects, shuffle=False)
    batch = next(iter(loader))

    model.eval()
    with torch.no_grad():
        logits = model(batch)

    assert logits.shape == (n_subjects, 2)


def test_leave_one_site_out_gnn_cv_smoke():
    """End-to-end LOSO-CV with synthetic data should produce a valid result."""
    rng = np.random.default_rng(42)
    n_subjects, n_regions = 20, 15
    connectivity = np.stack(
        [_random_symmetric_matrix(n_regions, rng) for _ in range(n_subjects)]
    )
    labels = np.array([i % 2 for i in range(n_subjects)])
    sites = np.array(["A"] * 10 + ["B"] * 10)

    result = leave_one_site_out_gnn_cv(
        connectivity=connectivity,
        labels=labels,
        sites=sites,
        n_epochs=3,  # tiny for speed
        batch_size=4,
    )

    assert len(result.fold_results) == 2
    assert 0.0 <= result.mean_accuracy <= 1.0
    for fr in result.fold_results:
        assert fr.n_train == 10
        assert fr.n_test == 10
        assert 0.0 <= fr.accuracy <= 1.0
