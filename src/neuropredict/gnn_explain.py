"""GNNExplainer-based explainability for the BrainGCN.

Trains a single BrainGCN on all data (no CV split, mirroring the linear
baseline explain.py approach), then runs GNNExplainer on each subject to
identify which edges in the graph most influenced its prediction.

Edge attributions are aggregated across subjects to produce a global
ranking comparable to the SHAP analysis of the linear baseline.

Note: this is for interpretation. The LOSO-CV in gnn.py is the rigorous
evaluation; this is asking 'when the GNN does make a decision, which
connections drive it?'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.loader import DataLoader

from neuropredict.gnn import BrainGCN, connectivity_to_graph

logger = logging.getLogger(__name__)


@dataclass
class EdgeAttribution:
    """One ranked edge with its connection metadata."""
    region_a: int
    region_b: int
    mean_importance: float    # mean attribution across all subjects
    n_subjects_present: int   # how many subjects had this edge in their graph


def fit_full_gnn(
    connectivity: np.ndarray,
    labels: np.ndarray,
    edge_threshold_percentile: float = 90.0,
    n_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 5e-4,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[BrainGCN, list[Data]]:
    """Train a single BrainGCN on all subjects (no CV split).

    Returns the trained model and the list of per-subject graphs used,
    so they can be re-fed into GNNExplainer.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info(
        "Converting %d subjects to graphs (edge threshold percentile=%.1f)...",
        len(connectivity),
        edge_threshold_percentile,
    )
    graphs = [
        connectivity_to_graph(m, int(y), edge_threshold_percentile=edge_threshold_percentile)
        for m, y in zip(connectivity, labels, strict=True)
    ]

    n_features = graphs[0].x.shape[1]
    model = BrainGCN(in_features=n_features).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

    logger.info("Training BrainGCN on %d subjects for %d epochs...", len(graphs), n_epochs)
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 25 == 0:
            logger.info("epoch %d | mean loss %.4f", epoch + 1, epoch_loss / len(loader))

    return model, graphs


def compute_edge_attributions(
    model: BrainGCN,
    graphs: list[Data],
    n_explainer_epochs: int = 100,
    device: str = "cpu",
) -> dict[tuple[int, int], list[float]]:
    """Run GNNExplainer on each subject's graph; return per-edge importance lists.

    Returns a dict mapping (region_a, region_b) tuples (canonicalized with
    region_a < region_b) to a list of importance values across all subjects
    where that edge was present.
    """
    model = model.to(device)
    model.eval()

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=n_explainer_epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config={
            "mode": "binary_classification",
            "task_level": "graph",
            "return_type": "raw",
        },
    )

    edge_imports: dict[tuple[int, int], list[float]] = {}

    logger.info("Running GNNExplainer on %d subjects...", len(graphs))
    for i, data in enumerate(graphs):
        if (i + 1) % 50 == 0:
            logger.info("  ... processed %d/%d subjects", i + 1, len(graphs))
        data = data.to(device)

        with torch.enable_grad():
            explanation = explainer(
                x=data.x,
                edge_index=data.edge_index,
                batch=torch.zeros(data.x.shape[0], dtype=torch.long, device=device),
            )

        edge_mask = explanation.edge_mask.detach().cpu().numpy()
        edges = data.edge_index.cpu().numpy()

        for k in range(edges.shape[1]):
            a, b = int(edges[0, k]), int(edges[1, k])
            key = (a, b) if a < b else (b, a)
            edge_imports.setdefault(key, []).append(float(edge_mask[k]))

    return edge_imports


def rank_edges_by_attribution(
    edge_imports: dict[tuple[int, int], list[float]],
    top_k: int = 50,
    min_subjects: int = 10,
) -> list[EdgeAttribution]:
    """Sort edges by mean attribution, keeping only those present in enough subjects."""
    ranked = []
    for (a, b), values in edge_imports.items():
        if len(values) < min_subjects:
            continue
        ranked.append(
            EdgeAttribution(
                region_a=a,
                region_b=b,
                mean_importance=float(np.mean(values)),
                n_subjects_present=len(values),
            )
        )
    ranked.sort(key=lambda e: e.mean_importance, reverse=True)
    return ranked[:top_k]
