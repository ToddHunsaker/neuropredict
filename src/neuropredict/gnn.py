"""Graph Neural Network (GNN) model for connectivity-based classification.

Treats each subject's connectivity matrix as a graph: nodes are brain regions,
edges are functional connections. A Graph Convolutional Network (GCN) learns
node embeddings by aggregating information from neighbors, then mean-pools
into a graph-level embedding for binary classification.

Design choices:
- Edge thresholding: keep top-k% of edges by absolute value per subject.
  Connectivity matrices are dense by construction; sparsifying makes GCN
  message-passing meaningful and reduces overfitting.
- Node features: each node's row of the connectivity matrix. This gives
  every node a 200-dim feature vector encoding its connection profile.
- Architecture: 2 GCN layers (200 -> 64 -> 32) + global mean pool +
  linear classifier. Deliberately small; N=871 punishes deeper models.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

logger = logging.getLogger(__name__)


def connectivity_to_graph(
    matrix: np.ndarray,
    label: int,
    edge_threshold_percentile: float = 90.0,
) -> Data:
    """Convert one subject's connectivity matrix to a PyG Data object.

    Parameters
    ----------
    matrix
        Square connectivity matrix of shape (n_regions, n_regions).
    label
        Binary diagnosis label (0 or 1).
    edge_threshold_percentile
        Keep edges with |weight| above this percentile (e.g. 90.0 = top 10%).

    Returns
    -------
    A torch_geometric.data.Data object with:
        x         : (n_regions, n_regions) float — per-node features
        edge_index: (2, n_edges) long — source/target indices
        edge_attr : (n_edges,) float — edge weights
        y         : (1,) long — graph label
    """
    n_regions = matrix.shape[0]
    abs_mat = np.abs(matrix)
    # Mask out the diagonal so self-loops don't dominate the threshold
    np.fill_diagonal(abs_mat, 0)

    threshold = np.percentile(abs_mat[abs_mat > 0], edge_threshold_percentile)
    edge_mask = abs_mat >= threshold
    np.fill_diagonal(edge_mask, False)  # no self-loops

    src, dst = np.where(edge_mask)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_attr = torch.tensor(matrix[src, dst], dtype=torch.float)
    x = torch.tensor(matrix, dtype=torch.float)
    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=n_regions)


class BrainGCN(nn.Module):
    """Two-layer Graph Convolutional Network with mean pooling."""

    def __init__(
        self,
        in_features: int = 200,
        hidden_features: int = 64,
        out_features: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, out_features)
        self.dropout = dropout
        self.classifier = nn.Linear(out_features, 2)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Pool node embeddings into a single graph-level vector
        x = global_mean_pool(x, batch)
        return self.classifier(x)


@dataclass
class GNNFoldResult:
    test_site: str
    n_train: int
    n_test: int
    n_test_case: int
    n_test_controls: int
    accuracy: float
    auc: float | None


@dataclass
class GNNCVResult:
    fold_results: list[GNNFoldResult] = field(default_factory=list)
    mean_accuracy: float = 0.0
    mean_auc: float = 0.0
    valid_auc_folds: int = 0

    def summary(self) -> str:
        lines = [
            f"GNN leave-one-site-out CV results ({len(self.fold_results)} folds)",
            f"  Mean accuracy: {self.mean_accuracy:.3f}",
            f"  Mean AUC:      {self.mean_auc:.3f} "
            f"(across {self.valid_auc_folds}/{len(self.fold_results)} folds with both classes)",
            "",
            "Per-fold breakdown:",
        ]
        for fr in self.fold_results:
            auc_str = f"{fr.auc:.3f}" if fr.auc is not None else "  N/A"
            lines.append(
                f"  {fr.test_site:>12}  "
                f"n_train={fr.n_train:>3}  n_test={fr.n_test:>2} "
                f"({fr.n_test_case} case, {fr.n_test_controls} ctrl)  "
                f"acc={fr.accuracy:.3f}  auc={auc_str}"
            )
        return "\n".join(lines)


def _train_one_fold(
    train_graphs: list[Data],
    test_graphs: list[Data],
    n_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 5e-4,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[float, float | None]:
    """Train a BrainGCN on one fold and return (accuracy, auc)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_features = train_graphs[0].x.shape[1]
    model = BrainGCN(in_features=n_features).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 25 == 0:
            logger.debug("epoch %d | loss %.4f", epoch + 1, epoch_loss / len(train_loader))

    # Evaluate
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    auc = float(roc_auc_score(all_labels, all_probs)) if len(set(all_labels)) >= 2 else None
    return float(accuracy), auc


def leave_one_site_out_gnn_cv(
    connectivity: np.ndarray,
    labels: np.ndarray,
    sites: np.ndarray,
    edge_threshold_percentile: float = 90.0,
    n_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 5e-4,
    min_train_size: int = 5,
    device: str = "cpu",
    seed: int = 42,
    checkpoint_dir: str | Path | None = None,
) -> GNNCVResult:
    """Run leave-one-site-out CV with a BrainGCN.

    If `checkpoint_dir` is provided, completed folds are written to disk as
    JSON. Re-running with the same `checkpoint_dir` will skip folds whose
    checkpoint exists, allowing crash recovery and incremental long runs.

    Mirrors the baseline LOSO-CV in models.py but with the GNN as classifier.
    """
    if not (len(connectivity) == len(labels) == len(sites)):
        raise ValueError(
            f"Length mismatch: connectivity={len(connectivity)}, "
            f"labels={len(labels)}, sites={len(sites)}"
        )

    ckpt_path = Path(checkpoint_dir) if checkpoint_dir else None
    if ckpt_path is not None:
        ckpt_path.mkdir(parents=True, exist_ok=True)
        logger.info("Checkpoints will be written to %s", ckpt_path)

    logger.info(
        "Converting %d subjects to graphs (edge threshold percentile=%.1f)...",
        len(connectivity),
        edge_threshold_percentile,
    )
    graphs = [
        connectivity_to_graph(m, int(y), edge_threshold_percentile=edge_threshold_percentile)
        for m, y in zip(connectivity, labels, strict=True)
    ]

    logo = LeaveOneGroupOut()
    fold_results: list[GNNFoldResult] = []

    total_folds = len(np.unique(sites))
    for fold_idx, (train_idx, test_idx) in enumerate(
        logo.split(connectivity, labels, groups=sites)
    ):
        test_site = str(sites[test_idx][0])

        # Resume support: if a checkpoint exists for this site, load and skip
        if ckpt_path is not None:
            fold_ckpt = ckpt_path / f"fold_{test_site}.json"
            if fold_ckpt.exists():
                logger.info("Resuming: found checkpoint for site %s, skipping", test_site)
                with fold_ckpt.open() as fh:
                    cached = json.load(fh)
                fold_results.append(
                    GNNFoldResult(
                        test_site=cached["test_site"],
                        n_train=cached["n_train"],
                        n_test=cached["n_test"],
                        n_test_case=cached["n_test_case"],
                        n_test_controls=cached["n_test_controls"],
                        accuracy=cached["accuracy"],
                        auc=cached["auc"],
                    )
                )
                continue

        if len(train_idx) < min_train_size:
            logger.warning(
                "Skipping site %s: only %d training samples", test_site, len(train_idx)
            )
            continue

        train_graphs = [graphs[i] for i in train_idx]
        test_graphs = [graphs[i] for i in test_idx]

        train_labels = labels[train_idx]
        if len(np.unique(train_labels)) < 2:
            logger.warning("Skipping site %s: training set has only one class", test_site)
            continue

        logger.info(
            "Fold %d/%d: holding out %s (n_train=%d, n_test=%d)",
            fold_idx + 1,
            total_folds,
            test_site,
            len(train_idx),
            len(test_idx),
        )

        acc, auc = _train_one_fold(
            train_graphs,
            test_graphs,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            seed=seed,
        )

        y_test = labels[test_idx]
        fr = GNNFoldResult(
            test_site=test_site,
            n_train=len(train_idx),
            n_test=len(test_idx),
            n_test_case=int((y_test == 1).sum()),
            n_test_controls=int((y_test == 0).sum()),
            accuracy=acc,
            auc=auc,
        )
        fold_results.append(fr)

        # Persist fold result so a crash mid-run isn't catastrophic
        if ckpt_path is not None:
            fold_ckpt = ckpt_path / f"fold_{test_site}.json"
            with fold_ckpt.open("w") as fh:
                json.dump(
                    {
                        "test_site": fr.test_site,
                        "n_train": fr.n_train,
                        "n_test": fr.n_test,
                        "n_test_case": fr.n_test_case,
                        "n_test_controls": fr.n_test_controls,
                        "accuracy": fr.accuracy,
                        "auc": fr.auc,
                    },
                    fh,
                    indent=2,
                )
            logger.info("Saved checkpoint for site %s", test_site)

    if not fold_results:
        raise RuntimeError("No valid GNN CV folds produced. Check your data.")

    accuracies = np.array([fr.accuracy for fr in fold_results])
    valid_aucs = [fr.auc for fr in fold_results if fr.auc is not None]

    return GNNCVResult(
        fold_results=fold_results,
        mean_accuracy=float(accuracies.mean()),
        mean_auc=float(np.mean(valid_aucs)) if valid_aucs else float("nan"),
        valid_auc_folds=len(valid_aucs),
    )
    """Run leave-one-site-out CV with a BrainGCN.

    Mirrors the baseline LOSO-CV in models.py but with the GNN as classifier.
    """
    if not (len(connectivity) == len(labels) == len(sites)):
        raise ValueError(
            f"Length mismatch: connectivity={len(connectivity)}, "
            f"labels={len(labels)}, sites={len(sites)}"
        )

    logger.info(
        "Converting %d subjects to graphs (edge threshold percentile=%.1f)...",
        len(connectivity),
        edge_threshold_percentile,
    )
    graphs = [
        connectivity_to_graph(m, int(y), edge_threshold_percentile=edge_threshold_percentile)
        for m, y in zip(connectivity, labels, strict=True)
    ]

    logo = LeaveOneGroupOut()
    fold_results: list[GNNFoldResult] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        logo.split(connectivity, labels, groups=sites)
    ):
        test_site = str(sites[test_idx][0])

        if len(train_idx) < min_train_size:
            logger.warning(
                "Skipping site %s: only %d training samples", test_site, len(train_idx)
            )
            continue

        train_graphs = [graphs[i] for i in train_idx]
        test_graphs = [graphs[i] for i in test_idx]

        train_labels = labels[train_idx]
        if len(np.unique(train_labels)) < 2:
            logger.warning("Skipping site %s: training set has only one class", test_site)
            continue

        logger.info(
            "Fold %d/%d: holding out %s (n_train=%d, n_test=%d)",
            fold_idx + 1,
            len(np.unique(sites)),
            test_site,
            len(train_idx),
            len(test_idx),
        )

        acc, auc = _train_one_fold(
            train_graphs,
            test_graphs,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            seed=seed,
        )

        y_test = labels[test_idx]
        fold_results.append(
            GNNFoldResult(
                test_site=test_site,
                n_train=len(train_idx),
                n_test=len(test_idx),
                n_test_case=int((y_test == 1).sum()),
                n_test_controls=int((y_test == 0).sum()),
                accuracy=acc,
                auc=auc,
            )
        )

    if not fold_results:
        raise RuntimeError("No valid GNN CV folds produced. Check your data.")

    accuracies = np.array([fr.accuracy for fr in fold_results])
    valid_aucs = [fr.auc for fr in fold_results if fr.auc is not None]

    return GNNCVResult(
        fold_results=fold_results,
        mean_accuracy=float(accuracies.mean()),
        mean_auc=float(np.mean(valid_aucs)) if valid_aucs else float("nan"),
        valid_auc_folds=len(valid_aucs),
    )
