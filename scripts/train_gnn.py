"""Train BrainGCN on a cached Dataset with leave-one-site-out CV.

Usage:
    python scripts/train_gnn.py --cached data/processed/abide.npz
    python scripts/train_gnn.py --cached data/processed/abide.npz --n-epochs 50 --tag quick
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neuropredict.datasets.base import Dataset
from neuropredict.gnn import leave_one_site_out_gnn_cv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BrainGCN with LOSO-CV")
    parser.add_argument(
        "--cached", type=str, required=True, help="Path to a cached Dataset .npz file"
    )
    parser.add_argument(
        "--edge-threshold-percentile",
        type=float,
        default=90.0,
        help="Keep top (100-x)%% of edges by |weight|. Default 90 = top 10%%.",
    )
    parser.add_argument("--n-epochs", type=int, default=100, help="Epochs per fold")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Where to save outputs"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag suffix for output files (e.g. 'gnn_v1')",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="If set, write per-fold checkpoints here for crash recovery",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logger = logging.getLogger(__name__)

    path = Path(args.cached)
    if not path.exists():
        raise SystemExit(f"{path} not found.")
    logger.info("Loading cached dataset from %s...", path)
    ds = Dataset.load_cached(path)

    logger.info(
        "Loaded %s: %d subjects, %d regions, %d sites",
        ds.name,
        ds.n_subjects,
        ds.n_regions,
        ds.subjects.site.nunique(),
    )
    logger.info(
        "Class balance: %d controls, %d cases",
        int((ds.subjects.diagnosis == 0).sum()),
        int((ds.subjects.diagnosis == 1).sum()),
    )

    labels = ds.subjects["diagnosis"].to_numpy()
    sites = ds.subjects["site"].to_numpy()

    logger.info(
        "Running GNN leave-one-site-out CV "
        "(epochs=%d, batch_size=%d, lr=%.4f, edge_pct=%.1f)...",
        args.n_epochs,
        args.batch_size,
        args.learning_rate,
        args.edge_threshold_percentile,
    )
    result = leave_one_site_out_gnn_cv(
        connectivity=ds.connectivity,
        labels=labels,
        sites=sites,
        edge_threshold_percentile=args.edge_threshold_percentile,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )

    print()
    print(result.summary())
    print()

    results_path = Path(args.results_dir)
    results_path.mkdir(exist_ok=True)
    tag = args.tag or f"gnn_{ds.name}"
    metrics_path = results_path / f"gnn_metrics_{tag}.json"

    metrics = {
        "dataset": ds.name,
        "model": "brain_gcn",
        "edge_threshold_percentile": args.edge_threshold_percentile,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "n_subjects": ds.n_subjects,
        "n_regions": ds.n_regions,
        "mean_accuracy": result.mean_accuracy,
        "mean_auc": result.mean_auc,
        "valid_auc_folds": result.valid_auc_folds,
        "total_folds": len(result.fold_results),
        "per_fold": [
            {
                "site": fr.test_site,
                "n_train": fr.n_train,
                "n_test": fr.n_test,
                "n_test_case": fr.n_test_case,
                "n_test_controls": fr.n_test_controls,
                "accuracy": fr.accuracy,
                "auc": fr.auc,
            }
            for fr in result.fold_results
        ],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved metrics to %s", metrics_path)

    sites_list = [fr.test_site for fr in result.fold_results]
    accuracies = [fr.accuracy for fr in result.fold_results]
    aucs = [fr.auc if fr.auc is not None else np.nan for fr in result.fold_results]

    fig, ax = plt.subplots(figsize=(max(10, len(sites_list) * 0.6), 5))
    x = np.arange(len(sites_list))
    width = 0.35
    ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="seagreen")
    ax.bar(x + width / 2, aucs, width, label="AUC", color="purple")
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(sites_list, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"{ds.name} GNN | per-site performance "
        f"(mean acc={result.mean_accuracy:.3f}, mean AUC={result.mean_auc:.3f})"
    )
    ax.legend()
    plt.tight_layout()

    fig_path = results_path / f"gnn_per_site_{tag}.png"
    plt.savefig(fig_path, dpi=120)
    logger.info("Saved per-site figure to %s", fig_path)


if __name__ == "__main__":
    main()
