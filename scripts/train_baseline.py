"""Run the baseline logistic regression with leave-one-site-out CV.

Usage:
    python scripts/train_baseline.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neuropredict.features import fisher_z_transform, upper_triangle_features
from neuropredict.models import leave_one_site_out_cv


def main(
    data_path: str = "data/processed/dev_subset.npz",
    results_dir: str = "results",
    C: float = 1.0,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    # Load processed data
    path = Path(data_path)
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run:\n  "
            "python -m neuropredict.data --fetch-small --compute-connectivity"
        )

    data = np.load(path, allow_pickle=True)
    connectivity = data["connectivity"]
    labels = data["diagnosis"]
    sites = data["sites"]

    logger.info("Loaded %d subjects with connectivity shape %s", len(labels), connectivity.shape)
    logger.info("Sites: %s", dict(zip(*np.unique(sites, return_counts=True), strict=True)))
    logger.info("Class balance: %d controls, %d ADHD", (labels == 0).sum(), (labels == 1).sum())

    # Feature extraction
    raw_features = upper_triangle_features(connectivity)
    features = fisher_z_transform(raw_features)
    logger.info("Feature matrix shape: %s", features.shape)

    # Run CV
    logger.info("Running leave-one-site-out CV with L1 logistic regression (C=%.2f)...", C)
    result = leave_one_site_out_cv(features, labels, sites, C=C)

    print()
    print(result.summary())
    print()

    # Save metrics
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    metrics_path = results_path / "baseline_metrics.json"

    metrics = {
        "model": "logistic_regression_l1",
        "C": C,
        "n_subjects": int(len(labels)),
        "n_features": int(features.shape[1]),
        "mean_accuracy": result.mean_accuracy,
        "mean_auc": result.mean_auc,
        "valid_auc_folds": result.valid_auc_folds,
        "total_folds": len(result.fold_results),
        "per_fold": [
            {
                "site": fr.test_site,
                "n_train": fr.n_train,
                "n_test": fr.n_test,
                "n_test_adhd": fr.n_test_adhd,
                "n_test_controls": fr.n_test_controls,
                "accuracy": fr.accuracy,
                "auc": fr.auc,
            }
            for fr in result.fold_results
        ],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved metrics to %s", metrics_path)

    # Plot per-site performance
    sites_list = [fr.test_site for fr in result.fold_results]
    accuracies = [fr.accuracy for fr in result.fold_results]
    aucs = [fr.auc if fr.auc is not None else np.nan for fr in result.fold_results]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sites_list))
    width = 0.35
    ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="steelblue")
    ax.bar(x + width / 2, aucs, width, label="AUC", color="coral")
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(sites_list, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Per-site performance (mean acc={result.mean_accuracy:.3f}, "
        f"mean AUC={result.mean_auc:.3f})"
    )
    ax.legend()
    plt.tight_layout()

    fig_path = results_path / "baseline_per_site.png"
    plt.savefig(fig_path, dpi=120)
    logger.info("Saved per-site figure to %s", fig_path)


if __name__ == "__main__":
    main()
