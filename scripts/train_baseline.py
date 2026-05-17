"""Run the baseline logistic regression with leave-one-site-out CV.

Usage:
    # Load a cached dataset:
    python scripts/train_baseline.py --cached data/processed/abide.npz

    # Or load fresh (will download/compute):
    python scripts/train_baseline.py --dataset abide
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neuropredict.data import load_dataset
from neuropredict.datasets.base import Dataset
from neuropredict.features import fisher_z_transform, upper_triangle_features
from neuropredict.models import leave_one_site_out_cv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline classifier with LOSO-CV")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--cached", type=str, help="Path to a cached Dataset .npz file"
    )
    source.add_argument(
        "--dataset",
        type=str,
        choices=["abide", "adhd200_dev", "adhd200_full"],
        help="Dataset name to load fresh",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=None,
        help="Limit subjects (only used with --dataset; default: all)",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength for L1 logistic regression",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Where to save outputs"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag suffix for output files (e.g. 'abide_v1')",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Load the dataset
    if args.cached:
        path = Path(args.cached)
        if not path.exists():
            raise SystemExit(f"{path} not found.")
        logger.info("Loading cached dataset from %s...", path)
        ds = Dataset.load_cached(path)
    else:
        kwargs = {}
        if args.n_subjects is not None:
            kwargs["n_subjects"] = args.n_subjects
        ds = load_dataset(args.dataset, **kwargs)

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

    # Feature extraction
    raw_features = upper_triangle_features(ds.connectivity)
    features = fisher_z_transform(raw_features)
    labels = ds.subjects["diagnosis"].to_numpy()
    sites = ds.subjects["site"].to_numpy()
    logger.info("Feature matrix shape: %s", features.shape)

    # Run CV
    logger.info(
        "Running leave-one-site-out CV with L1 logistic regression (C=%.2f)...", args.C
    )
    result = leave_one_site_out_cv(features, labels, sites, C=args.C)

    print()
    print(result.summary())
    print()

    # Save metrics
    results_path = Path(args.results_dir)
    results_path.mkdir(exist_ok=True)
    tag = args.tag or ds.name
    metrics_path = results_path / f"baseline_metrics_{tag}.json"

    metrics = {
        "dataset": ds.name,
        "model": "logistic_regression_l1",
        "C": args.C,
        "n_subjects": ds.n_subjects,
        "n_regions": ds.n_regions,
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
                "n_test_case": fr.n_test_adhd,
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

    fig, ax = plt.subplots(figsize=(max(10, len(sites_list) * 0.6), 5))
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
        f"{ds.name} | per-site performance "
        f"(mean acc={result.mean_accuracy:.3f}, mean AUC={result.mean_auc:.3f})"
    )
    ax.legend()
    plt.tight_layout()

    fig_path = results_path / f"baseline_per_site_{tag}.png"
    plt.savefig(fig_path, dpi=120)
    logger.info("Saved per-site figure to %s", fig_path)


if __name__ == "__main__":
    main()
