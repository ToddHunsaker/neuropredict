"""Run SHAP analysis on the linear baseline trained on a cached dataset.

Usage:
    python scripts/run_explain.py --cached data/processed/abide.npz
    python scripts/run_explain.py --cached data/processed/abide.npz --top-k 100
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neuropredict.datasets.base import Dataset
from neuropredict.explain import (
    compute_shap_values,
    fit_full_model,
    rank_features_by_shap,
)
from neuropredict.features import fisher_z_transform, upper_triangle_features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SHAP analysis for the linear baseline"
    )
    parser.add_argument(
        "--cached", type=str, required=True, help="Path to a cached Dataset .npz"
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength for L1 logistic regression",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="How many top features to extract and report",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=100,
        help="SHAP background sample size",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Where to save outputs"
    )
    parser.add_argument("--tag", type=str, default=None)
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
        "Loaded %s: %d subjects, %d regions", ds.name, ds.n_subjects, ds.n_regions
    )

    raw_features = upper_triangle_features(ds.connectivity)
    features = fisher_z_transform(raw_features)
    labels = ds.subjects["diagnosis"].to_numpy()
    logger.info("Feature matrix shape: %s", features.shape)

    model, scaler = fit_full_model(features, labels, C=args.C, seed=args.seed)
    shap_values = compute_shap_values(
        model, scaler, features,
        background_size=args.background_size,
        seed=args.seed,
    )
    logger.info("SHAP values shape: %s", shap_values.shape)

    top_features = rank_features_by_shap(
        shap_values=shap_values,
        coefficients=model.coef_[0],
        n_regions=ds.n_regions,
        top_k=args.top_k,
    )

    print()
    print(f"Top {args.top_k} features by mean |SHAP|:")
    print(f"{'rank':>4}  {'region_a':>8}  {'region_b':>8}  "
          f"{'mean|SHAP|':>11}  {'signed_SHAP':>12}  {'coef':>8}")
    for rank, f in enumerate(top_features, 1):
        print(
            f"{rank:>4}  {f.region_a:>8}  {f.region_b:>8}  "
            f"{f.mean_abs_shap:>11.4f}  {f.mean_signed_shap:>+12.4f}  "
            f"{f.coefficient:>+8.4f}"
        )
    print()

    # Save artifacts
    results_path = Path(args.results_dir)
    results_path.mkdir(exist_ok=True)
    tag = args.tag or ds.name

    attribution_path = results_path / f"shap_top_features_{tag}.json"
    payload = {
        "dataset": ds.name,
        "model": "logistic_regression_l1",
        "C": args.C,
        "top_k": args.top_k,
        "n_subjects": ds.n_subjects,
        "n_regions": ds.n_regions,
        "n_features": int(features.shape[1]),
        "n_nonzero_coefs": int(np.sum(model.coef_[0] != 0)),
        "features": [
            {
                "rank": rank,
                "feature_idx": f.feature_idx,
                "region_a": f.region_a,
                "region_b": f.region_b,
                "mean_abs_shap": f.mean_abs_shap,
                "mean_signed_shap": f.mean_signed_shap,
                "coefficient": f.coefficient,
            }
            for rank, f in enumerate(top_features, 1)
        ],
    }
    attribution_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved top-feature attributions to %s", attribution_path)

    # Save raw SHAP values for downstream visualization
    shap_array_path = results_path / f"shap_values_{tag}.npy"
    np.save(shap_array_path, shap_values)
    logger.info("Saved raw SHAP value array to %s", shap_array_path)

    # Summary plot: SHAP value distributions for top features
    top_idx = [f.feature_idx for f in top_features[: min(20, args.top_k)]]
    top_shap = shap_values[:, top_idx]
    labels_pretty = [
        f"R{f.region_a}-R{f.region_b}"
        for f in top_features[: min(20, args.top_k)]
    ]

    fig, ax = plt.subplots(figsize=(10, max(6, len(top_idx) * 0.3)))
    ax.boxplot(
        top_shap,
        vert=False,
        labels=labels_pretty,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "steelblue", "alpha": 0.6},
        medianprops={"color": "black"},
    )
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("SHAP value (impact on log-odds of autism)")
    ax.set_title(
        f"{ds.name} | Top {len(top_idx)} most influential connections "
        f"(L1 logistic regression)"
    )
    ax.invert_yaxis()
    plt.tight_layout()

    fig_path = results_path / f"shap_top_features_{tag}.png"
    plt.savefig(fig_path, dpi=120)
    logger.info("Saved top-feature SHAP figure to %s", fig_path)


if __name__ == "__main__":
    main()
