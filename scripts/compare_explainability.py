"""Side-by-side comparison of SHAP (linear) vs. GNNExplainer (GNN) network involvement.

Reads the annotated SHAP and GNNExplainer outputs and produces a single
grouped bar chart showing how each model attributes importance across
Yeo's 7 resting-state networks.

Usage:
    python scripts/compare_explainability.py \\
        --shap-annotated results/shap_top_features_abide_annotated.json \\
        --gnn-annotated results/gnn_top_edges_gnnexp_abide.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neuropredict.atlas import YEO_NETWORK_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SHAP vs GNNExplainer network involvement"
    )
    parser.add_argument("--shap-annotated", type=str, required=True)
    parser.add_argument("--gnn-annotated", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--tag", type=str, default="comparison")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logger = logging.getLogger(__name__)

    shap_path = Path(args.shap_annotated)
    gnn_path = Path(args.gnn_annotated)
    if not shap_path.exists():
        raise SystemExit(f"{shap_path} not found.")
    if not gnn_path.exists():
        raise SystemExit(f"{gnn_path} not found.")

    with shap_path.open() as fh:
        shap_data = json.load(fh)
    with gnn_path.open() as fh:
        gnn_data = json.load(fh)

    shap_counts = shap_data.get("network_counts", {})
    gnn_counts = gnn_data.get("network_counts", {})

    names = [n for n in YEO_NETWORK_NAMES.values() if n != "background"]
    shap_vals = [shap_counts.get(n, 0) for n in names]
    gnn_vals = [gnn_counts.get(n, 0) for n in names]

    print()
    print(f"{'network':>18}  {'SHAP':>6}  {'GNNExp':>6}")
    for name, s, g in zip(names, shap_vals, gnn_vals, strict=True):
        print(f"{name:>18}  {s:>6}  {g:>6}")
    print()

    # Grouped bar chart
    x = np.arange(len(names))
    width = 0.4

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars1 = ax.bar(
        x - width / 2,
        shap_vals,
        width,
        label="Linear (SHAP)",
        color="steelblue",
        alpha=0.9,
    )
    bars2 = ax.bar(
        x + width / 2,
        gnn_vals,
        width,
        label="BrainGCN (GNNExplainer)",
        color="darkorange",
        alpha=0.9,
    )

    ax.set_ylabel(
        f"Endpoints in top {shap_data['top_k']} connections"
    )
    ax.set_title(
        f"{shap_data['dataset']} | Network involvement: linear vs. GNN"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.3,
                str(int(height)),
                ha="center",
                fontsize=9,
            )

    plt.tight_layout()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(exist_ok=True)
    fig_path = out_dir / f"explainability_{args.tag}.png"
    plt.savefig(fig_path, dpi=120)
    logger.info("Saved comparison figure to %s", fig_path)


if __name__ == "__main__":
    main()
