"""Annotate SHAP top features with Yeo network membership.

Takes the output of run_explain.py (a JSON of top features) and the
CC200->Yeo mapping, and produces:
- An annotated JSON with each feature's network labels
- A console summary of network involvement
- A bar plot of network involvement

Usage:
    python scripts/annotate_shap.py --shap-json results/shap_top_features_abide.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from neuropredict.atlas import (
    YEO_NETWORK_NAMES,
    compute_cc200_to_yeo_mapping,
    summarize_top_features_by_network,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate SHAP features with Yeo network membership"
    )
    parser.add_argument(
        "--shap-json",
        type=str,
        required=True,
        help="Path to a shap_top_features_*.json from run_explain.py",
    )
    parser.add_argument(
        "--atlas-cache",
        type=str,
        default="data/processed/cc200_to_yeo.json",
        help="Where to cache (or load) the CC200->Yeo mapping",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Where to save outputs"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logger = logging.getLogger(__name__)

    shap_path = Path(args.shap_json)
    if not shap_path.exists():
        raise SystemExit(f"{shap_path} not found.")

    logger.info("Loading SHAP top features from %s", shap_path)
    with shap_path.open() as fh:
        shap_payload = json.load(fh)

    annotations = compute_cc200_to_yeo_mapping(cache_path=args.atlas_cache)
    region_to_network = {a.region_idx: a.dominant_network_name for a in annotations}
    region_to_overlap = {a.region_idx: a.overlap_fraction for a in annotations}

    # Attach network labels to each feature
    features_annotated = []
    for f in shap_payload["features"]:
        net_a = region_to_network.get(f["region_a"], "unknown")
        net_b = region_to_network.get(f["region_b"], "unknown")
        features_annotated.append({
            **f,
            "network_a": net_a,
            "network_b": net_b,
            "overlap_a": region_to_overlap.get(f["region_a"], 0.0),
            "overlap_b": region_to_overlap.get(f["region_b"], 0.0),
            "within_network": net_a == net_b,
        })

    # Reconstruct a simple object for summary helper
    class _F:
        def __init__(self, d):
            self.region_a = d["region_a"]
            self.region_b = d["region_b"]

    network_counts = summarize_top_features_by_network(
        [_F(f) for f in features_annotated],
        region_to_network,
    )

    print()
    print(f"Top {len(features_annotated)} connections involve these networks:")
    print(f"{'network':>18}  {'count':>5}")
    for name in YEO_NETWORK_NAMES.values():
        if name == "background":
            continue
        print(f"{name:>18}  {network_counts.get(name, 0):>5}")
    print()

    # Save annotated JSON
    out_path = Path(args.results_dir) / (shap_path.stem + "_annotated.json")
    out_payload = {
        **shap_payload,
        "features": features_annotated,
        "network_counts": network_counts,
    }
    out_path.write_text(json.dumps(out_payload, indent=2))
    logger.info("Saved annotated features to %s", out_path)

    # Bar chart of network involvement
    names = [n for n in YEO_NETWORK_NAMES.values() if n != "background"]
    counts = [network_counts.get(n, 0) for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, counts, color="steelblue", alpha=0.85)
    ax.set_ylabel(f"Endpoints in top {len(features_annotated)} connections")
    ax.set_title(
        f"{shap_payload['dataset']} | Network involvement in top SHAP features"
    )
    ax.set_xticklabels(names, rotation=30, ha="right")
    for bar, count in zip(bars, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(count),
            ha="center",
            fontsize=10,
        )
    plt.tight_layout()

    fig_path = Path(args.results_dir) / (shap_path.stem + "_networks.png")
    plt.savefig(fig_path, dpi=120)
    logger.info("Saved network involvement figure to %s", fig_path)


if __name__ == "__main__":
    main()
