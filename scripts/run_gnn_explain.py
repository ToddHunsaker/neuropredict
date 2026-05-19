"""Run GNNExplainer analysis on the BrainGCN trained on a cached dataset.

Usage:
    python scripts/run_gnn_explain.py --cached data/processed/abide.npz
    python scripts/run_gnn_explain.py --cached data/processed/abide.npz --n-epochs 50
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neuropredict.atlas import (
    YEO_NETWORK_NAMES,
    compute_cc200_to_yeo_mapping,
    summarize_top_features_by_network,
)
from neuropredict.datasets.base import Dataset
from neuropredict.gnn_explain import (
    compute_edge_attributions,
    fit_full_gnn,
    rank_edges_by_attribution,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GNNExplainer analysis for the BrainGCN"
    )
    parser.add_argument(
        "--cached", type=str, required=True, help="Path to a cached Dataset .npz"
    )
    parser.add_argument(
        "--edge-threshold-percentile",
        type=float,
        default=90.0,
        help="Keep top (100-x)%% of edges by |weight|. Default 90 = top 10%%.",
    )
    parser.add_argument("--n-epochs", type=int, default=100, help="GNN training epochs")
    parser.add_argument(
        "--explainer-epochs",
        type=int,
        default=100,
        help="Optimization epochs per subject for GNNExplainer",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--top-k", type=int, default=50, help="How many top edges to report"
    )
    parser.add_argument(
        "--min-subjects",
        type=int,
        default=10,
        help="Min subjects in which edge must appear to be ranked",
    )
    parser.add_argument(
        "--atlas-cache",
        type=str,
        default="data/processed/cc200_to_yeo.json",
        help="CC200->Yeo mapping cache",
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

    labels = ds.subjects["diagnosis"].to_numpy()

    # Train one GNN on all data
    model, graphs = fit_full_gnn(
        connectivity=ds.connectivity,
        labels=labels,
        edge_threshold_percentile=args.edge_threshold_percentile,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    # Explainer: per-subject edge attributions
    edge_imports = compute_edge_attributions(
        model, graphs, n_explainer_epochs=args.explainer_epochs
    )

    top_edges = rank_edges_by_attribution(
        edge_imports, top_k=args.top_k, min_subjects=args.min_subjects
    )

    print()
    print(f"Top {args.top_k} edges by GNNExplainer mean importance:")
    print(f"{'rank':>4}  {'region_a':>8}  {'region_b':>8}  "
          f"{'mean_imp':>10}  {'n_subj':>6}")
    for rank, e in enumerate(top_edges, 1):
        print(
            f"{rank:>4}  {e.region_a:>8}  {e.region_b:>8}  "
            f"{e.mean_importance:>10.4f}  {e.n_subjects_present:>6}"
        )
    print()

    # Network annotation, parallel to SHAP analysis
    annotations = compute_cc200_to_yeo_mapping(cache_path=args.atlas_cache)
    region_to_network = {a.region_idx: a.dominant_network_name for a in annotations}

    class _F:
        def __init__(self, ra, rb):
            self.region_a = ra
            self.region_b = rb

    network_counts = summarize_top_features_by_network(
        [_F(e.region_a, e.region_b) for e in top_edges],
        region_to_network,
    )

    print()
    print(f"Top {len(top_edges)} edges involve these networks:")
    print(f"{'network':>18}  {'count':>5}")
    for name in YEO_NETWORK_NAMES.values():
        if name == "background":
            continue
        print(f"{name:>18}  {network_counts.get(name, 0):>5}")
    print()

    # Save artifacts
    results_path = Path(args.results_dir)
    results_path.mkdir(exist_ok=True)
    tag = args.tag or f"gnnexp_{ds.name}"

    edges_path = results_path / f"gnn_top_edges_{tag}.json"
    payload = {
        "dataset": ds.name,
        "model": "brain_gcn",
        "edge_threshold_percentile": args.edge_threshold_percentile,
        "n_epochs": args.n_epochs,
        "explainer_epochs": args.explainer_epochs,
        "top_k": args.top_k,
        "min_subjects": args.min_subjects,
        "n_subjects": ds.n_subjects,
        "n_regions": ds.n_regions,
        "edges": [
            {
                "rank": rank,
                "region_a": e.region_a,
                "region_b": e.region_b,
                "network_a": region_to_network.get(e.region_a, "unknown"),
                "network_b": region_to_network.get(e.region_b, "unknown"),
                "mean_importance": e.mean_importance,
                "n_subjects_present": e.n_subjects_present,
            }
            for rank, e in enumerate(top_edges, 1)
        ],
        "network_counts": network_counts,
    }
    edges_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved top-edge attributions to %s", edges_path)

    # Network involvement bar chart
    names = [n for n in YEO_NETWORK_NAMES.values() if n != "background"]
    counts = [network_counts.get(n, 0) for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, counts, color="darkorange", alpha=0.85)
    ax.set_ylabel(f"Endpoints in top {len(top_edges)} edges")
    ax.set_title(
        f"{ds.name} BrainGCN | GNNExplainer network involvement"
    )
    ax.set_xticks(np.arange(len(names)))
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

    fig_path = results_path / f"gnn_top_edges_{tag}_networks.png"
    plt.savefig(fig_path, dpi=120)
    logger.info("Saved network involvement figure to %s", fig_path)


if __name__ == "__main__":
    main()
