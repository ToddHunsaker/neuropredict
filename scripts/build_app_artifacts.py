"""Build all artifacts the Streamlit app needs.

Trains a single L1 logistic regression on the full ABIDE dataset,
serializes it (and its scaler) with joblib, and picks a handful of
example subjects (mix of cases and controls) to bundle with the app.

Output goes to `app_artifacts/` which the Streamlit app loads at startup.

Usage:
    python scripts/build_app_artifacts.py --cached data/processed/abide.npz
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np

from neuropredict.atlas import compute_cc200_to_yeo_mapping
from neuropredict.datasets.base import Dataset
from neuropredict.explain import fit_full_model
from neuropredict.features import fisher_z_transform, upper_triangle_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build artifacts for the Streamlit app")
    parser.add_argument("--cached", type=str, required=True)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument(
        "--n-examples-per-class",
        type=int,
        default=4,
        help="How many example subjects to bundle from each class",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="app_artifacts",
        help="Where to write the serialized artifacts",
    )
    parser.add_argument(
        "--atlas-cache",
        type=str,
        default="data/processed/cc200_to_yeo.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    cached_path = Path(args.cached)
    if not cached_path.exists():
        raise SystemExit(f"{cached_path} not found.")
    logger.info("Loading cached dataset from %s...", cached_path)
    ds = Dataset.load_cached(cached_path)

    raw_features = upper_triangle_features(ds.connectivity)
    features = fisher_z_transform(raw_features)
    labels = ds.subjects["diagnosis"].to_numpy()
    sites = ds.subjects["site"].to_numpy()

    # Train model on all data
    logger.info("Fitting linear model on full dataset...")
    model, scaler = fit_full_model(features, labels, C=args.C, seed=args.seed)

    model_path = output_dir / "linear_model.joblib"
    joblib.dump({"model": model, "scaler": scaler, "C": args.C}, model_path)
    logger.info("Saved model to %s", model_path)

    # Build atlas mapping (or load if cached)
    annotations = compute_cc200_to_yeo_mapping(cache_path=args.atlas_cache)
    region_to_network = {a.region_idx: a.dominant_network_name for a in annotations}
    region_overlap = {a.region_idx: a.overlap_fraction for a in annotations}

    atlas_path = output_dir / "atlas_mapping.json"
    atlas_path.write_text(json.dumps({
        "region_to_network": {str(k): v for k, v in region_to_network.items()},
        "region_overlap": {str(k): v for k, v in region_overlap.items()},
        "n_regions": ds.n_regions,
    }, indent=2))
    logger.info("Saved atlas mapping to %s", atlas_path)

    # Pick example subjects: a balanced selection from a few different sites
    rng = np.random.default_rng(args.seed)
    examples = []
    target_sites = ["NYU", "UM_1", "UCLA_1", "PITT"]  # large sites with mixed populations
    for class_label, class_name in [(0, "control"), (1, "autism")]:
        for site in target_sites:
            mask = (labels == class_label) & (sites == site)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            if len(examples) >= args.n_examples_per_class * 2:
                break
            idx = int(rng.choice(indices))
            examples.append({
                "subject_idx": idx,
                "diagnosis": int(class_label),
                "diagnosis_name": class_name,
                "site": site,
                "age": float(ds.subjects.iloc[idx]["age"]),
                "sex": int(ds.subjects.iloc[idx]["sex"]),
            })

    # Save the connectivity matrices for these examples
    example_indices = [e["subject_idx"] for e in examples]
    example_matrices = ds.connectivity[example_indices]
    np.save(output_dir / "example_connectivity.npy", example_matrices)

    # Save metadata
    examples_path = output_dir / "examples.json"
    examples_path.write_text(json.dumps({
        "examples": examples,
        "n_examples": len(examples),
    }, indent=2))
    logger.info("Saved %d example subjects to %s", len(examples), output_dir)

    # Also save metadata about the dataset for the app to display
    meta_path = output_dir / "dataset_meta.json"
    meta_path.write_text(json.dumps({
        "name": ds.name,
        "n_subjects": ds.n_subjects,
        "n_regions": ds.n_regions,
        "n_features": int(features.shape[1]),
        "n_controls": int((labels == 0).sum()),
        "n_cases": int((labels == 1).sum()),
        "sites": sorted(set(sites.tolist())),
    }, indent=2))
    logger.info("Saved dataset metadata to %s", meta_path)

    print()
    print(f"Built artifacts in {output_dir}/:")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<30}  {size_kb:>9.1f} KB")


if __name__ == "__main__":
    main()
