"""Data loading and preprocessing for ADHD-200.

Development path: nilearn.datasets.fetch_adhd() pulls a 40-subject curated
subset preprocessed by the Athena pipeline. Good enough to develop the full
pipeline end-to-end before scaling up.

Production path: full ADHD-200 Preprocessed Repository (~973 subjects across
8 sites) downloaded from NITRC. Requires a free NITRC account and a more
involved download script — see scripts/download_full_adhd200.py.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ADHDSubject:
    """A single subject's data."""

    subject_id: str
    site: str
    age: float
    sex: int  # 1 = male, 2 = female (ADHD-200 convention)
    diagnosis: int  # 0 = control, 1 = ADHD (collapsed from 1/2/3 subtypes)
    func_path: Path
    confounds_path: Path | None = None


def fetch_dev_subset(data_dir: str | Path | None = None, n_subjects: int = 40) -> pd.DataFrame:
    """Fetch nilearn's curated 40-subject ADHD-200 development subset.

    Pulls preprocessed resting-state NIfTI files plus a phenotypic CSV from
    S3. Used for fast iteration; results are not publishable but are
    representative of pipeline behavior.

    Parameters
    ----------
    data_dir
        Where to cache the data. Defaults to ~/nilearn_data.
    n_subjects
        How many subjects to pull. Max is 40 in this subset.

    Returns
    -------
    DataFrame with columns: subject_id, site, age, sex, diagnosis, func_path.
    """
    from nilearn import datasets

    logger.info("Fetching ADHD-200 development subset (n=%d)...", n_subjects)
    adhd = datasets.fetch_adhd(n_subjects=n_subjects, data_dir=str(data_dir) if data_dir else None)

    # nilearn's phenotypic field is a structured numpy array; convert to a tidy df
    pheno = pd.DataFrame(adhd.phenotypic)

    # Normalize column names (nilearn uses inconsistent casing across versions)
    pheno.columns = [c.lower() for c in pheno.columns]

    # The 'adhd' column in nilearn's subset: 0 = control, 1 = ADHD (any subtype)
    diagnosis_col = "adhd" if "adhd" in pheno.columns else "dx"

    records = []
    for i, func_path in enumerate(adhd.func):
        row = pheno.iloc[i]
        records.append(
            {
                "subject_id": str(row.get("subject", row.get("sub", f"sub_{i:04d}"))),
                "site": str(row.get("site", "unknown")),
                "age": float(row.get("age", np.nan)),
                "sex": int(row.get("sex", 0)) if not pd.isna(row.get("sex")) else 0,
                "diagnosis": int(row[diagnosis_col]),
                "func_path": Path(func_path),
            }
        )

    df = pd.DataFrame(records)
    logger.info(
        "Loaded %d subjects: %d controls, %d ADHD across %d sites",
        len(df),
        (df.diagnosis == 0).sum(),
        (df.diagnosis == 1).sum(),
        df.site.nunique(),
    )
    return df


def compute_connectivity_matrices(
    subjects_df: pd.DataFrame,
    atlas: str = "msdl",
    kind: str = "correlation",
) -> tuple[np.ndarray, list[str]]:
    """Compute functional connectivity matrices for each subject.

    Parameters
    ----------
    subjects_df
        Output of `fetch_dev_subset`.
    atlas
        Brain parcellation. 'msdl' (39 regions, probabilistic) is a good
        default for development. For the full ADHD-200 we'll likely switch
        to 'aal' (116 regions) or 'cc200' for higher resolution.
    kind
        Connectivity measure: 'correlation', 'partial correlation', or
        'tangent'. Tangent space embedding is the most principled choice
        for downstream ML but requires fitting a reference; start with
        correlation for the MVP.

    Returns
    -------
    connectivity : array of shape (n_subjects, n_regions, n_regions)
    region_labels : list of region names
    """
    from nilearn import datasets as nilearn_datasets
    from nilearn.connectome import ConnectivityMeasure
    from nilearn.maskers import NiftiMapsMasker

    if atlas == "msdl":
        atlas_data = nilearn_datasets.fetch_atlas_msdl()
        masker = NiftiMapsMasker(
            maps_img=atlas_data.maps,
            standardize="zscore_sample",
            memory="nilearn_cache",
            verbose=0,
        )
        labels = list(atlas_data.labels)
    else:
        raise NotImplementedError(f"Atlas {atlas!r} not wired up yet")

    logger.info("Extracting time series with %s atlas...", atlas)
    time_series = []
    for func_path in subjects_df["func_path"]:
        ts = masker.fit_transform(str(func_path))
        time_series.append(ts)

    logger.info("Computing %s connectivity...", kind)
    connectivity_measure = ConnectivityMeasure(kind=kind, standardize="zscore_sample")
    matrices = connectivity_measure.fit_transform(time_series)

    return matrices, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and process ADHD-200 data")
    parser.add_argument(
        "--fetch-small",
        action="store_true",
        help="Fetch the 40-subject nilearn development subset",
    )
    parser.add_argument(
        "--n-subjects", type=int, default=40, help="Number of subjects to fetch (max 40 for --fetch-small)"
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Where to cache data")
    parser.add_argument(
        "--compute-connectivity",
        action="store_true",
        help="Also compute connectivity matrices after fetching",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/dev_subset.npz",
        help="Where to save processed connectivity arrays",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not args.fetch_small:
        parser.error("Pass --fetch-small to fetch the development subset")

    df = fetch_dev_subset(data_dir=args.data_dir, n_subjects=args.n_subjects)
    print("\nPhenotype summary:")
    print(df[["subject_id", "site", "age", "sex", "diagnosis"]].head(10))
    print(f"\nTotal: {len(df)} subjects | "
          f"Controls: {(df.diagnosis == 0).sum()} | "
          f"ADHD: {(df.diagnosis == 1).sum()}")

    if args.compute_connectivity:
        matrices, labels = compute_connectivity_matrices(df)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            connectivity=matrices,
            labels=np.array(labels),
            diagnosis=df["diagnosis"].values,
            subject_ids=df["subject_id"].values,
            sites=df["site"].values,
        )
        logger.info("Saved %d connectivity matrices to %s", len(matrices), out_path)


if __name__ == "__main__":
    main()
