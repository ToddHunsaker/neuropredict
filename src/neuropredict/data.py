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


def _parse_sex(value) -> int:
    """Map nilearn's sex codes (M/F strings or numeric) to ints. 1=M, 2=F, 0=unknown."""
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        return {"M": 1, "F": 2}.get(value.strip().upper(), 0)
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def fetch_dev_subset(data_dir: str | Path | None = None, n_subjects: int = 40) -> pd.DataFrame:
    """Fetch nilearn's curated 40-subject ADHD-200 development subset."""
    from nilearn import datasets

    logger.info("Fetching ADHD-200 development subset (n=%d)...", n_subjects)
    adhd = datasets.fetch_adhd(
        n_subjects=n_subjects, data_dir=str(data_dir) if data_dir else None
    )

    pheno_raw = adhd.phenotypic
    if isinstance(pheno_raw, pd.DataFrame):
        pheno = pheno_raw.copy().reset_index(drop=True)
    elif hasattr(pheno_raw, "dtype") and getattr(pheno_raw.dtype, "names", None):
        pheno = pd.DataFrame.from_records(pheno_raw)
    else:
        pheno = pd.DataFrame(pheno_raw).reset_index(drop=True)

    pheno.columns = [str(c).lower().strip() for c in pheno.columns]
    logger.info("Phenotype shape: %s | columns: %s", pheno.shape, list(pheno.columns))

    diagnosis_col = None
    for candidate in ("adhd", "dx", "diagnosis"):
        if candidate in pheno.columns:
            diagnosis_col = candidate
            break
    if diagnosis_col is None:
        raise ValueError(f"No diagnosis column found. Available: {list(pheno.columns)}")

    n_func = len(adhd.func)
    n_pheno = len(pheno)
    if n_func != n_pheno:
        logger.warning("Mismatch: %d func files vs %d phenotype rows; using min", n_func, n_pheno)
    n = min(n_func, n_pheno)

    records = []
    for i in range(n):
        row = pheno.iloc[i]
        records.append(
            {
                "subject_id": str(row.get("subject", row.get("sub", f"sub_{i:04d}"))),
                "site": str(row.get("site", "unknown")),
                "age": float(row.get("age", np.nan)) if not pd.isna(row.get("age")) else np.nan,
                "sex": _parse_sex(row.get("sex")),
                "diagnosis": int(row[diagnosis_col]) if not pd.isna(row[diagnosis_col]) else 0,
                "func_path": Path(adhd.func[i]),
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
        "--n-subjects",
        type=int,
        default=40,
        help="Number of subjects to fetch (max 40 for --fetch-small)",
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
