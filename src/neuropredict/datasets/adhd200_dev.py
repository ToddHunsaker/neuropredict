"""ADHD-200 development subset via nilearn (capped at 40 subjects).

This is the same data we've been using since phase 1 — useful for fast
iteration and CI tests where we don't want a multi-GB download. Not
suitable for publishable results; use ABIDE for those.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from neuropredict.datasets.base import Dataset

logger = logging.getLogger(__name__)


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


def _fetch_subjects(data_dir: str | Path | None, n_subjects: int) -> pd.DataFrame:
    """Pull nilearn's curated ADHD subset and normalize the phenotype DataFrame."""
    from nilearn import datasets

    logger.info("Fetching ADHD-200 development subset (n=%d)...", n_subjects)
    adhd = datasets.fetch_adhd(
        n_subjects=n_subjects,
        data_dir=str(data_dir) if data_dir else None,
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

    diagnosis_col = next(
        (c for c in ("adhd", "dx", "diagnosis") if c in pheno.columns),
        None,
    )
    if diagnosis_col is None:
        raise ValueError(f"No diagnosis column found. Available: {list(pheno.columns)}")

    n = min(len(adhd.func), len(pheno))
    if len(adhd.func) != len(pheno):
        logger.warning(
            "Mismatch: %d func files vs %d phenotype rows; using min",
            len(adhd.func),
            len(pheno),
        )

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

    return pd.DataFrame(records)


def _compute_connectivity(
    subjects: pd.DataFrame,
    atlas: str = "msdl",
    kind: str = "correlation",
) -> tuple[np.ndarray, list[str]]:
    """Extract per-subject connectivity matrices using a nilearn atlas."""
    from nilearn import datasets as nilearn_datasets
    from nilearn.connectome import ConnectivityMeasure
    from nilearn.maskers import NiftiMapsMasker

    if atlas != "msdl":
        raise NotImplementedError(f"Atlas {atlas!r} not wired up yet for adhd200_dev")

    atlas_data = nilearn_datasets.fetch_atlas_msdl()
    masker = NiftiMapsMasker(
        maps_img=atlas_data.maps,
        standardize="zscore_sample",
        memory="nilearn_cache",
        verbose=0,
    )
    labels = list(atlas_data.labels)

    logger.info("Extracting time series with %s atlas (%d subjects)...", atlas, len(subjects))
    time_series = [masker.fit_transform(str(p)) for p in subjects["func_path"]]

    logger.info("Computing %s connectivity...", kind)
    measure = ConnectivityMeasure(kind=kind, standardize="zscore_sample")
    matrices = measure.fit_transform(time_series)

    return matrices, labels


def load(
    data_dir: str | Path | None = None,
    n_subjects: int = 40,
    atlas: str = "msdl",
    kind: str = "correlation",
) -> Dataset:
    """Load the ADHD-200 development subset as a Dataset.

    Parameters
    ----------
    data_dir
        Where nilearn should cache raw downloads.
    n_subjects
        Number of subjects to fetch (max 40 in this subset).
    atlas
        Brain parcellation. Currently only 'msdl' (39 regions) is supported.
    kind
        Connectivity measure passed to nilearn's ConnectivityMeasure.
    """
    subjects = _fetch_subjects(data_dir=data_dir, n_subjects=n_subjects)
    connectivity, labels = _compute_connectivity(subjects, atlas=atlas, kind=kind)

    subjects_clean = subjects.drop(columns=["func_path"])

    logger.info(
        "Loaded %d subjects: %d controls, %d ADHD across %d sites",
        len(subjects_clean),
        (subjects_clean.diagnosis == 0).sum(),
        (subjects_clean.diagnosis == 1).sum(),
        subjects_clean.site.nunique(),
    )

    return Dataset(
        name="adhd200_dev",
        subjects=subjects_clean,
        connectivity=connectivity,
        atlas_labels=labels,
    )
