"""ABIDE Preprocessed dataset loader via nilearn.

The Autism Brain Imaging Data Exchange (ABIDE) PCP release provides
pre-extracted ROI time series for ~870 QC-passed subjects across 17 sites,
hosted on AWS S3. Nilearn's wrapper handles S3 fetching, caching, and
QC filtering automatically.

Pipeline choices baked in here:
- pipeline='cpac' — most widely benchmarked
- band_pass_filtering=True — standard for resting-state connectivity
- global_signal_regression=False — controversial; literature mixed
- quality_checked=True — only subjects passing visual QC
- derivatives=['rois_cc200'] — Craddock 200-region functional atlas

These match the most common configuration in published ABIDE papers,
which makes our results comparable to the literature.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from neuropredict.datasets.base import Dataset

logger = logging.getLogger(__name__)


# ABIDE's DX_GROUP coding: 1 = autism, 2 = control.
# We normalize to project convention: 0 = control, 1 = case.
_ABIDE_DX_TO_BINARY = {1: 1, 2: 0}


def _normalize_sex(value) -> int:
    """ABIDE sex coding: 1 = male, 2 = female. Match project convention."""
    if pd.isna(value):
        return 0
    try:
        v = int(value)
        return v if v in (1, 2) else 0
    except (ValueError, TypeError):
        return 0


def _phenotype_to_subjects_df(pheno: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical subjects DataFrame from ABIDE's phenotypic CSV.

    Nilearn returns one phenotype row per subject in the same order as
    the time-series list, so we can rely on positional alignment.
    """
    pheno = pheno.copy().reset_index(drop=True)

    records = []
    for _, row in pheno.iterrows():
        dx_raw = row.get("DX_GROUP")
        if pd.isna(dx_raw) or int(dx_raw) not in _ABIDE_DX_TO_BINARY:
            # Keep the row so positional alignment with time series holds;
            # we'll drop these later after we know which time series loaded.
            diagnosis = -1
        else:
            diagnosis = _ABIDE_DX_TO_BINARY[int(dx_raw)]

        records.append(
            {
                "subject_id": str(row.get("FILE_ID", row.get("SUB_ID", ""))).strip(),
                "site": str(row.get("SITE_ID", "unknown")).strip(),
                "age": float(row["AGE_AT_SCAN"])
                if not pd.isna(row.get("AGE_AT_SCAN"))
                else np.nan,
                "sex": _normalize_sex(row.get("SEX")),
                "diagnosis": diagnosis,
            }
        )

    return pd.DataFrame(records)


def _validate_time_series(
    time_series_list: list[np.ndarray],
    subjects: pd.DataFrame,
    min_timepoints: int = 50,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Drop subjects with malformed time series or unknown diagnosis.

    Filters in lockstep so the returned list and DataFrame stay aligned.
    """
    keep_ts = []
    keep_indices = []
    for i, ts in enumerate(time_series_list):
        sub = subjects.iloc[i]
        if sub["diagnosis"] not in (0, 1):
            logger.warning("Dropping %s: unknown diagnosis", sub["subject_id"])
            continue
        if not isinstance(ts, np.ndarray) or ts.ndim != 2:
            logger.warning("Dropping %s: bad time-series shape", sub["subject_id"])
            continue
        if ts.shape[0] < min_timepoints:
            logger.warning(
                "Dropping %s: only %d timepoints (need >= %d)",
                sub["subject_id"],
                ts.shape[0],
                min_timepoints,
            )
            continue
        keep_ts.append(ts)
        keep_indices.append(i)

    return keep_ts, subjects.iloc[keep_indices].reset_index(drop=True)


def _compute_connectivity(
    time_series_list: list[np.ndarray],
    kind: str = "correlation",
) -> np.ndarray:
    """Compute per-subject connectivity matrices from in-memory time series."""
    from nilearn.connectome import ConnectivityMeasure

    logger.info(
        "Computing %s connectivity for %d subjects...", kind, len(time_series_list)
    )
    measure = ConnectivityMeasure(kind=kind, standardize="zscore_sample")
    return measure.fit_transform(time_series_list)


def load(
    data_dir: str | Path | None = None,
    n_subjects: int | None = None,
    kind: str = "correlation",
) -> Dataset:
    """Load ABIDE Preprocessed as a Dataset.

    Parameters
    ----------
    data_dir
        Where nilearn caches downloads (default ~/nilearn_data).
    n_subjects
        If None, fetch all QC-passed subjects (~870 with default settings).
        Pass a small int (e.g. 50) for fast iteration.
    kind
        Connectivity measure: 'correlation', 'partial correlation', or 'tangent'.
    """
    from nilearn import datasets as nilearn_datasets

    logger.info(
        "Fetching ABIDE (pipeline=cpac, filt+noglobal, QC-checked, cc200) %s",
        f"n_subjects={n_subjects}" if n_subjects else "ALL subjects",
    )
    abide = nilearn_datasets.fetch_abide_pcp(
        data_dir=str(data_dir) if data_dir else None,
        n_subjects=n_subjects,
        pipeline="cpac",
        band_pass_filtering=True,
        global_signal_regression=False,
        derivatives=["rois_cc200"],
        quality_checked=True,
        verbose=1,
    )

    time_series_list = list(abide["rois_cc200"])
    pheno = abide["phenotypic"]
    if not isinstance(pheno, pd.DataFrame):
        pheno = pd.DataFrame(pheno)

    logger.info(
        "Loaded %d time series, %d phenotype rows",
        len(time_series_list),
        len(pheno),
    )

    if len(time_series_list) != len(pheno):
        raise RuntimeError(
            f"Alignment mismatch: {len(time_series_list)} time series "
            f"vs {len(pheno)} phenotype rows"
        )

    subjects = _phenotype_to_subjects_df(pheno)
    time_series_list, subjects = _validate_time_series(time_series_list, subjects)

    if not time_series_list:
        raise RuntimeError("No valid subjects after filtering. Check inputs.")

    connectivity = _compute_connectivity(time_series_list, kind=kind)

    n_regions = connectivity.shape[1]
    atlas_labels = [f"cc200_region_{i:03d}" for i in range(n_regions)]

    logger.info(
        "Final dataset: %d subjects x %d regions (%d controls, %d autism, %d sites)",
        len(subjects),
        n_regions,
        int((subjects.diagnosis == 0).sum()),
        int((subjects.diagnosis == 1).sum()),
        subjects.site.nunique(),
    )

    return Dataset(
        name="abide",
        subjects=subjects,
        connectivity=connectivity,
        atlas_labels=atlas_labels,
    )
