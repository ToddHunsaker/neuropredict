"""Common interface for dataset loaders.

Every dataset module in this package exposes a `load(...)` function that
returns a `Dataset` instance. Downstream code (feature extraction, models,
training scripts) consumes only this interface and stays dataset-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """A loaded dataset ready for modeling.

    Attributes
    ----------
    name
        Short identifier (e.g. 'abide', 'adhd200_dev').
    subjects
        DataFrame with one row per subject. Required columns:
        - subject_id (str)
        - site (str)
        - age (float, may be NaN)
        - sex (int, 1=male, 2=female, 0=unknown)
        - diagnosis (int, 0=control, 1=case)
    connectivity
        Array of shape (n_subjects, n_regions, n_regions) with the
        per-subject functional connectivity matrix. Rows align with
        the order of `subjects`.
    atlas_labels
        List of region labels of length n_regions.
    """

    name: str
    subjects: pd.DataFrame
    connectivity: np.ndarray
    atlas_labels: list[str]

    def __post_init__(self) -> None:
        n_subjects = len(self.subjects)
        if self.connectivity.shape[0] != n_subjects:
            raise ValueError(
                f"Length mismatch: subjects has {n_subjects} rows, "
                f"connectivity has {self.connectivity.shape[0]} matrices"
            )
        if self.connectivity.ndim != 3:
            raise ValueError(
                f"Connectivity must be 3D (n_subjects, n_regions, n_regions), "
                f"got shape {self.connectivity.shape}"
            )
        required = {"subject_id", "site", "age", "sex", "diagnosis"}
        missing = required - set(self.subjects.columns)
        if missing:
            raise ValueError(f"subjects DataFrame is missing required columns: {missing}")

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)

    @property
    def n_regions(self) -> int:
        return self.connectivity.shape[1]

    def save(self, path: str | Path) -> None:
        """Cache to a single .npz file for fast reload."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            name=self.name,
            connectivity=self.connectivity,
            atlas_labels=np.array(self.atlas_labels),
            subjects_csv=self.subjects.to_csv(index=False),
        )

    @classmethod
    def load_cached(cls, path: str | Path) -> Dataset:
        """Load a previously-cached dataset."""
        from io import StringIO

        data = np.load(path, allow_pickle=True)
        subjects = pd.read_csv(StringIO(str(data["subjects_csv"])))
        return cls(
            name=str(data["name"]),
            subjects=subjects,
            connectivity=data["connectivity"],
            atlas_labels=list(data["atlas_labels"]),
        )
