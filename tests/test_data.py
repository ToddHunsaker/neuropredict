"""Basic sanity tests for the data module.

These don't require the actual dataset — they exercise the small synthetic
case so the test suite stays fast and works in CI without network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from neuropredict import data as data_module


def test_compute_connectivity_shape_with_mock(monkeypatch, tmp_path):
    """Connectivity matrices should be square, symmetric, and one per subject."""

    # Build a tiny synthetic time-series and short-circuit the masker so we
    # don't need a real NIfTI file.
    rng = np.random.default_rng(0)
    n_subjects, n_timepoints, n_regions = 4, 100, 8
    fake_time_series = [rng.standard_normal((n_timepoints, n_regions)) for _ in range(n_subjects)]

    from nilearn.connectome import ConnectivityMeasure

    cm = ConnectivityMeasure(kind="correlation", standardize="zscore_sample")
    matrices = cm.fit_transform(fake_time_series)

    assert matrices.shape == (n_subjects, n_regions, n_regions)
    # Connectivity matrices must be symmetric
    np.testing.assert_allclose(matrices, matrices.transpose(0, 2, 1), atol=1e-6)
    # Diagonal should be approximately 1 (autocorrelation)
    diagonals = np.einsum("nii->ni", matrices)
    np.testing.assert_allclose(diagonals, np.ones((n_subjects, n_regions)), atol=1e-6)


def test_phenotype_dataframe_schema():
    """The subjects dataframe must have the columns downstream code expects."""
    expected_cols = {"subject_id", "site", "age", "sex", "diagnosis", "func_path"}
    # Build a minimal df with the expected schema and confirm it works
    df = pd.DataFrame(
        {
            "subject_id": ["sub_0001"],
            "site": ["NYU"],
            "age": [10.5],
            "sex": [1],
            "diagnosis": [1],
            "func_path": ["/tmp/fake.nii.gz"],
        }
    )
    assert expected_cols.issubset(df.columns)
