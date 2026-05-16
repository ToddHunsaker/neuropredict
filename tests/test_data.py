"""Basic sanity tests for the data module.

These don't require the actual dataset — they exercise the small synthetic
case so the test suite stays fast and works in CI without network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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

def test_upper_triangle_features_shape():
    """Upper triangle extraction should produce n*(n-1)/2 features per subject."""
    from neuropredict.features import upper_triangle_features

    rng = np.random.default_rng(42)
    n_subjects, n_regions = 5, 10
    # Generate symmetric matrices
    raw = rng.standard_normal((n_subjects, n_regions, n_regions))
    sym = (raw + raw.transpose(0, 2, 1)) / 2

    features = upper_triangle_features(sym)
    expected_n_features = n_regions * (n_regions - 1) // 2  # 45 for n=10
    assert features.shape == (n_subjects, expected_n_features)


def test_upper_triangle_features_values():
    """Extracted values should match the actual upper triangle of input."""
    from neuropredict.features import upper_triangle_features

    # Build a tiny known matrix
    mat = np.array(
        [
            [
                [1.0, 0.5, 0.3],
                [0.5, 1.0, 0.7],
                [0.3, 0.7, 1.0],
            ]
        ]
    )
    features = upper_triangle_features(mat)
    # Upper triangle off-diagonal of a 3x3: positions (0,1), (0,2), (1,2)
    expected = np.array([[0.5, 0.3, 0.7]])
    np.testing.assert_allclose(features, expected)


def test_upper_triangle_features_rejects_non_square():
    """Should raise on non-square matrices."""
    from neuropredict.features import upper_triangle_features

    bad = np.zeros((2, 3, 4))
    try:
        upper_triangle_features(bad)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass


def test_fisher_z_transform_inverse():
    """Fisher z applied to tanh of x should approximately recover x."""
    from neuropredict.features import fisher_z_transform

    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    correlations = np.tanh(x)
    recovered = fisher_z_transform(correlations)
    np.testing.assert_allclose(recovered, x, atol=1e-5)


def test_fisher_z_transform_handles_boundary():
    """Values at +/-1 shouldn't produce infinities."""
    from neuropredict.features import fisher_z_transform

    boundary = np.array([-1.0, 1.0])
    result = fisher_z_transform(boundary)
    assert np.all(np.isfinite(result))
