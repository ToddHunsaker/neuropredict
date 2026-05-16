"""Feature extraction from connectivity matrices.

Connectivity matrices are symmetric (corr(A, B) == corr(B, A)) with a
trivial diagonal (corr(A, A) == 1). For ML, we extract only the upper
triangle off-diagonal, which contains all the unique information.

For a 39-region atlas, this gives 39*38/2 = 741 unique edges per subject.
"""

from __future__ import annotations

import numpy as np


def upper_triangle_features(connectivity: np.ndarray) -> np.ndarray:
    """Flatten the upper triangle (off-diagonal) of each subject's matrix.

    Parameters
    ----------
    connectivity
        Array of shape (n_subjects, n_regions, n_regions). Must be square
        per subject. Symmetry is assumed but not enforced (we only read
        the upper triangle anyway).

    Returns
    -------
    features : array of shape (n_subjects, n_regions * (n_regions - 1) / 2)
    """
    if connectivity.ndim != 3:
        raise ValueError(
            f"Expected 3D array (n_subjects, n_regions, n_regions), "
            f"got shape {connectivity.shape}"
        )
    if connectivity.shape[1] != connectivity.shape[2]:
        raise ValueError(
            f"Connectivity matrices must be square, got "
            f"{connectivity.shape[1]}x{connectivity.shape[2]}"
        )

    n_regions = connectivity.shape[1]
    # Indices of the upper triangle, excluding the diagonal
    triu_idx = np.triu_indices(n_regions, k=1)
    # Vectorized extraction across all subjects
    return connectivity[:, triu_idx[0], triu_idx[1]]


def fisher_z_transform(features: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Fisher z-transform correlation values to stabilize variance.

    Correlation values are bounded in [-1, 1] which makes them poorly
    suited for linear models. Fisher's z-transform (arctanh) maps them
    to (-inf, +inf) with approximately Gaussian distribution.

    Parameters
    ----------
    features
        Array of correlation values, typically in (-1, 1).
    eps
        Clipping value to avoid infinities at exactly +/-1.

    Returns
    -------
    z-transformed features of the same shape.
    """
    clipped = np.clip(features, -1 + eps, 1 - eps)
    return np.arctanh(clipped)
