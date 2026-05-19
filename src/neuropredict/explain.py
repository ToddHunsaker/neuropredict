"""SHAP-based explainability for the linear baseline.

Trains a single L1 logistic regression on the full dataset (no CV split),
computes SHAP values to attribute predictions to input features, and maps
the top features back to brain region pairs.

Note: this is for interpretation, not generalization assessment. The
LOSO-CV in models.py is the rigorous evaluation; this analysis is about
asking 'what does the model focus on when it makes its decisions?'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class FeatureAttribution:
    """One ranked feature with its connection metadata."""
    feature_idx: int          # index into the flattened upper-triangle vector
    region_a: int             # one endpoint of the connection (atlas region index)
    region_b: int             # the other endpoint
    mean_abs_shap: float      # mean absolute SHAP value across subjects (importance)
    mean_signed_shap: float   # mean signed SHAP value (direction)
    coefficient: float        # the model's raw L1 coefficient


def upper_triangle_index_map(n_regions: int) -> np.ndarray:
    """Return a (n_features, 2) array mapping each upper-triangle feature
    to its (region_a, region_b) atlas indices.

    The order matches upper_triangle_features() in features.py.
    """
    rows, cols = np.triu_indices(n_regions, k=1)
    return np.stack([rows, cols], axis=1)


def fit_full_model(
    features: np.ndarray,
    labels: np.ndarray,
    C: float = 1.0,
    seed: int = 42,
) -> tuple[LogisticRegression, StandardScaler]:
    """Fit a single L1 logistic regression on all training data.

    Mirrors the per-fold setup in models.py but without leave-out splitting.
    Returns the fitted model and the scaler used to transform features.
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model = LogisticRegression(
        solver="saga",
        l1_ratio=1.0,
        C=C,
        max_iter=20000,
        tol=1e-3,
        random_state=seed,
    )
    model.fit(features_scaled, labels)
    n_nonzero = int(np.sum(model.coef_[0] != 0))
    logger.info(
        "Fit logistic regression on %d subjects, %d features (%d non-zero coefs)",
        features.shape[0],
        features.shape[1],
        n_nonzero,
    )
    return model, scaler


def compute_shap_values(
    model: LogisticRegression,
    scaler: StandardScaler,
    features: np.ndarray,
    background_size: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Compute SHAP values for each subject and feature.

    Uses LinearExplainer with a random sample of subjects as the background
    distribution. Returns an array of shape (n_subjects, n_features).
    """
    rng = np.random.default_rng(seed)
    features_scaled = scaler.transform(features)

    n_subjects = features_scaled.shape[0]
    bg_size = min(background_size, n_subjects)
    bg_idx = rng.choice(n_subjects, size=bg_size, replace=False)
    background = features_scaled[bg_idx]

    logger.info(
        "Computing SHAP values (background sample size=%d)...", bg_size
    )
    explainer = shap.LinearExplainer(model, background)
    shap_values = explainer.shap_values(features_scaled)
    return shap_values


def rank_features_by_shap(
    shap_values: np.ndarray,
    coefficients: np.ndarray,
    n_regions: int,
    top_k: int = 50,
) -> list[FeatureAttribution]:
    """Return the top-k features ranked by mean absolute SHAP value."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)

    idx_map = upper_triangle_index_map(n_regions)
    if idx_map.shape[0] != shap_values.shape[1]:
        raise ValueError(
            f"Index map has {idx_map.shape[0]} features but SHAP values "
            f"have {shap_values.shape[1]}. Check n_regions."
        )

    top_idx = np.argsort(mean_abs)[::-1][:top_k]
    return [
        FeatureAttribution(
            feature_idx=int(i),
            region_a=int(idx_map[i, 0]),
            region_b=int(idx_map[i, 1]),
            mean_abs_shap=float(mean_abs[i]),
            mean_signed_shap=float(mean_signed[i]),
            coefficient=float(coefficients[i]),
        )
        for i in top_idx
    ]
