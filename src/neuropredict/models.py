"""Baseline classifiers for connectivity-based ADHD classification.

Leave-one-site-out cross-validation is the rigorous choice for multi-site
fMRI: it tests whether the model generalizes across scanners, populations,
and acquisition protocols, not just within a homogeneous sample. Standard
k-fold CV is known to inflate performance on multi-site fMRI by leaking
site-specific signal across folds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Performance metrics for a single CV fold."""

    test_site: str
    n_train: int
    n_test: int
    n_test_adhd: int
    n_test_controls: int
    accuracy: float
    auc: float | None  # None when test fold has only one class


@dataclass
class CVResult:
    """Aggregated results across all CV folds."""

    fold_results: list[FoldResult] = field(default_factory=list)
    mean_accuracy: float = 0.0
    mean_auc: float = 0.0
    valid_auc_folds: int = 0

    def summary(self) -> str:
        lines = [
            f"Leave-one-site-out CV results ({len(self.fold_results)} folds)",
            f"  Mean accuracy: {self.mean_accuracy:.3f}",
            f"  Mean AUC:      {self.mean_auc:.3f} "
            f"(across {self.valid_auc_folds}/{len(self.fold_results)} folds with both classes)",
            "",
            "Per-fold breakdown:",
        ]
        for fr in self.fold_results:
            auc_str = f"{fr.auc:.3f}" if fr.auc is not None else "  N/A"
            lines.append(
                f"  {fr.test_site:>12}  "
                f"n_train={fr.n_train:>3}  n_test={fr.n_test:>2} "
                f"({fr.n_test_adhd} case, {fr.n_test_controls} ctrl)  "
                f"acc={fr.accuracy:.3f}  auc={auc_str}"
            )
        return "\n".join(lines)


def make_baseline_pipeline(C: float = 1.0, random_state: int = 42) -> Pipeline:
    """Construct the baseline preprocessing + classifier pipeline.

    Logistic regression with L1 (Lasso) penalty was chosen because:
    - Connectivity features are high-dimensional relative to n_subjects
    - L1 induces sparsity, which aids interpretability (most edges -> 0)
    - It's the standard baseline in the ADHD-200 literature
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
               "clf",
                LogisticRegression(
                    l1_ratio=1.0,
                    solver="saga",
                    C=C,
                    random_state=random_state,
                    max_iter=5000,
                    tol=1e-3,
                ),
            ),
        ]
    )


def leave_one_site_out_cv(
    features: np.ndarray,
    labels: np.ndarray,
    sites: np.ndarray,
    C: float = 1.0,
    min_train_size: int = 5,
    random_state: int = 42,
) -> CVResult:
    """Run leave-one-site-out CV with logistic regression.

    Parameters
    ----------
    features
        Feature matrix of shape (n_subjects, n_features).
    labels
        Binary labels (0 = control, 1 = ADHD) of shape (n_subjects,).
    sites
        Site identifier per subject of shape (n_subjects,).
    C
        Inverse regularization strength for logistic regression.
    min_train_size
        Skip folds where the training set would be smaller than this.
        Sites with all-or-most subjects are uninformative as held-out folds.

    Returns
    -------
    CVResult with per-fold and aggregate metrics.
    """
    if not (len(features) == len(labels) == len(sites)):
        raise ValueError(
            f"Length mismatch: features={len(features)}, labels={len(labels)}, sites={len(sites)}"
        )

    logo = LeaveOneGroupOut()
    fold_results: list[FoldResult] = []

    for train_idx, test_idx in logo.split(features, labels, groups=sites):
        test_site = str(sites[test_idx][0])

        if len(train_idx) < min_train_size:
            logger.warning(
                "Skipping site %s: only %d training samples (need >= %d)",
                test_site,
                len(train_idx),
                min_train_size,
            )
            continue

        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Skip folds where training set has only one class
        if len(np.unique(y_train)) < 2:
            logger.warning("Skipping site %s: training set has only one class", test_site)
            continue

        pipeline = make_baseline_pipeline(C=C, random_state=random_state)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # AUC undefined when the test set is single-class
        if len(np.unique(y_test)) >= 2:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, y_proba))
        else:
            auc = None

        fold_results.append(
            FoldResult(
                test_site=test_site,
                n_train=len(train_idx),
                n_test=len(test_idx),
                n_test_adhd=int((y_test == 1).sum()),
                n_test_controls=int((y_test == 0).sum()),
                accuracy=float(accuracy_score(y_test, y_pred)),
                auc=auc,
            )
        )

    if not fold_results:
        raise RuntimeError("No valid CV folds were produced. Check your data.")

    accuracies = np.array([fr.accuracy for fr in fold_results])
    valid_aucs = [fr.auc for fr in fold_results if fr.auc is not None]

    return CVResult(
        fold_results=fold_results,
        mean_accuracy=float(accuracies.mean()),
        mean_auc=float(np.mean(valid_aucs)) if valid_aucs else float("nan"),
        valid_auc_folds=len(valid_aucs),
    )
