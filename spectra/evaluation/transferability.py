"""Transferability metrics: NLEEP, LogME, GBC.

Migrated from utils/transferability_metrics.py into the spectra package.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder

try:
    _logme_root = os.environ.get("SPECTRA_LOGME_ROOT") or os.environ.get("LOGME_ROOT")
    if _logme_root:
        sys.path.insert(0, _logme_root)
    from LogME import LogME as _LogMERef
    _LOGME_AVAILABLE = True
except ImportError:
    _LOGME_AVAILABLE = False


def compute_nleep(features: np.ndarray, labels: np.ndarray, n_components: Optional[int] = None) -> float:
    """NLEEP score via GMM cluster assignment. Higher = better transferability."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)
    if n_classes < 2:
        return 0.0

    n_comp = n_components or min(n_classes, 10)
    gmm = GaussianMixture(
        n_components=n_comp, random_state=42, max_iter=200,
        reg_covar=1e-4, covariance_type="diag",
    )
    gmm.fit(features.astype(np.float64))
    resp = gmm.predict_proba(features)

    p_y_given_z = np.zeros((n_comp, n_classes))
    for z in range(n_comp):
        for c in range(n_classes):
            mask = y == c
            if mask.sum() > 0:
                p_y_given_z[z, c] = resp[mask, z].sum()
        if p_y_given_z[z].sum() > 0:
            p_y_given_z[z] /= p_y_given_z[z].sum()

    p_y_given_x = resp @ p_y_given_z
    log_probs = np.log(p_y_given_x[np.arange(len(y)), y] + 1e-10)
    return float(np.mean(log_probs))


def compute_logme(features: np.ndarray, labels: np.ndarray, normalize: bool = True) -> float:
    """LogME score (Bayesian evidence). Higher = better transferability."""
    if not _LOGME_AVAILABLE:
        raise ImportError("LogME not found. Set SPECTRA_LOGME_ROOT or LOGME_ROOT to the LogME implementation.")
    le = LabelEncoder()
    y = le.fit_transform(labels)
    if len(le.classes_) < 2:
        return 0.0
    logme = _LogMERef(regression=False)
    score = float(logme.fit(features.astype(np.float64), y))
    if normalize:
        score /= features.shape[1] * len(le.classes_)
    return score


def compute_gbc(features: np.ndarray, labels: np.ndarray, pca_dim: int = 64) -> float:
    """GBC score (Gaussian Bhattacharyya Coefficient). Higher (less negative) = better."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)
    if n_classes < 2:
        return 0.0

    n_comp = min(pca_dim, features.shape[1], features.shape[0] - 1)
    pca = PCA(n_components=n_comp)
    feats = pca.fit_transform(features)

    means, variances = [], []
    for c in range(n_classes):
        Xc = feats[y == c]
        if len(Xc) <= 1:
            return float("nan")
        means.append(Xc.mean(axis=0))
        variances.append(np.var(Xc, axis=0) + 1e-6)

    bc_sum = 0.0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            var_avg = (variances[i] + variances[j]) / 2
            diff = means[i] - means[j]
            term1 = 0.125 * np.sum(diff**2 / var_avg)
            term2 = 0.5 * np.sum(
                np.log(var_avg) - 0.5 * (np.log(variances[i]) + np.log(variances[j]))
            )
            db = np.clip(term1 + term2, 0, 50)
            bc_sum += np.exp(-db)

    return float(-bc_sum)


def compute_all_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    do_nleep: bool = True,
    do_logme: bool = True,
    do_gbc: bool = True,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if do_nleep:
        try:
            metrics["NLEEP"] = compute_nleep(features, labels)
        except Exception as e:
            metrics["NLEEP"] = float("nan")
            print(f"  NLEEP error: {e}")
    if do_logme:
        try:
            metrics["LogME"] = compute_logme(features, labels)
        except Exception as e:
            metrics["LogME"] = float("nan")
            print(f"  LogME error: {e}")
    if do_gbc:
        try:
            metrics["GBC"] = compute_gbc(features, labels)
        except Exception as e:
            metrics["GBC"] = float("nan")
            print(f"  GBC error: {e}")
    return metrics
