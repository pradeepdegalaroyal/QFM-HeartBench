"""Per-fold classification metrics computed from one confusion matrix.

Balanced accuracy is computed as (sensitivity + specificity) / 2 and checked
against sklearn, so reported metrics are always internally consistent. Probability-based metrics are NaN when a model gives no probabilities.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss, cohen_kappa_score,
                             confusion_matrix, log_loss, matthews_corrcoef, roc_auc_score)


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


def classification_metrics(y: np.ndarray, y_pred: np.ndarray, score: Optional[np.ndarray] = None,
                           proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    bal = (sens + spec) / 2
    assert np.isnan(bal) or abs(bal - balanced_accuracy_score(y, y_pred)) < 1e-12
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    out = {
        "accuracy": (tp + tn) / len(y), "balanced_accuracy": bal, "sensitivity": sens,
        "specificity": spec, "precision": ppv, "npv": npv, "f1": f1,
        "mcc": matthews_corrcoef(y, y_pred), "kappa": cohen_kappa_score(y, y_pred),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "pred_positive_rate": float(y_pred.mean()),
        "collapsed": int(len(np.unique(y_pred)) == 1),
    }
    s = score if score is not None else proba
    out["auc"] = roc_auc_score(y, s) if s is not None and len(np.unique(y)) == 2 else np.nan
    if proba is not None:
        pc = np.clip(proba, 1e-7, 1 - 1e-7)
        out.update({"brier": brier_score_loss(y, proba), "log_loss": log_loss(y, pc, labels=[0, 1]),
                    "ece": ece(y, proba)})
    else:
        out.update({"brier": np.nan, "log_loss": np.nan, "ece": np.nan})
    return {k: (float(v) if not isinstance(v, int) else v) for k, v in out.items()}
