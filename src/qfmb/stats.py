"""Statistical comparison of models evaluated on identical repeated-CV folds.

* Corrected resampled t-test (Nadeau & Bengio, Mach. Learn. 2003; Bouckaert &
  Frank, PAKDD 2004): Var = (1/J + n_test/n_train) s^2 over the J = r*k paired
  fold differences, df = J - 1. The naive paired t-test ignores the overlap of
  training sets and is anti-conservative.
* Wilcoxon signed-rank test on the same differences (normal approximation for many folds, exact for few).
* Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR) adjustment over the
  complete, pre-specified family of comparisons.
* Effect sizes: mean difference with corrected 95 % CI; Cohen's d_z (corrected).
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from scipy import stats


def corrected_ttest(diff: np.ndarray, n_train: float, n_test: float, alpha: float = 0.05) -> Dict[str, float]:
    d = np.asarray(diff, float)
    d = d[~np.isnan(d)]
    J = len(d)
    mean = float(d.mean())
    s2 = float(d.var(ddof=1)) if J > 1 else 0.0
    var = (1.0 / J + n_test / n_train) * s2
    if var <= 0:
        return {"mean_diff": mean, "t": np.inf if mean else 0.0, "p": 0.0 if mean else 1.0,
                "ci_low": mean, "ci_high": mean, "d_z": np.nan, "J": J}
    t = mean / np.sqrt(var)
    p = 2 * stats.t.sf(abs(t), J - 1)
    h = stats.t.ppf(1 - alpha / 2, J - 1) * np.sqrt(var)
    return {"mean_diff": mean, "t": float(t), "p": float(p), "ci_low": mean - h, "ci_high": mean + h,
            "d_z": mean / np.sqrt(s2 * (1 + J * n_test / n_train)) if s2 > 0 else np.nan, "J": J}


def naive_paired_ttest(diff: np.ndarray) -> float:
    d = np.asarray(diff, float)
    return float(stats.ttest_1samp(d, 0.0).pvalue) if np.any(d != 0) else 1.0


def wilcoxon_p(diff: np.ndarray) -> float:
    d = np.asarray(diff, float)
    d = d[~np.isnan(d)]
    if np.all(d == 0):
        return 1.0
    return float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def holm(p: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p), float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def benjamini_hochberg(p: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p), float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    running = 1.0
    for rank, i in enumerate(order):
        running = min(running, p[i] * m / (m - rank))
        adj[i] = running
    return adj


def mean_ci(values: np.ndarray, n_train: float, n_test: float, alpha: float = 0.05):
    """Corrected CI for a model's mean CV metric (same variance correction)."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    J = len(v)
    m = float(v.mean())
    if J < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(1 - alpha / 2, J - 1) * np.sqrt((1.0 / J + n_test / n_train) * v.var(ddof=1))
    return m, m - h, m + h


def compare(per_fold: Dict[str, np.ndarray], pairs: List[tuple], n_train: float, n_test: float) -> List[Dict]:
    """Run every (a, b) comparison in ``pairs`` (a - b) and adjust over the whole family."""
    rows = []
    for a, b in pairs:
        diff = per_fold[a] - per_fold[b]
        r = corrected_ttest(diff, n_train, n_test)
        r.update({"model": a, "reference": b, "p_wilcoxon": wilcoxon_p(diff),
                  "p_naive_paired_t": naive_paired_ttest(diff)})
        rows.append(r)
    if rows:
        for key in ("p", "p_wilcoxon"):
            ps = [r[key] for r in rows]
            for r, h, bh in zip(rows, holm(ps), benjamini_hochberg(ps)):
                r[f"{key}_holm"] = float(h)
                r[f"{key}_bh"] = float(bh)
    return rows
