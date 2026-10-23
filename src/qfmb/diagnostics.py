"""Quantitative kernel diagnostics that explain differences between feature maps.

For every feature map and bandwidth, on a training split of each cohort:
  * centered kernel-target alignment (KTA)
  * eigen-spectrum: participation-ratio / entropy effective dimension, numerical
    rank, condition number, top-k mass
  * concentration of off-diagonal kernel values (mean, variance)
  * geometric difference g(K_RBF || K_Q) (Huang et al., Nat. Commun. 2021)
  * data-driven expressibility: KL(P_data(F) || P_Haar(F)) (Sim et al., Adv. Quantum Technol. 2019)
  * Meyer-Wallach entangling capability averaged over encoded data states
  * pairwise kernel alignment between maps (e.g. RX vs Pauli-Z)
  * trailing data-independent gate invariance of the fidelity kernel
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedShuffleSplit

from . import data as data_mod
from .circuits import FEATURE_MAPS, feature_map
from .evaluation import BANDWIDTHS
from .kernels import (concentration_stats, fidelity_kernel, geometric_difference, kernel_alignment,
                      kernel_target_alignment, spectrum_stats)
from .preprocessing import make_preprocessor
from .simulator import statevectors


def expressibility_kl(K: np.ndarray, n_qubits: int, bins: int = 75) -> float:
    F = K[np.triu_indices_from(K, k=1)]
    edges = np.linspace(0, 1, bins + 1)
    p, _ = np.histogram(F, edges)
    p = p / p.sum()
    N = 2 ** n_qubits
    # Haar bin mass (1-a)^(N-1) - (1-b)^(N-1), in log space: 1 - (1-F)^(N-1) rounds to 1 above F ~ 0.45
    with np.errstate(divide="ignore"):
        la, lb = (N - 1) * np.log1p(-edges[:-1]), (N - 1) * np.log1p(-edges[1:])
    log_q = la + np.log1p(-np.exp(lb - la))
    m = p > 0
    return float(np.sum(p[m] * (np.log(p[m]) - log_q[m])))


def meyer_wallach(psi: np.ndarray, n: int) -> float:
    """Q = 2 (1 - mean_k Tr rho_k^2), averaged over states."""
    B = psi.shape[0]
    T = psi.reshape((B,) + (2,) * n)                # axis 1 + j  <->  qubit n-1-j
    pur = np.zeros(B)
    for q in range(n):
        ax = 1 + (n - 1 - q)
        M = np.moveaxis(T, ax, 1).reshape(B, 2, -1)
        rho = np.einsum("bik,bjk->bij", M, M.conj())
        pur += np.real(np.einsum("bij,bji->b", rho, rho))
    return float(np.mean(2 * (1 - pur / n)))


def strip_trailing_fixed(ops):
    k = len(ops)
    while k and ops[k - 1].angle is None:
        k -= 1
    return ops[:k], len(ops) - k


def run_diagnostics(task: Dict) -> tuple:
    ds = data_mod.load(task["dataset"])
    r = task.get("repeat", 0)
    n_max = task.get("n_max", 1500)
    sss = StratifiedShuffleSplit(1, train_size=min(0.8, n_max / len(ds.y)), random_state=500 + r)
    tr, _ = next(sss.split(ds.X, ds.y))
    pre = make_preprocessor(ds.numeric, ds.categorical, "pca", 6, seed=r)
    Z = pre.fit_transform(ds.X.iloc[tr])
    y = ds.y[tr]
    n = Z.shape[1]
    rows: List[Dict] = []
    spectra = {}
    K_rbf = rbf_kernel(Z, gamma=1.0 / n)
    base = {"model": "RBF (gamma=1/d)", "bandwidth": None, "kta": kernel_target_alignment(K_rbf, y),
            **spectrum_stats(K_rbf), **concentration_stats(K_rbf)}
    rows.append(base)
    kernels = {}
    for lam in BANDWIDTHS:
        for name in FEATURE_MAPS:
            ops = feature_map(name, n, 2)
            psi = statevectors(ops, n, lam * Z)
            K = fidelity_kernel(psi)
            kernels[(name, lam)] = K
            stripped, n_trailing = strip_trailing_fixed(ops)
            K2 = fidelity_kernel(statevectors(stripped, n, lam * Z))
            row = {"model": name, "bandwidth": lam, "kta": kernel_target_alignment(K, y),
                   **spectrum_stats(K), **concentration_stats(K),
                   "geometric_difference_vs_rbf": geometric_difference(K_rbf, K),
                   "expressibility_kl": expressibility_kl(K, n),
                   "meyer_wallach": meyer_wallach(psi[:400], n),
                   "trailing_fixed_gates": n_trailing,
                   "max_abs_kernel_change_without_trailing": float(np.max(np.abs(K - K2)))}
            rows.append(row)
            w = np.clip(np.linalg.eigvalsh(K)[::-1], 0, None)
            spectra[f"{name}|{lam}"] = (w / w.sum())[:64]
    pairs = []
    names = list(FEATURE_MAPS)
    for lam in BANDWIDTHS:
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pairs.append({"bandwidth": lam, "map_a": a, "map_b": b,
                              "alignment": kernel_alignment(kernels[(a, lam)], kernels[(b, lam)]),
                              "max_abs_diff": float(np.max(np.abs(kernels[(a, lam)] - kernels[(b, lam)])))})
    summary = {"task": task, "n_train": int(len(tr)), "rows": rows, "pairs": pairs,
               "retained_variance": float(np.sum(pre.named_steps["pca"].explained_variance_ratio_))}
    arrays = {k.replace("|", "__"): v for k, v in spectra.items()}
    return summary, arrays
