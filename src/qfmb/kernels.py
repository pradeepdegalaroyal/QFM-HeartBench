"""Fidelity quantum kernels: exact, finite-shot, PSD repair, explicit feature map.

For pure encoded states |psi(x)> the fidelity kernel is
    k(x, x') = |<psi(x)|psi(x')>|^2 = Tr[rho(x) rho(x')],
the Hilbert-Schmidt inner product of the density matrices, hence PSD.
For n qubits it is therefore *exactly* a linear kernel on the real 4**n-dim
vector vec(rho(x)) (``explicit_features``), which lets the full 68k-record CVD
cohort be handled in primal / weight space without approximation.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def fidelity_kernel(psi_a: np.ndarray, psi_b: Optional[np.ndarray] = None) -> np.ndarray:
    psi_b = psi_a if psi_b is None else psi_b
    ov = psi_a.conj() @ psi_b.T
    K = ov.real ** 2 + ov.imag ** 2
    if psi_b is psi_a:
        K = 0.5 * (K + K.T)
        np.fill_diagonal(K, 1.0)
    return K


def shot_noise(K: np.ndarray, shots: int, rng: np.random.Generator, symmetric: bool) -> np.ndarray:
    """Compute-uncompute estimator: k_hat = Binomial(shots, k) / shots per entry."""
    Kc = np.clip(K, 0.0, 1.0)
    if not symmetric:
        return rng.binomial(shots, Kc) / shots
    iu = np.triu_indices_from(Kc, k=1)
    out = np.ones_like(Kc)  # <psi(x)|psi(x)> = 1 -> all-zero outcome every shot
    vals = rng.binomial(shots, Kc[iu]) / shots
    out[iu] = vals
    out[(iu[1], iu[0])] = vals
    return out


def psd_project(K: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Nearest PSD matrix in Frobenius norm (clip negative eigenvalues to 0)."""
    w, V = np.linalg.eigh(0.5 * (K + K.T))
    neg = w < 0
    info = {"min_eig": float(w.min()), "n_neg_eig": int(neg.sum()),
            "neg_mass": float(-w[neg].sum() / np.abs(w).sum()) if neg.any() else 0.0}
    if neg.any():
        K = (V * np.clip(w, 0, None)) @ V.T
    return K, info


def explicit_features(psi: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Real phi(x) with phi(x).phi(x') = |<psi(x)|psi(x')>|^2 exactly (dim 4**n)."""
    B, d = psi.shape
    iu = np.triu_indices(d, k=1)
    out = np.empty((B, d * d), dtype=dtype)
    out[:, :d] = np.abs(psi) ** 2
    m = len(iu[0])
    for s in range(0, B, 2048):
        e = min(B, s + 2048)
        rho_off = psi[s:e, iu[0]] * psi[s:e, iu[1]].conj()
        out[s:e, d:d + m] = np.sqrt(2.0) * rho_off.real
        out[s:e, d + m:] = np.sqrt(2.0) * rho_off.imag
    return out


# ------------------------------- diagnostics ------------------------------- #
def centered(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - 1.0 / n
    return H @ K @ H


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Centered kernel-target alignment (Cortes, Mohri & Rostamizadeh, JMLR 2012)."""
    t = np.where(y == 1, 1.0, -1.0)
    Kc = centered(K)
    Y = np.outer(t, t)
    Yc = centered(Y)
    return float(np.sum(Kc * Yc) / (np.linalg.norm(Kc) * np.linalg.norm(Yc) + 1e-300))


def spectrum_stats(K: np.ndarray) -> Dict[str, float]:
    w = np.clip(np.linalg.eigvalsh(K)[::-1], 0, None)
    p = w / w.sum()
    nz = w[w > 1e-12 * w[0]]
    return {
        "eff_dim_participation": float(w.sum() ** 2 / np.sum(w ** 2)),
        "spectral_entropy_dim": float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))),
        "top1_frac": float(p[0]), "top5_frac": float(p[:5].sum()),
        "numerical_rank": int(len(nz)),
        "condition_number": float(nz[0] / nz[-1]),
        "condition_number_reg1e-3": float((w[0] + 1e-3) / (w[-1] + 1e-3)),
    }


def concentration_stats(K: np.ndarray) -> Dict[str, float]:
    off = K[~np.eye(K.shape[0], dtype=bool)]
    return {"offdiag_mean": float(off.mean()), "offdiag_var": float(off.var()),
            "offdiag_median": float(np.median(off))}


def geometric_difference(K_c: np.ndarray, K_q: np.ndarray, reg: float = 1e-3) -> float:
    """g(K_c || K_q) = sqrt(|| sqrt(K_q) (K_c + reg I)^-1 sqrt(K_q) ||_inf) (Huang et al., Nat Commun 2021).
    Both kernels are trace-normalised to Tr K = N first."""
    n = K_c.shape[0]
    Kc = K_c * n / np.trace(K_c)
    Kq = K_q * n / np.trace(K_q)
    w, V = np.linalg.eigh(Kq)
    sq = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    M = sq @ np.linalg.solve(Kc + reg * np.eye(n), sq)
    return float(np.sqrt(np.max(np.linalg.eigvalsh(0.5 * (M + M.T)))))


def kernel_alignment(K1: np.ndarray, K2: np.ndarray) -> float:
    """Centered alignment between two kernels (similarity of the induced geometries)."""
    A, B = centered(K1), centered(K2)
    return float(np.sum(A * B) / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-300))
