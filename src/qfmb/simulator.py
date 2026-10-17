"""Batched, exact statevector simulator for the gate lists in ``circuits.py``.

States are arrays of shape (B, 2**n) in Qiskit's little-endian convention
(basis index = sum_q b_q 2**q). Every gate is applied to all B samples at
once, with per-sample (data-dependent) or shared (trainable) angles. Results
match ``qiskit.quantum_info.Statevector`` to machine precision
(``tests/test_circuits.py``).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional, Sequence, Tuple

import numpy as np

from .circuits import Op

_SQ2 = 1.0 / np.sqrt(2.0)


@lru_cache(maxsize=None)
def _pairs(n: int, target: int, control: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(2 ** n)
    mask = ((idx >> target) & 1) == 0
    if control >= 0:
        mask &= ((idx >> control) & 1) == 1
    i0 = idx[mask]
    return i0, i0 | (1 << target)


@lru_cache(maxsize=None)
def _cz_sign(n: int, a: int, b: int) -> np.ndarray:
    idx = np.arange(2 ** n)
    return np.where(((idx >> a) & 1) & ((idx >> b) & 1), -1.0, 1.0)


def _angle(spec, X: Optional[np.ndarray], theta: Optional[np.ndarray]):
    kind = spec[0]
    if kind == "x":
        return X[:, spec[1]]
    if kind == "2x":
        return 2.0 * X[:, spec[1]]
    if kind == "zz":
        return 2.0 * (np.pi - X[:, spec[1]]) * (np.pi - X[:, spec[2]])
    if kind == "t":
        return theta[spec[1]]
    raise ValueError(spec)


def _col(a):
    """Shape an angle-derived quantity for broadcasting against (B, m) slices."""
    a = np.asarray(a)
    return a[:, None] if a.ndim == 1 else a


def _apply_1q(psi, n, q, u00, u01, u10, u11, control=-1):
    i0, i1 = _pairs(n, q, control)
    s0 = psi[:, i0]
    s1 = psi[:, i1]
    psi[:, i0] = _col(u00) * s0 + _col(u01) * s1
    psi[:, i1] = _col(u10) * s0 + _col(u11) * s1


def apply_ops(psi: np.ndarray, ops: Sequence[Op], n: int, X: Optional[np.ndarray] = None,
              theta: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply ``ops`` in place to ``psi`` (B, 2**n) and return it."""
    for op in ops:
        name = op.name
        if name == "h":
            q = op.qubits[0]
            i0, i1 = _pairs(n, q)
            s0, s1 = psi[:, i0], psi[:, i1]
            psi[:, i0], psi[:, i1] = _SQ2 * (s0 + s1), _SQ2 * (s0 - s1)
        elif name == "x":
            i0, i1 = _pairs(n, op.qubits[0])
            psi[:, i0], psi[:, i1] = psi[:, i1], psi[:, i0].copy()
        elif name == "cx":
            c, t = op.qubits
            i0, i1 = _pairs(n, t, c)
            tmp = psi[:, i0].copy()
            psi[:, i0] = psi[:, i1]
            psi[:, i1] = tmp
        elif name == "cz":
            psi *= _cz_sign(n, *op.qubits)
        else:
            a = _angle(op.angle, X, theta)
            if name == "rz":
                i0, i1 = _pairs(n, op.qubits[0])
                psi[:, i0] *= _col(np.exp(-0.5j * a))
                psi[:, i1] *= _col(np.exp(0.5j * a))
            elif name == "p":
                _, i1 = _pairs(n, op.qubits[0])
                psi[:, i1] *= _col(np.exp(1j * a))
            elif name in ("rx", "crx"):
                c, s = np.cos(0.5 * a), np.sin(0.5 * a)
                q = op.qubits[-1]
                ctrl = op.qubits[0] if name == "crx" else -1
                _apply_1q(psi, n, q, c, -1j * s, -1j * s, c, control=ctrl)
            elif name == "ry":
                c, s = np.cos(0.5 * a), np.sin(0.5 * a)
                _apply_1q(psi, n, op.qubits[0], c, -s, s, c)
            else:
                raise ValueError(f"unsupported gate {name}")
    return psi


def statevectors(ops: Sequence[Op], n: int, X: Optional[np.ndarray] = None,
                 theta: Optional[np.ndarray] = None, batch: Optional[int] = None,
                 chunk: int = 4096) -> np.ndarray:
    """Return |psi(x)> = U(x, theta)|0...0> for every row of X: shape (B, 2**n)."""
    B = X.shape[0] if X is not None else (batch or 1)
    out = np.empty((B, 2 ** n), dtype=np.complex128)
    for s in range(0, B, chunk):
        e = min(B, s + chunk)
        psi = np.zeros((e - s, 2 ** n), dtype=np.complex128)
        psi[:, 0] = 1.0
        out[s:e] = apply_ops(psi, ops, n, None if X is None else X[s:e], theta)
    return out


def unitary_rows(ops: Sequence[Op], n: int, theta: Optional[np.ndarray] = None) -> np.ndarray:
    """R = U^T for a data-independent circuit U, so that states (rows) evolve as S @ R."""
    psi = np.eye(2 ** n, dtype=np.complex128)
    return apply_ops(psi, ops, n, None, theta)


def prob_qubit_one(psi: np.ndarray, n: int, qubit: int = 0) -> np.ndarray:
    """P(measure 1 on ``qubit``) per sample = VQC class-1 probability (parity of qubit 0)."""
    _, i1 = _pairs(n, qubit)
    return np.sum(np.abs(psi[:, i1]) ** 2, axis=1)


def expval_z(psi: np.ndarray, n: int, qubit: int = 0) -> np.ndarray:
    return 1.0 - 2.0 * prob_qubit_one(psi, n, qubit)
