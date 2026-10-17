"""Gradient-variance scaling of the VQC (barren-plateau check).

For each feature map, number of qubits n and ansatz depth L, draw random
parameters theta ~ U[0, 2 pi) and data x ~ N(0, I) (whitened PCA scale, angle
0.5 x) and compute exact parameter-shift derivatives of
  * the local observable  <Z_0>          (the VQC read-out qubit), and
  * the global observable  P(|0...0>)    (fidelity-type cost),
with respect to the first and the final-layer Ry parameter on qubit 0. Barren plateaus
appear as Var[d<O>/d theta] decaying exponentially in n.
"""
from __future__ import annotations

import zlib
from typing import Dict

import numpy as np

from .circuits import feature_map, two_local
from .simulator import expval_z, statevectors


def _observables(ops, n, x, theta):
    psi = statevectors(ops, n, x[None, :], theta)
    return expval_z(psi, n, 0)[0], float(np.abs(psi[0, 0]) ** 2)


def run_gradient_variance(task: Dict) -> tuple:
    m, n = task["feature_map"], task["n_qubits"]
    rng = np.random.default_rng(zlib.crc32(f"{m}|{n}".encode()))
    rows = []
    for L in task["layers"]:
        ans, k = two_local(n, L)
        ops = feature_map(m, n, 2) + ans
        # theta_0 = first Ry on qubit 0; 2nL = Ry on qubit 0 in the final rotation layer
        # (the final Rz gates commute with Z_0 / the |0..0> projector and have zero gradient)
        idx = {"first": 0, "last": 2 * n * L}
        g = {(o, w): [] for o in ("local", "global") for w in idx}
        for _ in range(task.get("n_samples", 200)):
            theta = rng.uniform(0, 2 * np.pi, k)
            x = 0.5 * rng.normal(size=n)
            for w, j in idx.items():
                e = np.zeros(k)
                e[j] = np.pi / 2
                lp, gp = _observables(ops, n, x, theta + e)
                lm, gm = _observables(ops, n, x, theta - e)
                g[("local", w)].append(0.5 * (lp - lm))
                g[("global", w)].append(0.5 * (gp - gm))
        row = {"feature_map": m, "n_qubits": n, "layers": L, "n_params": k}
        for (o, w), v in g.items():
            row[f"var_{o}_{w}"] = float(np.var(v))
            row[f"meanabs_{o}_{w}"] = float(np.mean(np.abs(v)))
        rows.append(row)
    return {"task": task, "rows": rows}, None
