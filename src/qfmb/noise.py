"""Hardware-noise study with Qiskit Aer density-matrix simulation.

Noise levels
------------
shots   ideal gates, finite sampling only (2048 shots)
low     0.25 x the "medium" error rates
medium  1q depolarising 3e-4 (sx, x); 2q depolarising 8e-3 (cx); thermal
        relaxation T1 = 150 us, T2 = 100 us with gate times 50 / 400 ns;
        readout P(1|0) = 1.5 %, P(0|1) = 2.5 %  (typical of current
        superconducting devices)
high    3 x the "medium" error rates (T1, T2 divided by 3)

Circuits are transpiled to {rz, sx, x, cx} on a linear 6-qubit coupling map
(optimization_level 1), so full / circular entanglement incurs SWAP overhead
as it would on hardware.

Kernels are estimated with the compute-uncompute circuit U(x) U(x')^dagger,
k = P(0...0) from 2048 shots (diagonal included, i.e. not forced to 1); the
training Gram matrix is projected to the nearest PSD matrix. The VQC is
trained and evaluated on noisy circuits. Hyper-parameters (bandwidth, C) are
selected with the noiseless nested-CV procedure, as when a model tuned on a
simulator is deployed on hardware.
"""
from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, Hyperparameter, Kernel
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from . import data as data_mod
from .circuits import feature_map, to_qiskit, two_local
from .evaluation import ModelSpec, select_hyperparameters
from .kernels import fidelity_kernel, kernel_target_alignment, psd_project
from .metrics import classification_metrics
from .models.quantum import _sigmoid_gauss_integral
from .preprocessing import make_preprocessor
from .simulator import statevectors

SHOTS = 2048
LEVEL_SCALE = {"low": 0.25, "medium": 1.0, "high": 3.0}


def noise_model(level: str):
    if level == "shots":
        return None
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, thermal_relaxation_error
    s = LEVEL_SCALE[level]
    p1, p2 = 3e-4 * s, 8e-3 * s
    T1, T2 = 150e3 / s, 100e3 / s          # ns
    t1q, t2q = 50.0, 400.0
    r01, r10 = 0.015 * s, 0.025 * s
    nm = NoiseModel(basis_gates=["rz", "sx", "x", "cx"])
    e1 = depolarizing_error(p1, 1).compose(thermal_relaxation_error(T1, T2, t1q))
    tr2 = thermal_relaxation_error(T1, T2, t2q)
    e2 = depolarizing_error(p2, 2).compose(tr2.expand(tr2))
    nm.add_all_qubit_quantum_error(e1, ["sx", "x"])
    nm.add_all_qubit_quantum_error(e2, ["cx"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - r01, r01], [r10, 1 - r10]]))
    return nm


def _simulator(level: str, seed: int):
    from qiskit_aer import AerSimulator
    nm = noise_model(level)
    method = "statevector" if nm is None else "density_matrix"
    return AerSimulator(method=method, noise_model=nm, seed_simulator=seed,
                        max_parallel_threads=1, max_parallel_experiments=1)


def _transpile(qc, n):
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap
    return transpile(qc, basis_gates=["rz", "sx", "x", "cx"], coupling_map=CouplingMap.from_line(n),
                     optimization_level=1, seed_transpiler=0)


def _run_binds(sim, tqc, params, values: np.ndarray, chunk: int = 1500):
    """Run ``tqc`` for every row of ``values`` (bound to ``params``); return list of counts."""
    out = []
    for s in range(0, len(values), chunk):
        v = values[s:s + chunk]
        binds = [{p: v[:, j].tolist() for j, p in enumerate(params)}]
        res = sim.run(tqc, parameter_binds=binds, shots=SHOTS).result()
        out += [res.get_counts(i) for i in range(len(v))]
    return out


class KernelEstimator:
    def __init__(self, fmap: str, n: int, level: str, seed: int):
        fm = feature_map(fmap, n, 2)
        a, xa, _ = to_qiskit(fm, n, x_name="a")
        b, xb, _ = to_qiskit(fm, n, x_name="b")
        qc = a.compose(b.inverse())
        qc.measure_all()
        self.tqc = _transpile(qc, n)
        self.params = list(xa) + list(xb)
        self.sim = _simulator(level, seed)
        self.zero = "0" * n
        self.cx = self.tqc.count_ops().get("cx", 0)
        self.depth = self.tqc.depth()

    def pairs(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        vals = np.hstack([A, B])
        counts = _run_binds(self.sim, self.tqc, self.params, vals)
        return np.array([c.get(self.zero, 0) / SHOTS for c in counts])

    def gram(self, Z: np.ndarray) -> np.ndarray:
        n = len(Z)
        iu = np.triu_indices(n)                  # include diagonal (noisy self-overlap)
        k = self.pairs(Z[iu[0]], Z[iu[1]])
        K = np.zeros((n, n))
        K[iu] = k
        K[(iu[1], iu[0])] = k
        return K

    def cross(self, Zt: np.ndarray, Z: np.ndarray) -> np.ndarray:
        I, J = np.meshgrid(np.arange(len(Zt)), np.arange(len(Z)), indexing="ij")
        return self.pairs(Zt[I.ravel()], Z[J.ravel()]).reshape(len(Zt), len(Z))


class _IndexKernel(Kernel):
    """Kernel over integer indices into a precomputed matrix (train rows first, then test)."""

    def __init__(self, M=None):
        self.M = M

    @property
    def hyperparameters(self):
        return []

    def __call__(self, X, Y=None, eval_gradient=False):
        i = np.asarray(X).ravel().astype(int)
        j = i if Y is None else np.asarray(Y).ravel().astype(int)
        K = self.M[np.ix_(i, j)]
        return (K, np.empty((len(i), len(i), 0))) if eval_gradient else K

    def diag(self, X):
        i = np.asarray(X).ravel().astype(int)
        return self.M[i, i]

    def is_stationary(self):
        return False


def _noisy_vqc(fmap, n, Ztr, ytr, Zte, lam, level, seed, maxiter=200):
    from .models.quantum import spsa_minimize
    fm = feature_map(fmap, n, 2)
    ans, k = two_local(n, 2)
    qc, x, t = to_qiskit(fm + ans, n, n_theta=k)
    qc.measure_all()
    tqc = _transpile(qc, n)
    params = list(x) + list(t)
    sim = _simulator(level, seed)
    rng = np.random.default_rng(seed)

    def p1(Z, theta):
        vals = np.hstack([lam * Z, np.tile(theta, (len(Z), 1))])
        counts = _run_binds(sim, tqc, params, vals)
        return np.array([sum(v for key, v in c.items() if key[-1] == "1") / SHOTS for c in counts])

    def loss(theta):
        p = np.clip(p1(Ztr, theta), 1e-10, 1 - 1e-10)
        return float(np.mean(-(ytr * np.log(p) + (1 - ytr) * np.log(1 - p))))

    hist = []
    theta, nfev = spsa_minimize(loss, rng.uniform(0, 1, k), maxiter, rng,
                                callback=lambda it, th, nf: hist.append(it))
    pt = p1(Zte, theta)
    return pt, {"nfev": nfev, "cx": tqc.count_ops().get("cx", 0), "depth": tqc.depth()}


def _kernel_models(Kp, Kt, ytr, C, seed):
    """QSVM and QGPC on the same (noisy, PSD-repaired) kernel."""
    out = {}
    svc = SVC(kernel="precomputed", C=C, class_weight="balanced", probability=True,
              random_state=seed).fit(Kp, ytr)
    s = svc.decision_function(Kt)
    out["qsvm"] = ((s > 0).astype(int), s, svc.predict_proba(Kt)[:, 1])
    m, mt = Kp.shape[0], Kt.shape[0]
    M = np.zeros((m + mt, m + mt))
    M[:m, :m] = Kp
    M[m:, :m] = Kt
    M[:m, m:] = Kt.T
    M[np.arange(m, m + mt), np.arange(m, m + mt)] = 1.0
    gpc = GaussianProcessClassifier(ConstantKernel(1.0, (1e-2, 1e4)) * _IndexKernel(M),
                                    random_state=seed).fit(np.arange(m)[:, None], ytr)
    be = gpc.base_estimator_
    Xt = np.arange(m, m + mt)[:, None]
    Ks = be.kernel_(be.X_train_, Xt)
    mu = Ks.T @ (be.y_train_ - be.pi_)
    v = np.linalg.solve(be.L_, be.W_sr_[:, None] * Ks)
    var = np.maximum(be.kernel_.diag(Xt) - np.einsum("ij,ij->j", v, v), 1e-12)
    p = _sigmoid_gauss_integral(mu, var)
    out["qgpc"] = ((p > 0.5).astype(int), mu, p)
    return out


def run_noise_task(task: Dict) -> Tuple[Dict, Dict]:
    """family 'kernel': one noisy kernel -> QSVM and QGPC (bandwidth and C from the
    QSVM noiseless nested CV); family 'vqc': noisy training and inference."""
    ds = data_mod.load(task["dataset"])
    fam, fmap, level, f = task["family"], task["feature_map"], task["level"], task["fold"]
    skf = StratifiedKFold(task.get("n_splits", 5), shuffle=True, random_state=1000)
    tr, te = list(skf.split(ds.X, ds.y))[f]
    seed = 7000 + f
    if task.get("max_train") and len(tr) > task["max_train"]:
        rng = np.random.default_rng(seed)
        keep = np.concatenate([rng.choice(np.flatnonzero(ds.y[tr] == c),
                                          int(round(task["max_train"] * np.mean(ds.y[tr] == c))), replace=False)
                               for c in (0, 1)])
        tr = np.sort(tr[keep])
    t0 = time.perf_counter()
    sel_family = "vqc" if fam == "vqc" else "qsvm"
    ms = ModelSpec({"family": sel_family, "params": {"feature_map": fmap}, "tune": "default"}, len(tr))
    hp, _ = select_hyperparameters(ms, ds, {"mode": "pca", "k": 6}, tr, 2000 + f)
    pre = make_preprocessor(ds.numeric, ds.categorical, "pca", 6, seed)
    Ztr, Zte = pre.fit_transform(ds.X.iloc[tr]), pre.transform(ds.X.iloc[te])
    ytr, yte = ds.y[tr], ds.y[te]
    n = Ztr.shape[1]
    lam = hp["bandwidth"]
    base = {"repeat": 0, "fold": f, "n_train": int(len(tr)), "n_test": int(len(te)), "hyperparameters": hp}
    folds, arrays = {}, {"index": te, "y": yte}
    if fam == "vqc":
        proba, info = _noisy_vqc(fmap, n, Ztr, ytr, Zte, lam, level, seed)
        res = {"vqc": ((proba >= 0.5).astype(int), proba, proba)}
        base.update(info)
    else:
        est = KernelEstimator(fmap, n, level, seed)
        K = est.gram(lam * Ztr)
        Kt = est.cross(lam * Zte, lam * Ztr)
        K_exact = fidelity_kernel(statevectors(feature_map(fmap, n, 2), n, lam * Ztr))
        Kp, psd = psd_project(K)
        base.update({"cx": est.cx, "depth": est.depth, "psd_info": psd,
                     "kernel_mae_vs_exact": float(np.mean(np.abs(K - K_exact))),
                     "mean_diag": float(np.mean(np.diag(K))),
                     "kta_noisy": kernel_target_alignment(Kp, ytr),
                     "kta_exact": kernel_target_alignment(K_exact, ytr)})
        res = _kernel_models(Kp, Kt, ytr, hp["C"], seed)
    for name, (y_pred, score, proba) in res.items():
        folds[name] = {**base, "metrics": classification_metrics(yte, y_pred, score, proba),
                       "seconds": time.perf_counter() - t0}
        arrays[f"{name}_y_pred"] = y_pred
        arrays[f"{name}_proba"] = np.asarray(proba, float)
    return {"task": task, "models": folds}, arrays
