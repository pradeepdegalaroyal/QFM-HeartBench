"""Quantum classifiers as scikit-learn estimators.

QSVM  : SVC on the fidelity kernel (dual), or an exactly equivalent primal
        linear SVM on vec(rho(x)) when the training set is large.
QGPC  : Gaussian-process classification with a logistic likelihood and Laplace
        approximation (Rasmussen & Williams 2006, Alg. 3.1-3.2) whose covariance
        is s^2 * fidelity kernel; s^2 is set by type-II maximum likelihood. For
        large N the identical model is fitted in weight space.
VQC   : feature map followed by the TwoLocal(ry, rz; cz linear) ansatz. P(y=1|x)
        = P(qubit 0 measures 1), as in qiskit-machine-learning's VQC, trained by
        minimising (optionally class-weighted) mean cross-entropy with SPSA
        (Qiskit defaults incl. learning-rate calibration), COBYLA or Adam with
        exact parameter-shift gradients.

Input X is the preprocessed, unit-variance feature matrix; rotation angles are
``bandwidth * X``.
"""
from __future__ import annotations

import hashlib
import time
from typing import Dict, List, Optional

import numpy as np
from scipy.special import erf, expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, Hyperparameter, Kernel
from sklearn.svm import SVC, LinearSVC

from ..circuits import feature_map, two_local
from ..kernels import explicit_features, fidelity_kernel, psd_project, shot_noise
from ..simulator import apply_ops, statevectors, unitary_rows

# Probit-mixture approximation of the logistic-Gaussian integral used by sklearn.
_LAMBDAS = np.array([0.41, 0.4, 0.37, 0.44, 0.39])[:, None]
_COEFS = np.array([-1854.8214151, 3516.89893646, 221.29346712, 128.12323805, -2010.49422654])[:, None]


def _sigmoid_gauss_integral(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    var = np.maximum(var, 1e-12)
    alpha = 1.0 / (2.0 * var)
    gamma = _LAMBDAS * mu
    integrals = (np.sqrt(np.pi / alpha) * erf(gamma * np.sqrt(alpha / (alpha + _LAMBDAS ** 2)))
                 / (2.0 * np.sqrt(var * 2.0 * np.pi)))
    return np.clip((_COEFS * integrals).sum(axis=0) + 0.5 * _COEFS.sum(), 0.0, 1.0)


def _seed_from(*arrays, base: int = 0) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(base).encode())
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return int.from_bytes(h.digest(), "little") % (2 ** 32)


class _QuantumMixin:
    def _ops(self, n):
        return feature_map(self.feature_map, n, self.n_layers, self.entanglement)

    def _states(self, X):
        X = np.asarray(X, dtype=float)
        return statevectors(self._ops(X.shape[1]), X.shape[1], self.bandwidth * X)


# --------------------------------------------------------------------------- #
class QSVM(_QuantumMixin, ClassifierMixin, BaseEstimator):
    def __init__(self, feature_map="RX", n_layers=2, entanglement=None, bandwidth=1.0, C=1.0,
                 class_weight="balanced", shots=None, probability=True, form="auto",
                 primal_threshold=12000, seed=0):
        self.feature_map = feature_map
        self.n_layers = n_layers
        self.entanglement = entanglement
        self.bandwidth = bandwidth
        self.C = C
        self.class_weight = class_weight
        self.shots = shots
        self.probability = probability
        self.form = form
        self.primal_threshold = primal_threshold
        self.seed = seed

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        psi = self._states(X)
        self.form_ = self.form if self.form != "auto" else (
            "primal" if len(y) > self.primal_threshold else "kernel")
        self.psd_info_ = {}
        if self.form_ == "kernel":
            K = fidelity_kernel(psi)
            if self.shots:
                K = shot_noise(K, self.shots, np.random.default_rng(_seed_from(X, base=self.seed)), True)
                K, self.psd_info_ = psd_project(K)
            self.svc_ = SVC(kernel="precomputed", C=self.C, class_weight=self.class_weight,
                            probability=self.probability, random_state=self.seed).fit(K, y)
            self.psi_train_ = psi
            self.X_train_ = np.asarray(X, float)
        else:
            if self.shots:
                raise ValueError("finite shots are only simulated in kernel form")
            self.svc_ = LinearSVC(C=self.C, loss="hinge", class_weight=self.class_weight, dual=True,
                                  intercept_scaling=10.0, max_iter=50000, tol=1e-4,
                                  random_state=self.seed).fit(explicit_features(psi), y)
        return self

    def _test_kernel(self, X):
        psi = self._states(X)
        K = fidelity_kernel(psi, self.psi_train_)
        if self.shots:
            K = shot_noise(K, self.shots, np.random.default_rng(_seed_from(X, base=self.seed + 1)), False)
        return K

    def decision_function(self, X):
        if self.form_ == "kernel":
            return self.svc_.decision_function(self._test_kernel(X))
        return self.svc_.decision_function(explicit_features(self._states(X)))

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

    def predict_proba(self, X):
        if self.form_ != "kernel" or not self.probability:
            raise AttributeError("probabilities unavailable")
        return self.svc_.predict_proba(self._test_kernel(X))


# --------------------------------------------------------------------------- #
class FidelityKernel(Kernel):
    """sklearn ``Kernel`` wrapper (no hyperparameters) around the fidelity kernel."""

    def __init__(self, feature_map="RX", n_layers=2, entanglement=None, bandwidth=1.0, shots=None, seed=0):
        self.feature_map = feature_map
        self.n_layers = n_layers
        self.entanglement = entanglement
        self.bandwidth = bandwidth
        self.shots = shots
        self.seed = seed

    @property
    def hyperparameters(self) -> List[Hyperparameter]:
        return []

    def _psi(self, X):
        key = (X.shape, hashlib.blake2b(np.ascontiguousarray(X).tobytes(), digest_size=16).hexdigest())
        cache = self.__dict__.setdefault("_cache", {})
        if key not in cache:
            if len(cache) > 4:
                cache.clear()
            ops = feature_map(self.feature_map, X.shape[1], self.n_layers, self.entanglement)
            cache[key] = statevectors(ops, X.shape[1], self.bandwidth * X)
        return cache[key]

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X, float)
        if Y is None:
            K = fidelity_kernel(self._psi(X))
            if self.shots:
                K = shot_noise(K, self.shots, np.random.default_rng(_seed_from(X, base=self.seed)), True)
                K, info = psd_project(K)
                self.__dict__["psd_info_"] = info
            return (K, np.empty((K.shape[0], K.shape[0], 0))) if eval_gradient else K
        if eval_gradient:
            raise ValueError("gradient only for Y=None")
        Y = np.asarray(Y, float)
        K = fidelity_kernel(self._psi(X), self._psi(Y))
        if self.shots:
            K = shot_noise(K, self.shots, np.random.default_rng(_seed_from(X, Y, base=self.seed + 1)), False)
        return K

    def diag(self, X):
        return np.ones(np.asarray(X).shape[0])

    def is_stationary(self):
        return False


class QGPC(_QuantumMixin, ClassifierMixin, BaseEstimator):
    def __init__(self, feature_map="RX", n_layers=2, entanglement=None, bandwidth=1.0, shots=None,
                 amplitude=None, threshold="half", form="auto", weight_threshold=8000, seed=0):
        self.feature_map = feature_map
        self.n_layers = n_layers
        self.entanglement = entanglement
        self.bandwidth = bandwidth
        self.shots = shots
        self.amplitude = amplitude          # None -> type-II ML; float -> fixed s^2
        self.threshold = threshold          # "half" (0.5) or "prior" (training prevalence)
        self.form = form
        self.weight_threshold = weight_threshold
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        self.prior_ = float(y.mean())
        self.form_ = self.form if self.form != "auto" else (
            "weight" if len(y) > self.weight_threshold else "kernel")
        qk = FidelityKernel(self.feature_map, self.n_layers, self.entanglement, self.bandwidth,
                            self.shots, self.seed)
        if self.form_ == "kernel":
            if self.amplitude is None:
                kern = ConstantKernel(1.0, (1e-2, 1e4)) * qk
                opt = "fmin_l_bfgs_b"
            else:
                kern = ConstantKernel(self.amplitude, "fixed") * qk
                opt = None
            self.gpc_ = GaussianProcessClassifier(kernel=kern, optimizer=opt, random_state=self.seed,
                                                  max_iter_predict=200).fit(X, y)
            self.amplitude_ = float(self.gpc_.kernel_.k1.constant_value)
            self.lml_ = float(self.gpc_.log_marginal_likelihood_value_)
            self.psd_info_ = getattr(self.gpc_.kernel_.k2, "psd_info_", {})
        else:
            if self.amplitude is None:
                raise ValueError("weight-space form needs a fixed amplitude (tuned on a subsample)")
            self.amplitude_ = float(self.amplitude)
            self._fit_weight(explicit_features(self._states(X), dtype=np.float64), y)
        return self

    # ---- exact weight-space Laplace GPC for the linear kernel s^2 phi.phi' ----
    def _fit_weight(self, Phi, y, max_iter=100, tol=1e-9):
        s2 = self.amplitude_
        n, D = Phi.shape
        w = np.zeros(D)
        obj_old = -np.inf
        for _ in range(max_iter):
            f = Phi @ w
            pi = expit(f)
            W = pi * (1 - pi)
            A = (Phi.T * W) @ Phi
            A[np.diag_indices(D)] += 1.0 / s2
            g = Phi.T @ (y - pi) - w / s2
            L = np.linalg.cholesky(A)
            w = w + np.linalg.solve(L.T, np.linalg.solve(L, g))
            f = Phi @ w
            obj = float(np.sum(y * f - np.logaddexp(0, f)) - 0.5 * w @ w / s2)
            if obj - obj_old < tol * n:
                break
            obj_old = obj
        pi = expit(Phi @ w)
        W = pi * (1 - pi)
        A = (Phi.T * W) @ Phi
        A[np.diag_indices(D)] += 1.0 / s2
        self.L_w_ = np.linalg.cholesky(A)
        self.w_ = w
        # Laplace evidence: log p(y|f) - w'w/(2 s2) - 1/2 log|I + s2 Phi'W Phi|
        logdet = 2.0 * np.sum(np.log(np.diag(self.L_w_))) + D * np.log(s2)
        self.lml_ = float(obj - 0.5 * logdet)

    def latent(self, X):
        """Posterior mean and variance of the latent function f(x)."""
        X = np.asarray(X, float)
        if self.form_ == "kernel":
            be = self.gpc_.base_estimator_
            K_star = be.kernel_(be.X_train_, X)
            mu = K_star.T @ (be.y_train_ - be.pi_)
            v = np.linalg.solve(be.L_, be.W_sr_[:, None] * K_star)
            var = be.kernel_.diag(X) - np.einsum("ij,ij->j", v, v)
            return mu, np.maximum(var, 1e-12)
        Phi = explicit_features(self._states(X), dtype=np.float64)
        mu = Phi @ self.w_
        V = np.linalg.solve(self.L_w_, Phi.T)
        return mu, np.maximum(np.einsum("ij,ij->j", V, V), 1e-12)

    def predict_proba(self, X):
        mu, var = self.latent(X)
        p = _sigmoid_gauss_integral(mu, var)
        return np.column_stack([1 - p, p])

    def decision_function(self, X):
        return self.latent(X)[0]

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        thr = self.prior_ if self.threshold == "prior" else 0.5
        return (p > thr).astype(int)


# --------------------------------------------------------------------------- #
def spsa_minimize(fun, x0, maxiter, rng, c=0.2, alpha=0.602, gamma=0.101, A=0.0,
                  target_magnitude=2 * np.pi / 10, calibration_steps=50, callback=None):
    """SPSA (Spall 1998) with Qiskit's default calibration of the learning rate."""
    x = np.array(x0, float)
    dim = len(x)
    mags = 0.0
    for _ in range(calibration_steps):
        d = rng.choice([-1.0, 1.0], size=dim)
        mags += abs((fun(x + c * d) - fun(x - c * d)) / (2 * c))
    mags /= calibration_steps
    a = target_magnitude / mags if mags > 1e-12 else target_magnitude
    nfev = 2 * calibration_steps
    for k in range(1, maxiter + 1):
        ak = a / (k + A) ** alpha
        ck = c / k ** gamma
        d = rng.choice([-1.0, 1.0], size=dim)
        g = (fun(x + ck * d) - fun(x - ck * d)) / (2 * ck) * d
        x = x - ak * g
        nfev += 2
        if callback is not None:
            callback(k, x, nfev)
    return x, nfev


class VQC(_QuantumMixin, ClassifierMixin, BaseEstimator):
    def __init__(self, feature_map="RX", n_layers=2, entanglement=None, ansatz_reps=None, bandwidth=1.0,
                 optimizer="spsa", maxiter=200, class_weight=None, shots=None, init="qiskit",
                 learning_rate=0.05, record_every=0, seed=0):
        self.feature_map = feature_map
        self.n_layers = n_layers
        self.entanglement = entanglement
        self.ansatz_reps = ansatz_reps
        self.bandwidth = bandwidth
        self.optimizer = optimizer
        self.maxiter = maxiter
        self.class_weight = class_weight
        self.shots = shots
        self.init = init
        self.learning_rate = learning_rate
        self.record_every = record_every
        self.seed = seed

    def _ansatz(self, n):
        return two_local(n, self.ansatz_reps if self.ansatz_reps is not None else self.n_layers)

    def _encode(self, X):
        """Encoded states psi(x) (fixed during training) and the odd-parity column mask."""
        n = X.shape[1]
        psi = statevectors(self._ops(n), n, self.bandwidth * X)
        return psi, (np.arange(2 ** n) & 1) == 1

    def _p1_from_states(self, psi, odd, theta, n, rng=None, exact=False):
        if n <= 8:
            # |phi> = U_ansatz(theta)|psi(x)>;  rows evolve as psi @ U^T
            R = unitary_rows(self._ansatz(n)[0], n, theta)
            amp = psi @ R[:, odd]
        else:
            amp = apply_ops(psi.copy(), self._ansatz(n)[0], n, None, theta)[:, odd]
        p = amp.real ** 2 + amp.imag ** 2
        p = p.sum(axis=1)
        if self.shots and not exact:
            p = rng.binomial(self.shots, np.clip(p, 0, 1)) / self.shots
        return p

    def _p1(self, X, theta, rng=None, exact=False):
        psi, odd = self._encode(X)
        return self._p1_from_states(psi, odd, theta, X.shape[1], rng, exact)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        n = X.shape[1]
        _, k = self._ansatz(n)
        rng = np.random.default_rng(self.seed)
        if self.class_weight == "balanced":
            cw = len(y) / (2.0 * np.bincount(y, minlength=2))
            sw = cw[y]
        else:
            sw = np.ones(len(y))
        psi, odd = self._encode(X)
        self._train_states = (psi, odd)

        def loss(theta, exact=False):
            p = np.clip(self._p1_from_states(psi, odd, theta, n, rng, exact), 1e-10, 1 - 1e-10)
            return float(np.mean(sw * -(y * np.log(p) + (1 - y) * np.log(1 - p))))

        x0 = rng.uniform(0, 1, k) if self.init == "qiskit" else rng.uniform(-np.pi, np.pi, k)
        hist: Dict[str, list] = {"iter": [], "loss": [], "nfev": []}

        def record(it, theta, nfev):
            if self.record_every and (it % self.record_every == 0 or it == 1):
                hist["iter"].append(it)
                hist["loss"].append(loss(theta, exact=True))
                hist["nfev"].append(nfev)

        record(0, x0, 0) if self.record_every else None
        t0 = time.perf_counter()
        if self.optimizer == "spsa":
            theta, nfev = spsa_minimize(loss, x0, self.maxiter, rng, callback=record)
        elif self.optimizer == "cobyla":
            from scipy.optimize import minimize
            cnt = {"n": 0}

            def f(t):
                cnt["n"] += 1
                if self.record_every and cnt["n"] % (self.record_every * 2) == 0:
                    record(cnt["n"] // 2, t, cnt["n"])
                return loss(t)
            res = minimize(f, x0, method="COBYLA", options={"maxiter": 2 * self.maxiter, "rhobeg": 1.0})
            theta, nfev = res.x, cnt["n"]
        elif self.optimizer == "adam":
            theta, nfev = self._adam(X, y, sw, x0, rng, record)
        else:
            raise ValueError(self.optimizer)
        self.train_seconds_ = time.perf_counter() - t0
        self.theta_ = theta
        self.nfev_ = nfev
        self.history_ = hist
        self.final_train_loss_ = loss(theta, exact=True)
        del self._train_states
        return self

    def _adam(self, X, y, sw, x0, rng, record, b1=0.9, b2=0.999, eps=1e-8):
        """Adam on exact parameter-shift gradients (each theta_k sits in one Pauli rotation)."""
        theta = x0.copy()
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)
        nfev = 0
        psi, odd = self._train_states
        n = X.shape[1]
        P = lambda t: self._p1_from_states(psi, odd, t, n, rng)
        for it in range(1, self.maxiter + 1):
            p = np.clip(P(theta), 1e-10, 1 - 1e-10)
            dl_dp = sw * (-(y / p) + (1 - y) / (1 - p)) / len(y)
            grad = np.empty_like(theta)
            for j in range(len(theta)):
                e = np.zeros_like(theta)
                e[j] = np.pi / 2
                dp = 0.5 * (P(theta + e) - P(theta - e))
                grad[j] = np.dot(dl_dp, dp)
            nfev += 1 + 2 * len(theta)
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad ** 2
            theta = theta - self.learning_rate * (m / (1 - b1 ** it)) / (np.sqrt(v / (1 - b2 ** it)) + eps)
            record(it, theta, nfev)
        return theta, nfev

    def predict_proba(self, X):
        X = np.asarray(X, float)
        rng = np.random.default_rng(_seed_from(X, base=self.seed + 7))
        p = self._p1(X, self.theta_, rng)
        return np.column_stack([1 - p, p])

    def decision_function(self, X):
        return self.predict_proba(X)[:, 1]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
