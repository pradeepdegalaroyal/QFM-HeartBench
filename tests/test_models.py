"""Equivalence of kernel vs. primal/weight-space forms, PSD repair, metrics, leakage."""
import numpy as np
import pytest
from sklearn.datasets import make_classification

from qfmb.circuits import feature_map
from qfmb.kernels import explicit_features, fidelity_kernel, psd_project, shot_noise
from qfmb.metrics import classification_metrics
from qfmb.models import QGPC, QSVM, VQC
from qfmb.simulator import statevectors

X, y = make_classification(300, 6, n_informative=4, random_state=0)
X = (X - X.mean(0)) / X.std(0)


def test_explicit_features_reproduce_fidelity_kernel():
    psi = statevectors(feature_map("RX", 6, 2), 6, 0.5 * X[:50])
    Phi = explicit_features(psi, dtype=np.float64)
    assert Phi.shape == (50, 4096)
    assert np.allclose(Phi @ Phi.T, fidelity_kernel(psi), atol=1e-10)


def test_primal_qsvm_matches_dual():
    a = QSVM("Angle Encoding", bandwidth=0.5, C=1.0, probability=False, form="kernel").fit(X[:240], y[:240])
    b = QSVM("Angle Encoding", bandwidth=0.5, C=1.0, probability=False, form="primal").fit(X[:240], y[:240])
    agree = np.mean(a.predict(X[240:]) == b.predict(X[240:]))
    assert agree >= 0.95
    assert np.corrcoef(a.decision_function(X[240:]), b.decision_function(X[240:]))[0, 1] > 0.98


def test_weight_space_qgpc_matches_kernel_form():
    k = QGPC("Pauli-Z", bandwidth=0.25, amplitude=4.0, form="kernel").fit(X[:200], y[:200])
    w = QGPC("Pauli-Z", bandwidth=0.25, amplitude=4.0, form="weight").fit(X[:200], y[:200])
    assert np.allclose(k.predict_proba(X[200:])[:, 1], w.predict_proba(X[200:])[:, 1], atol=1e-4)
    assert abs(k.lml_ - w.lml_) < 1e-4 * abs(k.lml_)
    mk, vk = k.latent(X[200:])
    mw, vw = w.latent(X[200:])
    assert np.allclose(mk, mw, atol=1e-5) and np.allclose(vk, vw, rtol=1e-4, atol=1e-7)


def test_shot_noise_and_psd_projection():
    psi = statevectors(feature_map("VQE", 6, 2), 6, 0.1 * X[:80])   # non-concentrated regime
    K = fidelity_kernel(psi)
    Kn = shot_noise(K, 16, np.random.default_rng(0), symmetric=True)
    assert np.allclose(Kn, Kn.T) and np.all(np.diag(Kn) == 1)
    Kp, info = psd_project(Kn)
    assert info["n_neg_eig"] > 0 and np.linalg.eigvalsh(Kp).min() > -1e-9


def test_vqc_trains_and_records_history():
    m = VQC("Angle Encoding", bandwidth=0.5, maxiter=30, record_every=10, seed=1).fit(X[:100], y[:100])
    assert m.history_["iter"][0] == 0 and len(m.history_["loss"]) >= 3
    assert m.predict_proba(X[100:120]).shape == (20, 2)


def test_metrics_consistency():
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 2, 200)
    yp = rng.integers(0, 2, 200)
    m = classification_metrics(yt, yp, rng.random(200), rng.random(200))
    assert abs(m["balanced_accuracy"] - (m["sensitivity"] + m["specificity"]) / 2) < 1e-12
    assert m["tp"] + m["tn"] + m["fp"] + m["fn"] == 200


def test_preprocessing_fitted_on_training_fold_only():
    """Perturbing test rows must not change the transform learned on the training rows."""
    from qfmb import data
    from qfmb.preprocessing import make_preprocessor
    d = data.load("chd")
    tr, te = np.arange(0, 250), np.arange(250, 303)
    p1 = make_preprocessor(d.numeric, d.categorical, "pca", 6).fit(d.X.iloc[tr])
    X2 = d.X.copy()
    X2.loc[te, "chol"] = 1e4
    p2 = make_preprocessor(d.numeric, d.categorical, "pca", 6).fit(X2.iloc[tr])
    assert np.allclose(p1.transform(d.X.iloc[tr]), p2.transform(X2.iloc[tr]))


def test_vqc_factorised_simulation_equals_full_circuit():
    from qfmb.circuits import two_local
    from qfmb.simulator import prob_qubit_one
    m = VQC("Strongly Entangling", bandwidth=0.7)
    theta = np.random.default_rng(3).uniform(0, 2 * np.pi, 36)
    fast = m._p1(X[:30], theta)
    ops = feature_map("Strongly Entangling", 6, 2) + two_local(6, 2)[0]
    full = prob_qubit_one(statevectors(ops, 6, 0.7 * X[:30], theta), 6, 0)
    assert np.allclose(fast, full, atol=1e-12)
