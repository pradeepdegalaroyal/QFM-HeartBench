import numpy as np

from qfmb.diagnostics import expressibility_kl
from qfmb.suites import suite_samplesize


def _haar_states(n_states, n_qubits, seed):
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=(n_states, 2 ** n_qubits)) + 1j * rng.normal(size=(n_states, 2 ** n_qubits))
    return psi / np.linalg.norm(psi, axis=1, keepdims=True)


def test_expressibility_is_near_zero_for_haar_states():
    psi = _haar_states(800, 6, 0)
    assert expressibility_kl(np.abs(psi @ psi.conj().T) ** 2, 6) < 0.05


def test_expressibility_is_finite_for_concentrated_kernels():
    # fidelities near 1 have a Haar bin mass far below double precision of 1 - (1-F)^63
    psi = _haar_states(300, 6, 1)
    K = 0.9 + 0.1 * np.abs(psi @ psi.conj().T) ** 2
    kl = expressibility_kl(K, 6)
    assert np.isfinite(kl) and 50 < kl < 600


def test_only_one_samplesize_task_caps_the_gpc():
    capped = [t for t in suite_samplesize() if "gpc_max_train" in t]
    assert [(t["dataset"], t["label"], t["gpc_max_train"]) for t in capped] == [("cvd_n10000_s1", "GPC", 2000)]
