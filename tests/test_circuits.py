"""The NumPy simulator must reproduce Qiskit exactly for every circuit used."""
import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qfmb.circuits import ENTANGLEMENTS, FEATURE_MAPS, feature_map, to_qiskit, two_local
from qfmb.simulator import prob_qubit_one, statevectors

RNG = np.random.default_rng(0)


def _qiskit_state(qc, params, values):
    bound = qc.assign_parameters(dict(zip(params, values)))
    return Statevector(bound).data


@pytest.mark.parametrize("name", list(FEATURE_MAPS))
@pytest.mark.parametrize("n,layers", [(2, 1), (4, 2), (6, 2), (5, 3), (12, 2), (13, 6)])
@pytest.mark.parametrize("ent", ENTANGLEMENTS)
def test_feature_maps_match_qiskit(name, n, layers, ent):
    ops = feature_map(name, n, layers, ent)
    X = RNG.normal(size=(3, n))
    psi = statevectors(ops, n, X)
    qc, x, _ = to_qiskit(ops, n)
    for b in range(3):
        ref = _qiskit_state(qc, list(x), X[b])
        assert np.allclose(psi[b], ref, atol=1e-10)


def test_zz_map_matches_qiskit_library():
    try:
        from qiskit.circuit.library import zz_feature_map
        lib = zz_feature_map(6, reps=2, entanglement="full")
    except ImportError:  # older Qiskit
        from qiskit.circuit.library import ZZFeatureMap
        lib = ZZFeatureMap(6, reps=2, entanglement="full").decompose()
    X = RNG.uniform(-1, 1, size=(4, 6))
    psi = statevectors(feature_map("VQE", 6, 2), 6, X)
    params = sorted(lib.parameters, key=lambda p: p.index)
    for b in range(4):
        assert np.allclose(psi[b], _qiskit_state(lib, params, X[b]), atol=1e-10)


def test_two_local_matches_qiskit_library():
    try:
        from qiskit.circuit.library import n_local
        lib = n_local(6, ["ry", "rz"], "cz", entanglement="linear", reps=2)
    except ImportError:
        from qiskit.circuit.library import TwoLocal
        lib = TwoLocal(6, ["ry", "rz"], "cz", entanglement="linear", reps=2).decompose()
    ops, k = two_local(6, 2)
    assert k == lib.num_parameters == 36
    theta = RNG.uniform(-np.pi, np.pi, size=k)
    psi = statevectors(ops, 6, None, theta, batch=1)[0]
    params = sorted(lib.parameters, key=lambda p: p.index)
    assert np.allclose(psi, _qiskit_state(lib, params, theta), atol=1e-10)


def test_vqc_probability_is_qubit0_parity():
    """qiskit-machine-learning VQC (2 classes) uses interpret = bitstring % 2 = qubit 0."""
    fm = feature_map("Angle Encoding", 6, 2)
    ans, k = two_local(6, 2)
    X = RNG.normal(size=(5, 6))
    theta = RNG.uniform(0, 1, size=k)
    psi = statevectors(fm + ans, 6, X, theta)
    probs = np.abs(psi) ** 2
    odd = np.arange(64) % 2 == 1
    assert np.allclose(prob_qubit_one(psi, 6, 0), probs[:, odd].sum(1))
