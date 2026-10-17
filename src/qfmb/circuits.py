"""Quantum feature maps and the variational ansatz, defined once as gate lists.

Every circuit is a list of ``Op`` records. The same list is
  * executed by the batched NumPy statevector simulator (``simulator.py``), and
  * converted to a Qiskit ``QuantumCircuit`` (``to_qiskit``) for verification
    against ``qiskit.quantum_info.Statevector`` and for Aer noise simulations.

Angle specifications
--------------------
``None``            fixed gate (h, x, cx, cz)
``("x", i)``        x_i            (data feature i, already bandwidth-scaled)
``("2x", i)``       2 x_i          (single-qubit term of Qiskit's ZZFeatureMap)
``("zz", i, j)``    2 (pi - x_i)(pi - x_j)   (pair term of Qiskit's ZZFeatureMap)
``("t", k)``        theta_k        (trainable ansatz parameter)

The map labelled "ZZ (VQE)" is Qiskit's ``ZZFeatureMap``; the other maps are
defined below and verified against Qiskit in ``tests/test_circuits.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

AngleSpec = Optional[Tuple]


@dataclass(frozen=True)
class Op:
    name: str
    qubits: Tuple[int, ...]
    angle: AngleSpec = None


ENTANGLEMENTS = ("none", "linear", "circular", "full")


def entangling_pairs(n: int, entanglement: str) -> List[Tuple[int, int]]:
    if entanglement == "none" or n < 2:
        return []
    if entanglement == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if entanglement == "circular":
        pairs = [(i, i + 1) for i in range(n - 1)]
        return pairs + [(n - 1, 0)] if n > 2 else pairs
    if entanglement == "full":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    raise ValueError(f"unknown entanglement {entanglement!r}")


# --------------------------------------------------------------------------- #
# Feature maps (generalised to n qubits / L layers / entanglement pattern)
# --------------------------------------------------------------------------- #
def rx_map(n: int, layers: int = 2, entanglement: str = "linear") -> List[Op]:
    ops: List[Op] = []
    for _ in range(layers):
        for i in range(n):
            ops += [Op("rx", (i,), ("x", i)), Op("rz", (i,), ("x", i))]
        ops += [Op("cx", p) for p in entangling_pairs(n, entanglement)]
    return ops


def pauli_z_map(n: int, layers: int = 2, entanglement: str = "linear") -> List[Op]:
    ops: List[Op] = []
    for _ in range(layers):
        for i in range(n):
            ops += [Op("h", (i,)), Op("rz", (i,), ("x", i)), Op("h", (i,))]
        ops += [Op("cz", p) for p in entangling_pairs(n, entanglement)]
    return ops


def hadamard_map(n: int, layers: int = 2, entanglement: str = "linear") -> List[Op]:
    ops: List[Op] = []
    for _ in range(layers):
        for i in range(n):
            ops += [Op("ry", (i,), ("x", i)), Op("h", (i,))]
        ops += [Op("cx", p) for p in entangling_pairs(n, entanglement)]
    return ops


def strongly_entangling_map(n: int, layers: int = 2, entanglement: str = "linear") -> List[Op]:
    ops: List[Op] = []
    for _ in range(layers):
        for i in range(n):
            ops += [Op("rx", (i,), ("x", i)), Op("ry", (i,), ("x", i))]
        ops += [Op("crx", (c, t), ("x", c)) for c, t in entangling_pairs(n, entanglement)]
    return ops


def zz_map(n: int, layers: int = 2, entanglement: str = "full") -> List[Op]:
    """Qiskit ``ZZFeatureMap`` (H, P(2x_i), CX-P(2(pi-x_i)(pi-x_j))-CX)."""
    ops: List[Op] = []
    for _ in range(layers):
        for i in range(n):
            ops += [Op("h", (i,)), Op("p", (i,), ("2x", i))]
        for i, j in entangling_pairs(n, entanglement):
            ops += [Op("cx", (i, j)), Op("p", (j,), ("zz", i, j)), Op("cx", (i, j))]
    return ops


def angle_encoding_map(n: int, layers: int = 2, entanglement: str = "linear") -> List[Op]:
    ops: List[Op] = []
    for _ in range(layers):
        for i in range(n):
            ops += [Op("ry", (i,), ("x", i)), Op("rz", (i,), ("x", i))]
        ops += [Op("cz", p) for p in entangling_pairs(n, entanglement)]
    return ops


FEATURE_MAPS: Dict[str, Callable[..., List[Op]]] = {
    "RX": rx_map,
    "Pauli-Z": pauli_z_map,
    "Hadamard": hadamard_map,
    "Strongly Entangling": strongly_entangling_map,
    "VQE": zz_map,
    "Angle Encoding": angle_encoding_map,
}
DEFAULT_ENTANGLEMENT = {name: ("full" if name == "VQE" else "linear") for name in FEATURE_MAPS}
SHORT = {"RX": "RX", "Pauli-Z": "PZ", "Hadamard": "Had", "Strongly Entangling": "SE",
         "VQE": "ZZ", "Angle Encoding": "AE"}


def feature_map(name: str, n: int, layers: int = 2, entanglement: Optional[str] = None) -> List[Op]:
    ent = entanglement or DEFAULT_ENTANGLEMENT[name]
    return FEATURE_MAPS[name](n, layers, ent)


# --------------------------------------------------------------------------- #
# Variational ansatz: Qiskit TwoLocal(n, ['ry','rz'], 'cz', 'linear', reps=L)
# --------------------------------------------------------------------------- #
def two_local(n: int, reps: int = 2, entanglement: str = "linear") -> Tuple[List[Op], int]:
    ops: List[Op] = []
    k = 0
    for r in range(reps + 1):
        for gate in ("ry", "rz"):
            for i in range(n):
                ops.append(Op(gate, (i,), ("t", k)))
                k += 1
        if r < reps:
            ops += [Op("cz", p) for p in entangling_pairs(n, entanglement)]
    return ops, k


# --------------------------------------------------------------------------- #
# Qiskit conversion and circuit statistics
# --------------------------------------------------------------------------- #
def to_qiskit(ops: Sequence[Op], n: int, n_features: Optional[int] = None, n_theta: int = 0,
              x_name: str = "x", t_name: str = "θ"):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    x = ParameterVector(x_name, n_features if n_features is not None else n)
    t = ParameterVector(t_name, n_theta) if n_theta else None
    qc = QuantumCircuit(n)

    def angle(spec):
        kind = spec[0]
        if kind == "x":
            return x[spec[1]]
        if kind == "2x":
            return 2 * x[spec[1]]
        if kind == "zz":
            return 2 * (np.pi - x[spec[1]]) * (np.pi - x[spec[2]])
        if kind == "t":
            return t[spec[1]]
        raise ValueError(spec)

    for op in ops:
        if op.angle is None:
            getattr(qc, op.name)(*op.qubits)
        else:
            getattr(qc, op.name)(angle(op.angle), *op.qubits)
    return qc, x, t


def circuit_stats(ops: Sequence[Op], n: int) -> Dict[str, int]:
    """Gate count, two-qubit gate count, data-dependent gates and depth."""
    depth = [0] * n
    for op in ops:
        level = max(depth[q] for q in op.qubits) + 1
        for q in op.qubits:
            depth[q] = level
    return {
        "gates": len(ops),
        "two_qubit_gates": sum(len(op.qubits) == 2 for op in ops),
        "data_gates": sum(op.angle is not None and op.angle[0] != "t" for op in ops),
        "depth": max(depth) if ops else 0,
    }
