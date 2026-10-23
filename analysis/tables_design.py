"""Design tables that do not depend on experiment results: cohorts and circuits.

Usage: python analysis/tables_design.py [output_dir]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qfmb import data  # noqa: E402
from qfmb.circuits import FEATURE_MAPS, circuit_stats, feature_map, to_qiskit, two_local  # noqa: E402
from qfmb.preprocessing import make_preprocessor  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results", "tables")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- cohorts
rows = []
for nm, lab in [("chd", "CHD (Cleveland)"), ("hdc_ext", "HDC-ext (Hungary, Switzerland, VA)"), ("cvd", "CVD")]:
    d = data.load(nm)
    enc = make_preprocessor(d.numeric, d.categorical, "full").fit(d.X)
    n_enc = enc.transform(d.X.iloc[:2]).shape[1]
    pca = make_preprocessor(d.numeric, d.categorical, "pca", 6).fit(d.X)
    var = float(np.sum(pca.named_steps["pca"].explained_variance_ratio_))
    rows.append((lab, len(d.y), int(d.y.sum()), d.X.shape[1], n_enc, int(d.X.isna().sum().sum()), var))
lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Cohort & $n$ & Positive & Raw features & Encoded features & Missing cells & Var.\ (6 PCs)$^{a}$ \\",
         r"\midrule"]
for lab, n, pos, raw, enc, miss, var in rows:
    lines.append(f"{lab} & {n:,} & {pos:,} ({100 * pos / n:.1f}\\%) & {raw} & {enc} & {miss} & {100 * var:.1f}\\% \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
with open(os.path.join(OUT, "datasets.tex"), "w", newline="\n") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(OUT, "audit.txt"), "w", newline="\n") as f:
    for k, v in data.audit().items():
        f.write(f"{k} = {v}\n")

# ---------------------------------------------------------------- circuits
from qiskit import transpile  # noqa: E402
from qiskit.transpiler import CouplingMap  # noqa: E402

desc = {"RX": (r"$R_X(\phi_i)\,R_Z(\phi_i)$", "CNOT, linear"),
        "Pauli-Z": (r"$H\,R_Z(\phi_i)\,H$", "CZ, linear"),
        "Hadamard": (r"$R_Y(\phi_i)\,H$", "CNOT, linear"),
        "Strongly Entangling": (r"$R_X(\phi_i)\,R_Y(\phi_i)$", r"CRX$(\phi_i)$, linear"),
        "VQE": (r"$H\,P(2\phi_i)$", r"CX--$P(2(\pi-\phi_i)(\pi-\phi_j))$--CX, full"),
        "Angle Encoding": (r"$R_Y(\phi_i)\,R_Z(\phi_i)$", "CZ, linear")}
lines = [r"\begin{tabular}{lllrrrrr}", r"\toprule",
         r"Feature map & Encoding block & Entangler & Gates & 2q gates & Depth & CX$^{a}$ & Depth$^{a}$ \\",
         r"\midrule"]
for name in FEATURE_MAPS:
    ops = feature_map(name, 6, 2)
    st = circuit_stats(ops, 6)
    a_, _, _ = to_qiskit(ops, 6, x_name="a")
    b_, _, _ = to_qiskit(ops, 6, x_name="b")
    t = transpile(a_.compose(b_.inverse()), basis_gates=["rz", "sx", "x", "cx"],
                  coupling_map=CouplingMap.from_line(6), optimization_level=1, seed_transpiler=0)
    lab = "ZZ (VQE)" if name == "VQE" else name
    lines.append(f"{lab} & {desc[name][0]} & {desc[name][1]} & {st['gates']} & {st['two_qubit_gates']} & "
                 f"{st['depth']} & {t.count_ops().get('cx', 0)} & {t.depth()} \\\\")
ans, k = two_local(6, 2)
st = circuit_stats(ans, 6)
lines.append(f"TwoLocal ansatz (VQC) & $R_Y(\\theta)\\,R_Z(\\theta)$ & CZ, linear & {st['gates']} & "
             f"{st['two_qubit_gates']} & {st['depth']} & -- & -- \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
with open(os.path.join(OUT, "circuits.tex"), "w", newline="\n") as f:
    f.write("\n".join(lines) + "\n")
print("written to", OUT)
