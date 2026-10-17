# QFM-HeartBench

**Quantum Feature Map Benchmarking for Clinical Heart Disease Classification** — code and
reproducibility package for the accompanying article (QSVM, VQC and a quantum-kernel
Gaussian-process classifier, six quantum feature maps, three cardiovascular cohorts).

Every number, table, and figure in the article is produced by this repository.

## What the pipeline does

| Step | Implementation |
|---|---|
| Cohorts | Cleveland (CHD, n = 303); Hungary + Switzerland + VA Long Beach (HDC-ext, n = 615, **independent of Cleveland**: the OpenML 43672 file is de-duplicated and Cleveland records are removed by exact match); Cardiovascular Disease (CVD, n = 68,576 after de-duplication and plausibility filters) — `src/qfmb/data.py` |
| Preprocessing | mean/mode imputation → one-hot → standardisation → PCA (whitened), **fitted inside every training fold** (outer and inner) — `preprocessing.py` |
| Circuits | six 6-qubit, 2-layer feature maps (RX, Pauli-Z, Hadamard, "Strongly Entangling", ZZ/"VQE", Angle Encoding) and the TwoLocal ansatz, defined once and checked against Qiskit `Statevector` to 1e-10 — `circuits.py`, `tests/test_circuits.py` |
| Simulation | exact batched statevector simulator (`simulator.py`); Qiskit Aer density-matrix simulation for hardware-noise runs (`noise.py`) |
| Models | QSVM (fidelity kernel; exact primal form for N > 12k), QGPC (Laplace GP classification with a quantum kernel; exact weight-space form for N > 8k), VQC (SPSA with Qiskit's calibration, COBYLA, Adam with parameter-shift gradients); tuned classical LR, SVM-RBF, MLP, GPC-RBF, RF, XGBoost and LightGBM — `models/` |
| Protocol | nested CV: 10 × 10 repeated stratified CV outside, 5-fold grid search inside (balanced accuracy), identical for all models — `evaluation.py` |
| Statistics | Nadeau–Bengio corrected resampled t-test, Wilcoxon signed-rank, Holm and Benjamini–Hochberg adjustment, corrected 95 % CIs, effect sizes — `stats.py` |
| Diagnostics | kernel-target alignment, spectra, effective dimension, concentration, geometric difference, expressibility, entangling capability, gradient-variance scaling — `diagnostics.py`, `trainability.py` |

## Experiment suites

| Suite | Purpose |
|---|---|
| `main` | 18 quantum configurations + 7 tuned classical models (PCA-6) + 7 classical models on all features |
| `external` | train on Cleveland, test on Hungary/Switzerland/VA (and the reverse) |
| `samplesize` | CVD learning curves, N = 100 … 10,000 (3 subsamples each) |
| `ablation` | no PCA, number of PCs/qubits, class weighting, bandwidth |
| `sens` | circuit depth, entanglement pattern, number of shots, SPSA iterations |
| `noise` | depolarising + thermal-relaxation + readout noise at three strengths |
| `diag`, `gradvar`, `train` | mechanistic diagnostics, barren-plateau check, VQC training curves over seeds and optimisers |
| `supp` | HDC de-duplicated (918 records, includes Cleveland) as a supplementary cohort |

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate        # or conda
pip install -e ".[test]"
pytest -q                                           # circuit/model equivalence tests
qfmb list --suites main                             # how many tasks
qfmb work --suites main --workers 8 --wait          # run (resumable; safe to kill and restart)
qfmb dashboard                                      # runs/dashboard.html
python analysis/analyze_all.py results              # tables (CSV/LaTeX) and key numbers
python analysis/figures.py results figs             # figures of the paper
```

Raw data are the public OpenML datasets 194, 43672 and 45547, cached in `data/raw/`.

## Running on a PBS cluster

`scripts/hpc/submit.sh <queue> <ncpus> <mem> <suites> <small|big>` submits a worker job. Jobs share a
file-based task queue (`runs/locks`), so any number of jobs on any nodes can cooperate. Finished tasks are
never recomputed, tasks orphaned by a killed job are reclaimed once their heartbeat is older than 48 hours, and a job that reaches
its walltime resubmits itself while work remains.

## License

MIT (see `LICENSE`).
