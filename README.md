# QFM-HeartBench

**Quantum Feature Map Benchmarking for Clinical Heart Disease Classification.** This repository holds the code, data, and result tables of the accompanying article. The study compares six quantum feature maps in three quantum classifiers (QSVM, VQC, and a quantum-kernel Gaussian process classifier, QGPC) with seven tuned classical models on three cardiovascular cohorts.

Every reported number, result table, and data figure of the article comes from this repository. The system-architecture diagram (Figure 1) is drawn separately.

## What the pipeline does

| Step | Implementation |
|---|---|
| Cohorts | Cleveland (CHD, n = 303). Hungary, Switzerland, and VA Long Beach (HDC-ext, n = 615); this is the OpenML 43672 file after de-duplication and after removing the records that also appear in Cleveland, so it is independent of CHD. Cardiovascular Disease (CVD, n = 68,576 after de-duplication and plausibility filters). See `src/qfmb/data.py` |
| Preprocessing | Mean or most-frequent imputation, one-hot encoding, standardization, and whitened PCA. Each step is **fitted inside every training fold**, in the outer and the inner loop (`preprocessing.py`) |
| Circuits | Six 6-qubit, 2-layer feature maps (RX, Pauli-Z, Hadamard, Strongly Entangling, ZZ (VQE), Angle Encoding) and the TwoLocal ansatz. They are defined once and checked against Qiskit `Statevector` to 1e-10 (`circuits.py`, `tests/test_circuits.py`) |
| Simulation | Exact batched statevector simulator (`simulator.py`), and Qiskit Aer density-matrix simulation for the hardware-noise runs (`noise.py`) |
| Models | QSVM (fidelity kernel, exact primal form above 12,000 training records). QGPC (Laplace Gaussian process classification with a quantum kernel, exact weight-space form above 8,000). VQC (SPSA with Qiskit's calibration, COBYLA, or Adam with parameter-shift gradients). Tuned classical LR, SVM (RBF), MLP, GPC (RBF), random forest, XGBoost, and LightGBM (`models/`) |
| Protocol | Nested cross-validation with the same grid search for every model. CHD and HDC-ext use 10 × 10 repeated stratified outer folds and 5-fold inner tuning. CVD uses one 10-fold outer loop, with tuning on a 4,000-record stratified subsample of each training fold. The secondary suites (`ablation`, `sens`, `samplesize`, `train`) use 3-fold inner tuning with a smaller grid (`evaluation.py`, `suites.py`) |
| Statistics | Corrected resampled t test (Nadeau and Bengio), Wilcoxon signed-rank test, Holm and Benjamini–Hochberg adjustment, corrected 95% confidence intervals, and effect sizes (`stats.py`) |
| Diagnostics | Kernel-target alignment, eigenvalue spectra, effective dimension, concentration, geometric difference, expressibility, entangling capability, and gradient-variance scaling (`diagnostics.py`, `trainability.py`) |

## Experiment suites

| Suite | Tasks | Purpose |
|---|---|---|
| `main` | 960 | 18 quantum configurations, 7 tuned classical models on six components, and the same 7 models on all features |
| `external` | 250 | Train on Cleveland and test on Hungary, Switzerland, and VA, and the reverse |
| `samplesize` | 525 | CVD learning curves for N = 100 to 10,000 (three subsamples each) |
| `ablation` | 2,224 | No PCA, number of components (qubits), class weighting, and fixed bandwidths |
| `sens` | 1,656 | Encoding layers, entanglement pattern, number of shots, and SPSA iterations |
| `noise` | 480 | Depolarizing, thermal-relaxation, and readout noise at three strengths |
| `diag`, `gradvar`, `train` | 9, 36, 720 | Kernel diagnostics, gradient-variance scaling, and VQC training curves over seeds and optimizers |
| `supp` | 320 | HDC de-duplicated (918 records, includes Cleveland) as a supplementary cohort |

All 7,180 tasks together need about 1,030 CPU-hours, so a full re-run needs a multi-core machine or a cluster.

## Installation

Python 3.11 was used for the article.

```bash
python -m venv .venv
. .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt   # the exact versions used for the article
pip install -e ".[test]"
pytest -q                         # 109 tests: circuits, models, statistics
```

The raw data are the public OpenML datasets 194, 43672, and 45547, cached in `data/raw/`.

## Reproduce the tables and figures from the shipped results

`results/tables/` holds every result table (CSV and LaTeX), and `results/key_numbers.json` holds the numbers quoted in the text. From these files:

| Output | Command |
|---|---|
| Figures 2–7 | `python analysis/figures.py results figs` |
| Cohort and circuit tables | `python analysis/tables_design.py <out_dir>` |
| Numbers in the text (LaTeX macros) and the compact comparison table | `python analysis/write_latex_macros.py results <out_dir>` |
| Supplementary Tables S1–S8 | `python analysis/write_supplementary.py results <out_dir>` |

## Re-run the experiments

```bash
qfmb list --suites main                      # number of tasks and how many are done
qfmb work --suites main --workers 8 --wait   # run them; resumable, safe to stop and restart
QFMB_RUNS=runs python analysis/analyze_all.py results   # rebuild results/tables and key_numbers.json
```

`qfmb work` writes one result file per task to `runs/`. Workers share a file-based task queue, so several processes, or several machines that share the folder, can run the same suites together. Finished tasks are never recomputed. `analyze_all.py` needs the finished runs and overwrites `results/tables/`.

## License

MIT (see `LICENSE`).
