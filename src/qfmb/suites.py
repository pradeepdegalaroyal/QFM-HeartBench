"""Experiment suites -> deterministic task lists.

Suite       Content
----------  ---------------------------------------------------------------
main        18 quantum configs + 7 tuned classical (PCA-6)
            + 7 tuned classical on all features; nested CV.
            CHD, HDC-ext: 10x10 CV. CVD (68,576): 10-fold.
external    Cleveland <-> Hungary/Switzerland/VA transfer.
diag        kernel diagnostics (diagnostics.py).
gradvar     gradient-variance scaling (trainability.py).
train       VQC loss curves x 10 seeds x optimisers/inits.
samplesize  CVD learning curves N = 100 ... 10,000.
ablation    no-PCA, #PCs, class weighting, bandwidth.
sens        layers, entanglement, shots, SPSA iterations.
noise       Aer density-matrix noise (noise.py).
supp        HDC de-duplicated (918) with the main protocol.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from .circuits import FEATURE_MAPS
from .models.classical import CLASSICAL

MAPS = list(FEATURE_MAPS)
FAMILIES = ["qsvm", "vqc", "qgpc"]
FAM_LABEL = {"qsvm": "QSVM", "vqc": "VQC", "qgpc": "QGPC"}
SUITE_PRIORITY = {"main": 0, "external": 1, "diag": 1, "gradvar": 1, "train": 2, "samplesize": 2,
                  "ablation": 3, "sens": 3, "noise": 4, "supp": 5}
# relative cost per (family, fold) used only to order tasks (expensive first)
FAM_COST = {"vqc": 10, "qgpc": 4, "qsvm": 3, "classical": 1}


def _id(*parts) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "-", "__".join(str(p) for p in parts if p is not None and p != ""))


def quantum_model(fam: str, fmap: str, params: Dict[str, Any] = None, tune="default") -> Dict:
    return {"family": fam, "params": {"feature_map": fmap, **(params or {})}, "tune": tune}


def classical_model(name: str, params: Dict[str, Any] = None, tune="default") -> Dict:
    return {"family": "classical", "name": name, "params": dict(params or {}), "tune": tune}


SECONDARY = {"ablation", "sens", "samplesize", "train"}


def _task(suite, dataset, label, model, repeat, n_splits, group=None, variant="", prep=None, folds=None,
          **extra) -> Dict:
    t = {"suite": suite, "dataset": dataset, "label": label, "group": group or label, "model": model,
         "prep": prep or {"mode": "pca", "k": 6},
         "cv": {"n_splits": n_splits, "repeat": repeat, "folds": folds}, "variant": variant,
         "priority": SUITE_PRIORITY[suite]}
    if suite in SECONDARY:
        t.update({"inner_splits": 3, "reduced_grid": True})
    t.update(extra)
    fold_tag = f"f{folds[0]}" if folds else None
    t["id"] = _id(suite, dataset, label, variant, f"r{repeat}", fold_tag)
    fam = model["family"]
    nf = len(folds) if folds else n_splits
    t["cost"] = FAM_COST[fam] * nf * (20 if dataset == "cvd" else 1)
    return t


def standard_models(include_full: bool = True):
    """(label, model, prep) for the 18 quantum configurations and the classical baselines."""
    out = []
    for fam in FAMILIES:
        for m in MAPS:
            out.append((f"{FAM_LABEL[fam]}-{m}", quantum_model(fam, m), None))
    for c in CLASSICAL:
        out.append((c, classical_model(c), None))
    if include_full:
        for c in CLASSICAL:
            out.append((f"{c} (all features)", classical_model(c), {"mode": "full"}))
    return out


def suite_main(datasets=("chd", "hdc_ext"), suite="main") -> List[Dict]:
    tasks = []
    for ds in datasets:
        for label, model, prep in standard_models():
            for r in range(10):
                tasks.append(_task(suite, ds, label, model, r, 10, prep=prep))
    return tasks


def suite_main_cvd() -> List[Dict]:
    tasks = []
    for label, model, prep in standard_models():
        fam = model["family"]
        threads = 1
        for f in range(10):
            tasks.append(_task("main", "cvd", label, model, 0, 10, prep=prep, folds=[f],
                               tune_subsample=4000, big=True, threads=threads, store_predictions=True))
    return tasks


def suite_external() -> List[Dict]:
    tasks = []
    for tr, te in (("common_chd", "common_hdc_ext"), ("common_hdc_ext", "common_chd")):
        for label, model, prep in standard_models(include_full=False):
            for r in range(5):
                t = _task("external", tr, label, model, r, 1, variant=f"to_{te}", test_dataset=te)
                tasks.append(t)
    return tasks


def suite_samplesize() -> List[Dict]:
    tasks = []
    for n in (100, 200, 500, 1000, 2000, 5000, 10000):
        for s in range(3):
            ds = f"cvd_n{n}_s{s}"
            for label, model, prep in standard_models(include_full=False):
                tasks.append(_task("samplesize", ds, label, model, 0, 5, group=label, variant=f"N{n}",
                                   tune_subsample=4000 if n >= 5000 else None))
    return tasks


ABL_DATASETS = ("chd", "hdc_ext", "cvd_n2000_s0", "cvd_imb25_n2000_s0")
SENS_DATASETS = ("chd", "hdc_ext", "cvd_n2000_s0")


def suite_ablation() -> List[Dict]:
    tasks = []
    for ds in ABL_DATASETS:
        for r in range(2):
            def add(label, model, variant, prep=None):
                tasks.append(_task("ablation", ds, label, model, r, 5, group=label, variant=variant, prep=prep))
            # main-protocol reference at 5x2 (identical settings, smaller CV)
            for label, model, prep in standard_models(include_full=True):
                add(label, model, "reference", prep)
            # (a) no PCA: one qubit per raw (ordinal) feature; VQC bandwidth fixed (cost)
            for fam in FAMILIES:
                if fam == "vqc" and r > 0:          # 11-13 qubit VQC: one repeat only (cost)
                    continue
                for m in MAPS:
                    tune = {"bandwidth": [0.25]} if fam == "vqc" else "default"
                    add(f"{FAM_LABEL[fam]}-{m}", quantum_model(fam, m, tune=tune), "noPCA", {"mode": "raw"})
            for c in CLASSICAL:
                add(c, classical_model(c), "noPCA", {"mode": "raw"})
            # (b) number of principal components = number of qubits
            for k in (2, 4, 8, 10):
                for fam in FAMILIES:
                    if fam == "vqc" and k == 10:    # 10-qubit VQC with nested tuning: too costly
                        continue
                    for m in MAPS:
                        add(f"{FAM_LABEL[fam]}-{m}", quantum_model(fam, m), f"k{k}", {"mode": "pca", "k": k})
                for c in CLASSICAL:
                    add(c, classical_model(c), f"k{k}", {"mode": "pca", "k": k})
            # (c) class weighting / threshold correction toggled
            for m in MAPS:
                add(f"QSVM-{m}", quantum_model("qsvm", m, {"class_weight": None}), "weighting_flipped")
                add(f"VQC-{m}", quantum_model("vqc", m, {"class_weight": "balanced"}), "weighting_flipped")
                add(f"QGPC-{m}", quantum_model("qgpc", m, {"threshold": "prior"}), "weighting_flipped")
            for c in ("LR", "SVM", "RF", "LGBM"):
                add(c, classical_model(c, {"class_weight": None}), "weighting_flipped")
            # (d) fixed bandwidth sweep (other hyper-parameters tuned as usual)
            for lam in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
                for fam in FAMILIES:
                    for m in MAPS:
                        grid = {"qsvm": {"C": [0.1, 1.0, 10.0, 100.0]}, "qgpc": {}, "vqc": {}}[fam]
                        add(f"{FAM_LABEL[fam]}-{m}", quantum_model(fam, m, {"bandwidth": lam}, tune=grid),
                            f"bw{lam}")
    return tasks


def suite_sensitivity() -> List[Dict]:
    tasks = []
    for ds in SENS_DATASETS:
        for r in range(2):
            def add(label, model, variant):
                tasks.append(_task("sens", ds, label, model, r, 5, group=label, variant=variant))
            for fam in FAMILIES:
                for m in MAPS:
                    lab = f"{FAM_LABEL[fam]}-{m}"
                    for L in (1, 2, 3, 4, 6):
                        add(lab, quantum_model(fam, m, {"n_layers": L}), f"layers{L}")
                    for ent in ("none", "linear", "circular", "full"):
                        add(lab, quantum_model(fam, m, {"entanglement": ent}), f"ent-{ent}")
                    shots = (64, 256, 1024, 2048, 8192) if fam != "vqc" else (256, 2048, 8192)
                    for s in shots:
                        add(lab, quantum_model(fam, m, {"shots": s}), f"shots{s}")
                    if fam == "vqc":
                        for T in (25, 50, 100, 200, 400, 800):
                            add(lab, quantum_model(fam, m, {"maxiter": T}), f"spsa{T}")
    return tasks


def suite_train(bandwidth: float = 0.25) -> List[Dict]:
    tasks = []
    variants = {"spsa": {"optimizer": "spsa", "maxiter": 200},
                "spsa-init-pi": {"optimizer": "spsa", "maxiter": 200, "init": "uniform_pi"},
                "cobyla": {"optimizer": "cobyla", "maxiter": 200},
                "adam-ps": {"optimizer": "adam", "maxiter": 100, "learning_rate": 0.05}}
    for ds in SENS_DATASETS:
        for m in MAPS:
            for vname, p in variants.items():
                for r in range(10):
                    tasks.append(_task("train", ds, f"VQC-{m}", quantum_model("vqc", m, {**p, "bandwidth": bandwidth},
                                                                             tune=None),
                                       r, 5, variant=vname, record_history=True,
                                       record_every=1 if vname != "adam-ps" else 1))
    return tasks


def suite_diag() -> List[Dict]:
    tasks = []
    for ds in ("chd", "hdc_ext", "cvd_n2000_s0"):
        for r in range(3):
            tasks.append({"suite": "diag", "kind": "diag", "dataset": ds, "repeat": r,
                          "id": _id("diag", ds, f"r{r}"), "label": f"diagnostics {ds}", "group": "diag",
                          "priority": SUITE_PRIORITY["diag"], "cost": 5})
    return tasks


def suite_gradvar() -> List[Dict]:
    tasks = []
    for m in MAPS:
        for n in (2, 4, 6, 8, 10, 12):
            tasks.append({"suite": "gradvar", "kind": "gradvar", "feature_map": m, "n_qubits": n,
                          "layers": [1, 2, 4, 8], "n_samples": 200,
                          "id": _id("gradvar", m, f"n{n}"), "label": f"gradvar {m} n={n}", "group": "gradvar",
                          "priority": SUITE_PRIORITY["gradvar"], "cost": 2 ** max(0, n - 6)})
    return tasks


def suite_noise() -> List[Dict]:
    tasks = []
    for ds in ("chd", "hdc_ext"):
        for level in ("shots", "low", "medium", "high"):
            for fam in ("kernel", "vqc"):
                for m in MAPS:
                    for f in range(5):
                        label = f"{'QSVM+QGPC' if fam == 'kernel' else 'VQC'}-{m}"
                        tasks.append({"suite": "noise", "kind": "noise", "dataset": ds, "level": level,
                                      "family": fam, "feature_map": m, "fold": f, "n_splits": 5,
                                      "max_train": 300,
                                      "id": _id("noise", ds, label, level, f"f{f}"), "label": label,
                                      "group": label, "priority": SUITE_PRIORITY["noise"],
                                      "cost": 30 if fam == "vqc" else 20})
    return tasks


SUITES = {
    "main": lambda: suite_main() + suite_main_cvd(),
    "external": suite_external,
    "diag": suite_diag,
    "gradvar": suite_gradvar,
    "train": suite_train,
    "samplesize": suite_samplesize,
    "ablation": suite_ablation,
    "sens": suite_sensitivity,
    "noise": suite_noise,
    "supp": lambda: suite_main(("hdc_dedup",), suite="supp"),
}


def all_tasks(names) -> List[Dict]:
    tasks = []
    for n in names:
        tasks += SUITES[n]()
    tasks.sort(key=lambda t: (t["priority"], -t.get("cost", 1), t["id"]))
    return tasks
