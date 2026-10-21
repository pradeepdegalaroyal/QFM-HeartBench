"""Load finished task results into tidy pandas tables.

``RUNS`` is the ``runs/`` folder written by ``qfmb work``; set the QFMB_RUNS
environment variable to read results from another location.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.environ.get("QFMB_RUNS") or os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "results")

QUANTUM_PREFIX = ("QSVM-", "VQC-", "QGPC-")
DATASET_LABEL = {"chd": "CHD (Cleveland)", "hdc_ext": "HDC-ext (Hungary/Switzerland/VA)",
                 "cvd": "CVD", "hdc_dedup": "HDC (de-duplicated)"}


def family(label: str) -> str:
    for p in QUANTUM_PREFIX:
        if label.startswith(p):
            return p[:-1]
    return "Classical (all features)" if "(all features)" in label else "Classical"


def load_json(suite: str) -> List[Dict]:
    out = []
    for f in sorted(glob.glob(os.path.join(RUNS, "results", suite, "*.json"))):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def fold_table(suite: str) -> pd.DataFrame:
    """One row per (task, fold) with all metrics and bookkeeping columns."""
    rows = []
    for r in load_json(suite):
        t = r["task"]
        for f in r.get("folds", []):
            row = {"suite": suite, "dataset": t.get("dataset"), "test_dataset": t.get("test_dataset"),
                   "label": t.get("label"), "variant": t.get("variant", ""), "family": family(t.get("label", "")),
                   "repeat": f.get("repeat", 0), "fold": f.get("fold", 0), "n_train": f.get("n_train"),
                   "n_test": f.get("n_test"), "fit_seconds": f.get("fit_seconds"),
                   "tune_seconds": f.get("tune_seconds"), "retained_variance": f.get("retained_variance"),
                   "task_id": t["id"]}
            row.update({f"hp_{k}": v for k, v in (f.get("hyperparameters") or {}).items()})
            row.update(f["metrics"])
            for k in ("amplitude", "lml", "form", "final_train_loss", "nfev"):
                if k in f:
                    row[k] = f[k]
            psd = f.get("psd_info") or {}
            for k, v in psd.items():
                row[f"psd_{k}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def predictions(suite: str, task_id: str) -> Dict[str, np.ndarray]:
    path = os.path.join(RUNS, "results", suite, task_id + ".npz")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def ensure_out(*sub) -> str:
    p = os.path.join(OUT, *sub)
    os.makedirs(p, exist_ok=True)
    return p
