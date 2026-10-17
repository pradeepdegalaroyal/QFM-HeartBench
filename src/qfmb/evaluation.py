"""Nested cross-validation engine shared by every model (classical and quantum).

Outer loop  : repeated stratified k-fold. Split seeds depend only on the
              repeat index, so every model sees identical folds (paired tests).
Inner loop  : stratified 5-fold grid search on the outer-training data,
              scored by balanced accuracy, with the preprocessing pipeline
              re-fitted inside every inner fold (no leakage at either level).
              GP models choose kernel hyper-parameters by type-II ML instead.
Large data  : optional ``tune_subsample`` — the search runs on a stratified
              subsample of the outer-training fold; the selected model is then
              refitted on the full outer-training fold.
"""
from __future__ import annotations

import itertools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from . import data as data_mod
from .metrics import classification_metrics
from .models import QGPC, QSVM, VQC, make_classical
from .preprocessing import make_preprocessor, retained_variance

QUANTUM = {"qsvm": QSVM, "qgpc": QGPC, "vqc": VQC}
# Pilot (5-fold, CHD / HDC-ext) showed the optimal angle scale differs by map by
# more than an order of magnitude (ZZ best near 0.1, Hadamard near >= 1), so the
# bandwidth is tuned for every map from the same wide grid.
BANDWIDTHS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
DEFAULT_GRIDS = {
    "qsvm": {"bandwidth": BANDWIDTHS, "C": [0.1, 1.0, 10.0, 100.0]},
    "qgpc": {"bandwidth": BANDWIDTHS},
    "vqc": {"bandwidth": BANDWIDTHS},
}


def _grid(grid: Dict[str, list]) -> List[Dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def _strat_subsample(y: np.ndarray, m: int, seed: int) -> np.ndarray:
    if m is None or m >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    out = []
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        out.append(rng.choice(idx, int(round(m * len(idx) / len(y))), replace=False))
    return np.sort(np.concatenate(out))


REDUCED_BANDWIDTHS = [0.1, 0.25, 0.5, 1.0]


class ModelSpec:
    """Builds estimators for one model configuration of a task."""

    def __init__(self, spec: Dict[str, Any], n_train: int, reduced: bool = False):
        self.family = spec["family"]
        self.params = dict(spec.get("params", {}))
        self.name = spec.get("name")
        tune = spec.get("tune", "default")
        if self.family == "classical":
            self.base, grid = make_classical(self.name, n_train=n_train,
                                             class_weight=self.params.pop("class_weight", "balanced"))
            self.base.set_params(**self.params)
            self.grid = grid if tune == "default" else (tune or {})
        else:
            self.grid = dict(DEFAULT_GRIDS[self.family]) if tune == "default" else dict(tune or {})
            if reduced and tune == "default":
                self.grid["bandwidth"] = REDUCED_BANDWIDTHS

    def build(self, seed: int, **hp):
        if self.family == "classical":
            est = clone(self.base)
            if "random_state" in est.get_params():
                est.set_params(random_state=seed)
            return est.set_params(**hp)
        return QUANTUM[self.family](**{**self.params, **hp, "seed": seed})


def _fit_predict(est, Ztr, ytr, Zte):
    t0 = time.perf_counter()
    est.fit(Ztr, ytr)
    t1 = time.perf_counter()
    y_pred = est.predict(Zte)
    score = proba = None
    try:
        proba = est.predict_proba(Zte)[:, 1]
    except (AttributeError, NotImplementedError):
        proba = None
    if hasattr(est, "decision_function"):
        try:
            score = est.decision_function(Zte)
        except (AttributeError, NotImplementedError):
            score = None
    if score is None:
        score = proba
    t2 = time.perf_counter()
    return y_pred, score, proba, t1 - t0, t2 - t1


def select_hyperparameters(ms: ModelSpec, ds, prep: Dict, tr_idx: np.ndarray, seed: int,
                           inner_splits: int = 5) -> Tuple[Dict[str, Any], Dict[str, float]]:
    combos = _grid(ms.grid)
    X, y = ds.X.iloc[tr_idx], ds.y[tr_idx]
    if ms.family == "qgpc" and len(combos) > 1:       # type-II ML over the bandwidth grid
        pre = make_preprocessor(ds.numeric, ds.categorical, prep["mode"], prep.get("k", 6), seed)
        Z = pre.fit_transform(X)
        scores = {}
        best, best_lml, best_amp = None, -np.inf, None
        for hp in combos:
            m = ms.build(seed, **hp).set_params(form="kernel")
            m.fit(Z, y)
            scores[str(hp)] = m.lml_
            if m.lml_ > best_lml:
                best, best_lml, best_amp = hp, m.lml_, m.amplitude_
        return {**best, "amplitude": best_amp}, scores
    if len(combos) == 1:
        return combos[0], {}
    skf = StratifiedKFold(inner_splits, shuffle=True, random_state=seed)
    totals = np.zeros(len(combos))
    for itr, iva in skf.split(X, y):
        pre = make_preprocessor(ds.numeric, ds.categorical, prep["mode"], prep.get("k", 6), seed)
        Ztr = pre.fit_transform(X.iloc[itr])
        Zva = pre.transform(X.iloc[iva])
        for j, hp in enumerate(combos):
            est = ms.build(seed, **hp)
            if ms.family == "qsvm":
                est.set_params(probability=False, form="kernel")
            est.fit(Ztr, y[itr])
            totals[j] += classification_metrics(y[iva], est.predict(Zva))["balanced_accuracy"]
    means = totals / inner_splits
    return combos[int(np.argmax(means))], {str(hp): float(s) for hp, s in zip(combos, means)}


def run_task(task: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Execute one task (one repeat, all or some folds) and return (summary, arrays)."""
    ds = data_mod.load(task["dataset"])
    prep = task.get("prep", {"mode": "pca", "k": 6})
    cv = task["cv"]
    repeat = cv.get("repeat", 0)
    store_pred = task.get("store_predictions", True)
    folds_out, arrays = [], {k: [] for k in ("fold", "index", "y", "y_pred", "score", "proba", "latent_var")}

    if task.get("test_dataset"):                       # external validation: single train/test split
        ext = data_mod.load(task["test_dataset"])
        splits = [(np.arange(len(ds.y)), None)]
    else:
        skf = StratifiedKFold(cv.get("n_splits", 10), shuffle=True, random_state=1000 + repeat)
        splits = list(skf.split(ds.X, ds.y))
        if cv.get("folds") is not None:
            splits = [splits[f] for f in cv["folds"]]
    fold_ids = cv.get("folds") or list(range(len(splits)))

    for fold, (tr, te) in zip(fold_ids, splits):
        seed = 3000 + 100 * repeat + fold
        ms = ModelSpec(task["model"], len(tr), reduced=task.get("reduced_grid", False))
        t0 = time.perf_counter()
        sub = _strat_subsample(ds.y[tr], task.get("tune_subsample"), seed)
        hp, inner = select_hyperparameters(ms, ds, prep, tr[sub], 2000 + 100 * repeat + fold,
                                           inner_splits=task.get("inner_splits", 5))
        tune_s = time.perf_counter() - t0
        pre = make_preprocessor(ds.numeric, ds.categorical, prep["mode"], prep.get("k", 6), seed)
        fit_idx = tr
        # Exact GPC is O(N^3) with several Cholesky factorizations per restart
        # (n_restarts_optimizer=2); empirically this is already far too slow
        # well below 8,000 training rows (one such fold ran >18h), so the cap
        # is set much lower and the boundary is inclusive.
        if ms.family == "classical" and ms.name == "GPC" and len(tr) >= 2500:
            fit_idx = tr[_strat_subsample(ds.y[tr], 2000, seed)]
        Ztr = pre.fit_transform(ds.X.iloc[fit_idx])
        if task.get("test_dataset"):
            Zte, yte, te_index = pre.transform(ext.X), ext.y, np.arange(len(ext.y))
        else:
            Zte, yte, te_index = pre.transform(ds.X.iloc[te]), ds.y[te], te
        est = ms.build(seed, **{k: v for k, v in hp.items() if k != "amplitude"})
        if ms.family == "qgpc" and "amplitude" in hp and (
                len(fit_idx) > est.weight_threshold or task.get("tune_subsample")):
            est.set_params(amplitude=hp["amplitude"])
        if ms.family == "vqc" and task.get("record_history"):
            est.set_params(record_every=task.get("record_every", 1))
        y_pred, score, proba, fit_s, pred_s = _fit_predict(est, Ztr, ds.y[fit_idx], Zte)
        met = classification_metrics(yte, y_pred, score, proba)
        rec = {"fold": int(fold), "repeat": int(repeat), "n_train": int(len(fit_idx)),
               "n_test": int(len(yte)), "hyperparameters": hp, "inner_scores": inner,
               "tune_seconds": tune_s, "fit_seconds": fit_s, "predict_seconds": pred_s,
               "retained_variance": retained_variance(pre), "metrics": met}
        lat = None
        if ms.family == "qgpc":
            rec.update({"amplitude": est.amplitude_, "lml": est.lml_, "form": est.form_,
                        "psd_info": getattr(est, "psd_info_", {})})
            lat = est.latent(Zte)[1]
        if ms.family == "qsvm":
            rec.update({"form": est.form_, "psd_info": est.psd_info_})
        if ms.family == "vqc":
            rec.update({"final_train_loss": est.final_train_loss_, "nfev": est.nfev_,
                        "history": est.history_ if task.get("record_history") else None})
        folds_out.append(rec)
        if store_pred:
            arrays["fold"].append(np.full(len(yte), fold))
            arrays["index"].append(te_index)
            arrays["y"].append(yte)
            arrays["y_pred"].append(y_pred)
            arrays["score"].append(np.asarray(score, float) if score is not None else np.full(len(yte), np.nan))
            arrays["proba"].append(np.asarray(proba, float) if proba is not None else np.full(len(yte), np.nan))
            arrays["latent_var"].append(lat if lat is not None else np.full(len(yte), np.nan))
    summary = {"task": task, "folds": folds_out}
    arrays = {k: np.concatenate(v) for k, v in arrays.items() if v}
    return summary, arrays
