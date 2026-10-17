"""Dataset loading, cleaning and cohort construction.

Raw data are the OpenML datasets cached as pickles in ``data/raw``
(``dataset_<id>.pkl`` with keys ``X`` (DataFrame) and ``y`` (binary labels)):

* 194   Cleveland heart disease (CHD), 303 records, target = (num > 0)
* 43672 "Heart Disease Dataset (Comprehensive)", 1190 records = Cleveland
        + Hungarian + Switzerland + VA Long Beach + Statlog. Contains 272 exact
        duplicates and *all* 303 Cleveland records (verified by exact feature
        match, labels agree 100 %). Zero-coded cholesterol / blood pressure.
* 45547 Cardiovascular disease (CVD), 70 000 records, balanced.

Cohorts used in the analysis
----------------------------
``chd``        Cleveland (303).
``hdc_ext``    HDC de-duplicated, minus Cleveland-derived rows (615): Hungary,
               Switzerland, VA. Independent of ``chd``.
``hdc_dedup``  HDC de-duplicated (918), includes Cleveland — supplementary cohort.
``cvd``        CVD after de-duplication and physiological plausibility filters.
``cvd_n{N}_s{seed}``   stratified random subsample of ``cvd``.
``cvd_imb{P}_n{N}_s{seed}``  subsample with P % positives (imbalance ablation).
``common_chd`` / ``common_hdc_ext``  the 11 features shared by both cohorts,
               harmonised, for external validation.
"""
from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get(
    "QFMB_DATA", os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw"))

COMMON = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
          "oldpeak", "slope"]
COMMON_CATEGORICAL = ["cp", "restecg", "slope"]


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    categorical: List[str]
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)  # sex (1=male), age (years)
    notes: Dict[str, object] = field(default_factory=dict)

    @property
    def numeric(self) -> List[str]:
        return [c for c in self.X.columns if c not in self.categorical]


def _raw(openml_id: int):
    with open(os.path.join(DATA_DIR, f"dataset_{openml_id}.pkl"), "rb") as f:
        d = pickle.load(f)
    return d["X"].copy(), np.asarray(d["y"]).astype(int)


def _cleveland_frame() -> pd.DataFrame:
    X, y = _raw(194)
    df = pd.DataFrame({
        "age": X["age"].astype(float), "sex": X["sex"].astype(int),
        "cp": X["cp"].astype(int), "trestbps": X["trestbps"].astype(float),
        "chol": X["chol"].astype(float), "fbs": X["fbs"].astype(int),
        "restecg": X["restecg"].astype(int), "thalach": X["thalach"].astype(float),
        "exang": X["exang"].astype(int), "oldpeak": X["oldpeak"].astype(float).round(1),
        "slope": X["slope"].astype(int), "ca": X["ca"].astype(float),
        "thal": pd.to_numeric(X["thal"].astype(object), errors="coerce"),
    })
    df["y"] = y
    return df


def _hdc_frame() -> pd.DataFrame:
    X, y = _raw(43672)
    df = pd.DataFrame({
        "age": X["age"].astype(float), "sex": X["sex"].astype(int),
        "cp": X["chest_pain_type"].astype(int), "trestbps": X["resting_bp_s"].astype(float),
        "chol": X["cholesterol"].astype(float), "fbs": X["fasting_blood_sugar"].astype(int),
        "restecg": X["resting_ecg"].astype(int), "thalach": X["max_heart_rate"].astype(float),
        "exang": X["exercise_angina"].astype(int), "oldpeak": X["oldpeak"].astype(float).round(1),
        "slope": X["ST_slope"].astype(int),
    })
    df["y"] = y
    return df


@lru_cache(maxsize=None)
def _hdc_split():
    """Return (dedup frame with 'cleveland' flag, audit dict)."""
    hdc = _hdc_frame()
    chd = _cleveland_frame()
    n_raw = len(hdc)
    dedup = hdc.drop_duplicates().reset_index(drop=True)
    key = chd.drop_duplicates(COMMON)[COMMON + ["y"]].rename(columns={"y": "y_chd"})
    m = dedup.merge(key, on=COMMON, how="left", indicator=True)
    dedup["cleveland"] = (m["_merge"] == "both").values
    audit = {
        "hdc_raw": n_raw, "hdc_exact_duplicates": int(n_raw - len(dedup)),
        "hdc_dedup": len(dedup), "hdc_cleveland_rows": int(dedup["cleveland"].sum()),
        "hdc_label_agreement_with_cleveland": float(
            (m.loc[m["_merge"] == "both", "y"] == m.loc[m["_merge"] == "both", "y_chd"]).mean()),
        "hdc_chol_zero": int((dedup["chol"] == 0).sum()),
        "hdc_bp_zero": int((dedup["trestbps"] == 0).sum()),
        "hdc_slope_zero": int((dedup["slope"] == 0).sum()),
    }
    return dedup, audit


def _clean_hdc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["chol"] == 0, "chol"] = np.nan        # zero-coded missing
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan
    df.loc[df["slope"] == 0, "slope"] = np.nan      # invalid code (valid: 1-3)
    return df


def _meta(df: pd.DataFrame, sex_col="sex", age_col="age", male_value=1) -> pd.DataFrame:
    return pd.DataFrame({"sex_male": (df[sex_col].astype(float) == male_value).astype(int).values,
                         "age": df[age_col].astype(float).values})


def load_chd() -> Dataset:
    df = _cleveland_frame()
    X = df.drop(columns="y")
    return Dataset("chd", X, df["y"].to_numpy(), ["cp", "restecg", "slope", "thal"], _meta(df),
                   {"source": "OpenML 194 (Cleveland)", "n": len(df)})


def load_hdc(kind: str = "ext") -> Dataset:
    dedup, audit = _hdc_split()
    df = dedup[~dedup["cleveland"]] if kind == "ext" else dedup
    df = _clean_hdc(df.drop(columns="cleveland")).reset_index(drop=True)
    X = df.drop(columns="y")
    return Dataset(f"hdc_{kind}", X, df["y"].to_numpy(), COMMON_CATEGORICAL, _meta(df),
                   {"source": "OpenML 43672", **audit, "n": len(df)})


@lru_cache(maxsize=None)
def _cvd_clean():
    X, y = _raw(45547)
    df = pd.DataFrame({
        "age": X["age"].astype(float) / 365.25,
        "gender": X["gender"].astype(int),        # 1 = women, 2 = men
        "height": X["height"].astype(float), "weight": X["weight"].astype(float),
        "ap_hi": X["ap_hi"].astype(float), "ap_lo": X["ap_lo"].astype(float),
        "cholesterol": X["cholesterol"].astype(int), "gluc": X["gluc"].astype(int),
        "smoke": X["smoke"].astype(int), "alco": X["alco"].astype(int),
        "active": X["active"].astype(int),
    })
    df["y"] = y
    n0 = len(df)
    df = df.drop_duplicates()
    n_dup = n0 - len(df)
    ok = (df.ap_hi.between(80, 220) & df.ap_lo.between(40, 160) & (df.ap_lo < df.ap_hi)
          & df.height.between(120, 220) & df.weight.between(30, 200))
    audit = {"cvd_raw": n0, "cvd_duplicates": int(n_dup), "cvd_implausible_removed": int((~ok).sum()),
             "cvd_clean": int(ok.sum()), "cvd_positive_rate_raw": float(np.mean(y))}
    return df[ok].reset_index(drop=True), audit


def _subsample(y: np.ndarray, n: int, seed: int, pos_rate: Optional[float] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx0, idx1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    p = y.mean() if pos_rate is None else pos_rate
    n1 = int(round(n * p))
    take = np.concatenate([rng.choice(idx0, n - n1, replace=False), rng.choice(idx1, n1, replace=False)])
    return np.sort(take)


def load_cvd(n: Optional[int] = None, seed: int = 0, pos_rate: Optional[float] = None) -> Dataset:
    df, audit = _cvd_clean()
    name = "cvd"
    if n is not None and n < len(df):
        df = df.iloc[_subsample(df["y"].to_numpy(), n, seed, pos_rate)].reset_index(drop=True)
        name = f"cvd_n{n}_s{seed}" if pos_rate is None else f"cvd_imb{int(pos_rate*100)}_n{n}_s{seed}"
    X = df.drop(columns="y")
    return Dataset(name, X, df["y"].to_numpy(), ["cholesterol", "gluc"],
                   _meta(df, "gender", "age", male_value=2),
                   {"source": "OpenML 45547", **audit, "n": len(df)})


def load_common(cohort: str) -> Dataset:
    """11 harmonised features shared by Cleveland and HDC-ext (external validation)."""
    if cohort == "chd":
        df = _cleveland_frame()[COMMON + ["y"]]
    else:
        dedup, _ = _hdc_split()
        df = _clean_hdc(dedup[~dedup["cleveland"]].drop(columns="cleveland"))
    df = df.reset_index(drop=True)
    return Dataset(f"common_{cohort}", df[COMMON], df["y"].to_numpy(), COMMON_CATEGORICAL, _meta(df),
                   {"n": len(df)})


def load(name: str) -> Dataset:
    if name == "chd":
        return load_chd()
    if name in ("hdc_ext", "hdc_dedup"):
        return load_hdc(name.split("_")[1])
    if name == "cvd":
        return load_cvd()
    m = re.fullmatch(r"cvd_n(\d+)_s(\d+)", name)
    if m:
        return load_cvd(int(m.group(1)), int(m.group(2)))
    m = re.fullmatch(r"cvd_imb(\d+)_n(\d+)_s(\d+)", name)
    if m:
        return load_cvd(int(m.group(2)), int(m.group(3)), int(m.group(1)) / 100)
    m = re.fullmatch(r"common_(chd|hdc_ext)", name)
    if m:
        return load_common(m.group(1))
    raise KeyError(name)


def audit() -> Dict[str, object]:
    out = dict(_hdc_split()[1])
    out.update(_cvd_clean()[1])
    for nm in ("chd", "hdc_ext", "hdc_dedup", "cvd"):
        d = load(nm)
        out[f"{nm}_n"] = len(d.y)
        out[f"{nm}_pos_rate"] = float(d.y.mean())
        out[f"{nm}_n_features_raw"] = d.X.shape[1]
        out[f"{nm}_missing_cells"] = int(d.X.isna().sum().sum())
    return out
