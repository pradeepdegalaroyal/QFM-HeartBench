"""Fold-wise preprocessing. Every transformer is fitted on the training fold only.

Modes
-----
``pca``   impute (mean / most frequent) -> one-hot categorical -> standardise
          -> PCA(k, whiten=True). Output: k unit-variance components, one per qubit.
``raw``   impute -> standardise all columns as numeric (categorical codes kept
          ordinal) — the "no PCA" quantum ablation, one qubit per raw feature.
``full``  impute -> one-hot -> standardise, no PCA (full-feature classical reference).

Quantum models multiply the output by a bandwidth lambda (rotation angle = lambda * z).
"""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(numeric: List[str], categorical: List[str], mode: str = "pca",
                      n_components: int = 6, seed: int = 0) -> Pipeline:
    if mode == "raw":
        cols = list(numeric) + list(categorical)
        return Pipeline([
            ("impute", ColumnTransformer([
                ("num", SimpleImputer(strategy="mean"), list(numeric)),
                ("cat", SimpleImputer(strategy="most_frequent"), list(categorical))])),
            ("scale", StandardScaler()),
        ]) if categorical else Pipeline([
            ("impute", ColumnTransformer([("num", SimpleImputer(strategy="mean"), cols)])),
            ("scale", StandardScaler()),
        ])
    steps = [("encode", ColumnTransformer([
        ("num", SimpleImputer(strategy="mean"), list(numeric)),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         list(categorical)),
    ])), ("scale", StandardScaler())]
    if mode == "pca":
        steps.append(("pca", PCA(n_components=n_components, whiten=True, random_state=seed)))
    elif mode != "full":
        raise ValueError(mode)
    return Pipeline(steps)


def retained_variance(pipe: Pipeline) -> float:
    pca = pipe.named_steps.get("pca")
    return float(np.sum(pca.explained_variance_ratio_)) if pca is not None else 1.0
