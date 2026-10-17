"""Classical baselines and their nested-CV hyper-parameter grids.

All baselines are tuned by an inner stratified 5-fold grid search
(balanced accuracy), mirroring the quantum models. GP baselines also optimise
kernel hyper-parameters by type-II maximum likelihood.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

CLASSICAL = ["LR", "SVM", "MLP", "GPC", "RF", "XGB", "LGBM"]


def make_classical(name: str, seed: int = 0, n_train: int = 0, class_weight="balanced"):
    """Return (estimator, param_grid). ``n_train`` only switches off costly options on 60k-row data."""
    big = n_train > 20000
    if name == "LR":
        return (LogisticRegression(class_weight=class_weight, max_iter=5000),
                {"C": [1e-3, 1e-2, 1e-1, 1, 10, 100]})
    if name == "SVM":
        return (SVC(kernel="rbf", class_weight=class_weight, probability=not big, random_state=seed,
                    cache_size=2000),
                {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1.0]})
    if name == "MLP":
        return (MLPClassifier(max_iter=1000, early_stopping=True, random_state=seed),
                {"hidden_layer_sizes": [(32,), (64, 32), (100, 50)], "alpha": [1e-4, 1e-2, 1.0],
                 "learning_rate_init": [1e-3, 1e-2]})
    if name == "GPC":
        kern = ConstantKernel(1.0, (1e-2, 1e4)) * RBF(1.0, (1e-2, 1e3))
        return (GaussianProcessClassifier(kernel=kern, n_restarts_optimizer=2, random_state=seed,
                                          max_iter_predict=200), {})
    if name == "RF":
        return (RandomForestClassifier(n_estimators=500, class_weight=class_weight, n_jobs=1,
                                       random_state=seed),
                {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 5, 10],
                 "max_features": ["sqrt", 0.5]})
    if name == "XGB":
        from xgboost import XGBClassifier
        return (XGBClassifier(subsample=0.8, colsample_bytree=0.8, n_jobs=1, random_state=seed,
                              tree_method="hist", eval_metric="logloss", verbosity=0),
                {"n_estimators": [100, 300], "max_depth": [2, 3, 5], "learning_rate": [0.03, 0.1]})
    if name == "LGBM":
        from lightgbm import LGBMClassifier
        return (LGBMClassifier(subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               class_weight=class_weight, n_jobs=1, random_state=seed, verbose=-1),
                {"num_leaves": [7, 15, 31], "learning_rate": [0.03, 0.1], "n_estimators": [100, 300],
                 "min_child_samples": [10, 20]})
    raise KeyError(name)
