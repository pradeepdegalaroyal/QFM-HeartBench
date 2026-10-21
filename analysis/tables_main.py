"""Main benchmark: summary tables, pre-specified comparisons, confusion matrices.

Outputs (results/tables/):
  main_<dataset>.csv / .tex   mean and corrected 95 % CI per model and metric
  comparisons_main.csv / .tex every quantum configuration vs (i) the strongest tuned
                              classical model on the same PCA-6 inputs and (ii) its
                              paradigm-matched classical model; corrected resampled
                              t-test, Wilcoxon, Holm and BH over all comparisons
  confusion_per_fold.csv      TP/TN/FP/FN for every model, repeat and fold
  hyperparameters_main.csv    selection frequency of bandwidth / C
  retained_variance.csv       PCA variance retained (fold-wise)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collect import DATASET_LABEL, ensure_out, family, fold_table  # noqa: E402
from qfmb.stats import compare, mean_ci  # noqa: E402

METRICS = ["balanced_accuracy", "auc", "sensitivity", "specificity", "f1", "mcc", "brier", "ece"]
MATCHED = {"QSVM": "SVM", "QGPC": "GPC", "VQC": "MLP"}
PRIMARY = "balanced_accuracy"


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, label), g in df.groupby(["dataset", "label"]):
        n_tr, n_te = g["n_train"].mean(), g["n_test"].mean()
        row = {"dataset": ds, "model": label, "family": family(label), "folds": len(g),
               "fit_s_per_fold": g["fit_seconds"].mean(), "tune_s_per_fold": g["tune_seconds"].mean()}
        for m in METRICS:
            mu, lo, hi = mean_ci(g[m].to_numpy(float), n_tr, n_te)
            row.update({m: mu, f"{m}_lo": lo, f"{m}_hi": hi})
        row["collapsed_folds"] = int(g["collapsed"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def comparisons(df: pd.DataFrame, summary: pd.DataFrame, metric: str = PRIMARY) -> pd.DataFrame:
    out = []
    for ds, g in df.groupby("dataset"):
        key = g.assign(k=list(zip(g["repeat"], g["fold"])))
        wide = key.pivot_table(index="k", columns="label", values=metric)
        wide = wide.dropna(axis=0, how="any")            # folds finished by every model only
        s = summary[summary.dataset == ds]
        classical = s[s.family == "Classical"].sort_values(metric, ascending=False)
        if classical.empty or wide.empty:
            continue
        best = classical.iloc[0]["model"]
        pairs = []
        for lab in sorted(wide.columns):
            fam = family(lab)
            if fam in MATCHED:
                pairs.append((lab, best))
                if MATCHED[fam] != best and MATCHED[fam] in wide.columns:
                    pairs.append((lab, MATCHED[fam]))
        per_fold = {c: wide[c].to_numpy() for c in wide.columns}
        n_tr, n_te = g["n_train"].mean(), g["n_test"].mean()
        for r in compare(per_fold, pairs, n_tr, n_te):
            r.update({"dataset": ds, "metric": metric, "J_folds": len(wide),
                      "reference_type": "strongest classical" if r["reference"] == best else "paradigm-matched"})
            out.append(r)
    res = pd.DataFrame(out)
    if res.empty:
        return res
    from qfmb.stats import benjamini_hochberg, holm
    for key in ("p", "p_wilcoxon"):              # global family over all datasets
        res[f"{key}_holm_global"] = holm(res[key])
        res[f"{key}_bh_global"] = benjamini_hochberg(res[key])
    return res


def fmt_ci(r, m, scale=1.0, nd=3):
    if pd.isna(r[m]):
        return "--"
    return f"{r[m]*scale:.{nd}f} [{r[m+'_lo']*scale:.{nd}f}, {r[m+'_hi']*scale:.{nd}f}]"


def latex_main(s: pd.DataFrame, ds: str) -> str:
    s = s[s.dataset == ds].copy()
    order = {"QSVM": 0, "VQC": 1, "QGPC": 2, "Classical": 3, "Classical (all features)": 4}
    s["o"] = s["family"].map(order)
    s = s.sort_values(["o", "model"])
    best_cl = s[s.family == "Classical"][PRIMARY].max()
    lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
             r"Model & Balanced accuracy [95\% CI] & AUC-ROC & Sensitivity & Specificity & MCC & Brier score & Fit time (s) \\", r"\midrule"]
    last = None
    for _, r in s.iterrows():
        if last is not None and r["family"] != last:
            lines.append(r"\midrule")
        last = r["family"]
        bal = fmt_ci(r, PRIMARY)
        if r["family"] in ("QSVM", "VQC", "QGPC") and r[PRIMARY] > best_cl:
            bal = r"\textbf{" + bal + "}"
        brier = "--" if pd.isna(r["brier"]) else f"{r['brier']:.3f}"
        name = r["model"]
        name = name.replace("VQE", "ZZ (VQE)")
        lines.append(f"{name} & {bal} & {r['auc']:.3f} & {r['sensitivity']:.3f} & {r['specificity']:.3f} & "
                     f"{r['mcc']:.3f} & {brier} & {fit_s(r['fit_s_per_fold'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def fit_s(x):
    """Fit time: two decimals below 10 s so sub-second fits do not print as 0.0."""
    return f"{x:.2f}" if x < 10 else f"{x:.1f}"


def latex_comparisons(c: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Dataset & Quantum model vs.\ reference & $\Delta$ bal. acc. (pp) [95\% CI] & $d_z$ & $p_{\mathrm{corr}}$ & "
             r"$p_{\mathrm{Holm}}$ & $p_{\mathrm{BH}}$ & $p_{\mathrm{Wilcoxon}}$ \\", r"\midrule"]
    for _, r in c.sort_values(["dataset", "reference_type", "mean_diff"], ascending=[True, False, False]).iterrows():
        lines.append(f"{r['dataset']} & {r['model']} vs.\\ {r['reference']} & "
                     f"{100*r['mean_diff']:+.1f} [{100*r['ci_low']:+.1f}, {100*r['ci_high']:+.1f}] & {r['d_z']:.2f} & "
                     f"{r['p']:.3f} & {r['p_holm_global']:.3f} & {r['p_bh_global']:.3f} & {r['p_wilcoxon']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    df = fold_table("main")
    if df.empty:
        print("no main results yet")
        return
    tdir = ensure_out("tables")
    s = summarise(df)
    s.to_csv(os.path.join(tdir, "main_summary.csv"), index=False)
    for ds in s.dataset.unique():
        with open(os.path.join(tdir, f"main_{ds}.tex"), "w") as f:
            f.write(latex_main(s, ds))
    c = comparisons(df, s)
    if not c.empty:
        c.to_csv(os.path.join(tdir, "comparisons_main.csv"), index=False)
        with open(os.path.join(tdir, "comparisons_main.tex"), "w") as f:
            f.write(latex_comparisons(c))
    df[["dataset", "label", "repeat", "fold", "n_test", "tp", "tn", "fp", "fn", "sensitivity", "specificity",
        "balanced_accuracy", "accuracy"]].sort_values(["dataset", "label", "repeat", "fold"]).to_csv(
        os.path.join(tdir, "confusion_per_fold.csv"), index=False)
    hp_cols = [c for c in df.columns if c.startswith("hp_") and c != "hp_amplitude"]
    if hp_cols:
        melted = df.melt(id_vars=["dataset", "label"], value_vars=hp_cols).dropna()
        melted["value"] = melted["value"].apply(lambda v: str(v) if isinstance(v, (list, tuple)) else v)
        freq = (melted.groupby(["dataset", "label", "variable", "value"]).size()
                .rename("count").reset_index())
        freq.to_csv(os.path.join(tdir, "hyperparameters_main.csv"), index=False)
    (df.groupby("dataset")["retained_variance"].agg(["mean", "std", "min", "max"])
       .to_csv(os.path.join(tdir, "retained_variance.csv")))
    print(s.sort_values(["dataset", PRIMARY], ascending=[True, False])
          [["dataset", "model", "folds", PRIMARY, f"{PRIMARY}_lo", f"{PRIMARY}_hi", "auc"]].to_string(index=False))
    if not c.empty:
        print(c[["dataset", "model", "reference", "mean_diff", "ci_low", "ci_high", "p", "p_holm_global",
                 "p_wilcoxon"]].to_string(index=False))


if __name__ == "__main__":
    main()
