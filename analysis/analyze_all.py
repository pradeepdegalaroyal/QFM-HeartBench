"""Analyse every experiment suite and write tables + a key-numbers JSON.

Usage:
  QFMB_RUNS=<path to runs> python analysis/analyze_all.py <out_dir>

Outputs in <out_dir>:
  tables/*.csv, tables/*.tex      tables used in the manuscript and the results report
  key_numbers.json                every number quoted in the manuscript text
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import collect  # noqa: E402
from collect import RUNS, family, fold_table, load_json  # noqa: E402
from qfmb import data as data_mod  # noqa: E402
from qfmb.stats import benjamini_hochberg, compare, holm, mean_ci  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
import tables_main  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results")
T = os.path.join(OUT, "tables")
os.makedirs(T, exist_ok=True)
KEY: dict = {}
P = "balanced_accuracy"
MAPS = ["RX", "Pauli-Z", "Hadamard", "Strongly Entangling", "VQE", "Angle Encoding"]


def save(df: pd.DataFrame, name: str):
    df.to_csv(os.path.join(T, name + ".csv"), index=False)


def mean_by(df, keys, metric=P):
    return df.groupby(keys)[metric].agg(["mean", "std", "count"]).reset_index()


# ---------------------------------------------------------------- 1 main + supp
def main_and_supp():
    for suite in ("main", "supp"):
        df = fold_table(suite)
        s = tables_main.summarise(df)
        c = tables_main.comparisons(df, s)
        tag = "" if suite == "main" else "_supp"
        s.to_csv(os.path.join(T, f"main_summary{tag}.csv"), index=False)
        c.to_csv(os.path.join(T, f"comparisons{tag}.csv"), index=False)
        for ds in s.dataset.unique():
            with open(os.path.join(T, f"main_{ds}.tex"), "w") as f:
                f.write(tables_main.latex_main(s, ds))
        if suite == "main":
            with open(os.path.join(T, "comparisons_main.tex"), "w") as f:
                f.write(tables_main.latex_comparisons(c))
            df[["dataset", "label", "repeat", "fold", "n_train", "n_test", "tp", "tn", "fp", "fn", "sensitivity",
                "specificity", "balanced_accuracy", "accuracy"]].sort_values(
                ["dataset", "label", "repeat", "fold"]).to_csv(os.path.join(T, "confusion_per_fold.csv"), index=False)
            pca = df[~df.label.str.contains("all features")]          # PCA pipelines only
            rv = pca.groupby("dataset")["retained_variance"].agg(["mean", "std", "min", "max"]).reset_index()
            save(rv, "retained_variance")
            with open(os.path.join(T, "retained_variance.tex"), "w") as f:
                f.write("\\begin{tabular}{lcccc}\n\\toprule\nCohort & Mean & SD & Min & Max \\\\\n\\midrule\n")
                for _, r in rv.iterrows():
                    f.write(f"{r['dataset']} & {100*r['mean']:.1f}\\% & {100*r['std']:.1f} & {100*r['min']:.1f}\\% & {100*r['max']:.1f}\\% \\\\\n")
                f.write("\\bottomrule\n\\end{tabular}\n")
            KEY["retained_variance"] = {r["dataset"]: float(r["mean"]) for _, r in rv.iterrows()}
            # hyper-parameter selection frequency for quantum bandwidth
            hp = df[df.family.isin(["QSVM", "VQC", "QGPC"])].groupby(["dataset", "label", "hp_bandwidth"]).size()
            hp = hp.rename("count").reset_index()
            save(hp, "bandwidth_selection")
            # runtime
            rt = df.groupby(["dataset", "family"])[["fit_seconds", "tune_seconds"]].median().reset_index()
            save(rt, "runtime_median")
            KEY["runtime_median_fit_s"] = {f"{r['dataset']}|{r['family']}": float(r["fit_seconds"])
                                           for _, r in rt.iterrows()}
            KEY["collapsed_folds"] = {f"{r['dataset']}|{r['model']}": int(r["collapsed_folds"])
                                      for _, r in s.iterrows() if r["collapsed_folds"] > 0}
        # key numbers per dataset
        k = {}
        for ds, g in s.groupby("dataset"):
            g = g.sort_values(P, ascending=False)
            best = g.iloc[0]
            cl = g[g.family == "Classical"].iloc[0]
            q = g[g.family.isin(["QSVM", "VQC", "QGPC"])]
            bq = q.iloc[0]
            k[ds] = {
                "n_models": int(len(g)), "folds": int(best["folds"]),
                "best_overall": [best["model"], float(best[P]), float(best[P + "_lo"]), float(best[P + "_hi"])],
                "best_classical_pca6": [cl["model"], float(cl[P]), float(cl[P + "_lo"]), float(cl[P + "_hi"])],
                "best_quantum": [bq["model"], float(bq[P]), float(bq[P + "_lo"]), float(bq[P + "_hi"])],
                "best_per_family": {fam: [r["model"], float(r[P]), float(r["auc"])]
                                    for fam, gg in g.groupby("family") for r in [gg.iloc[0]]},
                "worst_quantum": [q.iloc[-1]["model"], float(q.iloc[-1][P])],
                "n_quantum_above_best_classical_mean": int((q[P] > cl[P]).sum()),
                "vqc_range": [float(q[q.family == "VQC"][P].min()), float(q[q.family == "VQC"][P].max())],
            }
            cc = c[c.dataset == ds] if not c.empty else c
            if not cc.empty:
                k[ds]["comparisons"] = int(len(cc))
                k[ds]["sig_holm_pos"] = int(((cc.p_holm_global < 0.05) & (cc.mean_diff > 0)).sum())
                k[ds]["sig_holm_neg"] = int(((cc.p_holm_global < 0.05) & (cc.mean_diff < 0)).sum())
                k[ds]["sig_bh_pos"] = int(((cc.p_bh_global < 0.05) & (cc.mean_diff > 0)).sum())
                k[ds]["sig_bh_neg"] = int(((cc.p_bh_global < 0.05) & (cc.mean_diff < 0)).sum())
                k[ds]["sig_unadjusted_pos"] = int(((cc.p < 0.05) & (cc.mean_diff > 0)).sum())
                k[ds]["max_positive_diff"] = cc.sort_values("mean_diff").iloc[-1][
                    ["model", "reference", "mean_diff", "ci_low", "ci_high", "p", "p_holm_global"]].tolist()
        KEY["main" if suite == "main" else "supp"] = k
        if suite == "main" and not c.empty:
            KEY["comparisons_total"] = int(len(c))
            KEY["comparisons_sig_holm_pos"] = int(((c.p_holm_global < 0.05) & (c.mean_diff > 0)).sum())
            KEY["comparisons_sig_holm_neg"] = int(((c.p_holm_global < 0.05) & (c.mean_diff < 0)).sum())
            KEY["comparisons_sig_bh_pos"] = int(((c.p_bh_global < 0.05) & (c.mean_diff > 0)).sum())
            KEY["comparisons_unadjusted_pos"] = int(((c.p < 0.05) & (c.mean_diff > 0)).sum())
            # naive paired t vs corrected: how many comparisons would look significant
            KEY["comparisons_naive_t_sig_pos"] = int(((c.p_naive_paired_t < 0.05) & (c.mean_diff > 0)).sum())


# ---------------------------------------------------------------- 2 external validation
def external():
    df = fold_table("external")
    g = df.groupby(["dataset", "test_dataset", "label", "family"])[[P, "auc", "sensitivity", "specificity"]].mean()
    g = g.reset_index()
    save(g, "external_validation")
    k = {}
    for (tr, te), gg in g.groupby(["dataset", "test_dataset"]):
        gg = gg.sort_values(P, ascending=False)
        q = gg[gg.family.isin(["QSVM", "VQC", "QGPC"])]
        cl = gg[gg.family == "Classical"]
        k[f"{tr}->{te}"] = {"best": [gg.iloc[0]["label"], float(gg.iloc[0][P])],
                            "best_quantum": [q.iloc[0]["label"], float(q.iloc[0][P])],
                            "best_classical": [cl.iloc[0]["label"], float(cl.iloc[0][P])],
                            "range": [float(gg[P].min()), float(gg[P].max())]}
    KEY["external"] = k
    piv = g.pivot_table(index="label", columns="dataset", values=P)
    with open(os.path.join(T, "external.tex"), "w") as f:
        f.write("\\begin{tabular}{lcc}\n\\toprule\nModel & CHD$\\rightarrow$HDC-ext & HDC-ext$\\rightarrow$CHD \\\\\n\\midrule\n")
        order = [m for fam in ("QSVM", "VQC", "QGPC") for m in [f"{fam}-{x}" for x in MAPS]] + \
                ["LR", "SVM", "MLP", "GPC", "RF", "XGB", "LGBM"]
        for m in order:
            if m in piv.index:
                a, b = piv.loc[m].get("common_chd", np.nan), piv.loc[m].get("common_hdc_ext", np.nan)
                f.write(f"{m.replace('VQE', 'ZZ (VQE)')} & {a:.3f} & {b:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------------------------------- 3 sample size
def samplesize():
    df = fold_table("samplesize")
    df["N"] = df["variant"].str.replace("N", "").astype(int)
    g = df.groupby(["N", "label", "family"]).agg(bal=(P, "mean"), sd=(P, "std"), n_train=("n_train", "max"))
    g = g.reset_index()
    save(g, "samplesize")
    fam = df.groupby(["N", "family"])[P].mean().unstack()
    save(fam.reset_index(), "samplesize_family_mean")
    best = g.loc[g.groupby(["N", "family"])["bal"].idxmax()]
    save(best, "samplesize_best_per_family")
    KEY["samplesize"] = {str(n): {r["family"]: [r["label"], float(r["bal"])] for _, r in gg.iterrows()}
                         for n, gg in best.groupby("N")}
    gpc = df[(df.label == "GPC")].groupby("N")["n_train"].max()
    KEY["samplesize_gpc_ntrain"] = {str(k): int(v) for k, v in gpc.items()}


# ---------------------------------------------------------------- 4 ablation
def ablation():
    df = fold_table("ablation")
    g = df.groupby(["dataset", "variant", "label", "family"])[P].mean().reset_index()
    ref = g[g.variant == "reference"][["dataset", "label", P]].rename(columns={P: "ref"})
    m = g.merge(ref, on=["dataset", "label"], how="left")
    m["delta"] = m[P] - m["ref"]
    save(m, "ablation")
    k = {}
    for var in ("noPCA", "k2", "k4", "k8", "k10", "weighting_flipped"):
        mm = m[m.variant == var]
        k[var] = {fam: float(x) for fam, x in mm.groupby("family")["delta"].mean().items()}
    # bandwidth sweep: mean balanced accuracy per family and lambda (averaged over maps and datasets)
    bw = m[m.variant.str.startswith("bw")].copy()
    bw["lambda"] = bw.variant.str[2:].astype(float)
    bwt = bw.groupby(["family", "lambda"])[P].mean().unstack()
    save(bwt.reset_index(), "ablation_bandwidth_family")
    bwm = bw.groupby(["label", "lambda"])[P].mean().unstack()
    save(bwm.reset_index(), "ablation_bandwidth_map")
    k["bandwidth_best_lambda"] = {lab: float(row.idxmax()) for lab, row in bwm.iterrows()}
    k["bandwidth_spread"] = {lab: float(row.max() - row.min()) for lab, row in bwm.iterrows()}
    # imbalanced CVD variant: reference sensitivity of each family
    imb = df[(df.dataset == "cvd_imb25_n2000_s0") & (df.variant.isin(["reference", "weighting_flipped"]))]
    imbt = imb.groupby(["variant", "label", "family"])[["sensitivity", "specificity", P]].mean().reset_index()
    save(imbt, "ablation_imbalance")
    KEY["ablation"] = k


# ---------------------------------------------------------------- 5 sensitivity
def sensitivity():
    df = fold_table("sens")
    df["param"] = df.variant.str.extract(r"^([a-z]+)")[0].replace({"ent": "entanglement"})
    df["value"] = df.variant.str.replace(r"^(layers|ent-|shots|spsa)", "", regex=True)
    g = df.groupby(["param", "value", "family"])[P].mean().reset_index()
    save(g, "sensitivity_family")
    gm = df.groupby(["param", "value", "label"])[P].mean().reset_index()
    save(gm, "sensitivity_map")
    k = {}
    for prm, gg in g.groupby("param"):
        k[prm] = {fam: {v: float(x) for v, x in zip(ggg["value"], ggg[P])} for fam, ggg in gg.groupby("family")}
    sh = df[df.param == "shots"]
    if "psd_n_neg_eig" in sh.columns:
        neg = sh.groupby(["value"]).apply(lambda x: float((x["psd_n_neg_eig"].fillna(0) > 0).mean()))
        k["shots_frac_indefinite"] = {v: float(x) for v, x in neg.items()}
        mm = sh.groupby(["value"])["psd_neg_mass"].mean()
        k["shots_mean_neg_mass"] = {v: float(x) for v, x in mm.items()}
    KEY["sens"] = k


# ---------------------------------------------------------------- 6 noise
def noise():
    rows = []
    for r in load_json("noise"):
        t = r["task"]
        for name, f in r["models"].items():
            rows.append({"dataset": t["dataset"], "level": t["level"], "model": f"{name.upper()}-{t['feature_map']}",
                         "family": name.upper(), "map": t["feature_map"], "fold": t["fold"],
                         P: f["metrics"][P], "auc": f["metrics"]["auc"], "cx": f.get("cx"),
                         "kernel_mae": f.get("kernel_mae_vs_exact"), "mean_diag": f.get("mean_diag"),
                         "kta_noisy": f.get("kta_noisy"), "kta_exact": f.get("kta_exact"),
                         "collapsed": f["metrics"]["collapsed"]})
    df = pd.DataFrame(rows)
    save(df, "noise_folds")
    g = df.groupby(["dataset", "family", "map", "level"])[[P, "kernel_mae", "mean_diag", "cx", "kta_noisy",
                                                         "kta_exact"]].mean().reset_index()
    save(g, "noise")
    lv = ["shots", "low", "medium", "high"]
    fam = df.groupby(["family", "level"])[P].mean().unstack()[lv]
    save(fam.reset_index(), "noise_family")
    KEY["noise"] = {
        "family_mean": {f: {l: float(fam.loc[f, l]) for l in lv} for f in fam.index},
        "mean_diag_by_map_level": {f"{m}|{l}": float(x) for (m, l), x in
                                   df[df.family != "VQC"].groupby(["map", "level"])["mean_diag"].mean().items()},
        "cx_by_map": {m: float(x) for m, x in df[df.family != "VQC"].groupby("map")["cx"].mean().items()},
        "drop_shots_to_high": {f"{fa}|{m}": float(a - b) for (fa, m), gg in df.groupby(["family", "map"])
                               for a, b in [(gg[gg.level == "shots"][P].mean(), gg[gg.level == "high"][P].mean())]},
        "collapsed_frac_by_level": {l: float(x) for l, x in df.groupby("level")["collapsed"].mean().items()},
    }
    tab = df.groupby(["family", "map", "level"])[P].mean().unstack()[lv].reset_index()
    with open(os.path.join(T, "noise.tex"), "w") as f:
        f.write("\\begin{tabular}{llcccc}\n\\toprule\nModel & Map & Shots only & Low & Medium & High \\\\\n\\midrule\n")
        for _, r in tab.iterrows():
            f.write(f"{r['family']} & {r['map'].replace('VQE', 'ZZ (VQE)')} & " +
                    " & ".join(f"{r[l]:.3f}" for l in lv) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    md = df[df.family == "QSVM"].groupby(["map", "level"])[["mean_diag", "kernel_mae", "cx"]].mean().reset_index()
    save(md, "noise_kernel_quality")


# ---------------------------------------------------------------- 7 diagnostics
def diagnostics(main_summary: pd.DataFrame):
    rows, pairs = [], []
    for r in load_json("diag"):
        ds = r["task"]["dataset"]
        for row in r["rows"]:
            rows.append({"dataset": ds, "repeat": r["task"]["repeat"], **row})
        for p in r["pairs"]:
            pairs.append({"dataset": ds, **p})
    d = pd.DataFrame(rows)
    pr = pd.DataFrame(pairs)
    num = [c for c in d.columns if c not in ("dataset", "repeat", "model", "bandwidth")]
    agg = d.groupby(["dataset", "model", "bandwidth"], dropna=False)[num].mean().reset_index()
    save(agg, "diagnostics")
    prag = pr.groupby(["dataset", "bandwidth", "map_a", "map_b"])[["alignment", "max_abs_diff"]].mean().reset_index()
    save(prag, "diagnostics_pairs")
    # link to main: use, per dataset and map, the bandwidth most often selected for QSVM in the main suite
    bwsel = pd.read_csv(os.path.join(T, "bandwidth_selection.csv"))
    k = {"invariance_max_change": float(d["max_abs_kernel_change_without_trailing"].max()),
         "rx_pz_alignment": {f"{ds}|{bw}": float(x) for (ds, bw), x in
                             prag[(prag.map_a == "RX") & (prag.map_b == "Pauli-Z")].set_index(
                                 ["dataset", "bandwidth"])["alignment"].items()}}
    link = []
    dsmap = {"chd": "chd", "hdc_ext": "hdc_ext", "cvd_n2000_s0": "cvd"}
    for dds, mds in dsmap.items():
        for m in MAPS:
            for fam in ("QSVM", "QGPC"):
                lab = f"{fam}-{m}"
                sel = bwsel[(bwsel.dataset == mds) & (bwsel.label == lab)]
                if sel.empty:
                    continue
                # most often selected; ties go to the larger bandwidth
                lam = float(sel.sort_values(["count", "hp_bandwidth"], ascending=False).iloc[0]["hp_bandwidth"])
                row = agg[(agg.dataset == dds) & (agg.model == m) & (np.isclose(agg.bandwidth.astype(float), lam))]
                perf = main_summary[(main_summary.dataset == mds) & (main_summary.model == lab)]
                if row.empty or perf.empty:
                    continue
                link.append({"dataset": mds, "label": lab, "family": fam, "map": m, "lambda": lam,
                             "kta": float(row.kta.iloc[0]), "offdiag_mean": float(row.offdiag_mean.iloc[0]),
                             "eff_dim": float(row.eff_dim_participation.iloc[0]),
                             "expressibility_kl": float(row.expressibility_kl.iloc[0]),
                             "geometric_difference": float(row.geometric_difference_vs_rbf.iloc[0]),
                             "meyer_wallach": float(row.meyer_wallach.iloc[0]),
                             P: float(perf[P].iloc[0])})
    lk = pd.DataFrame(link)
    save(lk, "diagnostics_vs_performance")
    from scipy.stats import spearmanr
    corr = {}
    for col in ("kta", "offdiag_mean", "eff_dim", "expressibility_kl", "geometric_difference", "meyer_wallach"):
        rho, p = spearmanr(lk[col], lk[P])
        corr[col] = [float(rho), float(p)]
    k["spearman_vs_balacc"] = corr
    # within-cohort correlations (the pooled value mixes cohorts with different accuracy levels);
    # each cohort has 6 distinct kernels shared by QSVM and QGPC (12 points, 6 independent kernels)
    within = {}
    for ds_, gg in lk.groupby("dataset"):
        within[ds_] = {col: [float(x) for x in spearmanr(gg[col], gg[P])]
                       for col in ("kta", "offdiag_mean", "eff_dim", "expressibility_kl",
                                   "geometric_difference", "meyer_wallach")}
    k["spearman_within_cohort"] = within
    k["n_points"] = int(len(lk))
    # concentration of the ZZ map across bandwidths
    zz = agg[(agg.model == "VQE")].groupby("bandwidth")["offdiag_mean"].mean()
    k["zz_offdiag_by_lambda"] = {str(b): float(x) for b, x in zz.items()}
    rbf = agg[agg.model.str.startswith("RBF")].groupby("dataset")["kta"].mean()
    k["rbf_kta"] = {d_: float(x) for d_, x in rbf.items()}
    KEY["diag"] = k


# ---------------------------------------------------------------- 8 gradient variance + VQC training
def trainability():
    rows = []
    for r in load_json("gradvar"):
        rows += r["rows"]
    gv = pd.DataFrame(rows)
    save(gv, "gradvar")
    k = {}
    for L in sorted(gv.layers.unique()):
        g = gv[gv.layers == L].groupby("n_qubits")["var_local_first"].mean()
        h = gv[gv.layers == L].groupby("n_qubits")["var_global_first"].mean()
        k[f"local_L{L}"] = {str(n): float(x) for n, x in g.items()}
        k[f"global_L{L}"] = {str(n): float(x) for n, x in h.items()}
    KEY["gradvar"] = k
    df = fold_table("train")
    g = df.groupby(["dataset", "variant", "label"]).agg(bal=(P, "mean"), sd=(P, "std"),
                                                        loss=("final_train_loss", "mean"),
                                                        loss_sd=("final_train_loss", "std"),
                                                        collapsed=("collapsed", "mean")).reset_index()
    save(g, "vqc_training")
    v = df.groupby("variant").agg(bal=(P, "mean"), loss=("final_train_loss", "mean"),
                                  collapsed=("collapsed", "mean")).reset_index()
    save(v, "vqc_optimizer")
    # seed-to-seed spread of final loss per (dataset, map, optimizer)
    spread = df.groupby(["dataset", "label", "variant"])["final_train_loss"].std().groupby("variant").mean()
    KEY["train"] = {"by_optimizer": {r["variant"]: {"bal": float(r["bal"]), "loss": float(r["loss"]),
                                                    "collapsed_frac": float(r["collapsed"])} for _, r in v.iterrows()},
                    "final_loss_sd_across_seeds": {k_: float(x) for k_, x in spread.items()},
                    "by_map_spsa": {r["label"]: float(r["bal"]) for _, r in
                                    df[df.variant == "spsa"].groupby("label")[P].mean().reset_index().rename(
                                        columns={P: "bal"}).iterrows()}}
    # loss curves: mean over seeds/folds per (dataset, map, optimizer)
    curves = defaultdict(list)
    for r in load_json("train"):
        t = r["task"]
        for f in r["folds"]:
            h = f.get("history") or {}
            if h.get("loss"):
                curves[(t["dataset"], t["label"], t["variant"])].append((h["iter"], h["loss"]))
    out = []
    for (ds, lab, var), lst in curves.items():
        n = min(len(x[0]) for x in lst)
        L = np.array([x[1][:n] for x in lst])
        it = lst[0][0][:n]
        for i in range(n):
            out.append({"dataset": ds, "label": lab, "variant": var, "iter": it[i], "loss_mean": float(L[:, i].mean()),
                        "loss_sd": float(L[:, i].std())})
    save(pd.DataFrame(out), "vqc_loss_curves")


# ---------------------------------------------------------------- 9 calibration, uncertainty, subgroups
def clinical():
    rows, sub = [], []
    for ds in ("chd", "hdc_ext", "cvd"):
        meta = data_mod.load(ds).meta
        for f in sorted(glob.glob(os.path.join(RUNS, "results", "main", f"main__{ds}__*.json"))):
            js = json.load(open(f))
            lab = js["task"]["label"]
            if "(all features)" in lab:
                continue
            z = np.load(f[:-5] + ".npz")
            y, yp, pr, idx = z["y"], z["y_pred"], z["proba"], z["index"]
            lv = z["latent_var"] if "latent_var" in z.files else np.full(len(y), np.nan)
            err = (y != yp).astype(int)
            rec = {"dataset": ds, "label": lab, "family": family(lab), "n": len(y)}
            if not np.all(np.isnan(lv)) and err.sum() > 0 and err.sum() < len(err):
                rec["latentvar_error_auc"] = float(roc_auc_score(err, lv))
            if not np.all(np.isnan(pr)):
                conf = np.abs(pr - 0.5)
                order = np.argsort(-conf)
                keep = order[: int(0.8 * len(order))]
                rec["acc_all"] = float((y == yp).mean())
                rec["acc_top80_confident"] = float((y[keep] == yp[keep]).mean())
                if err.sum() > 0 and err.sum() < len(err):
                    rec["conf_error_auc"] = float(roc_auc_score(err, -conf))
            rows.append(rec)
            sex = meta["sex_male"].to_numpy()[idx]
            old = (meta["age"].to_numpy()[idx] >= 55).astype(int)
            for gname, gv in (("sex", sex), ("age55", old)):
                for val in (0, 1):
                    m = gv == val
                    pos, neg = m & (y == 1), m & (y == 0)
                    sub.append({"dataset": ds, "label": lab, "family": family(lab), "group": gname, "value": val,
                                "n": int(m.sum()),
                                "sensitivity": float((yp[pos] == 1).mean()) if pos.any() else np.nan,
                                "specificity": float((yp[neg] == 0).mean()) if neg.any() else np.nan})
    u = pd.DataFrame(rows)
    u = u.groupby(["dataset", "label", "family"]).mean(numeric_only=True).reset_index()
    save(u, "uncertainty")
    s = pd.DataFrame(sub).groupby(["dataset", "label", "family", "group", "value"]).mean(numeric_only=True)
    s = s.reset_index()
    save(s, "subgroups")
    gap = s.pivot_table(index=["dataset", "label", "family", "group"], columns="value",
                        values="sensitivity").reset_index()
    gap["sens_gap"] = gap[1] - gap[0]
    save(gap, "subgroup_sensitivity_gap")
    cal = fold_table("main")
    cal = cal[~cal.label.str.contains("all features")].groupby(["dataset", "label", "family"])[
        ["brier", "ece"]].mean().reset_index()
    save(cal, "calibration")
    q = u[u.family == "QGPC"]
    KEY["clinical"] = {
        "qgpc_latentvar_error_auc": {f"{r['dataset']}|{r['label']}": float(r["latentvar_error_auc"])
                                     for _, r in q.iterrows() if not pd.isna(r.get("latentvar_error_auc"))},
        "conf_error_auc_family_mean": {f"{d_}|{f_}": float(x) for (d_, f_), x in
                                       u.groupby(["dataset", "family"])["conf_error_auc"].mean().items()},
        "brier_family_mean": {f"{d_}|{f_}": float(x) for (d_, f_), x in
                              cal.groupby(["dataset", "family"])["brier"].mean().items()},
        "ece_family_mean": {f"{d_}|{f_}": float(x) for (d_, f_), x in
                            cal.groupby(["dataset", "family"])["ece"].mean().items()},
        "sex_sens_gap_family_mean": {f"{d_}|{f_}": float(x) for (d_, f_), x in
                                     gap[gap.group == "sex"].groupby(["dataset", "family"])["sens_gap"].mean().items()},
        "age_sens_gap_family_mean": {f"{d_}|{f_}": float(x) for (d_, f_), x in
                                     gap[gap.group == "age55"].groupby(["dataset", "family"])["sens_gap"].mean().items()},
    }


if __name__ == "__main__":
    print("RUNS =", RUNS)
    main_and_supp()
    print("main done")
    external(); print("external done")
    samplesize(); print("samplesize done")
    ablation(); print("ablation done")
    sensitivity(); print("sens done")
    noise(); print("noise done")
    diagnostics(pd.read_csv(os.path.join(T, "main_summary.csv"))); print("diag done")
    trainability(); print("train done")
    clinical(); print("clinical done")
    KEY["audit"] = data_mod.audit()
    with open(os.path.join(OUT, "key_numbers.json"), "w") as f:
        json.dump(KEY, f, indent=1, default=float)
    print("written", OUT)
