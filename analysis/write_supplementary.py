"""Write the supplementary material (LaTeX) from the analysis tables.

Usage: python analysis/write_supplementary.py <results_dir> <manuscript_dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

RES, MS = sys.argv[1], sys.argv[2]
T = os.path.join(RES, "tables")
P = "balanced_accuracy"
DS = {"chd": "CHD", "hdc_ext": "HDC-ext", "cvd": "CVD", "hdc_dedup": "HDC (918)"}


def esc(x):
    return str(x).replace("VQE", "ZZ (VQE)").replace("&", r"\&").replace("_", r"\_")


def pv(x):
    return "<0.001" if x < 0.001 else f"{x:.3f}"


out = [r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{booktabs,longtable,array,amsmath}
\usepackage[T1]{fontenc}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}
\renewcommand{\thetable}{S\arabic{table}}
\usepackage[raggedright]{titlesec}
\setlength{\emergencystretch}{3em}
\title{Supplementary material\\[4pt]\large Quantum Feature Map Benchmarking for Clinical Heart Disease Classification}
\date{}
\begin{document}
\maketitle
\noindent All tables are generated automatically from the result files of the 7,180 experiment tasks. The code repository also contains the per-fold confusion matrices of every model (\texttt{confusion\_per\_fold.csv}), the per-fold metrics, and every table as CSV. Balanced accuracy is in percent unless stated otherwise; pp means percentage points.
"""]

# ---------------------------------------------------------------- S1 all comparisons
c = pd.read_csv(os.path.join(T, "comparisons.csv"))
c = c.sort_values(["dataset", "reference_type", "mean_diff"], ascending=[True, False, False])
out.append(r"""\section*{Table S1. All pre-specified comparisons}
Every quantum configuration against the strongest classical model on six components and against its paradigm-matched classical model (QSVM--SVM, QGPC--GPC, VQC--MLP). $\Delta$: quantum minus classical (pp) with corrected 95\% CI; $d_z$: standardized effect size; $p_{\mathrm{corr}}$: corrected resampled $t$ test; Holm and BH adjust over all comparisons; $p_{\mathrm{W}}$: Wilcoxon signed-rank test (treats folds as independent); $p_{\mathrm{naive}}$: ordinary paired $t$ test, which ignores the overlap of training folds (shown for reference). A $p$ value below 0.0005 is shown as $<0.001$, and $d_z$ is rounded to two decimals, so $0.00$ means no difference.
{\scriptsize
\begin{longtable}{lllrrrrrrr}
\toprule
Cohort & Quantum model & Reference & $\Delta$ [95\% CI] & $d_z$ & $p_{\mathrm{corr}}$ & $p_{\mathrm{Holm}}$ & $p_{\mathrm{BH}}$ & $p_{\mathrm{W}}$ & $p_{\mathrm{naive}}$ \\
\midrule\endhead""")
for _, r in c.iterrows():
    out.append(f"{DS[r['dataset']]} & {esc(r['model'])} & {r['reference']} & {100*r['mean_diff']:+.1f} "
               f"[{100*r['ci_low']:+.1f}, {100*r['ci_high']:+.1f}] & {r['d_z']:.2f} & {pv(r['p'])} & "
               f"{pv(r['p_holm_global'])} & {pv(r['p_bh_global'])} & {pv(r['p_wilcoxon'])} & {pv(r['p_naive_paired_t'])} \\\\")
out.append(r"\bottomrule" + "\n" + r"\end{longtable}}")

# ---------------------------------------------------------------- S2 supplementary cohort
s = pd.read_csv(os.path.join(T, "main_summary_supp.csv")).sort_values(P, ascending=False)
out.append(r"""\section*{Table S2. Supplementary cohort: HDC de-duplicated (918 records, includes Cleveland)}
Same protocol as the main analysis (10$\times$10 repeated cross-validation, nested tuning).
{\scriptsize
\begin{longtable}{llrrrrr}
\toprule Model & Family & Bal.\ acc.\ [95\% CI] & AUC & Sens. & Spec. & MCC \\ \midrule\endhead""")
for _, r in s.iterrows():
    out.append(f"{esc(r['model'])} & {r['family']} & {100*r[P]:.1f} [{100*r[P+'_lo']:.1f}, {100*r[P+'_hi']:.1f}] & "
               f"{r['auc']:.3f} & {r['sensitivity']:.3f} & {r['specificity']:.3f} & {r['mcc']:.3f} \\\\")
out.append(r"\bottomrule" + "\n" + r"\end{longtable}}")

# ---------------------------------------------------------------- S3 external validation
e = pd.read_csv(os.path.join(T, "external_validation.csv"))
piv = e.pivot_table(index=["label", "family"], columns="dataset", values=[P, "sensitivity", "specificity"]).reset_index()
out.append(r"""\section*{Table S3. External validation}
Models trained on one cohort with the 11 shared features and tested on the other (mean over 5 model seeds; one train--test split per direction, so no significance test).
{\scriptsize
\begin{longtable}{llrrrrrr}
\toprule
 & & \multicolumn{3}{c}{CHD $\rightarrow$ HDC-ext} & \multicolumn{3}{c}{HDC-ext $\rightarrow$ CHD} \\
Model & Family & Bal.\ acc. & Sens. & Spec. & Bal.\ acc. & Sens. & Spec. \\ \midrule\endhead""")
piv = piv.sort_values((P, "common_chd"), ascending=False)
for _, r in piv.iterrows():
    out.append(f"{esc(r[('label', '')])} & {r[('family', '')]} & {100*r[(P, 'common_chd')]:.1f} & "
               f"{r[('sensitivity', 'common_chd')]:.3f} & {r[('specificity', 'common_chd')]:.3f} & "
               f"{100*r[(P, 'common_hdc_ext')]:.1f} & {r[('sensitivity', 'common_hdc_ext')]:.3f} & "
               f"{r[('specificity', 'common_hdc_ext')]:.3f} \\\\")
out.append(r"\bottomrule" + "\n" + r"\end{longtable}}")

# ---------------------------------------------------------------- S4 noise
n = pd.read_csv(os.path.join(T, "noise.csv"))
lv = ["shots", "low", "medium", "high"]
out.append(r"""\section*{Table S4. Simulated hardware noise per feature map}
Balanced accuracy (\%), mean over CHD and HDC-ext and 5 folds. CX: CNOT gates of the transpiled fidelity circuit (linear coupling); $k(x,x)$: mean self-overlap of the noisy kernel at medium noise.
{\scriptsize
\begin{longtable}{llrrrrrr}
\toprule Family & Map & Shots only & Low & Medium & High & CX & $k(x,x)$ \\ \midrule\endhead""")
q = n[n.family == "QSVM"].groupby(["map", "level"])[["mean_diag", "cx"]].mean()
for fam in ("QSVM", "QGPC", "VQC"):
    for m in ["RX", "Pauli-Z", "Hadamard", "Strongly Entangling", "VQE", "Angle Encoding"]:
        g = n[(n.family == fam) & (n["map"] == m)].groupby("level")[P].mean()
        cx = q.loc[(m, "medium"), "cx"]
        dg = q.loc[(m, "medium"), "mean_diag"]
        out.append(f"{fam} & {esc(m)} & " + " & ".join(f"{100*g[l]:.1f}" for l in lv) + f" & {cx:.0f} & {dg:.2f} \\\\")
out.append(r"\bottomrule" + "\n" + r"\end{longtable}}")

# ---------------------------------------------------------------- S5 ablation
a = pd.read_csv(os.path.join(T, "ablation.csv"))
var_lab = {"noPCA": "No PCA (raw features)", "k2": "2 PCs", "k4": "4 PCs", "k8": "8 PCs", "k10": "10 PCs",
           "weighting_flipped": "Class weighting reversed"}
out.append(r"""\section*{Table S5. Ablation: change in balanced accuracy against the six-component reference (pp)}
Mean over models of each family and over CHD, HDC-ext, CVD-2000 and CVD-2000 with 25\% positives (5-fold CV $\times$ 2).
{\small
\begin{longtable}{lrrrr}
\toprule Change & QSVM & QGPC & VQC & Classical \\ \midrule\endhead""")
for v, lab in var_lab.items():
    g = a[a.variant == v].groupby("family")["delta"].mean()
    cells = [("--" if fam not in g or pd.isna(g[fam]) else f"{100*g[fam]:+.1f}") for fam in ("QSVM", "QGPC", "VQC", "Classical")]
    out.append(f"{lab} & " + " & ".join(cells) + r" \\")
out.append(r"\bottomrule" + "\n" + r"\end{longtable}}")

bw = pd.read_csv(os.path.join(T, "ablation_bandwidth_map.csv")).set_index("label")
cols = [c_ for c_ in bw.columns]
out.append(r"""\section*{Table S6. Balanced accuracy (\%) with a fixed bandwidth $\lambda$}
Mean over the four ablation cohorts; the main analysis tunes $\lambda$ by nested cross-validation instead.
{\scriptsize
\begin{longtable}{l""" + "r" * len(cols) + r"""}
\toprule Model & """ + " & ".join(f"$\\lambda={float(x):g}$" for x in cols) + r""" \\ \midrule\endhead""")
for lab, r in bw.iterrows():
    out.append(f"{esc(lab)} & " + " & ".join(f"{100*r[x_]:.1f}" for x_ in cols) + r" \\")
out.append(r"\bottomrule" + "\n" + r"\end{longtable}}")

# ---------------------------------------------------------------- S7 bandwidth selected in the main analysis
hs = pd.read_csv(os.path.join(T, "bandwidth_selection.csv"))
mode = hs.loc[hs.groupby(["dataset", "label"])["count"].idxmax()]
tot = hs.groupby(["dataset", "label"])["count"].sum()
out.append(r"""\section*{Table S7. Bandwidth selected by nested cross-validation in the main analysis}
Most frequently selected $\lambda$ and its share of the outer folds.
{\scriptsize
\begin{longtable}{lrrrrrr}
\toprule Model & \multicolumn{2}{c}{CHD} & \multicolumn{2}{c}{HDC-ext} & \multicolumn{2}{c}{CVD} \\ & $\lambda$ & share & $\lambda$ & share & $\lambda$ & share \\ \midrule\endhead""")
for lab in sorted(mode.label.unique()):
    cells = []
    for ds in ("chd", "hdc_ext", "cvd"):
        r = mode[(mode.dataset == ds) & (mode.label == lab)]
        if r.empty:
            cells += ["--", "--"]
        else:
            r = r.iloc[0]
            cells += [f"{float(r['hp_bandwidth']):g}", f"{100*r['count']/tot[(ds, lab)]:.0f}\\%"]
    out.append(f"{esc(lab)} & " + " & ".join(cells) + r" \\")
out.append(r"\bottomrule" + "\n" + r"\end{longtable}}")

# ---------------------------------------------------------------- kernel diagnostics: table of the paper (not the supplement)
dg = pd.read_csv(os.path.join(T, "diagnostics.csv"))
dg["dataset"] = dg["dataset"].replace({"cvd_n2000_s0": "cvd"})
sel = pd.read_csv(os.path.join(T, "diagnostics_vs_performance.csv"))
sel = sel[sel.family == "QSVM"][["dataset", "map", "lambda", "balanced_accuracy"]]
m = sel.merge(dg, left_on=["dataset", "map", "lambda"], right_on=["dataset", "model", "bandwidth"], how="left")
rows = [r"\begin{tabular}{llrrrrrrrrrrr}", r"\toprule",
        r"Cohort & Map & $\lambda$ & KTA & $d_{\mathrm{eff}}$ & $\lambda_1$ share & $\log_{10}\kappa$ & $\bar k$ & $\sigma^2_k$ & $g$ & KL & MW & BA \\",
        r"\midrule"]
order = ["RX", "Pauli-Z", "Hadamard", "Strongly Entangling", "VQE", "Angle Encoding"]
for ds in ("chd", "hdc_ext", "cvd"):
    for mp in order:
        r = m[(m.dataset == ds) & (m["map"] == mp)]
        if r.empty or pd.isna(r.iloc[0]["kta"]):
            continue
        r = r.iloc[0]
        rows.append(f"{DS[ds]} & {esc(mp)} & {r['lambda']:g} & {r['kta']:.3f} & {r['eff_dim_participation']:.2f} & "
                    f"{r['top1_frac']:.2f} & {np.log10(r['condition_number']):.1f} & {r['offdiag_mean']:.3f} & "
                    f"{r['offdiag_var']:.4f} & {r['geometric_difference_vs_rbf']:.2f} & {r['expressibility_kl']:.0f} & "
                    f"{r['meyer_wallach']:.3f} & {100*r['balanced_accuracy']:.1f} \\\\")
    rows.append(r"\midrule")
rows[-1] = r"\bottomrule"
rows.append(r"\end{tabular}")
os.makedirs(os.path.join(MS, "tables"), exist_ok=True)
open(os.path.join(MS, "tables", "diagnostics.tex"), "w", encoding="utf-8", newline="\n").write("\n".join(rows) + "\n")

out.append(r"\end{document}")
open(os.path.join(MS, "supplementary.tex"), "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("written supplementary.tex")
