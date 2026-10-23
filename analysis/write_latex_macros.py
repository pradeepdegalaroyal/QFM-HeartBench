"""Write results_macros.tex (every number quoted in the manuscript) and compact LaTeX tables.

Usage: python analysis/write_latex_macros.py <results_dir> <manuscript_dir>
"""
import json


def pf(x):
    return "<0.001" if x < 0.0005 else f"{x:.3f}"

import os
import re
import shutil
import sys

import pandas as pd

RES, MS = sys.argv[1], sys.argv[2]
T = os.path.join(RES, "tables")
OT = os.path.join(MS, "tables")
os.makedirs(OT, exist_ok=True)
K = json.load(open(os.path.join(RES, "key_numbers.json")))
P = "balanced_accuracy"
M = {}


def name(s):
    s = re.sub(r"[^A-Za-z]", "", s.title())
    return s


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


def put(key, val):
    assert re.fullmatch(r"[A-Za-z]+", key), key
    if re.fullmatch(r"-[0-9][0-9.,]*", val):  # typographic minus in text and math
        val = r"\ensuremath{-}" + val[1:]
    M[key] = val


def signed(x):
    return f"{x:+.1f}".replace("-", r"\ensuremath{-}")


DSN = {"chd": "Chd", "hdc_ext": "Hdc", "cvd": "Cvd"}
# ---------------------------------------------------------------- main
for ds, tag in DSN.items():
    k = K["main"][ds]
    for role in ("best_overall", "best_classical_pca6", "best_quantum"):
        r = k[role]
        rn = {"best_overall": "Best", "best_classical_pca6": "BestCl", "best_quantum": "BestQ"}[role]
        put(f"main{tag}{rn}Model", r[0].replace("VQE", "ZZ (VQE)"))
        put(f"main{tag}{rn}", pct(r[1]))
        put(f"main{tag}{rn}Lo", pct(r[2]))
        put(f"main{tag}{rn}Hi", pct(r[3]))
    for fam, r in k["best_per_family"].items():
        f = {"Classical": "Cl", "Classical (all features)": "Full", "QSVM": "Qsvm", "QGPC": "Qgpc", "VQC": "Vqc"}[fam]
        put(f"main{tag}Fam{f}Model", r[0].replace("VQE", "ZZ (VQE)"))
        put(f"main{tag}Fam{f}", pct(r[1]))
    put(f"main{tag}VqcMin", pct(k["vqc_range"][0]))
    put(f"main{tag}VqcMax", pct(k["vqc_range"][1]))
    put(f"main{tag}WorstModel", k["worst_quantum"][0].replace("VQE", "ZZ (VQE)"))
    put(f"main{tag}Worst", pct(k["worst_quantum"][1]))
    put(f"main{tag}SigNeg", str(k["sig_holm_neg"]))
    put(f"main{tag}SigPos", str(k["sig_holm_pos"]))
    put(f"main{tag}Comparisons", str(k["comparisons"]))
    put(f"main{tag}Folds", str(k["folds"]))
    mp = k["max_positive_diff"]
    put(f"main{tag}MaxDiffModel", mp[0].replace("VQE", "ZZ (VQE)"))
    put(f"main{tag}MaxDiffRef", mp[1])
    put(f"main{tag}MaxDiff", f"{100 * mp[2]:+.1f}")
    put(f"main{tag}MaxDiffLo", f"{100 * mp[3]:+.1f}")
    put(f"main{tag}MaxDiffHi", f"{100 * mp[4]:+.1f}")
    put(f"main{tag}MaxDiffP", f"{mp[5]:.2f}")
    put(f"varKept{tag}", pct(K["retained_variance"][ds]))
put("cmpTotal", str(K["comparisons_total"]))
put("cmpSigPos", str(K["comparisons_sig_holm_pos"]))
put("cmpSigNeg", str(K["comparisons_sig_holm_neg"]))
put("cmpSigPosBH", str(K["comparisons_sig_bh_pos"]))
put("cmpNaivePos", str(K["comparisons_naive_t_sig_pos"]))
put("cmpUnadjPos", str(K["comparisons_unadjusted_pos"]))
s = K["supp"]["hdc_dedup"]
put("suppBestModel", s["best_overall"][0]); put("suppBest", pct(s["best_overall"][1]))
put("suppBestQModel", s["best_quantum"][0]); put("suppBestQ", pct(s["best_quantum"][1]))
put("suppBestClModel", s["best_classical_pca6"][0]); put("suppBestCl", pct(s["best_classical_pca6"][1]))
put("suppSigPos", str(s["sig_holm_pos"])); put("suppSigNeg", str(s["sig_holm_neg"]))
# ---------------------------------------------------------------- external
for key, tag in (("common_chd->common_hdc_ext", "ChdToHdc"), ("common_hdc_ext->common_chd", "HdcToChd")):
    e = K["external"][key]
    put(f"ext{tag}BestModel", e["best"][0].replace("VQE", "ZZ (VQE)")); put(f"ext{tag}Best", pct(e["best"][1]))
    put(f"ext{tag}BestQModel", e["best_quantum"][0].replace("VQE", "ZZ (VQE)")); put(f"ext{tag}BestQ", pct(e["best_quantum"][1]))
    put(f"ext{tag}BestClModel", e["best_classical"][0]); put(f"ext{tag}BestCl", pct(e["best_classical"][1]))
    put(f"ext{tag}Min", pct(e["range"][0])); put(f"ext{tag}Max", pct(e["range"][1]))
# ---------------------------------------------------------------- sample size
ss = K["samplesize"]
for n in ("100", "2000", "10000"):
    w = {"100": "Hundred", "2000": "TwoK", "10000": "TenK"}[n]
    for fam in ("Classical", "QSVM", "QGPC", "VQC"):
        put(f"ss{w}{name(fam)}", pct(ss[n][fam][1]))
# ---------------------------------------------------------------- ablation
a = K["ablation"]
for var, tag in (("noPCA", "NoPca"), ("k2", "KTwo"), ("k4", "KFour"), ("k8", "KEight"), ("k10", "KTen"),
                 ("weighting_flipped", "Weight")):
    for fam in ("QSVM", "QGPC", "VQC", "Classical"):
        v = a[var].get(fam)
        if v is not None:
            put(f"abl{tag}{name(fam)}", f"{100 * v:+.1f}")
put("ablBandwidthSpreadMax", pct(max(a["bandwidth_spread"].values())))
put("ablBandwidthSpreadMin", pct(min(a["bandwidth_spread"].values())))
# ---------------------------------------------------------------- sensitivity
sv = K["sens"]
for fam in ("QSVM", "QGPC", "VQC"):
    for L in ("1", "2", "3", "6"):
        put(f"sensLayer{ {'1': 'One', '2': 'Two', '3': 'Three', '6': 'Six'}[L]}{name(fam)}", pct(sv["layers"][fam][L]))
    for e in ("none", "linear", "circular", "full"):
        put(f"sensEnt{name(e)}{name(fam)}", pct(sv["entanglement"][fam][e]))
for sh in ("64", "1024", "8192"):
    w = {"64": "SixtyFour", "1024": "OneK", "8192": "EightK"}[sh]
    for fam in ("QSVM", "QGPC"):
        put(f"sensShots{w}{name(fam)}", pct(sv["shots"][fam][sh]))
        put(f"sensShotsLoss{w}{name(fam)}", f"{100 * (sv['shots'][fam][sh] - sv['layers'][fam]['2']):.1f}")
for it in ("25", "200", "800"):
    w = {"25": "TwentyFive", "200": "TwoHundred", "800": "EightHundred"}[it]
    put(f"sensSpsa{w}", pct(sv["spsa"]["VQC"][it]))
nm = K["sens"]["shots_mean_neg_mass"]
put("shotsNegMassSixtyFour", pct(nm["64"], 0)); put("shotsNegMassEightK", pct(nm["8192"], 0))
# ---------------------------------------------------------------- noise
n = K["noise"]
for fam in ("QSVM", "QGPC", "VQC"):
    for lv in ("shots", "low", "medium", "high"):
        put(f"noise{name(fam)}{name(lv)}", pct(n["family_mean"][fam][lv]))
put("noiseCxZz", f"{n['cx_by_map']['VQE']:.0f}"); put("noiseCxRx", f"{n['cx_by_map']['RX']:.0f}")
put("noiseCxSe", f"{n['cx_by_map']['Strongly Entangling']:.0f}")
put("noiseDiagZzMedium", f"{n['mean_diag_by_map_level']['VQE|medium']:.2f}")
put("noiseDiagZzHigh", f"{n['mean_diag_by_map_level']['VQE|high']:.2f}")
put("noiseDiagRxMedium", f"{n['mean_diag_by_map_level']['RX|medium']:.2f}")
put("noiseDiagRxHigh", f"{n['mean_diag_by_map_level']['RX|high']:.2f}")
put("noiseDiagSeHigh", f"{n['mean_diag_by_map_level']['Strongly Entangling|high']:.2f}")
put("noiseDropZzQgpc", pct(n["drop_shots_to_high"]["QGPC|VQE"]))
put("noiseDropZzQsvm", pct(n["drop_shots_to_high"]["QSVM|VQE"]))
put("noiseDropZzVqc", pct(n["drop_shots_to_high"]["VQC|VQE"]))
# ---------------------------------------------------------------- diagnostics
d = K["diag"]
for ds, tag in (("chd", "Chd"), ("hdc_ext", "Hdc"), ("cvd", "Cvd")):
    c = d["spearman_within_cohort"][ds]
    put(f"diagKta{tag}", f"{c['kta'][0]:.2f}"); put(f"diagKtaP{tag}", pf(c['kta'][1]))
    put(f"diagGd{tag}", f"{c['geometric_difference'][0]:.2f}"); put(f"diagGdP{tag}", pf(c['geometric_difference'][1]))
    put(f"diagMw{tag}", f"{c['meyer_wallach'][0]:.2f}"); put(f"diagMwP{tag}", pf(c['meyer_wallach'][1]))
    put(f"diagEx{tag}", f"{c['expressibility_kl'][0]:.2f}"); put(f"diagExP{tag}", pf(c['expressibility_kl'][1]))
put("diagRxPzAlignChd", f"{d['rx_pz_alignment']['chd|0.1']:.2f}")
put("diagRxPzAlignHdc", f"{d['rx_pz_alignment']['hdc_ext|0.1']:.2f}")
put("diagRxPzAlignChdLarge", f"{d['rx_pz_alignment']['chd|2.0']:.2f}")
put("diagZzOffSmall", f"{d['zz_offdiag_by_lambda']['0.05']:.2f}")
put("diagZzOffLarge", f"{d['zz_offdiag_by_lambda']['1.0']:.3f}")
# ---------------------------------------------------------------- trainability
g = K["gradvar"]
put("gvLocalTwo", f"{g['local_L2']['2']:.2f}"); put("gvLocalTwelve", f"{g['local_L2']['12']:.2f}")
put("gvGlobalTwo", f"{g['global_L2']['2']:.3f}")
gv6 = g["global_L2"]["6"]
mant, exp = f"{gv6:.1e}".split("e")
put("gvGlobalSix", f"{mant}\\times10^{{{int(exp)}}}")
gv12 = g["global_L2"]["12"]
mant, exp = f"{gv12:.1e}".split("e")
put("gvGlobalTwelve", f"{mant}\\times10^{{{int(exp)}}}")
put("gvLocalEightSix", f"{g['local_L8']['6']:.3f}")
gvm = pd.read_csv(os.path.join(T, "gradvar.csv"))
gvm = gvm[gvm.layers == 2].pivot_table(index="n_qubits", columns="feature_map", values="var_local_first")
put("gvLocalZzTwo", f"{gvm.loc[2, 'VQE']:.2f}"); put("gvLocalZzSix", f"{gvm.loc[6, 'VQE']:.3f}")
mant, exp = f"{gvm.loc[12, 'VQE']:.1e}".split("e")
put("gvLocalZzTwelve", f"{mant}\\times10^{{{int(exp)}}}")
oth = gvm.drop(columns="VQE")
put("gvLocalOtherMin", f"{oth.values.min():.2f}"); put("gvLocalOtherMax", f"{oth.values.max():.2f}")
t = K["train"]
for v, tag in (("spsa", "Spsa"), ("spsa-init-pi", "SpsaPi"), ("cobyla", "Cobyla"), ("adam-ps", "Adam")):
    put(f"trainBal{tag}", pct(t["by_optimizer"][v]["bal"]))
    put(f"trainCollapse{tag}", pct(t["by_optimizer"][v]["collapsed_frac"], 0))
for m, tag in (("VQC-RX", "Rx"), ("VQC-Hadamard", "Had"), ("VQC-VQE", "Zz"), ("VQC-Angle Encoding", "Ae"),
               ("VQC-Strongly Entangling", "Se"), ("VQC-Pauli-Z", "Pz")):
    put(f"trainMap{tag}", pct(t["by_map_spsa"][m]))
put("trainLossSd", f"{max(t['final_loss_sd_across_seeds'].values()):.2f}")
# ---------------------------------------------------------------- clinical
cl = K["clinical"]
lv = list(cl["qgpc_latentvar_error_auc"].values())
put("qgpcVarAucMin", f"{min(lv):.2f}"); put("qgpcVarAucMax", f"{max(lv):.2f}")
cf = [v for v in cl["conf_error_auc_family_mean"].values() if v == v]
put("confAucMin", f"{min(cf):.2f}"); put("confAucMax", f"{max(cf):.2f}")
cal = pd.read_csv(os.path.join(T, "calibration.csv"))
for ds, tag in (("chd", "Chd"), ("hdc_ext", "Hdc")):
    for lab, lt in (("GPC", "Gpc"), ("QGPC-Pauli-Z", "QgpcPz")):
        r = cal[(cal.dataset == ds) & (cal.label == lab)].iloc[0]
        put(f"brier{tag}{lt}", f"{r['brier']:.3f}")
sg = cl["sex_sens_gap_family_mean"]
put("sexGapHdcMin", pct(min(v for kk, v in sg.items() if kk.startswith("hdc_ext")), 0))
put("sexGapHdcMax", pct(max(v for kk, v in sg.items() if kk.startswith("hdc_ext")), 0))
ag = cl["age_sens_gap_family_mean"]
put("ageGapCvdMin", pct(min(v for kk, v in ag.items() if kk.startswith("cvd")), 0))
put("ageGapCvdMax", pct(max(v for kk, v in ag.items() if kk.startswith("cvd")), 0))
# ---------------------------------------------------------------- run time
rt = pd.read_csv(os.path.join(T, "runtime_median.csv"))
put("rtCvdVqc", f"{rt[(rt.dataset == 'cvd') & (rt.family == 'VQC')].fit_seconds.iloc[0]:.0f}")
put("rtCvdQgpc", f"{rt[(rt.dataset == 'cvd') & (rt.family == 'QGPC')].fit_seconds.iloc[0]:.0f}")
put("rtCvdQsvm", f"{rt[(rt.dataset == 'cvd') & (rt.family == 'QSVM')].fit_seconds.iloc[0]:.0f}")
put("rtCvdCl", f"{rt[(rt.dataset == 'cvd') & (rt.family == 'Classical')].fit_seconds.iloc[0]:.1f}")
au = K["audit"]
put("auditCvdClean", f"{au['cvd_clean']:,}")

with open(os.path.join(MS, "results_macros.tex"), "w", encoding="utf-8", newline="\n") as f:
    f.write("% Auto-generated by analysis/write_latex_macros.py from results/key_numbers.json -- do not edit.\n")
    for k_, v in sorted(M.items()):
        f.write(f"\\newcommand{{\\{k_}}}{{{v}}}\n")
print(len(M), "macros")

# ---------------------------------------------------------------- compact comparison table for the paper body
c = pd.read_csv(os.path.join(T, "comparisons.csv"))
lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Cohort & Comparison & $\Delta$ (pp) [95\% CI] & $p_{\mathrm{corr}}$ & $p_{\mathrm{Holm}}$ & $p_{\mathrm{Wilcoxon}}$ \\",
         r"\midrule"]
dsl = {"chd": "CHD", "hdc_ext": "HDC-ext", "cvd": "CVD"}
for ds in ("chd", "hdc_ext", "cvd"):
    cc = c[c.dataset == ds]
    best_ref = cc[cc.reference_type == "strongest classical"]
    rows = []
    for fam in ("QSVM", "QGPC", "VQC"):
        sub = best_ref[best_ref.model.str.startswith(fam + "-")].sort_values("mean_diff", ascending=False)
        rows.append(sub.iloc[0])
    worst = best_ref.sort_values("mean_diff").iloc[0]
    rows.append(worst)
    first = True
    for r in rows:
        lines.append(f"{dsl[ds] if first else ''} & {r['model'].replace('VQE', 'ZZ (VQE)')} vs.\\ {r['reference']} & "
                     f"{signed(100 * r['mean_diff'])} [{signed(100 * r['ci_low'])}, {signed(100 * r['ci_high'])}] & "
                     f"{pf(r['p'])} & {pf(r['p_holm_global'])} & {pf(r['p_wilcoxon'])} \\\\")
        first = False
    lines.append(r"\midrule" if ds != "cvd" else r"\bottomrule")
lines.append(r"\end{tabular}")
open(os.path.join(OT, "comparisons_compact.tex"), "w", newline="\n").write("\n".join(lines) + "\n")

# copy the auto-generated tables the manuscript inputs
for fn in ("main_chd.tex", "main_hdc_ext.tex", "main_cvd.tex", "main_hdc_dedup.tex", "comparisons_main.tex",
           "retained_variance.tex", "external.tex", "noise.tex"):
    src = os.path.join(T, fn)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OT, fn))
print("tables copied to", OT)
