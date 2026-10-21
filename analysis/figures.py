"""Manuscript figures (vector PDF) from the tables written by analyze_all.py.

Usage: python analysis/figures.py <results_dir> <figure_dir>

Colour roles (reference palette, slots 1-3, validated all-pairs in light and
dark mode): QSVM = blue, VQC = orange, QGPC = aqua; classical references are
neutral grey. Six-map line charts use slots 1-6 in fixed order (adjacent
pairlist validated for lines). Marks: 1.5 pt lines, >= 6 pt markers, solid
hairline grid, text in neutral ink only.
"""
from __future__ import annotations

import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RES, FIG = sys.argv[1], sys.argv[2]
T = os.path.join(RES, "tables")
os.makedirs(FIG, exist_ok=True)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FAM_COL = {"QSVM": "#2a78d6", "VQC": "#eb6834", "QGPC": "#1baf7a", "Classical": "#6b6a66",
           "Classical (all features)": "#b5b4ae"}
MAP_ORDER = ["RX", "Pauli-Z", "Hadamard", "Strongly Entangling", "VQE", "Angle Encoding"]
MAP_COL = dict(zip(MAP_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MAP_LAB = {m: ("ZZ (VQE)" if m == "VQE" else m) for m in MAP_ORDER}
DS_LAB = {"chd": "CHD (n = 303)", "hdc_ext": "HDC-ext (n = 615)", "cvd": "CVD (n = 68,576)"}
P = "balanced_accuracy"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": GRID, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "lines.linewidth": 1.5, "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round", "figure.dpi": 150,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def short(m):
    return (m.replace("Strongly Entangling", "Str. Ent.").replace("Angle Encoding", "Angle Enc.")
             .replace("VQE", "ZZ").replace(" (all features)", " (all feat.)"))


# ---------------------------------------------------------------- Fig. 2 main comparison
def fig_main():
    s = pd.read_csv(os.path.join(T, "main_summary.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 5.6), sharey=False)
    for ax, ds in zip(axes, ["chd", "hdc_ext", "cvd"]):
        g = s[s.dataset == ds].sort_values(P)
        y = np.arange(len(g))
        ref = g[g.family == "Classical"][P].max()
        ax.axvline(ref, color=INK2, lw=0.8, zorder=1)
        for yi, (_, r) in zip(y, g.iterrows()):
            c = FAM_COL[r["family"]]
            ax.plot([r[P + "_lo"], r[P + "_hi"]], [yi, yi], color=c, lw=1.2, alpha=0.55, zorder=2)
            hollow = r["family"] == "Classical (all features)"
            ax.plot(r[P], yi, "o", ms=4.5, mfc="white" if hollow else c, mec=c if hollow else "white",
                    mew=1.0 if hollow else 0.8, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([short(m) for m in g.model], fontsize=6)
        ax.set_title(DS_LAB[ds], loc="left", fontweight="bold")
        ax.set_xlabel("Balanced accuracy (95% CI)")
        ax.grid(axis="y", visible=False)
        lo = max(0.45, g[P + "_lo"].min() - 0.02)
        ax.set_xlim(lo, min(1.0, g[P + "_hi"].max() + 0.02))
    handles = [Line2D([], [], marker="o", ls="", ms=5, mfc=FAM_COL[f], mec="white", label=f)
               for f in ("QSVM", "VQC", "QGPC", "Classical")]
    handles.append(Line2D([], [], marker="o", ls="", ms=5, mfc="white", mec=FAM_COL["Classical (all features)"],
                          label="Classical (all features, no PCA)"))
    handles.append(Line2D([], [], color=INK2, lw=0.8, label="Best classical model on the same 6 PCs"))
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "fig_main_comparison")


# ---------------------------------------------------------------- Fig. 3 sample size
def fig_samplesize():
    g = pd.read_csv(os.path.join(T, "samplesize_best_per_family.csv"))
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    off = {"Classical": 0.97, "QSVM": 0.99, "QGPC": 1.01, "VQC": 1.03}      # small horizontal offsets
    for fam in ("Classical", "QSVM", "QGPC", "VQC"):
        d = g[g.family == fam].sort_values("N")
        ax.errorbar(d.N * off[fam], d.bal, yerr=d.sd, fmt="-o", color=FAM_COL[fam], ms=4, mec="white", mew=0.7,
                    elinewidth=0.7, capsize=1.5, alpha=0.95, label=fam)
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 500, 1000, 2000, 5000, 10000])
    ax.set_xticklabels(["100", "200", "500", "1K", "2K", "5K", "10K"])
    ax.set_xlabel("CVD subsample size N (records, 5-fold CV)")
    ax.set_ylabel("Balanced accuracy (best model)")
    ax.legend(loc="lower right", ncol=2)
    save(fig, "fig_sample_size")


# ---------------------------------------------------------------- Fig. 4 noise
def fig_noise():
    q = pd.read_csv(os.path.join(T, "noise_kernel_quality.csv"))
    n = pd.read_csv(os.path.join(T, "noise.csv"))
    lv = ["shots", "low", "medium", "high"]
    lvl = ["Shots\nonly", "Low", "Medium", "High"]
    mk = {"RX": "o", "Pauli-Z": "s", "Hadamard": "^", "Strongly Entangling": "D", "VQE": "v", "Angle Encoding": "P"}
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4))
    ax = axes[0]
    for m in MAP_ORDER:
        d = q[q["map"] == m].set_index("level").loc[lv]
        ax.plot(range(4), d.mean_diag, "-", marker=mk[m], color=MAP_COL[m], ms=4, mec="white", mew=0.5,
                label=MAP_LAB[m])
    ax.set_title("(a) Kernel self-overlap", loc="left")
    ax.set_ylabel(r"Mean $k(x,x)$")
    ax.set_ylim(-0.02, 1.05)
    for i, fam in enumerate(("QSVM", "QGPC", "VQC")):
        ax = axes[i + 1]
        for m in MAP_ORDER:
            d = n[(n.family == fam) & (n["map"] == m)].groupby("level")[P].mean().loc[lv]
            ax.plot(range(4), d.values, "-", marker=mk[m], color=MAP_COL[m], ms=4, mec="white", mew=0.5)
        ax.set_title(f"({'bcd'[i]}) {fam}", loc="left")
        if i == 0:
            ax.set_ylabel("Balanced accuracy")
        ax.set_ylim(0.45, 0.9)
    for ax in axes:
        ax.set_xticks(range(4))
        ax.set_xticklabels(lvl)
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1), fontsize=7)
    save(fig, "fig_noise")


# ---------------------------------------------------------------- Fig. 5 sensitivity + bandwidth
def fig_sensitivity():
    s = pd.read_csv(os.path.join(T, "sensitivity_family.csv"), dtype={"value": str})
    b = pd.read_csv(os.path.join(T, "ablation_bandwidth_family.csv"))
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.3), sharey=True)
    specs = [("layers", ["1", "2", "3", "4", "6"], "Encoding layers"),
             ("entanglement", ["none", "linear", "circular", "full"], "Entanglement"),
             ("shots", ["64", "256", "1024", "2048", "8192"], "Shots"),
             ("spsa", ["25", "50", "100", "200", "400", "800"], "SPSA iterations")]
    for k, (ax, (prm, vals, lab)) in enumerate(zip(axes, specs)):
        for fam in ("QSVM", "QGPC", "VQC"):
            d = s[(s.param == prm) & (s.family == fam)].set_index("value")
            v = [x for x in vals if x in d.index]
            if not v:
                continue
            ax.plot([vals.index(x) for x in v], d.loc[v, P], "-o", color=FAM_COL[fam], ms=4, mec="white", mew=0.6,
                    label=fam)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals, rotation=0 if prm != "entanglement" else 30)
        ax.set_xlabel(lab)
        ax.set_title(f"({'abcd'[k]})", loc="left")
    ax = axes[4]
    cols = [c for c in b.columns if c != "family"]
    for _, r in b.iterrows():
        ax.plot(range(len(cols)), [r[c] for c in cols], "-o", color=FAM_COL[r["family"]], ms=4,
                mec="white", mew=0.6)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{float(c):g}" for c in cols])
    ax.set_xlabel("Bandwidth λ (fixed)")
    ax.set_title("(e)", loc="left")
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    save(fig, "fig_sensitivity")


# ---------------------------------------------------------------- Fig. 6 diagnostics: KTA + gradient variance
def fig_diagnostics():
    d = pd.read_csv(os.path.join(T, "diagnostics_vs_performance.csv"))
    gv = pd.read_csv(os.path.join(T, "gradvar.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    ax = axes[0]
    mk = {"chd": "o", "hdc_ext": "s", "cvd": "^"}
    for _, r in d.iterrows():
        c = MAP_COL[r["map"]]
        qsvm = r["family"] == "QSVM"
        ax.plot(r["kta"], r[P], mk[r["dataset"]], ls="", ms=5.5, mfc=c if qsvm else "white", mec=c,
                mew=1.0)
    ax.set_xlabel("Centered kernel-target alignment (training split)")
    ax.set_ylabel("Balanced accuracy (main analysis)")
    ax.set_title("(a) Kernel-target alignment and accuracy", loc="left")
    lo, hi = d[P].min(), d[P].max()
    ax.set_ylim(lo - 0.015, hi + 0.015)
    h = [Line2D([], [], marker="o", ls="", ms=5, mfc=MAP_COL[m], mec=MAP_COL[m], label=MAP_LAB[m]) for m in MAP_ORDER]
    h += [Line2D([], [], marker=m, ls="", ms=5, mfc=INK2, mec=INK2, label=DS_LAB[ds].split(" (")[0])
          for ds, m in mk.items()]
    h += [Line2D([], [], marker="o", ls="", ms=5, mfc=INK2, mec=INK2, label="QSVM (filled)"),
          Line2D([], [], marker="o", ls="", ms=5, mfc="white", mec=INK2, label="QGPC (open)")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=6, frameon=False)
    ax = axes[1]
    for L, mk_ in ((2, "o"), (8, "s")):
        g = gv[gv.layers == L].groupby("n_qubits")[["var_local_first", "var_global_first"]].mean()
        ax.plot(g.index, g.var_local_first, "-", marker=mk_, color=FAM_COL["QSVM"], ms=4.5, mec="white", mew=0.6,
                alpha=1.0 if L == 2 else 0.6, label=rf"Local $\langle Z_0\rangle$, mean of maps, depth {L}")
        ax.plot(g.index, g.var_global_first, "-", marker=mk_, color=FAM_COL["VQC"], ms=4.5, mec="white", mew=0.6,
                alpha=1.0 if L == 2 else 0.6, label=rf"Global $|0\cdots0\rangle$, mean of maps, depth {L}")
    z = gv[(gv.layers == 2) & (gv.feature_map == "VQE")].groupby("n_qubits")["var_local_first"].mean()
    ax.plot(z.index, z.values, "--", marker="v", color=MAP_COL["VQE"], ms=4.5, mec="white", mew=0.6,
            label=r"Local $\langle Z_0\rangle$, ZZ (VQE) map, depth 2")
    ax.set_yscale("log")
    ax.axvline(6, color=INK2, lw=0.8)
    ax.text(6.15, 2.0, "6 qubits (this study)", fontsize=6, color=INK2, va="top")
    ax.set_ylim(1e-8, 3)
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel(r"Var$[\partial\langle O\rangle/\partial\theta_1]$")
    ax.set_title("(b) Gradient variance", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6, frameon=False)
    fig.tight_layout()
    save(fig, "fig_diagnostics")


# ---------------------------------------------------------------- Fig. 7 VQC training
def fig_training():
    c = pd.read_csv(os.path.join(T, "vqc_loss_curves.csv"))
    o = pd.read_csv(os.path.join(T, "vqc_training.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    for m in MAP_ORDER:
        d = c[(c.dataset == "hdc_ext") & (c.variant == "spsa") & (c.label == f"VQC-{m}")].sort_values("iter")
        ax.plot(d.iter, d.loss_mean, color=MAP_COL[m], lw=1.3, ls="--" if m == "RX" else "-", label=MAP_LAB[m])
        ax.fill_between(d.iter, d.loss_mean - d.loss_sd, d.loss_mean + d.loss_sd, color=MAP_COL[m], alpha=0.10,
                        lw=0)
    ax.set_xlabel("SPSA iteration")
    ax.set_ylabel("Training cross-entropy")
    ax.set_title("(a) HDC-ext, mean ± SD over 10 seeds × 5 folds", loc="left")
    ax.set_ylim(0.3, 1.6)
    ax.legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    ax = axes[1]
    var_lab = {"spsa": "SPSA", "spsa-init-pi": "SPSA, init U(−π, π)", "cobyla": "COBYLA",
               "adam-ps": "Adam, parameter shift"}
    x = np.arange(len(MAP_ORDER))
    w = 0.2
    for i, (v, lab) in enumerate(var_lab.items()):
        d = o[o.variant == v].groupby("label")["bal"].mean()
        vals = [d.get(f"VQC-{m}", np.nan) for m in MAP_ORDER]
        sds = o[o.variant == v].groupby("label")["sd"].mean()
        errs = [sds.get(f"VQC-{m}", np.nan) for m in MAP_ORDER]
        bars = ax.bar(x + (i - 1.5) * w, vals, w * 0.9, color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"][i], label=lab,
                      yerr=errs, error_kw={"elinewidth": 0.6, "capsize": 1.2, "ecolor": INK2})
        for b_, v_, e_ in zip(bars, vals, errs):      # value printed vertically just above each error bar
            if not np.isnan(v_):
                ax.text(b_.get_x() + b_.get_width() / 2, v_ + (0 if np.isnan(e_) else e_) + 0.015, f"{v_:.2f}",
                        rotation=90, ha="center",
                        va="bottom", fontsize=6, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([MAP_LAB[m].replace("Strongly Entangling", "Strongly\nEntangling")
                        .replace("Angle Encoding", "Angle\nEncoding") for m in MAP_ORDER], fontsize=6.5)
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("(b) Optimizer and initialization", loc="left")
    ax.legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_vqc_training")


if __name__ == "__main__":
    fig_main(); fig_samplesize(); fig_noise(); fig_sensitivity(); fig_diagnostics(); fig_training()
    print("figures written to", FIG)
