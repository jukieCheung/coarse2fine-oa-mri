#!/usr/bin/env python3
"""Regenerate the Dual-vs-single forest plots from stats_recomputed_with_fdr.csv.

Layout: error bars on the left axes; p/q annotations in a fixed right-hand
text column (axes-fraction coordinates), so labels never overlap the bars.
Significant rows (q<0.05) are highlighted in color with an asterisk.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = __import__("os").environ.get("OACTF_BASE", ".")
PAPER = __import__("os").environ.get("OACTF_FIGDIR", ".")
df = pd.read_csv(f"{BASE}/stats_recomputed_with_fdr.csv")

ORDER = ["resnet3d", "m3t", "mamba"]
LABEL = {"resnet3d": "ResNet3D", "m3t": "M3T", "mamba": "nnMamba"}
C_SIG = "#0072B2"     # blue
C_NS = "#999999"      # gray


def plot(task, metrics, single_label, fname):
    sub = df[df.task == task].set_index(["backbone", "metric"])
    rows = [(bb, m) for bb in ORDER for m in metrics]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    plt.subplots_adjust(left=0.20, right=0.72, top=0.86, bottom=0.13)

    ypos = np.arange(len(rows))[::-1].astype(float)

    # group separators between backbones
    for i in range(1, len(ORDER)):
        ax.axhline(len(metrics) * i - 0.5, color="0.9", lw=0.8, zorder=0)

    for y, (bb, m) in zip(ypos, rows):
        r = sub.loc[(bb, m)]
        sig = bool(r["sig_fdr_0.05"])
        col = C_SIG if sig else C_NS
        ax.errorbar(r.delta, y,
                    xerr=[[r.delta - r.ci_lo], [r.ci_hi - r.delta]],
                    fmt="o", ms=5.5, capsize=3.5, lw=1.4,
                    color=col, ecolor=col, zorder=3)

    ax.axvline(0, ls="--", lw=1, color="0.35", zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{LABEL[bb]} | {m}" for bb, m in rows], fontsize=9)
    ax.set_xlabel(f"Metric difference (Dual $-$ {single_label})", fontsize=9.5)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_ylim(-0.7, len(rows) - 0.1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # fixed right-hand annotation column (axes-fraction x, data y)
    ax.text(1.06, len(rows) - 0.35, "p       q (BH)",
            transform=ax.get_yaxis_transform(), fontsize=9, fontweight="bold",
            va="bottom", ha="left")
    for y, (bb, m) in zip(ypos, rows):
        r = sub.loc[(bb, m)]
        sig = bool(r["sig_fdr_0.05"])
        ptxt = "<0.001" if r.p_value < 0.0005 else f"{r.p_value:.3f}"
        qtxt = ("<0.001" if r.q_value_BH < 0.0005 else f"{r.q_value_BH:.3f}") + ("*" if sig else "")
        col = C_SIG if sig else "0.25"
        ax.text(1.06, y, f"{ptxt}  {qtxt}", transform=ax.get_yaxis_transform(),
                fontsize=8.5, va="center", ha="left", color=col,
                fontfamily="monospace")

    fig.suptitle(f"Dual vs single: effect size with 95% CI ({task})",
                 fontsize=11, x=0.55)
    fig.savefig(f"{PAPER}/{fname}", dpi=220)
    print("wrote", fname)


plot("OA", ["AUC", "Acc", "F1"], "Single-OA", "forest_dual_vs_single_OA.png")
plot("KL", ["M-AUC", "Acc", "M-F1"], "Single-KL", "forest_dual_vs_single_KL.png")
