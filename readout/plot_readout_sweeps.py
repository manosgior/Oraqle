"""
plot_readout_sweeps.py
======================
Paper (Oraqle, Sec. 4) figure merging the two integration-and-threshold baseline
figures into one full-width row:

  Truncation from onset (Sec. 4.1)      | Sliding window (Sec. 4.2)
  (a) fidelity vs readout duration      | (c) 250 ns window fidelity vs center
  (b) information-gain rate dF/dT       | (d) 500 ns window fidelity vs center

(a), (c), (d) share the fidelity y-axis; one legend (Q1-Q5 + geometric mean) at the
bottom, "Higher is better" on top. Reads the same CSVs as
baseline_no_ML.plot_baseline_sweep_from_csv / plot_sliding_window_from_csv, pinned to
the runs used in the paper. Drawn at the ACM \\textwidth (7.0 in), included 1:1.

Usage:  python plot_readout_sweeps.py [OUT_PDF]
"""
import os, re, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

import sys as _sys  # [unify] repo layout: reuse discriminators/disc_paths.py
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'discriminators'))
from disc_paths import RAW_TRAIN, RAW_TEST, RESULTS_DISC, REPORTS_OUT, ensure_dir, require_dataset  # [unify]

# [unify] was RES = "Discriminators/training_results" (run from the old repo root)
RES = os.environ.get("ORAQLE_READOUT_RES", str(RESULTS_DISC))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ensure_dir(REPORTS_OUT), "readout_sweeps_merged.pdf")
DFDT_CSV = os.path.join(RES, "Baseline_Threshold_dFdT_analysis_20260514_131943.csv")
WIN_CSV = {250: os.path.join(RES, "Baseline_Threshold_sliding_window_250ns_20260514_204419.csv"),
           500: os.path.join(RES, "Baseline_Threshold_sliding_window_500ns.csv")}
DT, NQ = 2e-9, 5
QLABELS = [f"Q{q + 1}" for q in range(NQ)]
PALETTE = sns.color_palette("deep")
MARKERS = ["o", "s", "^", "D", "P"]

TEXT_W = 7.0
TITLE_FS, GROUP_FS, LABEL_FS, TICK_FS, LEGEND_FS = 6.4, 7.5, 7, 5.8, 6.5
LW, MS, MEW = 0.9, 2.6, 0.35
MEAN_C = "0.3"                     # geometric mean: dark grey, not black
FIG_H = 1.66                       # inches
# vertical layout in inches (from the top / bottom edge) -> tight, size-independent
AX_TOP_IN, AX_BOT_IN = 0.40, 0.42  # axes top below fig top / axes bottom above fig bottom
HDR_IN, RULE_IN = 0.155, 0.140     # group header / rule offset above the axes top


def gmean(a):
    return np.exp(np.log(np.clip(a, 1e-6, 1.0)).mean(axis=1))


def load_truncation():
    rows = []
    pat = re.compile(r"Baseline_Threshold_len(\d+)_")
    for f in os.listdir(RES):
        m = pat.match(f)
        if m and f.endswith(".csv"):
            r = next(csv.DictReader(open(os.path.join(RES, f))))
            rows.append([int(m.group(1))] + [float(r[f"qubit_{q}_accuracy"]) / 100 for q in range(NQ)])
    a = np.array(sorted(rows))
    return a[:, 0] * DT * 1e6, a[:, 1:]


def load_dfdt():
    a = np.array(sorted([[int(r["trace_length"])] + [float(r[f"qubit_{q}_dFdT"]) for q in range(NQ)]
                         for r in csv.DictReader(open(DFDT_CSV))]))
    return a[:, 0] * DT * 1e6, a[:, 1:]


def load_window(w):
    a = np.array(sorted([[float(r["window_center_us"])] + [float(r[f"qubit_{q}_fidelity"]) for q in range(NQ)]
                         for r in csv.DictReader(open(WIN_CSV[w]))]))
    return a[:, 0], a[:, 1:]


def style(ax, title, xlabel, ylabel=None, ylabels=True):
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=2.5)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS - 0.5, labelpad=1)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FS - 0.5, labelpad=1.5)
    ax.set_xlim(0, 2)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.tick_params(axis="both", labelsize=TICK_FS, length=2, pad=1.5)
    if not ylabels:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_axisbelow(True)
    ax.grid(color="gray", ls="--", lw=0.3, alpha=0.4)
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def fid_lines(ax, x, F):
    for q in range(NQ):
        ax.plot(x, F[:, q], color=PALETTE[q], marker=MARKERS[q], ms=MS, mec="white", mew=MEW,
                lw=LW, alpha=0.85)
    ax.plot(x, gmean(F), color=MEAN_C, marker="X", ms=MS + 0.6, mec="white", mew=MEW, lw=LW + 0.3)


def group_title(fig, axes, text):
    x0 = min(a.get_position().x0 for a in axes)
    x1 = max(a.get_position().x1 for a in axes)
    y1 = max(a.get_position().y1 for a in axes)
    fig.text((x0 + x1) / 2, y1 + HDR_IN / FIG_H, text, ha="center", va="bottom",
             fontsize=GROUP_FS, fontweight="bold")
    fig.add_artist(Line2D([x0, x1], [y1 + RULE_IN / FIG_H] * 2, color="black", lw=0.5,
                          transform=fig.transFigure))


def main():
    left, right = 0.055, 0.995
    fig = plt.figure(figsize=(TEXT_W, FIG_H))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.12, 1, 1], left=left, right=right,
                          top=1 - AX_TOP_IN / FIG_H, bottom=AX_BOT_IN / FIG_H, wspace=0.26)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 3], sharey=ax_a)
    ax_d = fig.add_subplot(gs[0, 4], sharey=ax_a)

    x, F = load_truncation()
    fid_lines(ax_a, x, F)
    style(ax_a, "(a) Fidelity vs. duration", "Readout duration (µs)", "Assignment fidelity")

    xd, R = load_dfdt()
    for q in range(NQ):
        ax_b.plot(xd, R[:, q] / 1e5, color=PALETTE[q], marker=MARKERS[q], ms=MS, mec="white", mew=MEW,
                  lw=LW, alpha=0.85)
        ax_b.axvline(xd[np.nanargmax(R[:, q])], color=PALETTE[q], lw=0.7, ls=":", alpha=0.7)
    ax_b.axhline(0, color="0.3", lw=0.5)
    style(ax_b, "(b) Information-gain rate", "Readout duration (µs)", r"dF/dT ($\times10^5$ s$^{-1}$)")

    for ax, w, tag in [(ax_c, 250, "(c)"), (ax_d, 500, "(d)")]:
        c, Fw = load_window(w)
        fid_lines(ax, c, Fw)
        style(ax, f"{tag} {w} ns window", "Window center (µs)",
              "Assignment fidelity" if w == 250 else None, ylabels=(w == 250))
    ax_a.set_ylim(0.54, 0.99)
    ax_a.set_yticks([0.6, 0.7, 0.8, 0.9])

    group_title(fig, [ax_a, ax_b], "Truncation from onset (§4.1)")
    group_title(fig, [ax_c, ax_d], "Sliding window (§4.2)")
    fig.text((left + right) / 2, 1 - 0.02 / FIG_H, "Higher is better ↑", ha="center", va="top",
             fontsize=LEGEND_FS, fontweight="bold", color="blue")

    handles = [Line2D([0], [0], color=PALETTE[q], marker=MARKERS[q], ms=MS + 0.6, mec="white",
                      mew=MEW, lw=LW) for q in range(NQ)]
    handles.append(Line2D([0], [0], color=MEAN_C, marker="X", ms=MS + 1, mec="white", mew=MEW,
                          lw=LW + 0.3))
    fig.legend(handles, QLABELS + ["geometric mean"], loc="lower center", ncol=NQ + 1,
               bbox_to_anchor=((left + right) / 2, 0.0), fontsize=LEGEND_FS, frameon=False,
               handlelength=1.6, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.1)

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
    print("saved", OUT)


if __name__ == "__main__":
    main()
