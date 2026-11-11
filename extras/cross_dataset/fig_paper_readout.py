"""Paper figure for Sec. 4: readout duration and window position across devices (two panels, full width).

  (a) assignment fidelity of the integration-and-threshold baseline against readout duration from signal onset
  (b) fidelity of a 500 ns integration window against its position

Lienhard: published sweep (unfiltered) and the unfiltered window recomputation (lienhard_windows.py), both as the
geometric mean of Q1, Q3, Q4, Q5. Other devices: results/linear_*.json. Everything unfiltered.

    python extras/cross_dataset/fig_paper_readout.py ../oraqle-ics/plots/readout_datasets.pdf
"""
import csv
import glob
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
REPO = HERE.parents[1]
OUT = sys.argv[1] if len(sys.argv) > 1 else str(RES / "readout_datasets.pdf")

PAL = sns.color_palette("deep")
DEV = [  # name, colour, marker, linestyle
    ("Lienhard", PAL[0], "o", "-"),
    ("QICK", PAL[1], "s", "-"),
    ("Cao", PAL[2], "^", "-"),
    ("Hassani", PAL[3], "D", "-"),
    ("Zheng transmon", PAL[4], "v", "--"),
    ("Zheng fluxonium", PAL[5], "P", "--"),
]
TEXT_W, FIG_H = 7.0, 1.62
TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS = 6.6, 6.6, 5.8, 6.4
LW, MS, MEW = 0.9, 2.4, 0.35
CUE = "#1155cc"
Q4 = (0, 2, 3, 4)  # Q2 is SNR-limited for every model and is left out, as in Sec. 6 and 7


def load_json(name):
    return json.load(open(RES / f"{name}.json"))


def duration_curves():
    T, F = [], []
    for L in range(50, 1000, 50):
        f = sorted(glob.glob(str(REPO / "results" / "discriminators" / f"Baseline_Threshold_len{L}_*.csv")))[-1]
        r = next(csv.DictReader(open(f)))
        T.append(2 * L / 1e3)
        F.append(math.prod(float(r[f"qubit_{q}_accuracy"]) / 100 for q in Q4) ** 0.25)
    c = {"Lienhard": (np.array(T), np.array(F))}
    for n, k in (("QICK", "linear_qick"), ("Cao", "linear_cao")):
        r = load_json(k)
        c[n] = np.array(r["T_ns"]) / 1e3, np.array(r["F"])
    h = load_json("linear_hassani")["powers"]["neg16"]
    c["Hassani"] = np.array(h["T_ns"]) / 1e3, np.array(h["F"])
    z = load_json("linear_zheng")["devices"]
    c["Zheng transmon"] = np.array(z["transmon"]["T_ns"]) / 1e3, np.array(z["transmon"]["F"])
    c["Zheng fluxonium"] = np.array(z["fluxonium1"]["T_ns"]) / 1e3, np.array(z["fluxonium1"]["F"])
    return c


def window_curves():
    c = {}
    if (RES / "linear_lienhard_windows.json").exists():
        w = load_json("linear_lienhard_windows")
        x = np.array(w["Q1"]["window_500ns"]["center_us"])
        F = np.array([w[f"Q{q + 1}"]["window_500ns"]["F"] for q in Q4])
        c["Lienhard"] = (x, np.exp(np.log(F).mean(0)))
    for n, k in (("QICK", "linear_qick"), ("Cao", "linear_cao")):
        r = load_json(k)["window_500ns"]
        c[n] = np.array(r["center_us"]), np.array(r["F"])
    h = load_json("linear_hassani")["powers"]["neg16"]["window_500ns"]
    c["Hassani"] = np.array(h["center_us"]), np.array(h["F"])
    return c


def style(ax, title, xlabel, ylabel=None):
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=2.5)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS, labelpad=1)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FS, labelpad=1.5)
    ax.tick_params(axis="both", labelsize=TICK_FS, length=2, pad=1.5)
    ax.set_axisbelow(True)
    ax.grid(color="gray", ls="--", lw=0.3, alpha=0.4)
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def draw(ax, curves, mark_best):
    for n, col, mk, ls in DEV:
        if n not in curves:
            continue
        x, F = curves[n]
        every = max(1, len(x) // 12)
        ax.plot(x, F, color=col, ls=ls, lw=LW, marker=mk, ms=MS, mec="white", mew=MEW, markevery=every, alpha=0.9)
        if mark_best:
            i = int(np.argmax(F))
            ax.plot(x[i], F[i], marker=mk, ms=MS + 2.2, color=col, mec="black", mew=0.5, zorder=5)


def main():
    left, right = 0.055, 0.995
    fig = plt.figure(figsize=(TEXT_W, FIG_H))
    gs = fig.add_gridspec(1, 2, left=left, right=right, top=1 - 0.30 / FIG_H, bottom=0.45 / FIG_H, wspace=0.12)
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1], sharey=a)

    draw(a, duration_curves(), mark_best=True)
    a.set_xscale("log")
    a.set_xlim(0.09, 22)
    a.set_xticks([0.1, 0.3, 1, 3, 10])
    a.set_xticklabels(["0.1", "0.3", "1", "3", "10"])
    a.axvline(1, color="0.35", lw=0.6, ls=":")
    style(a, "(a) Fidelity vs. readout duration (black edge: best)", "Readout duration from signal onset (µs, log)",
          "Assignment fidelity")

    draw(b, window_curves(), mark_best=True)
    b.set_xlim(0, 2.7)
    b.tick_params(axis="y", labelleft=False)
    style(b, "(b) Fidelity of one 500 ns window vs. its position", "Window center from signal onset (µs)")
    a.set_ylim(0.48, 1.0)
    a.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    fig.text((left + right) / 2, 1 - 0.03 / FIG_H, "Higher is better ↑", ha="center", va="top",
             fontsize=LEGEND_FS, fontweight="bold", color=CUE)
    handles = [Line2D([0], [0], color=c, marker=m, ls=ls, ms=MS + 0.6, mec="white", mew=MEW, lw=LW) for _, c, m, ls in DEV]
    labels = [n + (" (sweep)" if n.startswith("Zheng") else "") for n, *_ in DEV]
    fig.legend(handles, labels, loc="lower center", ncol=len(DEV), bbox_to_anchor=((left + right) / 2, 0.0),
               fontsize=LEGEND_FS, frameon=False, handlelength=2.2, columnspacing=1.4)
    fig.savefig(OUT, bbox_inches=None)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
