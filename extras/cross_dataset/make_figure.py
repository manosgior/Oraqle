"""The study's figure, from the committed result files. Needs matplotlib; no datasets.

    python extras/cross_dataset/make_figure.py        # writes results/cross_dataset.{pdf,png} next to this file

Left: linear baseline, readout error against duration, per device. Right: spacetime per logical operation
through the paper's QEC model, with the published chip's readout and with QICK's.
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

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "system")]
from analyze import load, summarize  # noqa: E402  (system/analyze.py)

INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
# Categorical slots in their fixed order; a device keeps its colour in both panels.
COLORS = dict(lienhard="#2a78d6", qick="#eb6834", cao="#1baf7a", hassani="#eda100", zheng_t="#e87ba4", zheng_f="#008300", gunyho="#6250d6")
NAMES = dict(lienhard="Lienhard (published, Q1 Q3 Q4 Q5)", qick="QICK transmon", cao="Cao transmon", hassani="Hassani shunted transmon",
             zheng_t="Zheng transmon", zheng_f="Zheng fluxonium", gunyho="Gunyhó bolometer")


def lienhard_curve():
    T, E = [], []
    for L in range(50, 1000, 50):
        f = sorted(glob.glob(str(REPO / "results" / "discriminators" / f"Baseline_Threshold_len{L}_*.csv")))[-1]
        r = list(csv.DictReader(open(f)))[0]
        F = [float(r[f"qubit_{q}_accuracy"]) / 100 for q in (0, 2, 3, 4)]
        T.append(2 * L)
        E.append(1 - math.prod(F) ** 0.25)
    return np.array(T), np.array(E)


def curves():
    c = dict(lienhard=lienhard_curve())
    for n in ("qick", "cao", "gunyho"):
        r = json.load(open(RES / f"linear_{n}.json"))
        c[n] = np.array(r["T_ns"]), 1 - np.array(r["F"])
    r = json.load(open(RES / "linear_hassani.json"))["powers"]["neg16"]
    c["hassani"] = np.array(r["T_ns"]), 1 - np.array(r["F"])
    z = json.load(open(RES / "linear_zheng.json"))["devices"]
    c["zheng_t"] = np.array(z["transmon"]["T_ns"]), 1 - np.array(z["transmon"]["F"])
    c["zheng_f"] = np.array(z["fluxonium1"]["T_ns"]), 1 - np.array(z["fluxonium1"]["F"])
    return c


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, which="both", length=3)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.5), gridspec_kw=dict(width_ratios=[1.35, 1]), facecolor=SURFACE)
    handles = {}
    for name, (T, E) in curves().items():
        (ln,) = a.plot(T / 1e3, E, color=COLORS[name], lw=1.5, solid_capstyle="round", label=NAMES[name])
        handles[name] = ln
        i = int(np.argmin(E))
        a.plot(T[i] / 1e3, E[i], "o", ms=6, color=COLORS[name], mec=SURFACE, mew=1.2, zorder=5)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(0.09, 60)
    a.set_ylim(0.025, 0.55)
    a.set_yticks([0.03, 0.05, 0.1, 0.2, 0.5])
    a.set_yticklabels(["3%", "5%", "10%", "20%", "50%"])
    a.set_xticks([0.1, 0.3, 1, 3, 10, 30])
    a.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    a.minorticks_off()
    a.axvline(1, color=AXIS, lw=0.9, ls=(0, (3, 3)), zorder=1)
    a.text(1.05, 0.5, "1 µs", color=MUTED, fontsize=8, va="top")
    a.set_xlabel("Readout duration T from signal onset (µs), dots mark each device's minimum", color=MUTED, fontsize=9)
    a.set_ylabel("Readout error, 1 − F (linear baseline)", color=MUTED, fontsize=9)
    a.set_title("Where readout saturates depends on the device", color=INK, fontsize=10, loc="left")
    style(a)

    pub = [r for r in summarize(load(str(REPO / "results" / "system" / "ler_disc.jsonl"))) if r["label"] == "MCMit-CNN"]
    qk = [r for r in json.load(open(RES / "ler_qick.summary.json")) if r["label"] == "qick/cnn"]
    for rows, name, lab in ((pub, "lienhard", "Lienhard, MCMit-CNN (published)"), (qk, "qick", "QICK, MCMit-CNN")):
        ok = [r for r in rows if r.get("spacetime@1e-09")]
        x = np.array([r["T"] for r in ok]) / 1e3
        y = np.array([r["spacetime@1e-09"] for r in ok])
        b.plot(x, y, "-o", color=COLORS[name], lw=1.5, ms=5, mec=SURFACE, mew=1, label=lab)
        k = int(np.argmin(y))
        b.annotate(f"{lab.split(',')[0]}: {x[k]:g} µs", (x[k], y[k]), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=8, color=INK)
    b.text(0.33, 1.5e7, "QICK: no error suppression\nat T ≤ 0.6 µs", color=MUTED, fontsize=8, va="center")
    b.set_yscale("log")
    b.set_xlim(0.25, 2.55)
    b.set_ylim(9e4, 3e7)
    b.set_xlabel("Readout duration T (µs)", color=MUTED, fontsize=9)
    b.set_ylabel("Spacetime per logical op at 10⁻⁹ (qubit·µs)", color=MUTED, fontsize=9)
    b.set_title("The QEC optimum moves with it (same gates)", color=INK, fontsize=10, loc="left")
    style(b)
    fig.text(0.5, 0.965, "Lower is better ↓", ha="center", va="center", color=MUTED, fontsize=9)
    fig.legend(handles=list(handles.values()), loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(RES / "cross_dataset.pdf")
    fig.savefig(RES / "cross_dataset.png", dpi=110)


if __name__ == "__main__":
    main()
