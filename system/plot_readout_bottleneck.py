"""Figures for parts A and C from results/system/summary.json (analyze.py).

  plots/readout_spacetime.pdf   (a) Lambda over readout length x readout error, with each deployable
                                    discriminator's operating curve; (b) physical qubits per logical
                                    qubit and (c) spacetime per logical operation at 1e-9 per op
  plots/decoder_load.pdf        (a) PyMatching cores per patch to keep pace with the round, vs d, one
                                    line per readout length; (b) decode time per round relative to a
                                    1000 ns readout, one line per distance
  plots/system_cost.pdf         program runtime and size, syndrome rate and decoder FPGAs (system_cost.py)
  plots/soft_readout.pdf        soft readout as erasures (summary_soft.json)
  plots/decoder_choice.pdf      Lambda and cost per decoder type and per hardware decoder (I6)

Colors: validated categorical palette (dataviz reference, slots 1-4) for discriminators; one-hue blue
ordinal ramp for readout length and distance.
Usage: python system/plot_readout_bottleneck.py [--data results/system] [--out $ORAQLE_OUT/paper_figures/system]
       [--only spacetime decoder system soft decoder_choice]
(the plots/ names above are the file names the paper includes; they are written to --out)
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, sys_path)
from discriminators import DURATIONS, measure_error

sys.path.insert(1, os.path.dirname(sys_path))
from oraqle_env import OUT as _OUT_ROOT, RESULTS  # noqa: E402

DATA = str(RESULTS / "system")                         # was <oraqle-ics>/data/sims
OUT = str(_OUT_ROOT / "paper_figures" / "system")      # was <oraqle-ics>/plots
S = None                                               # summary.json, loaded by configure()


def configure(data=None, out=None):
    global DATA, OUT, S
    DATA, OUT = data or DATA, out or OUT
    os.makedirs(OUT, exist_ok=True)
    S = json.load(open(os.path.join(DATA, "summary.json")))

INK, MUTED, GRID = "#333333", "#666666", "#dddddd"
BETTER = "#1155cc"
DISC = [("MCMit-CNN", "#2a78d6", "o"), ("HERQULES", "#eb6834", "s"),
        ("KLiNQ", "#1baf7a", "^"), ("QubiCML", "#eda100", "D")]
RAMP = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]       # ordinal blue, light -> dark
TARGET = "1e-09"
REFERENCE = "Micro Blossom"                                         # hardware decoder of the main results
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "pdf.fonttype": 42})


def style(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=INK)


def banner(ax, text):
    ax.text(0.5, 1.03, text, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.8,
            fontweight="bold", color=BETTER)


def title(ax, text):
    ax.set_title(text, fontsize=7.2, color=INK, fontweight="bold", pad=13)


def spacetime_figure():
    disc = {(s["label"], s["T"]): s for s in S["disc_hi"]}
    grid = S["grid"]
    Ts = sorted({s["T"] for s in grid})
    ps = sorted({s["p"] for s in grid})
    lam = np.array([[next(s for s in grid if s["T"] == T and s["p"] == p).get("Lambda", np.nan) for T in Ts]
                    for p in ps], dtype=float)
    lam[lam < 1] = np.nan                                          # above threshold

    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.25), gridspec_kw=dict(wspace=0.62))

    ax = axs[0]
    ax.set_facecolor("#eeeeee")
    levels = [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 7]
    cmap = matplotlib.colors.ListedColormap(["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                                             "#5598e7", "#3987e5", "#256abf"])
    cf = ax.contourf(Ts, ps, lam, levels=levels, cmap=cmap)
    cs = ax.contour(Ts, ps, lam, levels=[2], colors=INK, linewidths=0.8, linestyles="--")
    ax.clabel(cs, fmt={2: r"$\Lambda{=}2$"}, fontsize=6, inline_spacing=2, manual=[(290, 0.062)])
    for name, c, m in DISC:
        ax.plot(DURATIONS, [measure_error(name, T) for T in DURATIONS], color=c, marker=m, ms=3.6, lw=1.2,
                mec="black", mew=0.4, zorder=4)
    ax.text(420, 0.205, "above threshold", fontsize=6, color=MUTED, ha="left", va="center")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 0.25)
    ax.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2], ["1%", "2%", "5%", "10%", "20%"])
    ax.minorticks_off()
    ax.set_xticks([200, 400, 600, 800, 1000])
    ax.set_xlabel("Readout length (ns)")
    ax.set_ylabel("Readout error")
    cb = fig.colorbar(cf, ax=ax, pad=0.025, fraction=0.05, ticks=[1, 1.5, 2, 3, 5, 7])
    cb.ax.tick_params(labelsize=6, colors=INK)
    cb.set_label(r"$\Lambda$", fontsize=7)
    cb.outline.set_edgecolor(GRID)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    banner(ax, "Higher is better ↑")
    title(ax, "(a) Error suppression $\\Lambda$")

    Tx = [400, 600, 800, 1000]
    base = disc[("MCMit-CNN", 1000)][f"spacetime@{TARGET}"]
    for ax, key, ylab, ttl in ((axs[1], f"qubits@{TARGET}", "Physical qubits / logical qubit",
                                "(b) Space"),
                               (axs[2], f"spacetime@{TARGET}", "Spacetime / logical op\n(rel. to CNN @ 1000 ns)",
                                "(c) Spacetime")):
        for name, c, m in DISC:
            ys = [disc[(name, T)][key] / (base if "spacetime" in key else 1) for T in Tx]
            ax.plot(Tx, ys, color=c, marker=m, ms=4.2, lw=1.4, mec="black", mew=0.5, zorder=3)
        ax.axvspan(150, 300, color="#eeeeee", lw=0, zorder=0)
        ax.text(225, 0.5, "above\nthreshold", transform=ax.get_xaxis_transform(), fontsize=6, color=MUTED,
                ha="center", va="center", rotation=90)
        ax.set_xlim(150, 1050)
        ax.set_xticks([200, 400, 600, 800, 1000])
        ax.set_yscale("log")
        ax.set_xlabel("Readout length (ns)")
        ax.set_ylabel(ylab)
        style(ax)
        banner(ax, "Lower is better ↓")
        title(ax, ttl)
    axs[2].axhline(1, color=MUTED, lw=0.7, ls=":", zorder=1)
    axs[1].set_ylim(2e3, 5e4)
    axs[1].set_yticks([2e3, 5e3, 1e4, 2e4, 5e4], ["2k", "5k", "10k", "20k", "50k"])
    axs[1].minorticks_off()
    axs[2].set_ylim(0.4, 20)
    axs[2].set_yticks([0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])

    handles = [Line2D([0], [0], color=c, marker=m, ms=4.2, mec="black", mew=0.5, lw=1.4, label=n)
               for n, c, m in DISC]
    handles.append(Patch(facecolor="#eeeeee", label="Above threshold ($\\Lambda<1$)"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=5, frameon=False,
               fontsize=6.6, handlelength=1.6, columnspacing=1.4)
    fig.subplots_adjust(bottom=0.26, top=0.83, left=0.07, right=0.99)
    fig.savefig(os.path.join(OUT, "readout_spacetime.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "readout_spacetime.png"), bbox_inches="tight", dpi=200)


def decoder_figure():
    dec = S["decode"]
    ds = sorted({r["d"] for r in dec})
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.4), gridspec_kw=dict(wspace=0.3))

    ax = axs[0]
    for i, T in enumerate(DURATIONS):
        rows = sorted((r for r in dec if r["T"] == T), key=lambda r: r["d"])
        # two patches per experiment; cores = per-patch decode time per round / round length
        ax.plot([r["d"] for r in rows], [r["batch_ns_per_round"] / 2 / (T + 340) for r in rows],
                color=RAMP[i], marker="o", ms=3.8, lw=1.4, mec="black", mew=0.4, label=f"{T} ns")
    ax.axhline(1, color=MUTED, lw=0.7, ls=":")
    ax.set_yscale("log")
    ax.set_xticks(ds)
    ax.set_xlabel("Code distance $d$")
    ax.set_ylabel("CPU cores per patch\n(to keep pace with the round)")
    style(ax)
    banner(ax, "Lower is better ↓")
    title(ax, "(a) Decoder provisioning")
    ax.legend(title="Readout length", title_fontsize=6.4, fontsize=6.2, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.3), ncol=5, handlelength=1.4, columnspacing=1.0)

    ax = axs[1]
    sel = [5, 9, 13, 17]
    for i, d in enumerate(sel):
        rows = {r["T"]: r for r in dec if r["d"] == d}
        ax.plot(DURATIONS, [rows[T]["batch_ns_per_round"] / rows[1000]["batch_ns_per_round"] for T in DURATIONS],
                color=RAMP[i + 1], marker="o", ms=3.8, lw=1.4, mec="black", mew=0.4, label=f"$d{{=}}{d}$")
    ax.axhline(1, color=MUTED, lw=0.7, ls=":")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_xticks(DURATIONS)
    ax.set_xlabel("Readout length (ns)")
    ax.set_ylabel("Decode time per round\n(rel. to 1000 ns readout)")
    style(ax)
    banner(ax, "Lower is better ↓")
    title(ax, "(b) Decoder work per round")
    ax.legend(title="Code distance", title_fontsize=6.4, fontsize=6.2, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.3), ncol=4, handlelength=1.4, columnspacing=1.0)
    fig.subplots_adjust(top=0.82, bottom=0.2, left=0.09, right=0.99)
    fig.savefig(os.path.join(OUT, "decoder_load.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "decoder_load.png"), bbox_inches="tight", dpi=200)


def system_figure():
    """plots/system_cost.pdf: (a) runtime and (b) physical qubits of each workload (1% failure budget,
    Helios reaction wait) for four readout operating points; (c) hard-decision syndrome bandwidth and
    (d) Helios V80 cards per logical qubit at 1e-9 per op, vs readout length (system_cost.py)."""
    C = json.load(open(os.path.join(DATA, "system_cost.json")))
    configs = [("MCMit-CNN/1000/Micro Blossom", "CNN, 1000 ns", "#2a78d6", None),
               ("MCMit-CNN/600/Micro Blossom", "CNN, 600 ns", "#86b6ef", None),
               ("MCMit-CNN/400/Micro Blossom", "CNN, 400 ns", "#2a78d6", "////"),
               ("QubiCML/1000/Micro Blossom", "QubiCML, 1000 ns", "#eda100", None)]
    names = list(C["workloads"])
    fig, axs = plt.subplots(1, 4, figsize=(7.1, 2.3), gridspec_kw=dict(wspace=0.62, width_ratios=[1.35, 1.35, 1, 1]))
    x = np.arange(len(names))
    wdt = 0.8 / len(configs)
    for ax, key, ylab, ttl, scale in ((axs[0], "runtime_s", "Program runtime (s)", "(a) Time-to-solution", 1),
                                      (axs[1], "qubits", "Physical qubits", "(b) Machine size", 1)):
        for i, (cfg, lab, c, hatch) in enumerate(configs):
            ys = [C["workloads"][n]["configs"][cfg][key] * scale for n in names]
            ax.bar(x + (i - 1.5) * wdt, ys, wdt * 0.92, color=c if not hatch else "white", edgecolor=c if hatch else "white",
                   hatch=hatch, lw=0.6 if hatch else 0.3, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(x, names, rotation=40, ha="right", fontsize=6)
        ax.set_ylabel(ylab)
        style(ax)
        ax.grid(axis="x", visible=False)
        banner(ax, "Lower is better ↓")
        title(ax, ttl)

    Tx = [400, 600, 800, 1000]
    for ax, key, ylab, ttl, ticks, lim in (
            (axs[2], "gbps_hard", "Syndrome Gb/s / logical qubit", "(c) Syndrome rate", [1, 2, 5, 10, 20, 50], (1.2, 60)),
            (axs[3], "v80_microblossom", "Decoder FPGAs / logical qubit\n(Micro Blossom on V80)", "(d) Decoding",
             [10, 20, 50, 100, 200, 500], (7, 800))):
        for name, c, m in DISC:
            ys = [C["per_lq"][f"{name}/{T}"][key] for T in Tx]
            ax.plot(Tx, ys, color=c, marker=m, ms=3.8, lw=1.3, mec="black", mew=0.4, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(Tx, ["400", "600", "800", "1000"], fontsize=6)
        ax.set_yticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_ylim(*lim)
        ax.set_xlabel("Readout length (ns)")
        ax.set_ylabel(ylab)
        style(ax)
        banner(ax, "Lower is better ↓")
        title(ax, ttl)

    h = [Patch(facecolor=c if not hatch else "white", edgecolor=c if hatch else "none", hatch=hatch, label=lab)
         for _, lab, c, hatch in configs]
    h += [Line2D([0], [0], color=c, marker=m, ms=3.8, mec="black", mew=0.4, lw=1.3, label=n) for n, c, m in DISC]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=8, frameon=False, fontsize=6,
               handlelength=1.4, columnspacing=0.9)
    fig.subplots_adjust(bottom=0.3, top=0.82, left=0.06, right=0.995)
    fig.savefig(os.path.join(OUT, "system_cost.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "system_cost.png"), bbox_inches="tight", dpi=200)


def soft_figure():
    """plots/soft_readout.pdf: (a) Lambda vs the flagged fraction, one line per readout length; (b) spacetime
    per logical op and (c) syndrome bandwidth per logical qubit at 1e-9 per op, hard vs the best flag."""
    SS = json.load(open(os.path.join(DATA, "summary_soft.json")))
    pts = {(s["T"], float(s["label"][6:])): s for s in SS}
    Ts = [400, 600, 800, 1000]
    fs = sorted({f for _, f in pts})
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.25), gridspec_kw=dict(wspace=0.42))

    ax = axs[0]
    for i, T in enumerate(Ts):
        ax.plot([100 * f for f in fs], [pts[(T, f)].get("Lambda", np.nan) for f in fs], color=RAMP[i + 1],
                marker="o", ms=3.8, lw=1.4, mec="black", mew=0.4, label=f"{T} ns")
    ax.axhline(2, color=MUTED, lw=0.7, ls=":")
    ax.set_xlabel("Flagged measurements (%)")
    ax.set_ylabel(r"$\Lambda$")
    ax.set_xticks([0, 5, 10, 15, 20])
    style(ax)
    banner(ax, "Higher is better ↑")
    title(ax, "(a) Error suppression")
    ax.legend(title="Readout length", title_fontsize=6.2, fontsize=6, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.3), ncol=4, handlelength=1.3, columnspacing=0.8)

    base = pts[(1000, 0.0)]["spacetime@1e-09"]
    best = {T: min((f for f in fs if pts[(T, f)].get("spacetime@1e-09")), key=lambda f: pts[(T, f)]["spacetime@1e-09"])
            for T in Ts}
    HARD, SOFT = ("#2a78d6", "o"), ("#eb6834", "s")
    ax = axs[1]
    ax.plot(Ts, [pts[(T, 0.0)]["spacetime@1e-09"] / base for T in Ts], color=HARD[0], marker=HARD[1], ms=4.2,
            lw=1.4, mec="black", mew=0.5, zorder=3)
    ax.plot(Ts, [pts[(T, best[T])]["spacetime@1e-09"] / base for T in Ts], color=SOFT[0], marker=SOFT[1], ms=4.2,
            lw=1.4, mec="black", mew=0.5, zorder=3)
    for T in Ts:
        ax.annotate(f"{100 * best[T]:.0f}%", (T, pts[(T, best[T])]["spacetime@1e-09"] / base), textcoords="offset points",
                    xytext=(0, -9), ha="center", fontsize=5.6, color=MUTED)
    ax.axhline(1, color=MUTED, lw=0.7, ls=":")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5], ["0.5", "1", "2", "5"])
    ax.minorticks_off()
    ax.set_ylim(0.3, 9)
    ax.set_xticks(Ts)
    ax.set_xlabel("Readout length (ns)")
    ax.set_ylabel("Spacetime / logical op\n(rel. to hard @ 1000 ns)")
    style(ax)
    banner(ax, "Lower is better ↓")
    title(ax, "(b) Spacetime")

    ax = axs[2]
    for (c, m), bits, sel in ((HARD, 1, {T: 0.0 for T in Ts}), (SOFT, 2, best)):
        ys = []
        for T in Ts:
            d = pts[(T, sel[T])]["d@1e-09"]
            ys.append((d * d - 1) * bits / ((T + 340) * 1e-9) / 1e9)
        ax.plot(Ts, ys, color=c, marker=m, ms=4.2, lw=1.4, mec="black", mew=0.5, zorder=3)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_xticks(Ts)
    ax.set_xlabel("Readout length (ns)")
    ax.set_ylabel("Syndrome Gb/s / logical qubit")
    style(ax)
    banner(ax, "Lower is better ↓")
    title(ax, "(c) Bandwidth")

    h = [Line2D([0], [0], color=HARD[0], marker=HARD[1], ms=4.2, mec="black", mew=0.5, lw=1.4, label="Hard readout (1 bit)"),
         Line2D([0], [0], color=SOFT[0], marker=SOFT[1], ms=4.2, mec="black", mew=0.5, lw=1.4,
                label="Soft readout, best flag fraction (2 bits)")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.68, -0.03), ncol=2, frameon=False, fontsize=6.6,
               handlelength=1.6, columnspacing=1.4)
    fig.subplots_adjust(bottom=0.27, top=0.83, left=0.06, right=0.99)
    fig.savefig(os.path.join(OUT, "soft_readout.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "soft_readout.png"), bbox_inches="tight", dpi=200)


def decoder_choice_figure(program="Ising-100"):
    """plots/decoder_choice.pdf (I6): (a) Lambda and (b) spacetime per logical op at 1e-9 per decoder type, hard CNN
    readout (decoders_summary.json); (c) runtime and (d) physical qubits of one program per hardware decoder, each
    with its own reaction wait and the Lambda of the simulated decoder it can match (system_cost.json)."""
    D = json.load(open(os.path.join(DATA, "decoders_summary.json")))
    C = json.load(open(os.path.join(DATA, "system_cost.json")))["decoder_choice"][program]
    rows = {(r["T"], r["f"], r["decoder"]): r for r in D}
    Tx = [400, 600, 800, 1000]
    # colour follows the accuracy class: blue = full weights, orange = uniform weights (as in panels c, d)
    SIM = [("mwpm_w", "Weighted matching", "#2a78d6", "o", "-", "black"),
           ("mwpm_q16", "16-level weights", "#2a78d6", "o", "--", "white"),
           ("mwpm_c", "Correlated matching", "#1baf7a", "^", "-", "black"),
           ("mwpm_q4", "4-level weights", "#eda100", "D", "-", "black"),
           ("mwpm_u", "Uniform weights", "#eb6834", "s", "-", "black")]
    fig, axs = plt.subplots(1, 4, figsize=(7.1, 2.3),
                            gridspec_kw=dict(wspace=0.55, width_ratios=[1, 1, 1.3, 1.3]))

    for ax, key in ((axs[0], "Lambda"), (axs[1], "spacetime_rel")):
        for dec, lab, c, m, ls, face in SIM:
            ys = [rows[(T, 0.0, dec)].get(key) or np.nan for T in Tx]
            ax.plot(Tx, ys, color=c, marker=m, ms=4 if face == "black" else 3.4, lw=1.3, ls=ls,
                    mfc=c if face == "black" else "white", mec="black" if face == "black" else c, mew=0.5 if face == "black" else 0.9,
                    zorder=3 if face == "black" else 4)
        ax.set_xticks(Tx, ["400", "600", "800", "1000"], fontsize=6)
        ax.set_xlim(330, 1070)
        ax.set_xlabel("Readout length (ns)")
        style(ax)
    ax = axs[0]
    ax.axhspan(0.5, 1, color="#eeeeee", lw=0, zorder=0)
    ax.text(700, 0.8, "above threshold", fontsize=5.8, color=MUTED, ha="center", va="center")
    ax.set_ylim(0.6, 3.0)
    ax.set_ylabel(r"$\Lambda$")
    banner(ax, "Higher is better ↑")
    title(ax, "(a) Error suppression")
    ax = axs[1]
    ax.axhline(1, color=MUTED, lw=0.7, ls=":", zorder=1)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_ylim(0.35, 40)
    ax.set_ylabel("Spacetime / logical op\n(rel. to weighted @ 1000 ns)")
    banner(ax, "Lower is better ↓")
    title(ax, "(b) Spacetime")

    HW = [("Micro Blossom", "Micro Blossom (exact)", "#2a78d6", None),
          ("Helios (weighted)", "Helios, 16-level weights", "white", "////"),
          ("Helios", "Helios, unweighted", "#eb6834", None),
          ("GPU", "GPU (reported)", "#9a9a9a", None)]
    groups = [(1000, 0.0, "1000"), (600, 0.0, "600"), (400, 0.0, "400"), (600, 0.05, "600\n+5%"),
              (400, 0.1, "400\n+10%")]
    base = C[f"1000/0.0/{REFERENCE}"]
    x = np.arange(len(groups))
    wdt = 0.8 / len(HW)
    for ax, key, ylab, ttl, ticks, lim in (
            (axs[2], "runtime_s", f"Runtime ({program})\n(rel. to Micro Blossom @ 1000 ns)", "(c) Time-to-solution",
             [0.5, 1, 2, 3], (0.4, 3.6)),
            (axs[3], "qubits", f"Physical qubits ({program})\n(rel. to Micro Blossom @ 1000 ns)", "(d) Machine size",
             [0.5, 1, 2, 5, 10, 20], (0.4, 25))):
        for i, (dec, lab, c, hatch) in enumerate(HW):
            for j, (T, f, _) in enumerate(groups):
                r = C.get(f"{T}/{f}/{dec}")
                xi = x[j] + (i - 1.5) * wdt
                if r is None:
                    ax.text(xi, lim[0] * 1.15, "×", ha="center", va="bottom", fontsize=6.5, color=HW[i][2] if c != "white" else "#2a78d6")
                    continue
                ax.bar(xi, r[key] / base[key], wdt * 0.88, bottom=0.0, color=c,
                       edgecolor="#2a78d6" if hatch else "white", hatch=hatch, lw=0.6 if hatch else 0.3, zorder=3)
        ax.axhline(1, color=MUTED, lw=0.7, ls=":", zorder=4)
        ax.set_yscale("log")
        ax.set_yticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_ylim(*lim)
        ax.set_xticks(x, [g[2] for g in groups], fontsize=5.8)
        ax.set_xlabel("Readout length (ns) + flagged")
        ax.set_ylabel(ylab, fontsize=6.4)
        style(ax)
        ax.grid(axis="x", visible=False)
        banner(ax, "Lower is better ↓")
        title(ax, ttl)

    h = [Line2D([0], [0], color=c, marker=m, ms=4 if face == "black" else 3.4, ls=ls, lw=1.3,
                mfc=c if face == "black" else "white", mec="black" if face == "black" else c,
                mew=0.5 if face == "black" else 0.9, label=lab) for _, lab, c, m, ls, face in SIM]
    h += [Patch(facecolor=c, edgecolor="#2a78d6" if hatch else "none", hatch=hatch, lw=0.6, label=lab)
          for _, lab, c, hatch in HW]
    h.append(Line2D([0], [0], color="none", marker="$×$", mfc=MUTED, mec=MUTED, ms=5, label="above threshold"))
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=6, frameon=False, fontsize=6,
               handlelength=1.6, columnspacing=1.1)
    fig.subplots_adjust(bottom=0.34, top=0.82, left=0.06, right=0.995)
    fig.savefig(os.path.join(OUT, "decoder_choice.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "decoder_choice.png"), bbox_inches="tight", dpi=200)


FIGURES = {"spacetime": lambda: spacetime_figure(), "decoder": lambda: decoder_figure(),
           "system": lambda: system_figure(), "soft": lambda: soft_figure(),
           "decoder_choice": lambda: decoder_choice_figure()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--only", nargs="*", choices=list(FIGURES), default=list(FIGURES))
    a = ap.parse_args()
    configure(a.data, a.out)
    for name in a.only:
        if name == "soft" and not os.path.exists(os.path.join(DATA, "summary_soft.json")):
            continue
        if name == "decoder_choice" and not os.path.exists(os.path.join(DATA, "decoders_summary.json")):
            continue
        FIGURES[name]()
    print(f"wrote {', '.join(a.only)} figures (.pdf, .png) to {OUT}")
