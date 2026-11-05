"""
plot_paper_pairs.py
===================
Paper (Oraqle, Sec. 6) versions of the LER-vs-measurement-duration figures, with two
experiments merged LEFT | RIGHT into one full-width figure that shares the y-axis,
the legend (bottom row) and the x-label:

  ler_pair_discriminators.pdf : d=5 (4 codes)        | d=12 (4 codes)          1x(4+4)
  ler_pair_noise.pdf          : current ibm_boston   | low-coherence + backlog  2x3 | 2x3
  ler_pair_future.pdf         : high-fidelity readout| futuristic noise         2x3 | 2x3

Data loading / styling is reused from plot_discriminators.py and plot_single_model.py,
so the curves are identical to the single-experiment figures. Drawn at the ACM
\\textwidth (7.0 in) so the PDF is included 1:1.

Usage:  python plot_paper_pairs.py [OUT_DIR]   (default: experiment_results/paper_pairs)
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

import plot_discriminators as PD
import plot_single_model as PS

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "experiment_results/paper_pairs"
NM = "experiment_results/noise_models"

TEXT_W = 7.0                       # ACM sigconf \textwidth in inches
TITLE_FS, GROUP_FS, LABEL_FS, TICK_FS, LEGEND_FS = 6.4, 7.5, 7, 5.8, 6.5
LW, MS, MEW = 1.0, 3.2, 0.4
BETTER = dict(fontsize=LEGEND_FS, fontweight="bold", color="blue")


def style_ax(ax, title, ylabels):
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=2.5)
    ax.set_xticks([200, 600, 1000])
    ax.set_xlim(90, 1110)          # keep edge tick labels inside the panel
    ax.tick_params(axis="both", labelsize=TICK_FS, length=2, pad=1.5)
    ax.tick_params(axis="y", which="minor", length=1)
    if not ylabels:
        ax.tick_params(axis="y", which="both", labelleft=False)
    ax.set_axisbelow(True)
    ax.grid(color="gray", ls="--", lw=0.3, alpha=0.55, which="both")
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def plot_line(ax, xs, ys, color, marker):
    ax.plot(xs, ys, marker=marker, color=color, markersize=MS, markeredgecolor="black",
            markeredgewidth=MEW, linewidth=LW)


def finish(fig, handles, labels, left, right, out_name, xlabel_y):
    """'Lower is better' centred on top, one y-label, one x-label, and the shared
    legend row centred at the BOTTOM; save pdf+png."""
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=((left + right) / 2, 0.0), fontsize=LEGEND_FS,
               title=None, frameon=False, handlelength=1.6, columnspacing=0.9,
               handletextpad=0.4, borderaxespad=0.1)
    fig.text((left + right) / 2, 0.995, "Lower is better ↓", ha="center", va="top", **BETTER)
    fig.supylabel("Logical error rate (log)", fontsize=LABEL_FS, x=0.004)
    fig.text((left + right) / 2, xlabel_y, "Measurement duration (ns)", ha="center",
             va="bottom", fontsize=LABEL_FS)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, out_name + ".pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print("saved", out)


def group_title(fig, axes, text):
    """Bold header centred over a block of axes."""
    x0 = min(a.get_position().x0 for a in axes)
    x1 = max(a.get_position().x1 for a in axes)
    y1 = max(a.get_position().y1 for a in axes)
    fig.text((x0 + x1) / 2, y1 + 0.075, text, ha="center", va="bottom",
             fontsize=GROUP_FS, fontweight="bold")
    # thin rule under the header to tie the block together
    fig.add_artist(Line2D([x0, x1], [y1 + 0.068] * 2, color="black", lw=0.5,
                          transform=fig.transFigure))


# ---------------------------------------------------------------- discriminators
def discriminators():
    cnn_rows = PD.load(PD.CNN_CSV)
    reuse = [r for r in PD.load(PD.DISC_CSV.format(D=5)) if r["code"] in ("gross", "steane")]

    def rows_for(D):
        own = PD.load(PD.DISC_CSV.format(D=D))
        have = {r["code"] for r in own}
        return own + [r for r in reuse if r["code"] not in have]

    halves = [(5, PD.CODES_4, "d = 5 (responsive)"),
              (12, PD.CODES_4_D12, "d = 12 (saturated)")]
    left, right = 0.055, 0.995
    fig = plt.figure(figsize=(TEXT_W, 1.85))
    gs = fig.add_gridspec(1, 9, width_ratios=[1, 1, 1, 1, 0.22, 1, 1, 1, 1],
                          left=left, right=right, top=0.74, bottom=0.30, wspace=0.17)
    shared = None
    blocks = []
    for h, (D, codes, header) in enumerate(halves):
        disc_rows = rows_for(D)
        axs = []
        for i, (code, title) in enumerate(codes):
            ax = fig.add_subplot(gs[0, i + 5 * h], sharey=shared)
            shared = shared or ax
            d = PD.code_dist(code, D)
            for disc in PD.DISCRIMINATORS:
                if disc == "MCMit-CNN":
                    xs, ys = PD.cnn_series(code, d, disc_rows, cnn_rows)
                else:
                    xs, ys = PD.series(disc_rows, code, d, disc, "discriminator",
                                       "logical_error_rate")
                if xs:
                    plot_line(ax, xs, ys, PD.COLORS[disc], PD.MARKERS[disc])
            style_ax(ax, title, ylabels=(i == 0))
            axs.append(ax)
        blocks.append((axs, header))
    for axs, header in blocks:
        group_title(fig, axs, header)
    handles = [Line2D([0], [0], color=PD.COLORS[s], marker=PD.MARKERS[s], markersize=MS + 0.6,
                      markeredgecolor="black", markeredgewidth=MEW, linewidth=LW)
               for s in PD.DISCRIMINATORS]
    finish(fig, handles, PD.DISCRIMINATORS, left, right,
           "ler_pair_discriminators", xlabel_y=0.115)


# ---------------------------------------------------------------- six-code pairs
def load_six(csv_path):
    rows = list(csv.DictReader(open(csv_path)))

    def series(code, d):
        pts = []
        for r in rows:
            if r["code"] == code and int(r["distance"]) == d:
                y = r["logical_error_rate"]
                y = float(y) if y not in ("", "None") else np.nan
                pts.append((int(r["readout_ns"]), np.nan if y == 0.0 else y))
        pts.sort()
        return [p[0] for p in pts], [p[1] for p in pts]

    def dists(code):
        return sorted({int(r["distance"]) for r in rows if r["code"] == code})

    return series, dists


def filter_csv(src, noise_model, dst):
    """ler_noise_models.csv mixes several models; keep one (e.g. heron_cnn)."""
    rows = [r for r in csv.DictReader(open(src)) if r["noise_model"] == noise_model]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    return dst


def six_pair(left_csv, left_hdr, right_csv, right_hdr, out_name):
    left, right = 0.055, 0.995
    fig = plt.figure(figsize=(TEXT_W, 2.65))
    gs = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 0.16, 1, 1, 1],
                          left=left, right=right, top=0.815, bottom=0.20,
                          wspace=0.12, hspace=0.42)
    shared = None
    blocks = []
    for h, (path, header) in enumerate([(left_csv, left_hdr), (right_csv, right_hdr)]):
        series, dists = load_six(path)
        axs = []
        for k, (code, title) in enumerate(PS.CODES):
            r, c = divmod(k, 3)
            ax = fig.add_subplot(gs[r, c + 4 * h], sharey=shared, sharex=shared)
            shared = shared or ax
            for d in dists(code):
                xs, ys = series(code, d)
                plot_line(ax, xs, ys, PS.COLORS[d], PS.MARKERS[d])
            style_ax(ax, title, ylabels=(c == 0))
            if r == 0:
                ax.tick_params(axis="x", labelbottom=False)
            axs.append(ax)
        blocks.append((axs, header))
    for axs, header in blocks:
        group_title(fig, axs, header)
    handles = [Line2D([0], [0], color=PS.COLORS[d], marker=PS.MARKERS[d], markersize=MS + 0.6,
                      markeredgecolor="black", markeredgewidth=MEW, linewidth=LW)
               for d in PS.DISTS]
    finish(fig, handles, [f"d = {d}" for d in PS.DISTS], left, right, out_name,
           xlabel_y=0.085)


if __name__ == "__main__":
    discriminators()
    cur = filter_csv(f"{NM}/ler_noise_models.csv", "heron_cnn",
                     os.path.join(OUT_DIR, "_heron_cnn.csv"))
    six_pair(cur, "Current ibm_boston noise",
             f"{NM}/ler_noise_models_boston_lowt_backlog.csv",
             "Low coherence + decoder backlog", "ler_pair_noise")
    six_pair(f"{NM}/ler_noise_models_highfid.csv", "High-fidelity readout",
             f"{NM}/ler_noise_models_futuristic.csv", "Futuristic noise",
             "ler_pair_future")
