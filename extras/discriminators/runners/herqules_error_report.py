"""Aggregate the HERQULES error-mode analysis CSVs into two tidy tables
(fake vs realistic) and plot the error-mode breakdown for Q3 & Q5 at both
readout lengths (1000 ns / 400 ns).

Reads the per-run CSVs written by analyze_herqules_errors.run():
  herqules_error_modes_len{500,200}_{scrambled,pertrace}_*.csv
(picks the most recent of each), and writes:
  herqules_error_breakdown_fake.csv
  herqules_error_breakdown_realistic.csv
  herqules_error_breakdown.pdf / .png   (2 panels: fake | realistic)
"""
import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

REPORTS = (str(REPORTS_OUT))

# exclusive error categories and the analyze-CSV column they live in
CATS = ["near_boundary", "t1_pattern", "amp_outlier"]   # "other" dropped (always empty)
CAT_COL = {
    "near_boundary": "excl_n_near_boundary",
    "t1_pattern":    "excl_n_t1_pattern",
    "amp_outlier":   "excl_n_amp_outlier",
    "other":         "excl_n_other",
}
CAT_LABEL = {
    "near_boundary": "Near boundary",
    "t1_pattern":    "Relaxation",
    "amp_outlier":   "Excitation",
    "other":         "Other",
}
# (trace_samples, readout_ns)
LENGTHS = [(500, 1000), (200, 400)]
FOCUS = [(2, "Q3"), (4, "Q5")]   # 0-indexed qubit -> paper label

AXIS_FS, TITLE_FS, SUP_FS, LEG_FS = 12, 14, 16, 10

# Distinct hatch per error category (redundant with colour -> survives grayscale).
HATCH = {
    "near_boundary": "",
    "t1_pattern":    "//",
    "amp_outlier":   "xx",
    "other":         "..",
}


def _latest(length, split):
    g = sorted(glob.glob(os.path.join(
        REPORTS, f"herqules_error_modes_len{length}_{split}_*.csv")))
    return g[-1] if g else None


def _load_mode(split):
    """split = 'scrambled' (fake) or 'pertrace' (realistic)."""
    rows = []
    for L, ns in LENGTHS:
        f = _latest(L, split)
        if f is None:
            print(f"[report] WARNING: no CSV for len{L} {split}")
            continue
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            q = int(r["qubit"])
            row = {
                "paper_qubit": f"Q{q+1}",
                "qubit_idx": q,
                "length_ns": ns,
                "trace_samples": L,
                "n_test": int(r["n_test"]),
                "n_error": int(r["n_error"]),
                "accuracy_pct": float(r["accuracy_pct"]),
            }
            for c in CATS:
                row[c] = int(r[CAT_COL[c]])
            rows.append(row)
    out = pd.DataFrame(rows).sort_values(["qubit_idx", "length_ns"])
    for c in CATS:
        out[c + "_rate_pct"] = 100.0 * out[c] / out["n_test"]      # % of shots
        out[c + "_share_pct"] = 100.0 * out[c] / out["n_error"]    # % of errors
    out["err_rate_pct"] = 100.0 * out["n_error"] / out["n_test"]
    return out


def _panel(ax, dfm, title, cat_colors):
    """100%-stacked composition of errors; bar annotated with total error rate."""
    order = [(q, ns) for q, _ in FOCUS for ns in (1000, 400)]
    xlabels = [f"{lbl}\n{ns} ns" for _, lbl in FOCUS for ns in (1000, 400)]
    x = list(range(len(order)))
    bottom = [0.0] * len(order)
    for c in CATS:
        vals = []
        for q, ns in order:
            row = dfm[(dfm.qubit_idx == q) & (dfm.length_ns == ns)]
            vals.append(float(row[c + "_share_pct"].iloc[0]) if len(row) else 0.0)
        ax.bar(x, vals, bottom=bottom, color=cat_colors[c],
               width=0.7, label=CAT_LABEL[c])
        bottom = [b + v for b, v in zip(bottom, vals)]
    # annotate each bar with the absolute error rate (magnitude)
    for xi, (q, ns) in zip(x, order):
        row = dfm[(dfm.qubit_idx == q) & (dfm.length_ns == ns)]
        if len(row):
            ax.text(xi, 101.5, f"err {float(row.err_rate_pct.iloc[0]):.1f}%",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Share of Errors (%)", fontsize=AXIS_FS)
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold")
    ax.grid(False)


def _panel_grouped(ax, dfm, title, cat_colors):
    """Grouped bars (one per error category) on a LOG y-axis so even the tiny
    categories (leakage, T1) are visible. Pastel fill + per-category hatch."""
    import numpy as np
    groups = [(q, ns) for q, _ in FOCUS for ns in (1000, 400)]
    glabels = [f"{lbl}\n{ns} ns" for _, lbl in FOCUS for ns in (1000, 400)]
    ng, nc = len(groups), len(CATS)
    width = 0.8 / nc
    xc = np.arange(ng)
    zero_floor = 7e-4   # X marker sits just above the axis bottom (5e-4)
    for ci, c in enumerate(CATS):
        xs = xc + (ci - (nc - 1) / 2.0) * width
        ys = []
        for q, ns in groups:
            row = dfm[(dfm.qubit_idx == q) & (dfm.length_ns == ns)]
            r = float(row[c + "_rate_pct"].iloc[0]) if len(row) else 0.0
            ys.append(r if r > 0 else np.nan)   # nan -> no bar (log can't show 0)
        ys = np.array(ys, dtype=float)
        ax.bar(xs, ys, width=width * 0.95, color=cat_colors[c], hatch=HATCH[c],
               edgecolor="0.3", linewidth=0.6, label=CAT_LABEL[c])
        # Zero categories: mark with an X at the floor (a bar of height 0 is
        # invisible on a log axis, so it would otherwise look like missing data).
        zmask = np.isnan(ys)
        if zmask.any():
            ax.scatter(xs[zmask], np.full(int(zmask.sum()), zero_floor),
                       marker="x", color="0.25", s=38, linewidths=1.6, zorder=5)
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 30)
    # Plain decimal tick labels (no 10^x), since there's room.
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xticks(xc)
    ax.set_xticklabels(glabels)
    ax.set_ylabel("Error Rate (% of Shots, log scale)", fontsize=AXIS_FS)
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold")
    ax.grid(False)


def plot_grouped(panels, out_path, cat_colors=None, figsize=(16, 4)):
    """panels: list of (title, dataframe). One subplot per panel, shared y."""
    if cat_colors is None:
        cat_colors = dict(zip(CATS, sns.color_palette("pastel", len(CATS))))
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, dfm) in zip(axes, panels):
        _panel_grouped(ax, dfm, title, cat_colors)
    for ax in axes[1:]:
        ax.set_ylabel("")
    handles, labels = axes[-1].get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="x", color="0.25", linestyle="None",
                          markersize=8, markeredgewidth=1.6))
    labels.append("Zero (no errors)")
    fig.tight_layout()
    fig.legend(handles, labels, title="Error Type", fontsize=LEG_FS + 3,
               title_fontsize=LEG_FS + 3, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=True, edgecolor="0.3", framealpha=1.0)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"[report] saved {out_path}")


def main():
    fake = _load_mode("scrambled")
    real = _load_mode("pertrace")
    fake.insert(0, "variant", "fake")
    real.insert(0, "variant", "realistic")
    fp = os.path.join(REPORTS, "herqules_error_breakdown_fake.csv")
    rp = os.path.join(REPORTS, "herqules_error_breakdown_realistic.csv")
    fake.to_csv(fp, index=False)
    real.to_csv(rp, index=False)
    print(f"[report] wrote {fp}")
    print(f"[report] wrote {rp}")

    sns.set_style("white")
    cat_colors = dict(zip(CATS, sns.color_palette("deep", len(CATS))))
    # Per-trace (deployment) HERQULES only -- shot-scrambled variant removed.
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    _panel(ax, real, "HERQULES Error Modes (Per-Trace)", cat_colors)
    ax.legend(title="Error Type", fontsize=LEG_FS, title_fontsize=LEG_FS,
              loc="lower right")
    fig.tight_layout()
    out = os.path.join(REPORTS, "herqules_error_breakdown.pdf")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"[report] saved {out} (legacy stacked; per-trace only)")

    # New style: grouped bars, pastel + hatch, log y (shows every category).
    plot_grouped(
        [("HERQULES (Per-Trace)", real)],
        os.path.join(REPORTS, "herqules_error_breakdown_grouped.pdf"),
        figsize=(8, 4),
    )

    # CNN-vs-HERQULES comparison: best real model (HybridCNN) vs the
    # deployment-faithful per-trace HERQULES. Only if the HybridCNN
    # error CSVs are present for both lengths.
    if all(_latest(L, "hybridcnn") for L, _ in LENGTHS):
        hyb = _load_mode("hybridcnn")
        hyb.insert(0, "variant", "hybridcnn")
        hp = os.path.join(REPORTS, "herqules_error_breakdown_hybridcnn.csv")
        hyb.to_csv(hp, index=False)
        print(f"[report] wrote {hp}")
        plot_grouped(
            [("(a) MCMit-CNN", hyb),
             ("(b) HERQULES (Per-Trace)", real)],
            os.path.join(REPORTS, "cnn_vs_herqules_error_breakdown.pdf"),
        )
    else:
        print("[report] HybridCNN CSVs not found yet; skipping CNN-HERQULES plot.")


if __name__ == "__main__":
    main()
