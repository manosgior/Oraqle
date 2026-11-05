"""Background figure: single-shot integrated IQ for one qubit at two readout
lengths, showing how the |0> / |1> clouds tighten & separate with integration.

Qubit 5 (highest quality). Each shot's demodulated trace is INTEGRATED
(time-averaged) to a single IQ point. Left panel integrates 50 samples (100 ns),
right integrates 500 samples (1000 ns) -- the SAME shots, two windows. A linear
threshold (perpendicular bisector of the two centroids) is drawn per panel.

extract (heavy): one demod at length 500, integrate over [:50] and [:500],
save iq_clouds_vs_length_q{Q}.npz.   plot (cheap): read npz, render.
Env: IQ_QUBIT (default 5), IQ_REPLOT=1 to skip extraction and just re-plot.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

import sys as _sys  # [unify]
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "discriminators"))
from disc_paths import REPORTS_OUT as _REPORTS_OUT, RESULTS_DISC as _RESULTS_DISC  # [unify]
# [unify] IQ_REPORTS (if set) is used for everything. Otherwise extract() writes the npz to
# REPORTS_OUT; plot() reads a fresh npz from REPORTS_OUT if present, else the published one
# in results/discriminators/, and writes the figure to REPORTS_OUT.
REPORTS = os.environ.get("IQ_REPORTS") or str(_REPORTS_OUT)
QB = int(os.environ.get("IQ_QUBIT", "5"))
WINDOWS = [(50, "100 ns"), (500, "1000 ns")]   # (samples, label)
N_PER_STATE = 3000
AXIS_FS, TITLE_FS, LEG_FS, TICK_FS = 18, 20, 15, 14


def _npz_path():
    os.makedirs(REPORTS, exist_ok=True)  # [unify]
    return os.path.join(REPORTS, f"iq_clouds_vs_length_q{QB}.npz")


def _npz_in():  # [unify]
    p = os.path.join(REPORTS, f"iq_clouds_vs_length_q{QB}.npz")
    if os.environ.get("IQ_REPORTS") or os.path.exists(p):
        return p
    return os.path.join(str(_RESULTS_DISC), f"iq_clouds_vs_length_q{QB}.npz")


def extract():
    import trainers.HERQULES_original as H
    traces, y, _ = H.get_data(QB)
    traces = np.asarray(traces)
    bit = (y.astype(np.int64) >> (QB - 1)) & 1
    rng = np.random.RandomState(0)
    s0 = rng.choice(np.where(bit == 0)[0], min(N_PER_STATE, int((bit == 0).sum())), replace=False)
    s1 = rng.choice(np.where(bit == 1)[0], min(N_PER_STATE, int((bit == 1).sum())), replace=False)
    logger.info(f"Q{QB}: integrating {len(s0)}+{len(s1)} shots at {[w for w,_ in WINDOWS]}")
    out = {}
    for W, _ in WINDOWS:
        iq0 = np.stack([traces[s0, :W, 0].mean(1), traces[s0, :W, 1].mean(1)], axis=1)
        iq1 = np.stack([traces[s1, :W, 0].mean(1), traces[s1, :W, 1].mean(1)], axis=1)
        out[f"iq0_{W}"], out[f"iq1_{W}"] = iq0, iq1
        out[f"cen0_{W}"], out[f"cen1_{W}"] = iq0.mean(0), iq1.mean(0)
    np.savez_compressed(_npz_path(), **out)
    logger.info(f"saved {_npz_path()}")


def plot():
    d = np.load(_npz_in())  # [unify]
    sns.set_style("white")
    pastel, deep = sns.color_palette("pastel"), sns.color_palette("deep")

    allpts = np.vstack([np.vstack([d[f"iq0_{W}"], d[f"iq1_{W}"]]) for W, _ in WINDOWS])
    pad = 0.10 * (allpts.max(0) - allpts.min(0))
    lo, hi = allpts.min(0) - pad, allpts.max(0) + pad

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), sharex=True, sharey=True)
    for i, (ax, (W, lab)) in enumerate(zip(axes, WINDOWS)):
        iq0, iq1 = d[f"iq0_{W}"], d[f"iq1_{W}"]
        cen0, cen1 = d[f"cen0_{W}"], d[f"cen1_{W}"]
        ax.scatter(iq0[:, 0], iq0[:, 1], s=7, color=pastel[0], alpha=0.30,
                   label=r"$|0\rangle$", rasterized=True)
        ax.scatter(iq1[:, 0], iq1[:, 1], s=7, color=pastel[3], alpha=0.30,
                   label=r"$|1\rangle$", rasterized=True)
        ax.scatter(*cen0, marker="X", s=200, color=deep[0], edgecolor="k", zorder=6)
        ax.scatter(*cen1, marker="X", s=200, color=deep[3], edgecolor="k", zorder=6)
        mid = (cen0 + cen1) / 2
        axis = (cen1 - cen0); axis = axis / (np.linalg.norm(axis) + 1e-12)
        pdir = np.array([-axis[1], axis[0]])
        span = 2.0 * np.linalg.norm(hi - lo)
        b0, b1 = mid - span * pdir, mid + span * pdir
        ax.plot([b0[0], b1[0]], [b0[1], b1[1]], "--", color="0.25", lw=2.0,
                zorder=5, label="Threshold")
        ax.set_xlabel("In-Phase (I)", fontsize=AXIS_FS)
        ax.set_title(f"({'ab'[i]}) Readout {lab} ({W} Samples)",
                     fontsize=TITLE_FS, fontweight="bold")
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)

    axes[0].set_ylabel("Quadrature (Q)", fontsize=AXIS_FS)
    axes[0].set_xlim(lo[0], hi[0])
    axes[0].set_ylim(lo[1], hi[1])
    axes[1].legend(fontsize=LEG_FS, loc="best", framealpha=0.92, markerscale=2)
    fig.tight_layout()
    os.makedirs(REPORTS, exist_ok=True)  # [unify]
    out = os.path.join(REPORTS, f"iq_clouds_vs_length_q{QB}.pdf")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    logger.info(f"saved {out}")


if __name__ == "__main__":
    if os.environ.get("IQ_REPLOT") != "1":
        extract()
    plot()
