"""Background figure: what each readout-error type looks like in the IQ plane.

Two phases, so plotting can be iterated without re-demodulating:
  * extract (heavy): demodulate via HERQULES front-end, select real shots per
    error category, save everything to iq_error_gallery_q{Q}.npz
  * plot (cheap): read the npz and render.

Drift errors (relaxation, excitation) are shown as MEAN trajectories (averaged
over the category -> smooth) with the demod ring-up trimmed; leakage is shown as
a scatter blob (|2> lands in its own region); near-boundary / state-prep are
example points.

Env: HERQULES_TRACE_LENGTH, IQ_QUBIT (1-based, default 5),
     IQ_REPLOT=1 to skip extraction and just re-plot from the npz.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from loguru import logger
except ImportError:                     # plotting-only machines
    import logging; logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

import sys as _sys  # [unify]
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from disc_paths import REPORTS_OUT as _REPORTS_OUT, RESULTS_DISC as _RESULTS_DISC  # [unify]
# [unify] IQ_REPORTS (if set) is used for everything, as before. Otherwise extract()
# writes to REPORTS_OUT and plot() reads a fresh npz from REPORTS_OUT if present,
# else the published one in results/discriminators/, and writes figures to REPORTS_OUT.
REPORTS = os.environ.get("IQ_REPORTS") or str(_REPORTS_OUT)


def _npz_in(name):  # [unify]
    p = os.path.join(REPORTS, name)
    return p if (os.environ.get("IQ_REPORTS") or os.path.exists(p)) else os.path.join(str(_RESULTS_DISC), name)


QB = int(os.environ.get("IQ_QUBIT", "5"))
TRIM = 0.12          # fraction of trace trimmed from the start (demod ring-up)
# Paper figure: drawn at the ACM column width (3.33 in) so it is included 1:1.
COL_W, FIG_H = 3.33, 1.75
AXIS_FS, LEG_FS, TICK_FS = 7, 6.2, 6


def _dist(p, c):
    return np.sqrt((p[:, 0] - c[0]) ** 2 + (p[:, 1] - c[1]) ** 2)


def _smooth2d(t, w):
    if w < 2:
        return t
    ker = np.ones(w) / w
    return np.stack([np.convolve(t[:, 0], ker, "valid"),
                     np.convolve(t[:, 1], ker, "valid")], axis=1)


def extract():
    import trainers.HERQULES_original as H
    traces, y, _ = H.get_data(QB)
    traces = np.asarray(traces)
    N, T, _ = traces.shape
    bit = (y.astype(np.int64) >> (QB - 1)) & 1
    mI, mQ = traces[:, :, 0].mean(1), traces[:, :, 1].mean(1)
    miq = np.stack([mI, mQ], axis=1)
    k = max(5, T // 10)
    start = np.stack([traces[:, :k, 0].mean(1), traces[:, :k, 1].mean(1)], axis=1)
    end = np.stack([traces[:, -k:, 0].mean(1), traces[:, -k:, 1].mean(1)], axis=1)
    cen0, cen1 = miq[bit == 0].mean(0), miq[bit == 1].mean(0)
    mid = (cen0 + cen1) / 2
    r = 0.5 * np.linalg.norm(cen1 - cen0)
    ax = (cen1 - cen0) / (np.linalg.norm(cen1 - cen0) + 1e-12)
    perp = np.abs((miq - mid) @ np.array([-ax[1], ax[0]]))

    d_e0, d_e1 = _dist(end, cen0), _dist(end, cen1)
    d_s0, d_s1 = _dist(start, cen0), _dist(start, cen1)
    d_m0, d_m1 = _dist(miq, cen0), _dist(miq, cen1)

    relax_m = (bit == 1) & (d_s1 < r) & (d_e0 < r)
    excite_m = (bit == 0) & (d_s0 < r) & (d_e1 < r)
    leak_m = (bit == 1) & (d_m0 > 1.3 * r) & (d_m1 > 1.3 * r) & (perp > 0.8 * r)
    logger.info(f"counts relax={int(relax_m.sum())} excite={int(excite_m.sum())} leak={int(leak_m.sum())}")

    idx_near = int(np.argmin(_dist(miq, mid)))
    idx_prep = int(np.argmin(np.where(bit == 0, d_s1 + d_e1, 1e9)))

    rng = np.random.RandomState(0)
    def sub(m, n=4000):
        idx = np.where(m)[0]
        return miq[rng.choice(idx, min(n, len(idx)), replace=False)] if len(idx) else np.empty((0, 2))

    os.makedirs(REPORTS, exist_ok=True)  # [unify]
    npz = os.path.join(REPORTS, f"iq_error_gallery_q{QB}.npz")
    np.savez_compressed(
        npz, cen0=cen0, cen1=cen1, mid=mid, radius=r,
        cloud0=sub(bit == 0), cloud1=sub(bit == 1), leak_cloud=sub(leak_m, 2500),
        traj_relax=traces[relax_m].mean(0) if relax_m.sum() else np.empty((0, 2)),
        traj_excite=traces[excite_m].mean(0) if excite_m.sum() else np.empty((0, 2)),
        pt_near=miq[idx_near], pt_prep=miq[idx_prep])
    logger.info(f"saved {npz}")
    return npz


def plot():
    """Compact paper layout: no title (the caption has it), axes cropped to the data,
    legend centred at the bottom. I/Q are NOT drawn to equal scale: the data is
    taller than wide, so equal aspect would make the figure tall and mostly empty."""
    d = np.load(_npz_in(f"iq_error_gallery_q{QB}.npz"))  # [unify]
    cen0, cen1, mid, r = d["cen0"], d["cen1"], d["mid"], float(d["radius"])
    sns.set_style("white")
    pastel, deep = sns.color_palette("pastel"), sns.color_palette("deep")
    fig, ax = plt.subplots(figsize=(COL_W, FIG_H))

    ax.scatter(d["cloud0"][:, 0], d["cloud0"][:, 1], s=1.5, lw=0, color=pastel[0], alpha=0.25,
               label=r"$|0\rangle$ cloud", rasterized=True)
    ax.scatter(d["cloud1"][:, 0], d["cloud1"][:, 1], s=1.5, lw=0, color=pastel[3], alpha=0.25,
               label=r"$|1\rangle$ cloud", rasterized=True)
    if len(d["leak_cloud"]):
        ax.scatter(d["leak_cloud"][:, 0], d["leak_cloud"][:, 1], s=2, lw=0, color=deep[4],
                   alpha=0.35, label=r"leakage $|2\rangle$", rasterized=True)
    ax.scatter(*cen0, marker="X", s=40, color=deep[0], edgecolor="k", lw=0.5, zorder=6)
    ax.scatter(*cen1, marker="X", s=40, color=deep[3], edgecolor="k", lw=0.5, zorder=6)

    axis = (cen1 - cen0) / (np.linalg.norm(cen1 - cen0) + 1e-12)
    pdir = np.array([-axis[1], axis[0]])
    b0, b1 = mid - 2.0 * r * pdir, mid + 2.0 * r * pdir
    ax.plot([b0[0], b1[0]], [b0[1], b1[1]], "--", color="0.4", lw=0.8,
            label="decision boundary", zorder=3)

    # Drift errors as clean curved arrows between the (real) cloud centroids;
    # the per-shot mean trajectory loops because the transition time varies, so
    # an arrow conveys the |1>->|0> / |0>->|1> direction far more legibly.
    from matplotlib.patches import FancyArrowPatch

    def arrow(p0, p1, color, label, rad):
        ax.add_patch(FancyArrowPatch(
            tuple(p0), tuple(p1), connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=8, lw=1.2, color=color, zorder=5))
        ax.plot([], [], color=color, lw=1.2, label=label)   # legend proxy

    arrow(cen1, cen0, deep[1], r"relaxation $|1\rangle\!\to\!|0\rangle$", 0.3)
    arrow(cen0, cen1, deep[2], r"excitation $|0\rangle\!\to\!|1\rangle$", 0.3)

    # 3 near-boundary examples, spread along the boundary line (real shots
    # whose IQ is ~equidistant from both clouds, at different perp positions).
    clouds = np.vstack([d["cloud0"], d["cloud1"]])
    proj = (clouds - mid) @ axis
    for tol in (0.15, 0.25, 0.4, 0.6):
        cand = clouds[np.abs(proj) < tol * r]
        if len(cand) >= 3:
            break
    perp_c = (cand - mid) @ pdir
    near3 = cand[np.argsort(perp_c)[[0, len(perp_c) // 2, -1]]]
    ax.scatter(near3[:, 0], near3[:, 1], marker="D", s=18, facecolor="none",
               edgecolor="k", linewidths=0.8, zorder=8, label="near-boundary")
    ax.scatter(*d["pt_prep"], marker="P", s=40, color=pastel[0], edgecolor=deep[3],
               linewidths=0.8, zorder=8, label="state-prep error")

    # Crop to the data (0.3-99.7th percentile of all shots) rather than padding I.
    allp = np.vstack([d["cloud0"], d["cloud1"]]
                     + ([d["leak_cloud"]] if len(d["leak_cloud"]) else []))
    lo, hi = np.percentile(allp, 0.3, axis=0), np.percentile(allp, 99.7, axis=0)
    pad = 0.05 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.tick_params(axis="both", labelsize=TICK_FS, length=2, pad=1.5)
    ax.set_xlabel("In-Phase (I)", fontsize=AXIS_FS, labelpad=1)
    ax.set_ylabel("Quadrature (Q)", fontsize=AXIS_FS, labelpad=1)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    ax.grid(False)

    leg = fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
                     bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=LEG_FS, frameon=False,
                     handlelength=1.3, columnspacing=0.7, handletextpad=0.3)
    for h in leg.legend_handles[:3]:            # cloud swatches: bigger + opaque
        h.set_sizes([10]); h.set_alpha(0.9)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.975, bottom=0.36)
    os.makedirs(REPORTS, exist_ok=True)  # [unify]
    out = os.path.join(REPORTS, f"iq_error_gallery_q{QB}.pdf")
    fig.savefig(out, dpi=600)
    fig.savefig(out.replace(".pdf", ".png"), dpi=250)
    logger.info(f"saved {out}")


if __name__ == "__main__":
    if os.environ.get("IQ_REPLOT") != "1":
        extract()
    plot()
