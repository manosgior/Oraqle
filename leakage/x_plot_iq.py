"""x_plot_iq.py -- step 5 diagnostics (runs locally; matplotlib).

Reads data/leakage/x_hist.npz + x_sanity.json (produced on graham by x_sanity.py)
and writes data/leakage/x_iq_q{0..4}.png. One figure per qubit:
rows = window (full 2 us = bins 0-19, paper "1 us" = bins 0-9, early 200 ns = bins 0-1),
cols = log-density of the window-mean IQ (along vs perp) for prepared bit 0, the
same for prepared bit 1, the 1-D perpendicular distribution in robust z units
(log counts, Gaussian reference), and the 1-D along-axis distribution in units
of the |0>-|1> separation (log counts). Histograms pool train + test (1.6 M shots).
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

D = sys.argv[1] if len(sys.argv) > 1 else str(LP.RES)
H = np.load(os.path.join(D, "x_hist.npz"))
S = json.load(open(os.path.join(D, "x_sanity.json")))
C0, C1 = "#2a78d6", "#eb6834"          # categorical slots 1/2 (prep |0>, prep |1>)
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
CM = {0: LinearSegmentedColormap.from_list("b", ["#dbe8f8", C0, "#0d3a73"]),
      1: LinearSegmentedColormap.from_list("o", ["#fbe1d6", C1, "#7a2a0b"])}
WIN = [("full_2us", "full 2 us (bins 0-19)"), ("1us", 'paper "1 us" (bins 0-9)'), ("early_200ns", "early 200 ns (bins 0-1)")]
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 8.5, "figure.facecolor": SURF, "axes.facecolor": SURF})

for q in range(5):
    fig, axs = plt.subplots(3, 4, figsize=(15, 10.5), gridspec_kw={"width_ratios": [1, 1, 1.15, 1.15]})
    for r, (w, wl) in enumerate(WIN):
        xe, ye = H[f"q{q}_{w}_xe"], H[f"q{q}_{w}_ye"]
        cen = S["per_qubit"][f"q{q}"]["windows"][w]["centroids_test"]
        vmax = max(H[f"q{q}_{w}_prep{b}_h2"].max() for b in (0, 1))
        for b in (0, 1):
            ax = axs[r, b]
            h = H[f"q{q}_{w}_prep{b}_h2"].T.astype(float); h[h == 0] = np.nan
            im = ax.pcolormesh(xe, ye, h, cmap=CM[b], norm=LogNorm(vmin=1, vmax=vmax), shading="flat", rasterized=True)
            for bb, col in ((0, C0), (1, C1)):
                ax.plot(cen[f"prep{bb}"]["median_along"], cen[f"prep{bb}"]["median_perp"], "+", color=INK, ms=9, mew=1.2)
            ax.set_aspect("equal"); ax.set_xlabel("along |0>→|1> axis (a.u.)"); ax.set_ylabel("perpendicular (a.u.)")
            n_out = int(H[f"q{q}_{w}_prep{b}_outside"])
            ax.set_title(f"{wl}\nprepared |{b}>  (log counts; {n_out} shots outside view)", color=INK)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=7)
        # 1-D perpendicular z
        ax = axs[r, 2]; pe = H["perp_edges"]; pc = 0.5 * (pe[1:] + pe[:-1]); dz = pe[1] - pe[0]
        for b, col in ((0, C0), (1, C1)):
            hh = H[f"q{q}_{w}_prep{b}_hperp"].astype(float)
            ax.step(pc, np.where(hh > 0, hh, np.nan), where="mid", color=col, lw=1.4, label=f"prepared |{b}>")
        n = H[f"q{q}_{w}_prep0_hperp"].sum()
        ax.plot(pc, n * dz * np.exp(-pc ** 2 / 2) / np.sqrt(2 * np.pi), color=INK2, lw=1, ls="--", label="Gaussian (MAD σ)")
        ax.set_yscale("log"); ax.set_ylim(0.5, None); ax.set_xlabel("(perp − median) / σ_perp,MAD  per prepared bit")
        ax.set_ylabel("shots / bin"); ax.set_title(f"{wl}: perpendicular distribution", color=INK)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=7)
        ax.grid(alpha=0.25, lw=0.5)
        # 1-D along in units of separation
        ax = axs[r, 3]; ae = H["along_edges"]; ac = 0.5 * (ae[1:] + ae[:-1])
        for b, col in ((0, C0), (1, C1)):
            hh = H[f"q{q}_{w}_prep{b}_halong"].astype(float)
            ax.step(ac, np.where(hh > 0, hh, np.nan), where="mid", color=col, lw=1.4, label=f"prepared |{b}>")
        ax.axvline(0, color=INK2, lw=0.8, ls=":"); ax.axvline(1, color=INK2, lw=0.8, ls=":")
        ax.set_yscale("log"); ax.set_ylim(0.5, None); ax.set_xlabel("(along − c0) / (c1 − c0)")
        ax.set_ylabel("shots / bin"); ax.set_title(f"{wl}: along-axis distribution", color=INK)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=7)
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle(f"Paper Q{q+1} (code q{q}): window-mean IQ after demodulation + 5 MHz FIR + rotation, "
                 f"train+test pooled (1.6 M shots; 800k per prepared bit). + = median centroids (test)", color=INK, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(D, f"x_iq_q{q}.png")
    fig.savefig(out, dpi=110); plt.close(fig)
    print("wrote", out, os.path.getsize(out) // 1024, "KB")
