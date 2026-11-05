"""m1tail_v2_plot.py -- local plots for the m1tail v2 off-axis tail analysis.

usage: python m1tail_v2_plot.py <dir with m1tail_v2_results_test.json, m1tail_v2_store_test.npz,
                                 [m1tail_v2_sliding_test.json]> [out dir]
writes m1tail_q{0..4}.png, m1tail_time.png, m1tail_traj.png

Per-qubit figure: rows = window (first 200 ns, paper "1 us" = samples 0-499, full 2 us).
  col 1: histogram of z_b = (perp - mu_s)/sigma_noise (log y) for prepared |0> and |1>, N(0,1) null.
  col 2: one-sided survival counts N(z_b > k) (solid) and N(z_b < -k) (dashed) with the Gaussian null.
  col 3/4: 2-D log density (noise-normalised frame of the neighbour configuration, origin c0) for
           prepared |0> / |1>: two-state segment, d_seg = 5 sigma contour, non-core density modes
           (circles), far-excess modes at d_seg cuts 5..10 (small markers; they follow the cut when
           there is no compact cloud).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from scipy.special import ndtr

D = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else D
R = json.load(open(os.path.join(D, "m1tail_v2_results_test.json")))
ST = np.load(os.path.join(D, "m1tail_v2_store_test.npz"))
SLp = os.path.join(D, "m1tail_v2_sliding_test.json")
SL = json.load(open(SLp)) if os.path.exists(SLp) else None
C0, C1 = "#2a78d6", "#eb6834"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
CM = {0: LinearSegmentedColormap.from_list("b", ["#dbe8f8", C0, "#0d3a73"]),
      1: LinearSegmentedColormap.from_list("o", ["#fbe1d6", C1, "#7a2a0b"])}
WIN = [("early", "first 200 ns (bins 0-1)"), ("1us", 'paper "1 us" (samples 0-499)'),
       ("2us", "full 2 us (samples 0-999)")]
PAPER = ["Q1", "Q2", "Q3", "Q4", "Q5"]
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 8.5, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "text.color": INK})
G = R["grid"]
XR, YR, GR = G["XR"], G["YR"], G["GRID"]
XE = np.arange(XR[0], XR[1] + 1e-9, GR)
YE = np.arange(YR[0], YR[1] + 1e-9, GR)
ze = ST["zedges"]
zc = 0.5 * (ze[1:] + ze[:-1])
dz = ze[1] - ze[0]


def segdist(X, Y, S, Dd):
    L2 = S * S + Dd * Dd
    t = np.clip((X * S + Y * Dd) / L2, 0, 1)
    return np.hypot(X - t * S, Y - t * Dd)


for q in range(5):
    fig, axs = plt.subplots(3, 4, figsize=(16.5, 11.5), gridspec_kw={"width_ratios": [1, 1, 1.15, 1.15]})
    H = {}
    for r, (w, wl) in enumerate(WIN):
        RW = R["windows"][w][f"q{q}"]
        N0 = RW["prep0"]["N"]
        # (1) z_b histogram
        ax = axs[r, 0]
        for b, col in ((0, C0), (1, C1)):
            h = ST[f"{w}_q{q}_b{b}_zb"].astype(float)
            H[f"p{b}"], = ax.step(zc, np.where(h > 0, h, np.nan), where="mid", color=col, lw=1.2,
                                  label=f"prepared |{b}>")
        H["g"], = ax.plot(zc, N0 * dz * np.exp(-zc ** 2 / 2) / np.sqrt(2 * np.pi), color=INK, lw=1, ls="--",
                          label="H0 null N(0,1)")
        ax.set_yscale("log"); ax.set_ylim(0.5, N0 * dz * 0.6); ax.set_xlim(-8, 25)
        ax.set_xlabel("z_b = (perp - mu_s) / sigma_noise")
        ax.set_ylabel("shots per 0.1 z")
        ax.set_title(f"{PAPER[q]} (q{q}), {wl}", color=INK)
        ax.grid(color=GRID, lw=0.5)
        # (2) one-sided survival
        ax = axs[r, 1]
        ks = np.arange(0, 20.01, 0.1)
        for b, col in ((0, C0), (1, C1)):
            h = ST[f"{w}_q{q}_b{b}_zb"].astype(float)
            cs = np.cumsum(h[::-1])[::-1]          # counts with z >= edge
            plus = np.interp(ks, ze[:-1], cs)
            cm = np.cumsum(h)
            minus = np.interp(-ks, ze[1:], cm)
            ax.step(ks, np.where(plus > 0, plus, np.nan), where="post", color=col, lw=1.3)
            H["minus"], = ax.step(ks, np.where(minus > 0, minus, np.nan), where="post", color=col, lw=1.0,
                                  ls="--", label="z < -k side (dashed)")
        ax.plot(ks, N0 * ndtr(-ks), color=INK, lw=1, ls="--")
        ax.set_yscale("log"); ax.set_ylim(0.5, N0 * 0.6); ax.set_xlim(0, 20)
        ax.set_xlabel("k (noise sigma)"); ax.set_ylabel("shots beyond +k (solid) / -k (dashed)")
        t3 = RW["prep1"]["tails_zb"]["3"]
        t30 = RW["prep0"]["tails_zb"]["3"]
        ax.set_title(f"one-sided tails; mirror excess E(3) = [N(z>3)-N(z<-3)]/N\n"
                     f"|0>: {100 * t30['mirror_excess_frac']:.3f}%   |1>: {100 * t3['mirror_excess_frac']:.3f}%",
                     color=INK)
        ax.grid(color=GRID, lw=0.5)
        # (3,4) 2-D maps
        for b in (0, 1):
            ax = axs[r, 2 + b]
            h2 = ST[f"{w}_q{q}_b{b}_H"].T.astype(float)
            h2[h2 == 0] = np.nan
            im = ax.pcolormesh(XE, YE, h2, cmap=CM[b], norm=LogNorm(vmin=1, vmax=np.nanmax(h2)), shading="flat",
                               rasterized=True)
            MC = RW[f"prep{b}"]["mode_census"]
            S, Dd = MC["S_median"], MC["D_median"]
            ax.plot([0, S], [0, Dd], color=INK, lw=1.0)
            gx = np.linspace(-10, 40, 300); gy = np.linspace(-15, 25, 240)
            GX, GY = np.meshgrid(gx, gy)
            cs_ = ax.contour(GX, GY, segdist(GX, GY, S, Dd), levels=[5], colors=[INK2], linewidths=0.8,
                             linestyles="--")
            H["d5"] = plt.Line2D([], [], color=INK2, lw=0.8, ls="--", label="d_seg = 5 sigma")
            for md in MC["modes"][:8]:
                if md["is_core0"] or md["is_core1"]:
                    continue
                ax.add_patch(plt.Circle((md["x_ms"], md["y_ms"]), 1.5, fill=False, color=INK, lw=1.2))
                ax.annotate(f"mode ({md['uv_sep_units'][0]:.2f}, {md['uv_sep_units'][1]:.2f}) sep\n"
                            f"excess {100 * md['excess_frac_lenient']:.2f}%",
                            (md["x_ms"], md["y_ms"]), xytext=(8, -18), textcoords="offset points", fontsize=7,
                            color=INK)
                H["mode"] = plt.Line2D([], [], marker="o", mfc="none", mec=INK, ls="", ms=9,
                                       label="non-core density mode (persistent)")
            fx = []; fy = []
            for kc, fe in MC["far_excess"].items():
                if "mode_xy" in fe:
                    fx.append(fe["mode_xy"][0]); fy.append(fe["mode_xy"][1])
            if fx:
                H["far"], = ax.plot(fx, fy, "^-", color=INK, ms=4, lw=0.6,
                                    label="excess mode for cuts d_seg>5,6,7,8,10")
            d5 = RW[f"prep{b}"]["dseg"]["5"]
            ax.set_xlim(-8, 40); ax.set_ylim(-15, 25); ax.set_aspect("equal")
            ax.set_xlabel("along / sigma (origin c0)"); ax.set_ylabel("perp / sigma")
            ax.set_title(f"prepared |{b}>: d_seg > 5: {d5['obs']} / {RW[f'prep{b}']['N']} = {100 * d5['frac']:.3f}%"
                         f"\n(H0 bound {d5['rayleigh_bound']:.1f} shots; sep = {S:.2f} sigma)", color=INK)
            cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cb.ax.tick_params(labelsize=7)
            cb.set_label("shots per 0.2 x 0.2 sigma", fontsize=7)
    hl = [H[k] for k in ("p0", "p1", "g", "minus", "d5", "mode", "far") if k in H]
    fig.legend(hl, [h.get_label() for h in hl], loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=7,
               frameon=False, fontsize=8)
    fig.suptitle(f"{PAPER[q]} (code q{q}): perpendicular tails vs H0 (Gaussian amplifier noise + crosstalk "
                 f"centres per neighbour configuration), TEST, {R['windows']['1us'][f'q{q}']['prep0']['N']:,} "
                 f"shots per prepared bit", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(os.path.join(OUT, f"m1tail_q{q}.png"), dpi=105)
    plt.close(fig)
    print("wrote", f"m1tail_q{q}.png")

# ---------------------------------------------------------------- time evolution
if SL is not None:
    fig, axs = plt.subplots(2, 5, figsize=(17, 6.5), sharex=True)
    hd = {}
    for q in range(5):
        S_ = SL[f"q{q}"]
        t = (np.array(S_["start_bin"]) * 100 + 250) / 1000.0
        for b, col in ((0, C0), (1, C1)):
            P = S_[f"prep{b}"]
            N = P["N"]
            e3 = 100 * np.array(P["mirror3"])
            se = 100 * 1.96 * np.sqrt(np.array(P["plus3"]) + np.array(P["minus3"])) / N
            hd[f"a{b}"], = axs[0, q].plot(t, e3, color=col, lw=1.6, marker="o", ms=3, label=f"prepared |{b}>")
            axs[0, q].fill_between(t, e3 - se, e3 + se, color=col, alpha=0.2, lw=0)
            ex = 100 * (np.array(P["dseg5"]) - np.array(P["dseg5_null_len"])) / N
            axs[1, q].plot(t, ex, color=col, lw=1.6, marker="o", ms=3)
        axs[0, q].set_title(f"{PAPER[q]} (q{q})", color=INK)
        for a in axs[:, q]:
            a.axhline(0, color=INK2, lw=0.6)
            a.grid(color=GRID, lw=0.5)
        axs[1, q].set_xlabel("window centre (us), 500 ns sliding window")
    axs[0, 0].set_ylabel("mirror excess E(3) (%)")
    axs[1, 0].set_ylabel("d_seg>5 excess over null (%)")
    fig.legend([hd["a0"], hd["a1"]], ["prepared |0>", "prepared |1>"], loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Build-up of the one-sided off-axis population during the readout (TEST, 500 ns windows). Top: E(3) = [N(z_b>3) - N(z_b<-3)]/N, bands = 95% CI. Bottom: N(d_seg>5) minus perp-resampled null, per shot",
                 color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(os.path.join(OUT, "m1tail_time.png"), dpi=105)
    plt.close(fig)
    print("wrote m1tail_time.png")

# ---------------------------------------------------------------- per-bin trajectories (2 us frame)
fig, axs = plt.subplots(3, 5, figsize=(17, 9.6))
hd = {}
styles = {"core0": (C0, "-", "|0> core (d < 1.5 sigma of c0)"), "core1": (C1, "-", "|1> core (d < 1.5 sigma of c1)"),
          "streak_5to7": ("#52514e", "--", "off-segment, 5 < d_seg < 7"),
          "streak_end_top10pct": (INK, "-", "off-segment, top 10% of d_seg")}
tb = np.arange(20) * 0.1 + 0.05
for q in range(5):
    TR = R["windows"]["2us"][f"q{q}"].get("trajectories")
    if TR is None:
        continue
    sep = TR["sep_median_raw"]
    for row, (bb, comp) in enumerate(((1, "A_med"), (1, "P_med"), (0, "P_med"))):
        rb = TR[f"prep{bb}"]
        ax = axs[row, q]
        for g, (col, ls, lab) in styles.items():
            if g not in rb:
                continue
            hd[g], = ax.plot(tb, np.array(rb[g][comp]) / sep, color=col, ls=ls, lw=1.5, label=lab)
        for g in rb:
            if g.startswith("mode"):
                hd["mode"], = ax.plot(tb, np.array(rb[g][comp]) / sep, color="#1baf7a", lw=1.5,
                                      label="non-core density mode (Q4 blob)")
        ax.grid(color=GRID, lw=0.5)
        if row == 0:
            ax.set_title(f"{PAPER[q]} (q{q})", color=INK)
        if row == 2:
            ax.set_xlabel("time-bin centre (us)")
axs[0, 0].set_ylabel("prepared |1>: along - c0\n(units of 2 us separation)")
axs[1, 0].set_ylabel("prepared |1>: perp - c0\n(units of 2 us separation)")
axs[2, 0].set_ylabel("prepared |0>: perp - c0\n(units of 2 us separation)")
hl = [hd[k] for k in ("core0", "core1", "streak_5to7", "streak_end_top10pct", "mode") if k in hd]
fig.legend(hl, [h.get_label() for h in hl], loc="lower center", ncol=5, frameon=False)
fig.suptitle("Per-bin median trajectories of shots grouped by their 2 us window mean (TEST). Bin 0 and 19 have "
             "zero-padding gain 0.87; bins 0-2 are resonator ring-up.", color=INK, fontsize=10)
fig.tight_layout(rect=(0, 0.04, 1, 0.96))
fig.savefig(os.path.join(OUT, "m1tail_traj.png"), dpi=105)
plt.close(fig)
print("wrote m1tail_traj.png")
