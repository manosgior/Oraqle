"""m1tail_plot.py -- local plots for the m1tail off-axis tail analysis.

Reads data/leakage/m1tail_{hist,excess}_test.npz, m1tail_results_test.json and
m1tail_models_test.json (computed on graham by m1tail_analysis.py) and writes
data/leakage/m1tail_q{0..4}.png and m1tail_time.png.

Per qubit figure: rows = window (first 200 ns, paper "1 us" = samples 0-499, full 2 us).
cols = (1) histogram of z_a (perp standardised per prepared 5-bit state) with the N(0,1)
null and the Student-t fit; (2) one-sided survival counts N(z>k), N(z<-k) vs the Gaussian
null, z_b (state-independent noise sigma); (3),(4) 2-D log density of all shots in the
noise-normalised frame of the neighbour configuration for prepared |0> and |1>, with the
two-state segment, the d_seg = 5 boundary, the excess shots and the cloud centres.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from scipy import stats
from scipy.special import ndtr

D = sys.argv[1] if len(sys.argv) > 1 else "/home/manosgior/Documents/GitHub/oraqle-ics/data/leakage"
H = np.load(os.path.join(D, "m1tail_hist_test.npz"))
E = np.load(os.path.join(D, "m1tail_excess_test.npz"))
R = json.load(open(os.path.join(D, "m1tail_results_test.json")))
M = json.load(open(os.path.join(D, "m1tail_models_test.json")))
C0, C1 = "#2a78d6", "#eb6834"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
CM = {0: LinearSegmentedColormap.from_list("b", ["#dbe8f8", C0, "#0d3a73"]),
      1: LinearSegmentedColormap.from_list("o", ["#fbe1d6", C1, "#7a2a0b"])}
WIN = [("early", "first 200 ns (bins 0-1)"), ("1us", 'paper "1 us" (samples 0-499)'), ("2us", "full 2 us (samples 0-999)")]
PAPER = ["Q1", "Q2", "Q3", "Q4", "Q5"]
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 8.5, "figure.facecolor": SURF, "axes.facecolor": SURF})

ze = H["zedges"]; zc = 0.5 * (ze[1:] + ze[:-1]); dz = ze[1] - ze[0]
XE = np.linspace(-8, 40, 241); YE = np.linspace(-15, 25, 201)

for q in range(5):
    fig, axs = plt.subplots(3, 4, figsize=(16, 11), gridspec_kw={"width_ratios": [1, 1, 1.2, 1.2]})
    handles = {}
    for r, (w, wl) in enumerate(WIN):
        RW = R["windows"][w][f"q{q}"]
        # (1) z_a histogram
        ax = axs[r, 0]
        for b, col in ((0, C0), (1, C1)):
            h = H[f"{w}_q{q}_b{b}_za"].astype(float)
            handles[f"p{b}"], = ax.step(zc, np.where(h > 0, h, np.nan), where="mid", color=col, lw=1.3,
                                        label=f"prepared |{b}>")
            mt = M.get(f"{w}_q{q}_b{b}", {}).get("models_za", {}).get("student_t")
            if mt and b == 1:
                handles["t"], = ax.plot(zc, h.sum() * dz * stats.t.pdf(zc, mt["df"], mt["loc"], mt["scale"]),
                                        color=C1, lw=0.9, ls=":", label="Student-t fit (prep |1>)")
        n = H[f"{w}_q{q}_b0_za"].sum()
        handles["g"], = ax.plot(zc, n * dz * np.exp(-zc ** 2 / 2) / np.sqrt(2 * np.pi), color=INK, lw=1, ls="--",
                                label="null N(0,1)")
        ax.set_yscale("log"); ax.set_ylim(0.5, n * dz * 0.6); ax.set_xlim(-10, 25)
        ax.set_xlabel("z_a = (perp − μ_s) / σ_s   (per prepared 5-bit state)"); ax.set_ylabel("shots / 0.1 z")
        ax.set_title(f"{PAPER[q]} (q{q}), {wl}\nstandardised perpendicular coordinate", color=INK)
        ax.grid(color=GRID, lw=0.5)
        # (2) one-sided survival counts, z_b
        ax = axs[r, 1]
        ks = np.arange(0, 20.01, 0.1)
        for b, col in ((0, C0), (1, C1)):
            h = H[f"{w}_q{q}_b{b}_zb"].astype(float)
            plus = np.array([h[zc > k].sum() for k in ks]); minus = np.array([h[zc < -k].sum() for k in ks])
            ax.step(ks, np.where(plus > 0, plus, np.nan), where="post", color=col, lw=1.3)
            handles["minus"], = ax.step(ks, np.where(minus > 0, minus, np.nan), where="post", color=col, lw=1.0,
                                        ls="--", label="− side (dashed)")
        N = R["windows"][w][f"q{q}"]["prep0"]["N"]
        ax.plot(ks, N * ndtr(-ks), color=INK, lw=1, ls="--")
        ax.set_yscale("log"); ax.set_ylim(0.5, N * 0.6); ax.set_xlim(0, 20)
        ax.set_xlabel("k  (z_b: noise σ of the prep-|0> state, same neighbours)")
        ax.set_ylabel("shots beyond ±k")
        sr = RW["frame"]["sP_ratio_prep1_over_prep0_median"]
        ax.set_title(f"one-sided tails: solid N(z>k), dashed N(z<−k)\nσ_perp(|1>)/σ_perp(|0>) = {sr:.2f}", color=INK)
        ax.grid(color=GRID, lw=0.5)
        # (3,4) 2-D maps
        for b in (0, 1):
            ax = axs[r, 2 + b]
            h2 = H[f"{w}_q{q}_b{b}_h2"].T.astype(float); h2[h2 == 0] = np.nan
            im = ax.pcolormesh(XE, YE, h2, cmap=CM[b], norm=LogNorm(vmin=1, vmax=np.nanmax(h2)), shading="flat",
                               rasterized=True)
            fr = RW["frame"]
            S = fr["S_sigma_units_median"]; Dd = fr["D_perp_offset_c1_minus_c0_sigma_median"]
            ax.plot([0, S], [0, Dd], color=INK, lw=1.0)
            # d_seg = 5 boundary (stadium)
            th = np.linspace(np.pi / 2, 3 * np.pi / 2, 60)
            bx = np.r_[5 * np.cos(th), S + 5 * np.cos(th[::-1] + np.pi), 5 * np.cos(th[0])]
            by = np.r_[5 * np.sin(th), Dd + 5 * np.sin(th[::-1] + np.pi), 5 * np.sin(th[0])]
            handles["d5"], = ax.plot(bx, by, color=INK2, lw=0.8, ls="--", label="d_seg = 5σ (two-state boundary)")
            ex = E[f"{w}_q{q}_b{b}"]
            for pk in R["windows"][w][f"q{q}"][f"prep{b}"]["peaks"][:3]:
                cx, cy = pk["centre_sigma"]
                handles["pk"], = ax.plot(cx, cy, "x", color=INK, ms=8, mew=1.6, label="excess-cloud centre (2σ disc)")
                ax.add_patch(plt.Circle((cx, cy), 2, fill=False, color=INK, lw=0.8))
            RB = R["windows"][w][f"q{q}"][f"prep{b}"]
            d5 = RB["dseg"]["5"]
            ax.set_xlim(-8, 40); ax.set_ylim(-15, 25); ax.set_aspect("equal")
            ax.set_xlabel("along / σ  (origin c0)"); ax.set_ylabel("perp / σ")
            ax.set_title(f"prepared |{b}>: d_seg>5σ: {d5['obs']}/{d5['N'] if 'N' in d5 else RB['N']} "
                         f"= {100 * d5['frac']:.3f}%", color=INK)
            cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.ax.tick_params(labelsize=7)
    hl = [handles[k] for k in ("p0", "p1", "g", "t", "minus", "d5", "pk") if k in handles]
    fig.legend(hl, [h.get_label() for h in hl], loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=7,
               frameon=False, fontsize=8)
    fig.suptitle(f"{PAPER[q]} (code q{q}): off-axis tails vs Gaussian+crosstalk null, TEST split "
                 f"(560,000 shots per prepared bit)", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(os.path.join(D, f"m1tail_q{q}.png"), dpi=110)
    plt.close(fig)
    print("wrote", f"m1tail_q{q}.png")

# time evolution: sliding 500 ns windows
fig, axs = plt.subplots(2, 5, figsize=(17, 6.5), sharex=True)
hd = {}
for q in range(5):
    SL = R["sliding_500ns"][f"q{q}"]
    t = (np.array(SL["start_bin"]) * 100 + 250) / 1000.0
    for b, col in ((0, C0), (1, C1)):
        P = SL[f"prep{b}"]; N = P["N"]
        one = (np.array(P["zb_plus4"]) - np.array(P["zb_minus4"])) / N
        d5 = np.array(P["dseg5"]) / N
        hd[f"a{b}"], = axs[0, q].plot(t, 100 * one, color=col, lw=1.6, marker="o", ms=3, label=f"prepared |{b}>")
        axs[1, q].plot(t, 100 * d5, color=col, lw=1.6, marker="o", ms=3)
    axs[0, q].set_title(f"{PAPER[q]} (q{q})", color=INK)
    axs[0, q].axhline(0, color=INK2, lw=0.6)
    for a in axs[:, q]:
        a.grid(color=GRID, lw=0.5)
    axs[1, q].set_yscale("symlog", linthresh=0.01)
    axs[1, q].set_xlabel("window centre (µs), 500 ns sliding window")
axs[0, 0].set_ylabel("one-sided excess [N(z_b>4) − N(z_b<−4)] / N  (%)")
axs[1, 0].set_ylabel("off two-state segment, d_seg > 5σ (% of shots)")
fig.legend([hd["a0"], hd["a1"]], ["prepared |0>", "prepared |1>"], loc="lower center", ncol=2, frameon=False)
fig.suptitle("Time evolution of the off-axis population (TEST, 560,000 shots per prepared bit; null ≤ 2e-4 % for d_seg>5)",
             color=INK, fontsize=10)
fig.tight_layout(rect=(0, 0.04, 1, 0.96))
fig.savefig(os.path.join(D, "m1tail_time.png"), dpi=110)
print("wrote m1tail_time.png")
