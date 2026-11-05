#!/usr/bin/env python3
"""m3time_plot.py -- local plots for the temporal-signature analysis (agent tag m3time).

Inputs  (oraqle-ics/data/leakage/): m3time_feat_xt1_q{q}.json, m3time_examples_raw.npz
Outputs (oraqle-ics/data/leakage/): m3time_q{q}_examples.png   per-bin/10 ns time traces of ~22 shots
                                    m3time_q{q}_examples_iq.png  same shots in the IQ plane
                                    m3time_survival.png          survival of L-starters vs |1>-starters
                                    m3time_occupancy.png         P(L at bin k | prepared bit)
Run: /home/manosgior/Documents/GitHub/Oraqle/Discriminators/.venv/bin/python sims/leakage/m3time_plot.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/home/manosgior/Documents/GitHub/oraqle-ics/data/leakage"
RUN = "feat_xt1"
NAMES = ["Q1", "Q2", "Q3", "Q4", "Q5"]
C0, C1, CL = "#2a78d6", "#eb6834", "#1baf7a"     # |0>, |1>, L (reference palette slots 1-3)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SC = {0: C0, 1: C1, 2: CL}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})
TB = (np.arange(20) + 0.5) * 0.1                 # bin centres, us
TF = np.arange(200) * 0.01                       # 10 ns fine grid, us


def frame(d):
    P = d["params_train"]
    mu = np.array(P["mu"])                        # (3,20,2)
    Sig = np.array(P["Sig"])
    m0, m1, mL = mu[0, 3:].mean(0), mu[1, 3:].mean(0), mu[2, 3:].mean(0)
    sep = np.linalg.norm(m1 - m0)
    u = (m1 - m0) / sep
    v = np.array([-u[1], u[0]])
    sb = np.sqrt(np.mean([np.trace(S) / 2 for S in Sig[3:]]))
    return mu, m0, m1, mL, sep, u, v, sb


def norm_traj(x, mu, u, v, sep, kbin):
    """x (...,2) at bins kbin -> (along, perp) in units of sep, centred on the per-bin |0> mean."""
    z = x - mu[0, kbin]
    return z @ u / sep, z @ v / sep


def plot_examples(q, d, E):
    mu, m0, m1, mL, sep, u, v, sb = frame(d)
    b100, fine = E[f"q{q}_b100"], E[f"q{q}_fine10ns"]
    cats, paths, rows, ys = E[f"q{q}_cats"], E[f"q{q}_paths"], E[f"q{q}_rows"], E[f"q{q}_y"]
    n = len(rows)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    Lal = (mL - m0) @ u / sep
    Lpe = (mL - m0) @ v / sep
    fig, axs = plt.subplots(nrow, ncol, figsize=(13, 2.0 * nrow + 0.8), sharex=True, sharey=True)
    axs = np.atleast_2d(axs)
    kf = np.minimum(np.arange(200) // 10, 19)
    for i in range(nrow * ncol):
        ax = axs.flat[i]
        if i >= n:
            ax.axis("off")
            continue
        for k, s in enumerate(paths[i]):
            ax.axvspan(k * 0.1, (k + 1) * 0.1, color=SC["01L".index(s)], alpha=0.13, lw=0)
        fa, fp = norm_traj(fine[i], mu, u, v, sep, kf)
        ba, bp = norm_traj(b100[i], mu, u, v, sep, np.arange(20))
        ax.axhline(0, color=C0, lw=0.8, ls="--")
        ax.axhline(1, color=C1, lw=0.8, ls="--")
        ax.axhline(Lal, color=CL, lw=0.8, ls="--")
        ax.axhline(Lpe, color=CL, lw=0.8, ls=":")
        ax.plot(TF, fa, color=INK, lw=0.6, alpha=0.45)
        ax.plot(TF, fp, color=INK2, lw=0.6, alpha=0.45, ls="-")
        ax.plot(TB, ba, "o-", color=INK, ms=2.8, lw=1.0, label="along (100 ns bins)")
        ax.plot(TB, bp, "s-", color="#8a8984", ms=2.6, lw=1.0, label="perp (100 ns bins)")
        ax.set_title(f"{cats[i]}  idx {rows[i]}  prep{(ys[i] >> q) & 1}\n{paths[i]}", fontsize=6.5, color=INK)
        ax.set_xlim(0, 2.0)
    for ax in axs[-1]:
        ax.set_xlabel("time (us)")
    for ax in axs[:, 0]:
        ax.set_ylabel("units of |0>-|1> sep")
    h = [plt.Line2D([], [], color=INK, marker="o", ms=3, label="along, 100 ns bin"),
         plt.Line2D([], [], color="#8a8984", marker="s", ms=3, label="perp, 100 ns bin"),
         plt.Line2D([], [], color=INK, lw=0.6, alpha=0.5, label="FIR trace, 10 ns"),
         plt.Line2D([], [], color=C0, ls="--", label="|0> along level"),
         plt.Line2D([], [], color=C1, ls="--", label="|1> along level"),
         plt.Line2D([], [], color=CL, ls="--", label="L along level"),
         plt.Line2D([], [], color=CL, ls=":", label="L perp level"),
         plt.Rectangle((0, 0), 1, 1, color=C0, alpha=0.25, label="Viterbi |0>"),
         plt.Rectangle((0, 0), 1, 1, color=C1, alpha=0.25, label="Viterbi |1>"),
         plt.Rectangle((0, 0), 1, 1, color=CL, alpha=0.25, label="Viterbi L")]
    fig.legend(handles=h, loc="lower center", ncol=10, frameon=False, fontsize=7)
    fig.suptitle(f"{NAMES[q]} (code q{q}): strongest TEST candidates, raw traces re-demodulated "
                 f"(L at along {Lal:.2f}, perp {Lpe:+.2f} sep; sep = {sep:.2f} = {sep/sb:.1f} sigma/bin)",
                 fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(f"{D}/m3time_q{q}_examples.png", dpi=130)
    plt.close(fig)
    # IQ-plane version
    fig, axs = plt.subplots(nrow, ncol, figsize=(13, 2.6 * nrow + 0.8), sharex=True, sharey=True)
    axs = np.atleast_2d(axs)
    th = np.linspace(0, 2 * np.pi, 80)
    for i in range(nrow * ncol):
        ax = axs.flat[i]
        if i >= n:
            ax.axis("off")
            continue
        for s, m in ((0, m0), (1, m1), (2, mL)):
            a_, p_ = (m - m0) @ u / sep, (m - m0) @ v / sep
            ax.plot(a_ + sb / sep * np.cos(th), p_ + sb / sep * np.sin(th), color=SC[s], lw=1.0)
        ba, bp = norm_traj(b100[i], mu, u, v, sep, np.arange(20))
        ax.plot(ba[1:], bp[1:], "-", color=GRID, lw=0.8, zorder=1)
        sc = ax.scatter(ba[1:], bp[1:], c=TB[1:], cmap="Greys", vmin=-0.4, vmax=2.0, s=14,
                        edgecolors=INK2, linewidths=0.3, zorder=2)
        ax.set_title(f"{cats[i]}  idx {rows[i]}\n{paths[i]}", fontsize=6.5, color=INK)
        ax.set_aspect("equal", adjustable="datalim")
    for ax in axs[-1]:
        ax.set_xlabel("along (sep)")
    for ax in axs[:, 0]:
        ax.set_ylabel("perp (sep)")
    cb = fig.colorbar(sc, ax=axs, shrink=0.5, pad=0.01)
    cb.set_label("bin centre time (us)")
    fig.suptitle(f"{NAMES[q]}: per-bin IQ trajectories (bins 1-19); circles = HMM state means "
                 f"(blue |0>, orange |1>, green L), radius 1 sigma per bin", fontsize=9, color=INK)
    fig.savefig(f"{D}/m3time_q{q}_examples_iq.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_survival(ds):
    fig, axs = plt.subplots(1, 5, figsize=(15, 3.4), sharey=True)
    for q, ax in enumerate(axs):
        d = ds[q]
        H = d["hard_test"]["viterbi_filtered"]["2us"]
        R = d["hmm_rates_test_onestep"]
        series = [("L-starters (pooled)", H["pooled_exit_from2"], CL, R["from2"]["tau_ns"]),
                  ("|1>-starters, prep 1", H["prep1"]["exit_from1"], C1, R["from1"]["tau_ns"])]
        for lab, r, c, tau in series:
            if not r:
                continue
            kb = np.array(r["survival_bins"])
            ax.plot(TB[kb], r["survival"], "o-", color=c, ms=3, lw=1.6,
                    label=f"{lab}: n={r['n_start']}, Viterbi tau={r['tau_ns']/1e3:.2f} us")
            if np.isfinite(tau):
                ax.plot(TB[kb], np.exp(-(TB[kb] - TB[2]) * 1e3 / tau), color=c, lw=1.0, ls="--",
                        label=f"HMM A: tau={tau/1e3:.2f} us")
        ax.set_yscale("log")
        ax.set_title(f"{NAMES[q]} (q{q})", color=INK)
        ax.set_xlabel("time (us)")
        ax.legend(fontsize=6, frameon=False, loc="lower left")
    axs[0].set_ylabel("P(still in start state)")
    fig.suptitle("Survival in the start state (TEST, 560k shots per prepared bit; start = Viterbi state at "
                 "bins 1-2; 1-bin blips removed)", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"{D}/m3time_survival.png", dpi=130)
    plt.close(fig)


def plot_occupancy(ds):
    fig, axs = plt.subplots(1, 5, figsize=(15, 3.2), sharey=True)
    for q, ax in enumerate(axs):
        O = ds[q]["soft_occupancy_test"]
        for b, c in ((0, C0), (1, C1)):
            ax.plot(TB, O[f"prep{b}"]["P_L_per_bin"], "o-", color=c, ms=3, lw=1.6, label=f"prepared |{b}>")
        ax.set_yscale("log")
        ax.set_title(f"{NAMES[q]} (q{q})", color=INK)
        ax.set_xlabel("time (us)")
        ax.axvline(1.0, color=INK2, lw=0.6, ls=":")
    axs[0].set_ylabel("posterior P(L at bin)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("HMM posterior occupancy of the third state per 100 ns bin (TEST, train-fitted parameters; "
                 "dotted = paper's 1 us)", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(f"{D}/m3time_occupancy.png", dpi=130)
    plt.close(fig)


def main():
    ds = [json.load(open(f"{D}/m3time_{RUN}_q{q}.json")) for q in range(5)]
    E = np.load(f"{D}/m3time_examples_raw.npz")
    for q in range(5):
        plot_examples(q, ds[q], E)
    plot_survival(ds)
    plot_occupancy(ds)
    print("plots written to", D)


if __name__ == "__main__":
    main()
