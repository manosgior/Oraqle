#!/usr/bin/env python3
"""m3time_plot_re.py -- local plots for the temporal-signature analysis with canonical labels (agent tag m3time).

Inputs  (copied from graham ~/oraqle-ics-sims/leakage/m3time/out_re/feat_xt1/ to oraqle-ics/data/leakage/):
  m3time_re_feat_xt1_q{q}.json, m3time_re_examples_raw.npz, m3time_xchk_feat_xt1.json
Outputs (oraqle-ics/data/leakage/):
  m3time_q{q}_examples.png   per-bin (100 ns) and 10 ns re-demodulated trajectories of 20 TEST shots
  m3time_survival.png        survival in L (L-starters) vs in |1> (|1>-starters), Viterbi + HMM exponential
  m3time_occupancy.png       posterior P(L at bin k | prepared bit)
  m3time_exchange.png        Q1/Q5 L transitions locked to Q3 transitions; Q3 decay rate vs Q1/Q5 prepared bits
Run: /home/manosgior/Documents/GitHub/Oraqle/Discriminators/.venv/bin/python sims/leakage/m3time_plot_re.py
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = str(LP.RES)
NAMES = ["Q1", "Q2", "Q3", "Q4", "Q5"]
C0, C1, CL, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"   # reference categorical slots 1,2,3,7
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SC = {0: C0, 1: C1, 2: CL}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})
TB = (np.arange(20) + 0.5) * 0.1
TF = np.arange(200) * 0.01


def frame(d):
    mu = np.array(d["mu_canonical"])
    m0, m1, mL = mu[0, 3:].mean(0), mu[1, 3:].mean(0), mu[2, 3:].mean(0)
    sep = np.linalg.norm(m1 - m0)
    u = (m1 - m0) / sep
    v = np.array([-u[1], u[0]])
    sb = d["geometry"]["2us"]["sigma_bin"]
    return mu, m0, m1, mL, sep, u, v, sb


def norm_traj(x, mu, u, v, sep, kbin):
    z = x - mu[0, kbin]
    return z @ u / sep, z @ v / sep


def plot_examples(q, ds, E):
    d = ds[q]
    mu, m0, m1, mL, sep, u, v, sb = frame(d)
    b100, fine = E[f"q{q}_b100"], E[f"q{q}_fine10ns"]
    cats, paths, rows, ys = E[f"q{q}_cats"], E[f"q{q}_paths"], E[f"q{q}_rows"], E[f"q{q}_y"]
    Fall, Pall = E[f"q{q}_feat_all"], E[f"q{q}_paths_all"]
    show3 = q in (0, 4)                                  # overlay Q3 for the qubits whose L is Q3-locked
    if show3:
        mu3, _, _, _, sep3, u3, v3, _ = frame(ds[2])
    n = len(rows)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    Lal, Lpe = (mL - m0) @ u / sep, (mL - m0) @ v / sep
    fig, axs = plt.subplots(nrow, ncol, figsize=(13, 2.05 * nrow + 1.0), sharex=True, sharey=True)
    axs = np.atleast_2d(axs)
    kf = np.minimum(np.arange(200) // 10, 19)
    for i in range(nrow * ncol):
        ax = axs.flat[i]
        if i >= n:
            ax.axis("off")
            continue
        for k, s in enumerate(paths[i]):
            ax.axvspan(k * 0.1, (k + 1) * 0.1, color=SC["01L".index(s)], alpha=0.14, lw=0)
        fa, fp = norm_traj(fine[i], mu, u, v, sep, kf)
        ba, bp = norm_traj(b100[i], mu, u, v, sep, np.arange(20))
        ax.axhline(0, color=C0, lw=0.8, ls="--")
        ax.axhline(1, color=C1, lw=0.8, ls="--")
        ax.axhline(Lal, color=CL, lw=0.8, ls="--")
        ax.axhline(Lpe, color=CL, lw=0.8, ls=":")
        ax.plot(TF, fa, color=INK, lw=0.5, alpha=0.35)
        ax.plot(TF, fp, color=INK2, lw=0.5, alpha=0.35)
        ax.plot(TB, ba, "o-", color=INK, ms=2.8, lw=1.0)
        ax.plot(TB, bp, "s-", color="#8a8984", ms=2.6, lw=1.0)
        ttl = f"{cats[i]}  idx {rows[i]}  prep{(ys[i] >> q) & 1}\n{NAMES[q]}: {paths[i]}"
        if show3:
            a3 = (Fall[i, 2] - mu3[0]) @ u3 / sep3          # Q3 along, units of Q3 sep (not xt-corrected)
            ax.plot(TB, a3, "-", color=C4, lw=1.1, alpha=0.9)
            ttl += f"\nQ3: {''.join('01L'[s] for s in Pall[i, 2])} (prep{(ys[i] >> 2) & 1})"
        ax.set_title(ttl, fontsize=6.3, color=INK, loc="left")
        ax.set_xlim(0, 2.0)
        ax.axvline(1.0, color=INK2, lw=0.6, ls=":")
    for ax in axs[-1]:
        ax.set_xlabel("time (us)")
    for ax in axs[:, 0]:
        ax.set_ylabel(f"{NAMES[q]} IQ / |0>-|1> sep")
    h = [plt.Line2D([], [], color=INK, marker="o", ms=3, label="along, 100 ns bin"),
         plt.Line2D([], [], color="#8a8984", marker="s", ms=3, label="perp, 100 ns bin"),
         plt.Line2D([], [], color=INK, lw=0.6, alpha=0.5, label="FIR trace, 10 ns"),
         plt.Line2D([], [], color=C0, ls="--", label="|0> level"),
         plt.Line2D([], [], color=C1, ls="--", label="|1> along level"),
         plt.Line2D([], [], color=CL, ls="--", label="L along level"),
         plt.Line2D([], [], color=CL, ls=":", label="L perp level"),
         plt.Rectangle((0, 0), 1, 1, color=C0, alpha=0.3, label="Viterbi |0>"),
         plt.Rectangle((0, 0), 1, 1, color=C1, alpha=0.3, label="Viterbi |1>"),
         plt.Rectangle((0, 0), 1, 1, color=CL, alpha=0.3, label="Viterbi L")]
    if show3:
        h.append(plt.Line2D([], [], color=C4, lw=1.1, label="Q3 along (Q3 sep units)"))
    fig.legend(handles=h, loc="lower center", ncol=11, frameon=False, fontsize=7)
    fig.suptitle(f"{NAMES[q]} (code q{q}): 20 TEST shots, raw traces re-demodulated. L sits at along {Lal:.2f}, "
                 f"perp {Lpe:+.2f} sep; sep = {sep:.2f} ADC units = {sep / sb:.1f} sigma per 100 ns bin. "
                 f"Dotted vertical = paper's 1 us.", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    fig.savefig(f"{D}/m3time_q{q}_examples.png", dpi=125)
    plt.close(fig)


def plot_survival(ds):
    fig, axs = plt.subplots(1, 5, figsize=(15, 3.6), sharey=True)
    for q, ax in enumerate(axs):
        d = ds[q]
        H = d["hard_test"]["viterbi_filtered"]["2us"]
        R = d["hmm_rates_test_onestep"]
        for lab, r, c, tau in (("L-starters", H["pooled_exit_from2"], CL, R["from2"]["tau_ns"]),
                               ("|1>-starters (prep 1)", H["prep1"]["exit_from1"], C1, R["from1"]["tau_ns"])):
            if not r:
                continue
            kb = np.array(r["survival_bins"])
            ax.plot(TB[kb], r["survival"], "o-", color=c, ms=3.5, lw=2,
                    label=f"{lab} n={r['n_start']}: tau={r['tau_ns'] / 1e3:.2f} us")
            ax.plot(TB[kb], np.exp(-(TB[kb] - TB[2]) * 1e3 / tau), color=c, lw=1, ls="--")
        t10 = R["from1"]["tau_to0_ns"]
        ax.plot(TB[2:], np.exp(-(TB[2:] - TB[2]) * 1e3 / t10), color=C0, lw=1, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(0.2, 1.05)
        ax.set_title(f"{NAMES[q]} (q{q})", color=INK)
        ax.set_xlabel("time (us)")
        ax.legend(fontsize=6, frameon=False, loc="lower left")
    axs[0].set_ylabel("P(still in start state)")
    h = [plt.Line2D([], [], color=INK2, ls="--", label="HMM exponential, total exit"),
         plt.Line2D([], [], color=C0, ls=":", label="HMM |1>->|0> partial (T1 under readout)")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Survival in the start state (TEST; start = filtered-Viterbi state in bins 1-2; markers = Viterbi, "
                 "tau = censored-geometric MLE over 2 us)", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(f"{D}/m3time_survival.png", dpi=130)
    plt.close(fig)


def plot_occupancy(ds):
    fig, axs = plt.subplots(1, 5, figsize=(15, 3.3), sharey=True)
    for q, ax in enumerate(axs):
        O = ds[q]["soft_occupancy_test"]
        for b, c in ((0, C0), (1, C1)):
            ax.plot(TB, O[f"prep{b}"]["P_L_per_bin"], "o-", color=c, ms=3, lw=1.8, label=f"prepared |{b}>")
        ax.set_yscale("log")
        ax.set_title(f"{NAMES[q]} (q{q})", color=INK)
        ax.set_xlabel("time (us)")
        ax.axvline(1.0, color=INK2, lw=0.6, ls=":")
    axs[0].set_ylabel("posterior P(L at bin)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("HMM posterior occupancy of the third state per 100 ns bin (TEST, 560000 shots per prepared bit, "
                 "train-fitted parameters; dotted = paper's 1 us)", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    fig.savefig(f"{D}/m3time_occupancy.png", dpi=130)
    plt.close(fig)


def plot_exchange(X):
    lags = np.arange(-6, 7)
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    series = [("Q5 1->L vs Q3 1->0 | prepq1_prepr1", C4, "-", "Q5 1->L  vs  Q3 1->0"),
              ("Q5 L->1 vs Q3 0->1 | prepq1_prepr1", C4, "--", "Q5 L->1  vs  Q3 0->1"),
              ("Q1 1->L vs Q3 1->0 | prepq1_prepr1", C1, "-", "Q1 1->L  vs  Q3 1->0"),
              ("Q1 L->1 vs Q3 0->1 | prepq1_prepr1", C1, "--", "Q1 L->1  vs  Q3 0->1")]
    ax = axs[0]
    for key, c, ls, lab in series:
        r = X["lags"][key]
        ax.plot(lags, [r[str(l)]["lift"] for l in lags], "o" + ls, color=c, ms=4, lw=1.8,
                label=f"{lab} (n={r['n_events']})")
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_xlabel("lag of the Q3 transition (100 ns bins)")
    ax.set_ylabel("lift over same-bin baseline")
    ax.set_title("Q1/Q5 L transitions are locked to Q3 transitions\n(both qubits prepared |1>)", color=INK)
    ax.legend(fontsize=6.5, frameon=False)
    ax = axs[1]
    ctrl = [("Q3 1->L vs Q4 1->L | prepq1_prepr1", C0, "Q3 1->L vs Q4 1->L"),
            ("Q3 1->L vs Q5 1->L | prepq1_prepr1", C4, "Q3 1->L vs Q5 1->L"),
            ("Q3 1->L vs Q1 1->L | prepq1_prepr1", C1, "Q3 1->L vs Q1 1->L"),
            ("Q2 1->L vs Q3 1->0 | prepq1_prepr1", "#8a8984", "Q2 1->L vs Q3 1->0"),
            ("Q4 1->L vs Q3 1->0 | prepq1_prepr1", CL, "Q4 1->L vs Q3 1->0")]
    for key, c, lab in ctrl:
        r = X["lags"][key]
        ax.plot(lags, [r[str(l)]["lift"] for l in lags], "o-", color=c, ms=3.5, lw=1.5,
                label=f"{lab} (n={r['n_events']})")
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_ylim(0, 3)
    ax.set_xlabel("lag (100 ns bins)")
    ax.set_title("Controls: Q3's own L, Q2's and Q4's L are not locked", color=INK)
    ax.legend(fontsize=6.5, frameon=False)
    ax = axs[2]
    dd = X["Q3_decay_by_Q1_Q5_prep"]
    keys = ["Q1prep0_Q5prep0", "Q1prep0_Q5prep1", "Q1prep1_Q5prep0", "Q1prep1_Q5prep1"]
    labs = ["Q1=0\nQ5=0", "Q1=0\nQ5=1", "Q1=1\nQ5=0", "Q1=1\nQ5=1"]
    p = np.array([dd[k]["p_1to0_per_100ns"]["frac"] for k in keys]) * 100
    lo = p - np.array([dd[k]["p_1to0_per_100ns"]["ci95"][0] for k in keys]) * 100
    hi = np.array([dd[k]["p_1to0_per_100ns"]["ci95"][1] for k in keys]) * 100 - p
    pl = np.array([dd[k]["p_1toL_per_100ns"]["frac"] for k in keys]) * 100
    x = np.arange(4)
    ax.bar(x - 0.18, p, 0.34, color=C0, yerr=[lo, hi], label="Q3 |1>->|0>")
    ax.bar(x + 0.18, pl, 0.34, color=CL, label="Q3 |1>->L")
    for xi, pi in zip(x, p):
        ax.text(xi - 0.18, pi + 0.15, f"{pi:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel("per-100 ns rate (%)")
    ax.set_title("Q3 (prepared |1>, started in |1>) loses its excitation\nfaster when Q1 / Q5 are prepared |1>",
                 color=INK)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{D}/m3time_exchange.png", dpi=130)
    plt.close(fig)


def main():
    ds = [json.load(open(f"{D}/m3time_re_feat_xt1_q{q}.json")) for q in range(5)]
    E = np.load(f"{D}/m3time_re_examples_raw.npz")
    X = json.load(open(f"{D}/m3time_xchk_feat_xt1.json"))
    for q in range(5):
        plot_examples(q, ds, E)
    plot_survival(ds)
    plot_occupancy(ds)
    plot_exchange(X)
    print("plots written to", D)


if __name__ == "__main__":
    main()
