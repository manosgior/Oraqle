"""m4conf_plots.py -- local plots for the m4conf confounder analysis.
Usage: python m4conf_plots.py <dir with m4conf_*.json/npz> <out dir>"""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, DST = sys.argv[1], sys.argv[2]
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
GRID = "#d9d8d4"; INK = "#0b0b0b"; INK2 = "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2})
R = json.load(open(f"{SRC}/m4conf_results.json"))
FIN = json.load(open(f"{SRC}/m4conf_final.json"))["funnel"]
FU = json.load(open(f"{SRC}/m4conf_followup.json"))
FU2 = json.load(open(f"{SRC}/m4conf_followup2.json"))
RP = json.load(open(f"{SRC}/m4conf_rawprobe.json"))
PD = np.load(f"{SRC}/m4conf_plotdata.npz")
QN = [f"Q{q + 1}\n(q{q})" for q in range(5)]


def wil(k, n, z=1.96):
    k = np.asarray(k, float); n = np.asarray(n, float)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, np.maximum(c - h, 0), c + h


def legend_bottom(fig, handles, labels, ncol, y=0.0):
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False)


# ---------------------------------------------------------------- 1 funnel
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
series = [("prior |2> labels (B)", None, C4), ("candidates |z_perp|>5 (A)", "candidates", C1),
          ("after 4 base screens", "BASE_survivor_rate", C2), ("single-qubit (no neighbour event)", "SINGLE_QUBIT_rate", C3)]
hs = []
for bi, b in enumerate((0, 1)):
    ax = axs[bi]
    for si, (lab, key, col) in enumerate(series):
        xs, ps, lo, hi = [], [], [], []
        for q in range(5):
            if key is None:
                e = R["d_prepdep"]["B"][f"q{q}"][f"prep{b}"]
            else:
                e = FIN["1us"][f"q{q}_prep{b}"][key]
            k, n = e["k"], e["n"]
            p, l, h = wil(k, n)
            xs.append(q + (si - 1.5) * 0.17); ps.append(max(p, 1e-7)); lo.append(max(l, 1e-7)); hi.append(h)
        ps, lo, hi = map(np.array, (ps, lo, hi))
        hh = ax.errorbar(xs, ps, yerr=[ps - lo, hi - ps], fmt="o", ms=6, color=col, ecolor=col, elinewidth=1.5, capsize=0, label=lab)
        if bi == 0:
            hs.append(hh)
    ax.set_yscale("log"); ax.set_ylim(5e-7, 0.2)
    ax.set_xticks(range(5)); ax.set_xticklabels(QN)
    ax.set_title(f"prepared |{b}>  (n = 800,000 shots per qubit; train+test)", fontsize=9)
    ax.axhline(5.7e-7, color=INK2, lw=1, ls=":")
    ax.text(4.45, 6.5e-7, "Gaussian |z|>5", ha="right", va="bottom", fontsize=7, color=INK2)
axs[0].set_ylabel("fraction of prepared shots (log)")
fig.suptitle("Off-axis candidates, 1 us window, before and after confounder screens (95% Wilson CI; points at 1e-7 = zero count)", fontsize=10)
legend_bottom(fig, hs, [h.get_label() for h in hs], 4)
fig.tight_layout(rect=(0, 0.08, 1, 0.95))
fig.savefig(f"{DST}/m4conf_funnel.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- 2 crosstalk per neighbour configuration
fig, axs = plt.subplots(2, 5, figsize=(15, 6), sharex=True)
for bi, b in enumerate((0, 1)):
    for q in range(5):
        ax = axs[bi, q]
        for name, col, lab in [("A", C1, "A: |z_perp|>5 (per-bit z)"), ("B", C4, "B: prior |2> labels")]:
            arr = PD[f"a_rates__{name}_q{q}_p{b}"]
            k, n, dm = arr
            p, l, h = wil(k, n)
            x = np.arange(16) + (0.15 if name == "B" else -0.15)
            ax.errorbar(x, np.maximum(p, 1e-7), yerr=[np.maximum(p, 1e-7) - np.maximum(l, 1e-7), h - np.maximum(p, 1e-7)], fmt="o", ms=4,
                        color=col, elinewidth=1, label=lab)
        k, n, dm = PD[f"a_rates__A_q{q}_p{b}"]
        from scipy.stats import norm
        pred = norm.sf(5 - dm) + norm.cdf(-5 - dm)
        ax.plot(np.arange(16), pred, color=INK2, lw=1.2, ls="--", label="Gaussian, centroid shifted by neighbours")
        ax.set_yscale("log"); ax.set_ylim(3e-7, 0.3)
        e = R["a_xtalk"]["A"][f"q{q}_prep{b}"]
        pv = "n/a" if e["p"] is None else f"{e['p']:.1g}"
        ax.set_title(f"Q{q + 1} prepared |{b}>   A: chi2 p={pv}", fontsize=8)
        others = [j for j in range(5) if j != q]
        ax.set_xticks(range(16))
        ax.set_xticklabels(["".join(str((s >> i) & 1) for i in range(4)) for s in range(16)], rotation=90, fontsize=6)
        if q == 0:
            ax.set_ylabel("candidate rate")
        if bi == 1:
            ax.set_xlabel("neighbour bits " + ",".join(f"Q{j + 1}" for j in others), fontsize=7)
h, l = axs[0, 1].get_legend_handles_labels()
legend_bottom(fig, h, l, 3)
fig.suptitle("(a) Candidate rate per neighbour configuration (16 per prepared bit, 50,000 shots each). Centroid shifts predict <= 1.4e-6.", fontsize=10)
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
fig.savefig(f"{DST}/m4conf_xtalk.png", dpi=120); plt.close(fig)

# ---------------------------------------------------------------- 3 simultaneity: lift matrices + neighbour-event O/E
fig, axs = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1.1, 1.5]})
for ax, name, ttl in [(axs[0], "A", "A candidates (1 us)"), (axs[1], "B", "B prior labels")]:
    b = R["b_simul"][name]
    L = np.array([[np.nan if v is None else v for v in r] for r in b["lift"]])
    both = np.array(b["both"])
    im = ax.imshow(np.log2(np.where(np.isnan(L), 1, np.maximum(L, 1 / 16))), cmap="RdBu_r", vmin=-3, vmax=3)
    for i in range(5):
        for j in range(5):
            if i != j:
                ax.text(j, i, f"{L[i, j]:.2f}\n({both[i, j]})", ha="center", va="center", fontsize=7, color=INK)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([f"Q{q + 1}" for q in range(5)]); ax.set_yticklabels([f"Q{q + 1}" for q in range(5)])
    ax.grid(False)
    ax.set_title(f"{ttl}: pairwise lift obs/exp\n(>=2 qubits: obs {b['ge2_observed']} vs exp {b['ge2_expected']:.0f})", fontsize=9)
cb = fig.colorbar(im, ax=axs[1], fraction=0.045, pad=0.03); cb.set_label("log2 lift")
ax = axs[2]
rows = []
for key, v in FU["F1_neighbour_event_OE"].items():
    q = int(key[1]); b = int(key[-1])
    for kk, e in v.items():
        if e["OE"] is None or e["obs"] < 15:
            continue
        lo, hi = e["OE_ci95"]
        if lo > 1.5 or hi < 0.67:
            rows.append((f"Q{q + 1}|{b}> & Q{int(kk[1]) + 1} {kk.split('_')[1]}", e["OE"], lo, hi, e["obs"]))
rows.sort(key=lambda r: -r[1])
rows = rows[:16]
yy = np.arange(len(rows))[::-1]
ax.errorbar([r[1] for r in rows], yy, xerr=[[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]], fmt="o", color=C1, ms=5)
ax.axvline(1, color=INK2, lw=1)
ax.set_yticks(yy); ax.set_yticklabels([f"{r[0]}  (n={r[4]})" for r in rows], fontsize=7)
ax.set_xscale("log"); ax.set_xlabel("observed / expected under within-state independence")
ax.set_title("Neighbour events coinciding with A candidates\n(flip = read on the other side, far = >1.5 sep, off = |z|>5)", fontsize=9)
fig.suptitle("(b) Simultaneity: no feedline-wide glitch (all-tone coincidences absent), but specific qubit pairs co-fire", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(f"{DST}/m4conf_simultaneity.png", dpi=120); plt.close(fig)

# ---------------------------------------------------------------- 4 time / drift / memory
fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))
ax = axs[0, 0]
cols = [C1, C2, C3, C4, C5]
for q in (1, 2, 4):                      # qubits with >= 1000 prepared-|1> candidates (Q1: 165, Q4: 2)
    tr = PD[f"time_rate__A_q{q}_p1"]
    test = tr[16:]                       # blocks are ordered train (16) then test (16) for this bit
    m = np.nanmean(test, 0)
    ok = ~np.isnan(m)
    base = np.nanmean(m[ok])
    if base > 0:
        ax.plot(np.arange(70)[ok] * 500 / 1000, m[ok] / base, color=cols[q], lw=1.5, label=f"Q{q + 1} (mean {100 * base:.3f}%)")
ax.axhline(1, color=INK2, lw=1)
ax.set_xlabel("row position inside a TEST block (thousand shots)"); ax.set_ylabel("candidate rate / block mean")
ax.set_title("A candidates, prepared |1>, 500-shot bins, mean over the 16 TEST blocks\n(Q1, Q4 omitted: 165 and 2 candidates)", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="upper right")
ax = axs[0, 1]
for q in range(5):
    k = f"q{q}_prep1"
    if k in FU["F3_memory"]:
        acf = FU["F3_memory"][k]["acf_lag1_20"]
        ax.plot(range(1, 21), acf, "o-", color=cols[q], ms=3, lw=1.2, label=f"Q{q + 1}")
se = FU["F3_memory"]["q1_prep1"]["se_iid"]
ax.fill_between([0.5, 20.5], -2 * se, 2 * se, color=GRID, alpha=0.6, lw=0, label="+-2 SE (iid)")
ax.set_xlabel("lag (shots)"); ax.set_ylabel("autocorrelation of candidate indicator")
ax.set_title("Shot-to-shot memory of A candidates (prepared |1>)", fontsize=9)
ax.legend(fontsize=7, frameon=False, ncol=3)
ax = axs[1, 0]
lags = [1, 2, 5, 10, 50, 200]
for q in range(5):
    for b, ls in ((1, "-"), (0, ":")):
        v = FU["F4_acf"][f"q{q}_along_prep{b}"]["median_over_blocks"]
        ax.plot(lags, v, ls, marker="o", color=cols[q], ms=3, lw=1.3, label=f"Q{q + 1} prep |{b}>" if q in (2, 3) or b == 1 else None)
ax.set_xscale("log"); ax.axhline(0, color=INK2, lw=1)
ax.set_xlabel("lag (shots)"); ax.set_ylabel("ACF of the along-axis window mean (median over blocks)")
ax.set_title("Q4 (and weakly Q3) prepared |1>: correlated over 1-2 consecutive shots", fontsize=9)
ax.legend(fontsize=7, frameon=False, ncol=2)
ax = axs[1, 1]
bb = PD["drift_bin_block"]; bp = PD["drift_bin_pos"]
for q, comp in [(3, "along"), (2, "along"), (0, "perp")]:
    med = PD[f"drift__q{q}_{comp}"]
    for blk_i in (32 + 31, 32 + 8):     # two TEST blocks (state 31: all |1>; state 8: only Q4 |1>)
        m = bb == blk_i
        ax.plot(bp[m] * 0.5, med[m] - np.median(med[m]), color=cols[q], lw=1.2, ls="-" if blk_i == 63 else "--",
                label=f"Q{q + 1} {comp}, state {blk_i - 32}")
ax.set_xlabel("row position inside the block (thousand shots)"); ax.set_ylabel("500-shot median - block median (ADC units)")
ax.set_title("Centroid drift inside two TEST blocks (sigma_shot ~ 0.55-0.75)", fontsize=9)
ax.legend(fontsize=7, frameon=False, ncol=2)
fig.suptitle("(c) Time structure: candidate rates are stationary (dispersion ~1, no bursts); small lag-1 memory for Q2/Q3; Q4 |1> has 1-2 shot memory", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{DST}/m4conf_time.png", dpi=120); plt.close(fig)

# ---------------------------------------------------------------- 5 B composition (overlap)
fig, ax = plt.subplots(figsize=(10, 4.2))
cats = [("offaxis_absz_ge5", "|z_perp|>=5 (= set A)", C1), ("tail_3_to_5", "3<=|z|<5 (Gaussian-tail range)", C4),
        ("absz_lt3_gaussian_core_within_3sigma_of_a_centroid", "|z|<3, inside a |0>/|1> core", C3),
        ("absz_lt3_on_axis_between_centroids_T1like", "|z|<3, on-axis between centroids (T1-like)", C2),
        ("absz_lt3_on_axis_beyond_centroids", "|z|<3, on-axis beyond centroids", C5)]
labels = []; y = 0
for q in range(5):
    for b in (0, 1):
        c = R["e_overlap"][f"q{q}"]["B_composition_in_m4conf_frame"][f"prep{b}"]
        left = 0
        for key, lab, col in cats:
            v = c[key]["rate"]
            ax.barh(y, v, left=left, color=col, edgecolor="white", linewidth=1.5, height=0.75, label=lab if y == 0 else None)
            left += v
        labels.append(f"Q{q + 1} |{b}>  n={c['n']}"); y += 1
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis()
ax.set_xlim(0, 1); ax.set_xlabel("fraction of prior |2> labels (B), classified in the m4conf 1 us frame")
ax.grid(axis="y", visible=False)
h, l = ax.get_legend_handles_labels()
legend_bottom(fig, h, l, 3)
fig.suptitle("(e) What the prior |2> labels are: A is a strict subset of B (Jaccard 0.008-0.52)", fontsize=10)
fig.tight_layout(rect=(0, 0.12, 1, 0.94))
fig.savefig(f"{DST}/m4conf_overlap.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- 6 Q5 x Q3 trajectories
G3 = FU2["G3_q4_cand_vs_q2_flip"]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
tb = np.arange(20) * 0.1 + 0.05
grp = [("q4cand_q2flip", "Q5 candidate & Q3 flipped", C1), ("q4cand_q2stay", "Q5 candidate & Q3 stayed |1>", C2),
       ("q4non_q2flip", "Q5 not candidate & Q3 flipped", C3), ("q4non_q2stay", "Q5 not candidate & Q3 stayed", C4)]
for k, lab, col in grp:
    axs[0].plot(tb, G3[k]["q5_perp_z_median_per_bin"], color=col, label=f"{lab} (n={G3['counts'][k]})")
    axs[1].plot(tb, G3[k]["q3_along_t_median_per_bin"], color=col, label=lab)
axs[0].set_ylabel("Q5 perpendicular, median z per 100 ns bin"); axs[1].set_ylabel("Q3 along, median position (0 = |0>, 1 = |1>)")
for ax in axs:
    ax.set_xlabel("time in the trace (us)"); ax.axvline(1.0, color=INK2, lw=1, ls=":")
axs[0].set_title("Q5 off-axis excursion: rises in 300 ns, relaxes over ~1.5 us", fontsize=9)
axs[1].set_title("Q3 in the coincident shots is at |0> from the first bins", fontsize=9)
h, l = axs[0].get_legend_handles_labels()
legend_bottom(fig, h, l, 2)
fig.suptitle("Q5 (q4) prepared |1> candidates mostly coincide with Q3 (q2) prepared |1> reading |0>; both qubits prepared |1>", fontsize=10)
fig.tight_layout(rect=(0, 0.13, 1, 0.94))
fig.savefig(f"{DST}/m4conf_q5q3.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- 7 raw probe
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
ax = axs[0]
rowsl = []; M = []
for q in range(5):
    for g in ("A_p1", "A_p0", "ctl"):
        e = RP[f"q{q}"].get(g, {})
        if not e.get("n") or e["n"] < 20:
            continue
        M.append([e[f"tone_q{j}"]["frac_2d_gt4"] for j in range(5)])
        rowsl.append(f"Q{q + 1} {g.replace('A_p', 'cand |').replace('ctl', 'controls')}{'>' if g != 'ctl' else ''} (n={e['n']})")
M = np.array(M)
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=0.5, aspect="auto")
for i in range(M.shape[0]):
    for j in range(5):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] > 0.3 else INK)
ax.set_xticks(range(5)); ax.set_xticklabels([f"Q{j + 1} tone" for j in range(5)])
ax.set_yticks(range(len(rowsl))); ax.set_yticklabels(rowsl, fontsize=7); ax.grid(False)
ax.set_title("fraction with residual > 4 sigma on each tone (1 us, residual = trace - state mean)", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.03)
ax = axs[1]
xs = []; ys = []; lab = []
for q in range(5):
    for g, col in (("A_p1", C1), ("A_p0", C2), ("Bonly", C4), ("ctl", INK2)):
        e = RP[f"q{q}"].get(g, {})
        if not e.get("n") or e["n"] < 20:
            continue
        ax.plot(q + {"A_p1": -0.2, "A_p0": -0.07, "Bonly": 0.07, "ctl": 0.2}[g], e["oob_over_ctl_median"]["median"], "o", color=col, ms=6,
                label={"A_p1": "A cand |1>", "A_p0": "A cand |0>", "Bonly": "B-only labels", "ctl": "controls"}[g] if q == 1 else None)
        ax.plot([q + {"A_p1": -0.2, "A_p0": -0.07, "Bonly": 0.07, "ctl": 0.2}[g]] * 2, [1, e["oob_over_ctl_median"]["p99"]], color=col, lw=1)
ax.set_xticks(range(5)); ax.set_xticklabels([f"Q{q + 1}" for q in range(5)]); ax.set_ylim(0.9, 1.25)
ax.set_ylabel("out-of-band residual power / control median"); ax.set_title("broadband (out-of-band) residual: median (dot), 99th pct (line)", fontsize=9)
h, l = ax.get_legend_handles_labels()
legend_bottom(fig, h, l, 4)
fig.suptitle("(f) Raw traces (TEST, up to 200 shots per group): no feedline-wide or broadband glitch; partner tone Q3 is elevated for Q2/Q4/Q5 candidates", fontsize=10)
fig.tight_layout(rect=(0, 0.08, 1, 0.94))
fig.savefig(f"{DST}/m4conf_rawprobe.png", dpi=130); plt.close(fig)
print("plots written")
