"""m4conf_followup2.py -- final estimates + two targeted confounder checks (agent tag m4conf).

G1  'quiet-neighbour' estimate: rate of A candidates that pass the 5 base screens (S_frame, S_xtalk, S_simul,
    S_raw, S_time) among shots in which every OTHER qubit is in the 'stay' class (not off-axis, not beyond
    1.5 separations, not on the opposite side of its prepared bit) in the same window. This removes all
    candidates that coincide with a neighbour event, without a chance-coincidence correction.
G2  Q4 (q3) prepared |1>: consecutive-shot transition matrix of the along class (low t<0.2 / blob 0.2-0.7 /
    main t>=0.7) and where the blob shots sit in the first 200 ns (present from t=0 or developing).
G3  Q5 (q4) candidates coincident with a Q3 (q2) 'flip': per-bin trajectories (20 bins, 2 us) of Q5 perp and
    Q3 along, for Q5-cand & Q3-flip, Q5-cand & Q3-stay, and controls (Q5 non-cand & Q3-flip / stay).
Usage: python m4conf_followup2.py /feat /work/out
"""
import json, sys
import numpy as np

FEAT, OUT = sys.argv[1], sys.argv[2]
D = np.load(f"{OUT}/m4conf_pershot.npz")
y = D["y"]; blk = D["blk"]; pos = D["pos"]; spi = D["spi"]; row = D["row"]
N = len(y); NQ = 5
bits = ((y[:, None] >> np.arange(NQ)) & 1).astype(np.int8)
R = {}


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0, c - h)), float(min(1, c + h))]


def rate(k, n):
    return {"k": int(k), "n": int(n), "rate": (float(k / n) if n else None), "ci95": wilson(k, n)}


def katz(k1, n1, k0, n0):
    if k1 == 0 or k0 == 0:
        return [None, None]
    rr = (k1 / n1) / (k0 / n0); se = np.sqrt(1 / k1 - 1 / n1 + 1 / k0 - 1 / n0)
    return [float(rr * np.exp(-1.96 * se)), float(rr * np.exp(1.96 * se))]


def classes(zA0, tA):
    CL = np.zeros((N, NQ), np.int8)
    for j in range(NQ):
        off = np.abs(zA0[:, j]) > 5
        far = (tA[:, j] >= 1.5) & ~off
        flip = np.where(bits[:, j] == 1, tA[:, j] < 0.5, tA[:, j] >= 0.5) & ~off & ~far
        CL[flip, j] = 1; CL[far, j] = 2; CL[off, j] = 3
    return CL


# ---------------- G1
R["G1_quiet_neighbour"] = {}
for wn, cand_key, surv_key, z0k, tk in [("1us", "A", "surv_A", "zA0", "tA"), ("2us", None, "surv_A2us", "zA0_2us", "tA_2us")]:
    CL = classes(D[z0k], D[tk])
    cand = D["A"] if cand_key else (np.abs(D["zA_2us"]) > 5)
    S = D[surv_key]
    out = {}
    for q in range(NQ):
        quiet = (np.delete(CL, q, 1) == 0).all(1)
        for b in (0, 1):
            mb = bits[:, q] == b
            nq_ = int((mb & quiet).sum())
            kc = int((cand[:, q] & mb).sum()); n = int(mb.sum())
            kcq = int((cand[:, q] & mb & quiet).sum())
            ksq = int((S[:, q] & mb & quiet).sum())
            ks = int((S[:, q] & mb).sum())
            rq = ksq / nq_ if nq_ else None
            out[f"q{q}_prep{b}"] = {
                "n_prepared": n, "n_quiet_neighbour_shots": nq_, "quiet_fraction": nq_ / n,
                "candidate_rate_all": rate(kc, n), "candidate_rate_quiet": rate(kcq, nq_),
                "survivor_rate_all_shots_5screens": rate(ks, n),
                "survivor_rate_quiet_LEAKAGE_ESTIMATE": rate(ksq, nq_),
                "surviving_fraction_of_candidates": ((rq / (kc / n)) if kc and rq is not None else None),
                "surviving_fraction_ci95_katz": katz(ksq, nq_, kc, n) if kc and ksq else [None, None],
                "fraction_of_candidates_in_quiet_shots_passing_screens": rate(ksq, kcq)}
    R["G1_quiet_neighbour"][wn] = out
    # sensitivity: noise-0 scale candidates (no further screens), quiet neighbours
    if wn == "1us":
        sens = {}
        c0 = np.abs(D["zA0"]) > 5
        for q in range(NQ):
            quiet = (np.delete(CL, q, 1) == 0).all(1)
            for b in (0, 1):
                mb = bits[:, q] == b
                sens[f"q{q}_prep{b}"] = {"all": rate((c0[:, q] & mb).sum(), mb.sum()), "quiet": rate((c0[:, q] & mb & quiet).sum(), (mb & quiet).sum())}
        R["G1_noise0scale_candidates"] = sens
    CL1 = CL if wn == "1us" else CL1

# ---------------- G2  Q4 prepared |1> consecutive-shot transitions
t3 = D["tA"][:, 3]; t3e = D["tA_early"][:, 3]
cat = np.where(t3 < 0.2, 0, np.where(t3 < 0.7, 1, 2))
names = ["low(t<0.2)", "blob(0.2-0.7)", "main(t>=0.7)"]
for b in (1, 0):
    T = np.zeros((3, 3)); marg = np.zeros(3)
    for bb in range(64):
        if ((bb % 32) >> 3) & 1 != b:
            continue
        m = np.flatnonzero(blk == bb); m = m[np.argsort(pos[m])]
        c = cat[m]
        np.add.at(T, (c[:-1], c[1:]), 1); marg += np.bincount(c, minlength=3)
    P = T / T.sum(1, keepdims=True)
    R[f"G2_q3_prep{b}_transition"] = {"categories": names, "P_next_given_prev": P.tolist(), "marginal": (marg / marg.sum()).tolist(),
                                      "counts": T.tolist(), "lift_diag": [float(P[i, i] / (marg[i] / marg.sum())) for i in range(3)]}
mb = bits[:, 3] == 1
R["G2_q3_prep1_early_t_by_class"] = {names[i]: {"n": int((mb & (cat == i)).sum()),
                                                "early200ns_t_percentiles_10_50_90": np.percentile(t3e[mb & (cat == i)], [10, 50, 90]).tolist()}
                                     for i in range(3)}
R["G2_q3_prep0_early_t_by_class"] = {names[i]: {"n": int(((~mb) & (cat == i)).sum()),
                                                "early200ns_t_percentiles_10_50_90": (np.percentile(t3e[(~mb) & (cat == i)], [10, 50, 90]).tolist() if ((~mb) & (cat == i)).any() else None)}
                                     for i in range(3)}

# ---------------- G3 Q5 cand x Q3 flip trajectories
A = D["A"]
mm = (bits[:, 4] == 1) & (bits[:, 2] == 1)
groups = {"q4cand_q2flip": mm & A[:, 4] & (CL1[:, 2] == 1), "q4cand_q2stay": mm & A[:, 4] & (CL1[:, 2] == 0),
          "q4non_q2flip": mm & ~A[:, 4] & (CL1[:, 2] == 1), "q4non_q2stay": mm & ~A[:, 4] & (CL1[:, 2] == 0)}
rng = np.random.default_rng(3)
Fm = {0: np.load(f"{FEAT}/feat_train.npy", mmap_mode="r"), 1: np.load(f"{FEAT}/feat_test.npy", mmap_mode="r")}


def fetch(sel):
    out = np.empty((len(sel), NQ, 20, 2), np.float32)
    for i in (0, 1):
        g = np.flatnonzero(spi[sel] == i)
        if len(g) == 0:
            continue
        rr = row[sel[g]]; o = np.argsort(rr)
        tmp = np.asarray(Fm[i][rr[o]])
        out[g[o]] = tmp
    return out


ref = fetch(rng.choice(np.flatnonzero(mm & (CL1[:, 2] == 0) & ~A[:, 4]), 4000, replace=False))
med = np.median(ref, 0); sig = 1.4826 * np.median(np.abs(ref - med), 0)
# Q3 along reference: separation per bin from prepared-0 vs prepared-1 (Q3) medians, Q5 prepared 1
ref0 = fetch(rng.choice(np.flatnonzero((bits[:, 4] == 1) & (bits[:, 2] == 0)), 4000, replace=False))
c0 = np.median(ref0[:, 2, :, 0], 0); c1 = med[2, :, 0]
G3 = {"counts": {k: int(v.sum()) for k, v in groups.items()}}
for k, v in groups.items():
    sel = np.flatnonzero(v)
    sel = rng.choice(sel, min(3000, len(sel)), replace=False)
    X = fetch(sel)
    q5perp = (X[:, 4, :, 1] - med[4, :, 1]) / sig[4, :, 1]
    q3t = (X[:, 2, :, 0] - c0) / (c1 - c0)
    q3perp = (X[:, 2, :, 1] - med[2, :, 1]) / sig[2, :, 1]
    G3[k] = {"q5_perp_z_median_per_bin": np.median(q5perp, 0).tolist(),
             "q5_perp_z_p90_per_bin": np.percentile(q5perp, 90, axis=0).tolist(),
             "q3_along_t_median_per_bin": np.median(q3t, 0).tolist(),
             "q3_perp_z_median_per_bin": np.median(q3perp, 0).tolist()}
    # onset: first bin where Q3 t < 0.5 (decay) and first bin where Q5 perp z > 3
    on3 = np.array([np.argmax(r < 0.5) if (r < 0.5).any() else -1 for r in q3t])
    on5 = np.array([np.argmax(r > 3) if (r > 3).any() else -1 for r in q5perp])
    G3[k]["q3_decay_onset_bin_hist"] = np.bincount(on3 + 1, minlength=21).tolist()
    G3[k]["q5_excursion_onset_bin_hist"] = np.bincount(on5 + 1, minlength=21).tolist()
    both = (on3 >= 0) & (on5 >= 0)
    G3[k]["q5_onset_minus_q3_onset_bins_percentiles_10_50_90"] = (np.percentile(on5[both] - on3[both], [10, 50, 90]).tolist() if both.any() else None)
R["G3_q4_cand_vs_q2_flip"] = G3


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


json.dump(clean(R), open(f"{OUT}/m4conf_followup2.json", "w"), indent=1)
for wn in ("1us", "2us"):
    for k, v in R["G1_quiet_neighbour"][wn].items():
        e = v["survivor_rate_quiet_LEAKAGE_ESTIMATE"]
        print(wn, k, "cand", v["candidate_rate_all"]["k"], "quietfrac %.3f" % v["quiet_fraction"], "cand_quiet", v["candidate_rate_quiet"]["rate"],
              "est", e["k"], e["n"], e["rate"], e["ci95"], "survfrac", v["surviving_fraction_of_candidates"], v["surviving_fraction_ci95_katz"])
print(json.dumps(R["G2_q3_prep1_transition"]), json.dumps(R["G2_q3_prep1_early_t_by_class"]))
print(json.dumps(G3["counts"]))
for k in groups:
    print(k, [round(x, 2) for x in G3[k]["q5_perp_z_median_per_bin"]], [round(x, 2) for x in G3[k]["q3_along_t_median_per_bin"]], G3[k]["q5_onset_minus_q3_onset_bins_percentiles_10_50_90"])
print("done")
