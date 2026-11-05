"""m4conf_analysis.py -- confounder analysis of leakage candidates (agent tag m4conf).

Candidate sets (per code qubit q = paper Q(q+1)):
  A    : |z_perp| > 5 in the paper "1 us" window (feat1us, bins 0-9 = raw samples 0-499, strictly causal),
         z = (perp - median_perp) / (1.4826 MAD_perp), median/MAD per (split, qubit, prepared bit of q).
  A2us : same rule on the full stored 2 us trace (feat bins 0-19), secondary.
  B    : prior labels, three_level_labels.hdf5 y3_{train,test}[:, q] == 2 (rows aligned with raw files).
Screens (applied in this order, every one recorded separately as well):
  S_frame : |zA| > 5 in OUR calibrated 1 us frame (trivially true for A; the key screen for B)
  S_xtalk : |z32| > 5 with centre/scale conditioned on the FULL 32-state prepared configuration
            (removes centroid shifts caused by neighbours' prepared states)
  S_simul : no other qubit j != q with |z32_perp_j| > 4 in the same shot (single-channel event)
  S_raw   : raw-trace glitch metrics normal: |robust z| < 6 (per state, per split) for out-of-band power,
            DC I, DC Q and max sample-to-sample jump (m4conf_rawstats.py)
  S_time  : not inside a burst (Poisson p >= 1e-4 for the candidate count in +-250 rows of the same block)
            AND still |z| > 5 when the centre is the local median of the 500-row bin (drift removed)
Tests: (a) per-neighbour-configuration rates + chi-square, (b) multi-qubit coincidence vs independence,
(c) runs test / dispersion / lag-1 autocorrelation within blocks + centroid drift, (d) P(.|0) vs P(.|1),
(e) A-vs-B overlap and B composition.
Usage: python m4conf_analysis.py /feat /data /work/out
"""
import json, os, sys, time
import numpy as np
import h5py
from scipy import stats

FEAT, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NQ = 5
SPL = ["train", "test"]
NPER = {"train": 15000, "test": 35000}
BINW = 500          # rows per time bin inside a block
HALF = 250          # burst window half width
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def rate(k, n):
    return {"k": int(k), "n": int(n), "rate": (float(k / n) if n else None), "ci95": wilson(k, n)}


def ratio_ci(k1, n1, k0, n0):
    if k1 == 0 or k0 == 0:
        return {"rr": (float((k1 / n1) / (k0 / n0)) if k0 else None), "ci95": [None, None]}
    rr = (k1 / n1) / (k0 / n0)
    se = np.sqrt(1 / k1 - 1 / n1 + 1 / k0 - 1 / n0)
    return {"rr": float(rr), "ci95": [float(rr * np.exp(-1.96 * se)), float(rr * np.exp(1.96 * se))]}


def pois_ci(k):
    lo = stats.chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0
    hi = stats.chi2.ppf(0.975, 2 * k + 2) / 2
    return float(lo), float(hi)


def robust(x):
    m = np.median(x)
    s = 1.4826 * np.median(np.abs(x - m))
    return m, (s if s > 0 else 1e-12)


# ------------------------------------------------------------------ load
log("loading features")
y_l, sp_l, row_l, w1_l, w2_l, wl_l, we_l = [], [], [], [], [], [], []
Fmm = {}
for i, sp in enumerate(SPL):
    y = np.load(f"{FEAT}/y_{sp}.npy")
    idx = np.load(f"{FEAT}/idx_{sp}.npy")
    assert np.array_equal(idx, np.arange(len(y)))
    assert np.array_equal(y, np.repeat(np.arange(32), NPER[sp])), "rows not in 32 sorted blocks"
    F1 = np.load(f"{FEAT}/feat1us_{sp}.npy", mmap_mode="r")
    F = np.load(f"{FEAT}/feat_{sp}.npy", mmap_mode="r")
    Fmm[sp] = F
    F1a = np.asarray(F1, dtype=np.float32)
    w1_l.append(F1a.mean(2, dtype=np.float64))
    we_l.append(F1a[:, :, 0:2].mean(2, dtype=np.float64))      # early 200 ns window (samples 0-99)
    del F1a
    Fa = np.asarray(F, dtype=np.float32)
    w2_l.append(Fa.mean(2, dtype=np.float64))
    wl_l.append(Fa[:, :, 10:20].mean(2, dtype=np.float64))
    del Fa
    y_l.append(y); sp_l.append(np.full(len(y), i, np.int8)); row_l.append(np.arange(len(y)))
y = np.concatenate(y_l); spi = np.concatenate(sp_l); row = np.concatenate(row_l)
N = len(y)
bits = ((y[:, None] >> np.arange(NQ)) & 1).astype(np.int8)
pos = row - y * np.where(spi == 0, NPER["train"], NPER["test"])         # position inside its block
blk = spi.astype(np.int64) * 32 + y                                     # 64 blocks
W = {"1us": np.concatenate(w1_l), "2us": np.concatenate(w2_l), "late": np.concatenate(wl_l), "early": np.concatenate(we_l)}
del w1_l, w2_l, wl_l, we_l
with h5py.File(f"{DATA}/three_level_labels.hdf5", "r") as h:
    y3 = np.concatenate([h["y3_train"][:], h["y3_test"][:]]).astype(np.int8)
    yy = np.concatenate([h["y_train"][:], h["y_test"][:]])
assert np.array_equal(yy, y), "prior label file not aligned"
RS = {}
for i, sp in enumerate(SPL):
    z = np.load(f"{OUT}/m4conf_rawstats_{sp}.npz")
    for k in ["dcI1", "dcQ1", "p1", "oob1", "oob2", "maxabs", "dmax"]:
        RS.setdefault(k, []).append(z[k].astype(np.float64))
RS = {k: np.concatenate(v) for k, v in RS.items()}
log("loaded", N, "shots; prior |2> labels per qubit:", (y3 == 2).sum(0).tolist())
assert np.all(RS["p1"] > 0), "raw stats incomplete"

# ------------------------------------------------------------------ standardisation
R = {"meta": {"N": int(N), "window_primary": "1us = feat1us bins 0-9 = raw samples 0-499 (strictly causal FIR)",
              "z_rule": "z = (perp - median)/(1.4826*MAD) per (split, qubit, prepared bit of that qubit)",
              "qubit_names": {f"q{q}": f"Q{q + 1}" for q in range(NQ)}}}


def zstats(Wn):
    """returns zA (per-bit), z32 (per-state), zA0 (per-bit centre, |0>-noise scale), z32_along, tA (along in sep units)."""
    A, P = Wn[:, :, 0], Wn[:, :, 1]
    zA = np.empty((N, NQ)); z32 = np.empty((N, NQ)); z32a = np.empty((N, NQ)); zA0 = np.empty((N, NQ)); tA = np.empty((N, NQ))
    cen = {}
    for i in range(2):
        ms = spi == i
        for q in range(NQ):
            c = {}
            for b in (0, 1):
                m = ms & (bits[:, q] == b)
                c[b] = robust(P[m, q]) + robust(A[m, q])          # (med_perp, sig_perp, med_along, sig_along)
                zA[m, q] = (P[m, q] - c[b][0]) / c[b][1]
            sig0 = []
            for s in range(32):
                m = ms & (y == s)
                mp, spp = robust(P[m, q]); ma, sa = robust(A[m, q])
                z32[m, q] = (P[m, q] - mp) / spp
                z32a[m, q] = (A[m, q] - ma) / sa
                if not (s >> q) & 1:
                    sig0.append(spp)
            s0 = float(np.median(sig0))
            for b in (0, 1):
                m = ms & (bits[:, q] == b)
                zA0[m, q] = (P[m, q] - c[b][0]) / s0
            sep = c[1][2] - c[0][2]
            tA[ms, q] = (A[ms, q] - c[0][2]) / sep
            cen[f"{SPL[i]}_q{q}"] = {"med_perp_p0": c[0][0], "sig_perp_p0": c[0][1], "med_perp_p1": c[1][0], "sig_perp_p1": c[1][1],
                                     "med_along_p0": c[0][2], "sig_along_p0": c[0][3], "med_along_p1": c[1][2], "sig_along_p1": c[1][3],
                                     "sig_perp_noise_prep0_32state_median": s0, "sep": sep}
    return zA, z32, zA0, z32a, tA, cen


Z = {}
for wn in ["1us", "2us", "late", "early"]:
    Z[wn] = zstats(W[wn])
    R["meta"][f"centroids_{wn}"] = Z[wn][5]
log("z computed")
zA, z32, zA0, z32a, tA, _ = Z["1us"]

# local-centre (drift-removed) perp z for 1us and 2us: centre = median of the 500-row bin inside the block
binid = blk * 100 + pos // BINW
ub, inv = np.unique(binid, return_inverse=True)


def local_z(Wn, zAw):
    P = Wn[:, :, 1]
    zl = np.empty((N, NQ)); binmed = np.empty((len(ub), NQ))
    order = np.argsort(inv, kind="stable"); cuts = np.flatnonzero(np.diff(inv[order])) + 1
    groups = np.split(order, cuts)
    for q in range(NQ):
        # scale: the same per-bit sigma as zA  -> sigma = (P - med_bit)/zA
        for g_i, g in enumerate(groups):
            binmed[g_i, q] = np.median(P[g, q])
        sig = np.empty(N)
        for i in range(2):
            for b in (0, 1):
                m = (spi == i) & (bits[:, q] == b)
                c = robust(P[m, q])
                sig[m] = c[1]
        zl[:, q] = (P[:, q] - binmed[inv, q]) / sig
    return zl, binmed, groups


zloc, binmed1, groups = local_z(W["1us"], zA)
zloc2, binmed2, _ = local_z(W["2us"], Z["2us"][0])
log("local z computed")

# raw-stat robust z per (split, state)
RZ = {}
for k, tf in [("oob1", np.log), ("dcI1", None), ("dcQ1", None), ("dmax", np.log), ("p1", np.log), ("maxabs", np.log)]:
    v = tf(RS[k]) if tf else RS[k]
    z = np.empty(N)
    for b_ in range(64):
        m = blk == b_
        c = robust(v[m]); z[m] = (v[m] - c[0]) / c[1]
    RZ[k] = z
GLITCH_KEYS = ["oob1", "dcI1", "dcQ1", "dmax"]
glitch = np.zeros(N, bool)
for k in GLITCH_KEYS:
    glitch |= np.abs(RZ[k]) >= 6
R["meta"]["raw_glitch_rate_all_shots"] = {k: rate(int((np.abs(RZ[k]) >= 6).sum()), N) for k in RZ}
R["meta"]["raw_glitch_any_used"] = rate(int(glitch.sum()), N)
log("raw z computed; glitch any:", int(glitch.sum()))

# ------------------------------------------------------------------ candidate sets
CS = {"A": np.abs(zA) > 5, "A2us": np.abs(Z["2us"][0]) > 5, "B": (y3 == 2),
      "A_noise0scale": np.abs(zA0) > 5, "A32": np.abs(z32) > 5}
ZSET = {"A": (zA, z32, zloc), "A2us": (Z["2us"][0], Z["2us"][1], zloc2), "B": (zA, z32, zloc),
        "A_noise0scale": (zA, z32, zloc), "A32": (zA, z32, zloc)}

# ------------------------------------------------------------------ (d) prepared-state dependence
R["d_prepdep"] = {}
for name, C in CS.items():
    R["d_prepdep"][name] = {}
    for q in range(NQ):
        e = {}
        for b in (0, 1):
            m = bits[:, q] == b
            e[f"prep{b}"] = rate(C[m, q].sum(), m.sum())
            for i in range(2):
                mm = m & (spi == i)
                e[f"prep{b}_{SPL[i]}"] = rate(C[mm, q].sum(), mm.sum())
            e[f"prep{b}_sign_pos_frac"] = (float((zA[m & C[:, q], q] > 0).mean()) if (m & C[:, q]).any() else None)
        e["overall"] = rate(C[:, q].sum(), N)
        e["rr_prep1_over_prep0"] = ratio_ci(e["prep1"]["k"], e["prep1"]["n"], e["prep0"]["k"], e["prep0"]["n"])
        R["d_prepdep"][name][f"q{q}"] = e
log("(d) done")

# ------------------------------------------------------------------ (a) crosstalk / neighbour configurations
R["a_xtalk"] = {}
PL = {"a_rates": {}}
for name in ["A", "B", "A32", "A2us"]:
    C = CS[name]
    zz = ZSET[name][0]
    R["a_xtalk"][name] = {}
    for q in range(NQ):
        for b in (0, 1):
            ss = [s for s in range(32) if ((s >> q) & 1) == b]
            k = np.array([C[y == s, q].sum() for s in ss]); n = np.array([(y == s).sum() for s in ss])
            # mean perp offset of the configuration relative to the per-bit centre, in per-bit sigma units (zA scale)
            dmean = np.array([np.median(zz[y == s, q]) for s in ss])
            # Gaussian prediction for |z|>5 given that shift (crosstalk-only hypothesis), for set A
            pred = stats.norm.sf(5 - dmean) + stats.norm.cdf(-5 - dmean)
            tab = np.vstack([k, n - k])
            if k.sum() > 0:
                chi2, p, dof, ex = stats.chi2_contingency(tab, correction=False)
                V = float(np.sqrt(chi2 / n.sum()))
                mine = float(ex[0].min())
            else:
                chi2 = p = V = mine = None; dof = 15
            rr = {}
            for j in range(NQ):
                if j == q:
                    continue
                g1 = np.array([(s >> j) & 1 for s in ss], bool)
                rr[f"q{j}(Q{j + 1})=1_vs_0"] = ratio_ci(k[g1].sum(), n[g1].sum(), k[~g1].sum(), n[~g1].sum())
            R["a_xtalk"][name][f"q{q}_prep{b}"] = {
                "states": ss, "k": k.tolist(), "n": n.tolist(), "rate": (k / n).tolist(),
                "median_zperp_of_config": dmean.tolist(), "gauss_pred_rate_from_shift": pred.tolist(),
                "chi2": (float(chi2) if chi2 is not None else None), "dof": int(dof), "p": (float(p) if p is not None else None),
                "cramers_V": V, "min_expected": mine,
                "rate_max_over_min": (float((k / n).max() / (k / n).min()) if (k > 0).all() else None),
                "neighbour_rate_ratio": rr}
            PL["a_rates"][f"{name}_q{q}_p{b}"] = np.vstack([k, n, dmean])
log("(a) done")

# ------------------------------------------------------------------ (b) simultaneity
R["b_simul"] = {}


def coincidence(Fl):
    """Fl (N,5) bool. Expectation under within-state independence."""
    p_s = np.array([Fl[y == s].mean(0) for s in range(32)])            # (32,5)
    n_s = np.array([(y == s).sum() for s in range(32)])
    out = {"lift": [[None] * NQ for _ in range(NQ)], "both": [[0] * NQ for _ in range(NQ)],
           "expected_both": [[0.0] * NQ for _ in range(NQ)], "lift_ci95": [[None] * NQ for _ in range(NQ)]}
    for q in range(NQ):
        for r in range(NQ):
            o = int((Fl[:, q] & Fl[:, r]).sum())
            e = float((n_s * p_s[:, q] * p_s[:, r]).sum()) if q != r else float(Fl[:, q].sum())
            out["both"][q][r] = o; out["expected_both"][q][r] = e
            if q != r and e > 0:
                lo, hi = pois_ci(o)
                out["lift"][q][r] = o / e; out["lift_ci95"][q][r] = [lo / e, hi / e]
    cnt = Fl.sum(1)
    obs = {k: int((cnt == k).sum()) for k in range(6)}
    # Poisson-binomial expectation per state
    exp = np.zeros(6)
    for s in range(32):
        dist = np.array([1.0])
        for q in range(NQ):
            dist = np.convolve(dist, [1 - p_s[s, q], p_s[s, q]])
        exp += n_s[s] * dist
    ge2_o = int((cnt >= 2).sum()); ge2_e = float(exp[2:].sum())
    lo, hi = pois_ci(ge2_o)
    out["count_hist_observed"] = obs; out["count_hist_expected_indep"] = exp.tolist()
    out["ge2_observed"] = ge2_o; out["ge2_expected"] = ge2_e
    out["ge2_ratio"] = (ge2_o / ge2_e if ge2_e > 0 else None); out["ge2_ratio_ci95"] = ([lo / ge2_e, hi / ge2_e] if ge2_e > 0 else None)
    out["ge3_observed"] = int((cnt >= 3).sum()); out["ge3_expected"] = float(exp[3:].sum())
    return out


F4 = np.abs(z32) > 4
p4_s = np.array([F4[y == s].mean(0) for s in range(32)])               # P(|z32|>4 | state) per qubit
for name in ["A", "A32", "B", "A2us"]:
    C = CS[name]
    res = coincidence(C)
    per_q = {}
    for q in range(NQ):
        oth = np.delete(F4, q, 1).any(1)
        chance = 1 - np.prod(1 - np.delete(p4_s, q, 1), axis=1)       # per state
        e = {}
        for b in (0, 1):
            m = C[:, q] & (bits[:, q] == b)
            ctl = (~C[:, q]) & (bits[:, q] == b)
            k = int((m & oth).sum()); n = int(m.sum())
            e[f"prep{b}"] = {"cand_with_other_qubit_z32perp_gt4": rate(k, n),
                             "expected_frac_by_chance": (float(chance[y[m]].mean()) if n else None),
                             "noncand_frac": float((ctl & oth).sum() / max(ctl.sum(), 1))}
            # raw glitch metrics
            g = {}
            for kk, z in RZ.items():
                g[kk] = {"cand_median_rz": (float(np.median(z[m])) if n else None),
                         "cand_frac_absrz_ge6": (float((np.abs(z[m]) >= 6).mean()) if n else None),
                         "cand_frac_absrz_ge4": (float((np.abs(z[m]) >= 4).mean()) if n else None),
                         "noncand_frac_absrz_ge6": float((np.abs(z[ctl]) >= 6).mean()),
                         "noncand_frac_absrz_ge4": float((np.abs(z[ctl]) >= 4).mean())}
            e[f"prep{b}_raw"] = g
            # other-tone energy from features: sum_j z32perp_j^2 over the 4 other qubits (chi2_4 if Gaussian)
            chi = (np.delete(z32, q, 1) ** 2).sum(1)
            e[f"prep{b}_other_tone_chi2"] = {"cand_median": (float(np.median(chi[m])) if n else None),
                                             "noncand_median": float(np.median(chi[ctl])),
                                             "cand_frac_gt_18.47": (float((chi[m] > 18.47).mean()) if n else None),
                                             "noncand_frac_gt_18.47": float((chi[ctl] > 18.47).mean()),
                                             "chi2_4dof_expected_median": 3.357}
            # other qubits' along deviation (|z32_along|) median
            oa = np.abs(np.delete(z32a, q, 1))
            e[f"prep{b}_other_along_absz_median"] = {"cand": (float(np.median(oa[m])) if n else None), "noncand": float(np.median(oa[ctl]))}
        per_q[f"q{q}"] = e
    res["per_qubit"] = per_q
    R["b_simul"][name] = res
log("(b) done")

# ------------------------------------------------------------------ (c) time / drift
R["c_time"] = {}
PL["time_rate"] = {}


def runs_stats(seq_list):
    Rr = Er = Vr = 0.0
    num = den = 0.0; nn = 0
    for c in seq_list:
        n = len(c); n1 = c.sum(); n0 = n - n1
        if n1 == 0 or n0 == 0:
            continue
        Rr += 1 + np.count_nonzero(c[1:] != c[:-1])
        Er += 1 + 2 * n1 * n0 / n
        Vr += 2 * n1 * n0 * (2 * n1 * n0 - n) / (n * n * (n - 1))
        p = n1 / n; d = c - p
        num += (d[1:] * d[:-1]).sum(); den += (d * d).sum(); nn += n
    if Vr == 0:
        return None
    zr = (Rr - Er) / np.sqrt(Vr)
    ac = num / den
    return {"runs_obs": float(Rr), "runs_exp": float(Er), "runs_z": float(zr), "runs_p_two_sided": float(2 * stats.norm.sf(abs(zr))),
            "lag1_autocorr": float(ac), "lag1_autocorr_se_iid": float(1 / np.sqrt(nn)), "lag1_z": float(ac * np.sqrt(nn))}


burst = {}
for name in ["A", "B", "A2us"]:
    C = CS[name]
    R["c_time"][name] = {}
    bmask = np.zeros((N, NQ), bool)
    for q in range(NQ):
        cs_all = np.zeros(N + 1)
        for b in (0, 1):
            blocks = [bb for bb in range(64) if ((bb % 32) >> q) & 1 == b]
            seqs = []; chi = 0.0; dof = 0; counts = []
            minp = 1.0; nburst = 0
            for bb in blocks:
                m = np.flatnonzero(blk == bb)
                c = C[m, q].astype(np.float64)
                seqs.append(c)
                nb = len(c) // BINW
                cnt = c[:nb * BINW].reshape(nb, BINW).sum(1)
                pr = c.mean()
                e = pr * BINW
                if e > 0:
                    chi += ((cnt - e) ** 2 / (e * (1 - pr))).sum(); dof += nb - 1
                counts.append(cnt / BINW)
                # burst: count in +-HALF rows (excluding self) vs Poisson(block rate excluding self)
                cs = np.concatenate([[0], np.cumsum(c)])
                lo = np.clip(np.arange(len(c)) - HALF, 0, len(c)); hi = np.clip(np.arange(len(c)) + HALF + 1, 0, len(c))
                loc = cs[hi] - cs[lo] - c
                mu = (c.sum() - 1) / (len(c) - 1) * (hi - lo - 1) if c.sum() > 1 else np.zeros(len(c))
                pv = np.where(c > 0, stats.poisson.sf(loc - 1, np.maximum(mu, 1e-12)), 1.0)
                isb = (c > 0) & (pv < 1e-4)
                bmask[m, q] = isb; nburst += int(isb.sum())
                if c.sum() > 0:
                    minp = min(minp, float(pv[c > 0].min()))
            rs = runs_stats(seqs)
            R["c_time"][name][f"q{q}_prep{b}"] = {
                "runs": rs, "dispersion_chi2": float(chi), "dispersion_dof": int(dof),
                "dispersion_index": (float(chi / dof) if dof else None),
                "dispersion_p": (float(stats.chi2.sf(chi, dof)) if dof else None),
                "n_candidates_in_bursts_p_lt_1e-4": nburst, "min_burst_p": minp}
            PL["time_rate"][f"{name}_q{q}_p{b}"] = np.array([np.pad(cc, (0, 70 - len(cc)), constant_values=np.nan) for cc in counts])
    burst[name] = bmask
# centroid drift per qubit per block in 500-row bins
R["c_drift"] = {}
PL["drift"] = {}
gb = ub // 100                       # block of each bin
for q in range(NQ):
    for comp, Wn in [("perp", W["1us"][:, q, 1]), ("along", W["1us"][:, q, 0])]:
        med = np.array([np.median(Wn[g]) for g in groups])
        sig_stat = []; spread = []; ac = []
        for bb in range(64):
            mb = gb == bb
            v = med[mb]
            sh = Wn[blk == bb]
            s = robust(sh)[1]
            sig_stat.append(1.2533 * s / np.sqrt(BINW)); spread.append(np.std(v, ddof=1))
            d = v - v.mean(); ac.append((d[1:] * d[:-1]).sum() / (d * d).sum())
        sig_stat = np.array(sig_stat); spread = np.array(spread)
        R["c_drift"][f"q{q}_{comp}"] = {"median_ratio_binmedian_spread_to_statistical": float(np.median(spread / sig_stat)),
                                         "ratio_per_block": (spread / sig_stat).tolist(),
                                         "lag1_autocorr_binmedians_median_over_blocks": float(np.median(ac)),
                                         "lag1_autocorr_binmedians_per_block": [float(a) for a in ac],
                                         "max_abs_binmedian_dev_over_sigma_shot": float(max(np.abs(med[gb == bb] - np.median(med[gb == bb])).max() / robust(Wn[blk == bb])[1] for bb in range(64)))}
        PL["drift"][f"q{q}_{comp}"] = med
PL["drift_bin_block"] = gb
PL["drift_bin_pos"] = ub % 100
# per-shot lag-1 autocorrelation of continuous quantities inside blocks (acquisition-order test)
R["c_order_test"] = {}
for nm, v in [("dcI1", RS["dcI1"]), ("dcQ1", RS["dcQ1"]), ("log_oob1", np.log(RS["oob1"]))] + \
             [(f"q{q}_perp", W["1us"][:, q, 1]) for q in range(NQ)] + [(f"q{q}_along", W["1us"][:, q, 0]) for q in range(NQ)]:
    num = den = 0.0; acs = []
    for bb in range(64):
        x = v[blk == bb]; d = x - x.mean()
        a = (d[1:] * d[:-1]).sum() / (d * d).sum(); acs.append(a)
    R["c_order_test"][nm] = {"lag1_autocorr_median_over_blocks": float(np.median(acs)), "min": float(np.min(acs)), "max": float(np.max(acs)),
                             "se_iid_test_block": float(1 / np.sqrt(35000))}
# binned drift of the raw DC (acquisition-order check)
for nm, v in [("dcI1", RS["dcI1"]), ("log_oob1", np.log(RS["oob1"]))]:
    med = np.array([np.median(v[g]) for g in groups])
    PL["drift"][nm] = med
log("(c) done")

# ------------------------------------------------------------------ (e) overlap A vs B, B composition
R["e_overlap"] = {}
for q in range(NQ):
    e = {}
    for b in (None, 0, 1):
        m = np.ones(N, bool) if b is None else bits[:, q] == b
        a = CS["A"][m, q]; bb_ = CS["B"][m, q]
        inter = int((a & bb_).sum()); uni = int((a | bb_).sum())
        e["all" if b is None else f"prep{b}"] = {"nA": int(a.sum()), "nB": int(bb_.sum()), "nAandB": inter,
                                                  "jaccard": (inter / uni if uni else None),
                                                  "frac_of_A_in_B": (inter / a.sum() if a.sum() else None),
                                                  "frac_of_B_in_A": (inter / bb_.sum() if bb_.sum() else None)}
    # composition of B in our frame
    cen = R["meta"]["centroids_1us"]
    comp = {}
    for b in (0, 1):
        m = CS["B"][:, q] & (bits[:, q] == b)
        az = np.abs(zA[m, q]); t = tA[m, q]
        # along distance (in |0>-noise sigma) to the nearest centroid
        s0 = np.where(spi[m] == 0, cen[f"train_q{q}"]["sig_perp_noise_prep0_32state_median"], cen[f"test_q{q}"]["sig_perp_noise_prep0_32state_median"])
        sep = np.where(spi[m] == 0, cen[f"train_q{q}"]["sep"], cen[f"test_q{q}"]["sep"])
        dnear = np.minimum(np.abs(t), np.abs(t - 1)) * sep / s0
        off = az >= 5
        tail = (az >= 3) & (az < 5)
        core = (az < 3) & (dnear < 3)
        between = (az < 3) & (dnear >= 3) & (t > 0) & (t < 1)
        beyond = (az < 3) & (dnear >= 3) & ~((t > 0) & (t < 1))
        n = int(m.sum())
        comp[f"prep{b}"] = {"n": n, "offaxis_absz_ge5": rate(off.sum(), n), "tail_3_to_5": rate(tail.sum(), n),
                            "absz_lt3_gaussian_core_within_3sigma_of_a_centroid": rate(core.sum(), n),
                            "absz_lt3_on_axis_between_centroids_T1like": rate(between.sum(), n),
                            "absz_lt3_on_axis_beyond_centroids": rate(beyond.sum(), n),
                            "median_absz": (float(np.median(az)) if n else None),
                            "median_t_along_sep_units": (float(np.median(t)) if n else None),
                            "gaussian_expected_count_absz_gt3_in_this_prep_bit": float(2 * stats.norm.sf(3) * (bits[:, q] == b).sum()),
                            "gaussian_expected_count_absz_gt4": float(2 * stats.norm.sf(4) * (bits[:, q] == b).sum())}
    e["B_composition_in_m4conf_frame"] = comp
    R["e_overlap"][f"q{q}"] = e
log("(e) done")

# ------------------------------------------------------------------ survival funnel
R["survival"] = {}
SURV = {}
PL["traj"] = {}
others4 = {q: np.delete(F4, q, 1).any(1) for q in range(NQ)}
F4_2us = np.abs(Z["2us"][1]) > 4
others4_2us = {q: np.delete(F4_2us, q, 1).any(1) for q in range(NQ)}
S_raw = ~glitch
rng = np.random.default_rng(0)
# per-bin reference medians/sigmas for trajectories (subsample)
for name in ["A", "B", "A2us"]:
    C = CS[name]
    zf, zc, zl = ZSET[name]
    R["survival"][name] = {}
    for q in range(NQ):
        chance = 1 - np.prod(1 - np.delete(p4_s, q, 1), axis=1)
        for b in (0, 1):
            mb = bits[:, q] == b
            nb = int(mb.sum())
            m0 = C[:, q] & mb
            s1 = m0 & (np.abs(zA[:, q]) > 5 if name != "A2us" else np.abs(zf[:, q]) > 5)
            s2 = s1 & (np.abs(zc[:, q]) > 5)
            oth = others4_2us[q] if name == "A2us" else others4[q]
            s3 = s2 & ~oth
            s4 = s3 & S_raw
            s5 = s4 & ~burst[name][:, q] & (np.abs(zl[:, q]) > 5)
            SURV.setdefault(name, np.zeros((N, NQ), bool))[:, q] |= s5
            k0 = int(m0.sum())
            each = {"S_frame": int((m0 & (np.abs(zf[:, q]) > 5)).sum()), "S_xtalk": int((m0 & (np.abs(zc[:, q]) > 5)).sum()),
                    "S_simul": int((m0 & ~oth).sum()), "S_raw": int((m0 & S_raw).sum()),
                    "S_time": int((m0 & ~burst[name][:, q] & (np.abs(zl[:, q]) > 5)).sum())}
            ch = float(chance[y[s4]].mean()) if s4.any() else float(chance[y[mb]].mean())
            k5 = int(s5.sum())
            corr = k5 / (1 - ch)
            late_z = Z["late"][0][:, q]
            e = {"n_prepared": nb, "candidates": rate(k0, nb),
                 "after_S_frame": int(s1.sum()), "after_S_xtalk": int(s2.sum()), "after_S_simul": int(s3.sum()),
                 "after_S_raw": int(s4.sum()), "after_S_time": k5,
                 "pass_each_screen_alone": each,
                 "surviving_fraction_of_candidates": rate(k5, k0) if k0 else {"k": 0, "n": 0, "rate": None, "ci95": [None, None]},
                 "survivor_rate_per_prepared_shot": rate(k5, nb),
                 "chance_coincidence_loss_S_simul": ch,
                 "survivor_rate_corrected_for_chance_coincidence": float(corr / nb),
                 "survivor_sign_pos_frac": (float((zA[s5, q] > 0).mean()) if k5 else None),
                 "survivor_median_t_along_sep_units": (float(np.median(tA[s5, q])) if k5 else None),
                 "survivor_median_zperp": (float(np.median(zA[s5, q])) if k5 else None),
                 "survivor_late_window_absz_gt5_frac": (float((np.abs(late_z[s5]) > 5).mean()) if k5 else None),
                 "survivor_late_window_absz_gt3_frac": (float((np.abs(late_z[s5]) > 3).mean()) if k5 else None)}
            R["survival"][name][f"q{q}_prep{b}"] = e
            # trajectories over 20 bins (2 us) for survivors (subsample <= 3000) vs random controls of the same bit
            if name in ("A", "B") and k5 > 0:
                sel = np.flatnonzero(s5); sel = rng.choice(sel, min(3000, len(sel)), replace=False)
                ctl = rng.choice(np.flatnonzero(mb & ~C[:, q]), 3000, replace=False)
                tr = []
                for grp in (sel, ctl):
                    arr = np.empty((len(grp), 20, 2))
                    for i in range(2):
                        gi = grp[spi[grp] == i]
                        arr[spi[grp] == i] = np.asarray(Fmm[SPL[i]][np.sort(row[gi])], dtype=np.float64)[np.argsort(np.argsort(row[gi]))][:, q]
                    tr.append(arr)
                # reference per-bin centre and noise from the controls
                ref_med = np.median(tr[1], 0); ref_sig = 1.4826 * np.median(np.abs(tr[1] - ref_med), 0)
                zs = (tr[0] - ref_med) / ref_sig
                PL["traj"][f"{name}_q{q}_p{b}"] = np.stack([np.percentile(zs[:, :, 1], [25, 50, 75], axis=0),
                                                            np.percentile(zs[:, :, 0], [25, 50, 75], axis=0)])
log("survival done")

# save candidate index list for the raw probe (TEST rows): A candidates and B-only candidates
cand_rows = {}
for q in range(NQ):
    for name in ["A", "B"]:
        for b in (0, 1):
            m = CS[name][:, q] & (bits[:, q] == b) & (spi == 1)
            cand_rows[f"{name}_q{q}_p{b}"] = row[m]
    cand_rows[f"Bonly_q{q}"] = row[CS["B"][:, q] & ~CS["A"][:, q] & (spi == 1)]
np.savez(f"{OUT}/m4conf_cand_rows_test.npz", **cand_rows)
f32 = lambda a: a.astype(np.float32)
np.savez(f"{OUT}/m4conf_pershot.npz", y=y, spi=spi, row=row, pos=pos, blk=blk,
         zA=f32(zA), z32=f32(z32), zA0=f32(zA0), z32a=f32(z32a), tA=f32(tA), zloc=f32(zloc),
         zA_2us=f32(Z["2us"][0]), zA0_late=f32(Z["late"][2]), tA_late=f32(Z["late"][4]),
         zA0_early=f32(Z["early"][2]), tA_early=f32(Z["early"][4]), zA0_2us=f32(Z["2us"][2]), tA_2us=f32(Z["2us"][4]),
         B=CS["B"], A=CS["A"], surv_A=SURV["A"], surv_A2us=SURV["A2us"], surv_B=SURV["B"], burst_A=burst["A"], glitch=glitch,
         **{f"rz_{k}": f32(v) for k, v in RZ.items()})
np.savez(f"{OUT}/m4conf_plotdata.npz", **{f"a_rates__{k}": v for k, v in PL["a_rates"].items()},
         **{f"time_rate__{k}": v for k, v in PL["time_rate"].items()},
         **{f"drift__{k}": v for k, v in PL["drift"].items()}, drift_bin_block=PL["drift_bin_block"], drift_bin_pos=PL["drift_bin_pos"],
         **{f"traj__{k}": v for k, v in PL["traj"].items()})


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


json.dump(clean(R), open(f"{OUT}/m4conf_results.json", "w"), indent=1)
log("saved")
# short console summary
for name in ["A", "B", "A2us"]:
    for q in range(NQ):
        for b in (0, 1):
            e = R["survival"][name][f"q{q}_prep{b}"]
            print(name, f"Q{q + 1}/q{q} prep{b}: cand {e['candidates']['k']}/{e['n_prepared']} -> frame {e['after_S_frame']} xtalk {e['after_S_xtalk']} "
                  f"simul {e['after_S_simul']} raw {e['after_S_raw']} time {e['after_S_time']}  surv.rate {e['survivor_rate_per_prepared_shot']['rate']}", flush=True)
