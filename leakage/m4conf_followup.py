"""m4conf_followup.py -- follow-up confounder tests on the A candidates (agent tag m4conf).

Reads m4conf_pershot.npz (written by m4conf_analysis.py).
F1  neighbour-event coincidence: for candidates of qubit q, the state of every other qubit j in the same shot,
    classified in the 1 us window as  off (|zA0_j|>5), far (t_j >= 1.5 separations, not off),
    flip (t_j on the other side of 0.5 from j's prepared bit, not off/far), stay (else).
    Observed vs expected under within-32-state independence (Mantel-Haenszel style O/E, Poisson CI).
F2  transfer: P(candidate on q | class of j) for each source j, within shots where j is prepared |1>.
F3  shot-to-shot memory: lag-k autocorrelation (k=1..20) of the candidate indicator inside blocks;
    P(cand_i | cand_{i-1}) vs P(cand_i); early-window (0-200 ns) perp of candidates that follow a candidate
    vs candidates that follow a non-candidate (carry-over of a long-lived state would be visible from t=0).
F4  Q4 temporal structure: per-block lag-k autocorrelation of Q4's along/perp (prepared |1> vs |0> blocks).
F5  extended survival: A survivors (all 5 screens) that also have no 'far'/'off' event on another qubit.
Usage: python m4conf_followup.py /work/out
"""
import json, sys
import numpy as np
from scipy import stats

OUT = sys.argv[1]
D = np.load(f"{OUT}/m4conf_pershot.npz")
y = D["y"]; blk = D["blk"]; pos = D["pos"]; spi = D["spi"]
N = len(y); NQ = 5
bits = ((y[:, None] >> np.arange(NQ)) & 1).astype(np.int8)
zA = D["zA"]; zA0 = D["zA0"]; tA = D["tA"]; A = D["A"]; S = D["surv_A"]
R = {}


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0, c - h)), float(min(1, c + h))]


def rate(k, n):
    return {"k": int(k), "n": int(n), "rate": (float(k / n) if n else None), "ci95": wilson(k, n)}


def pois_ci(k):
    lo = stats.chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0
    return float(lo), float(stats.chi2.ppf(0.975, 2 * k + 2) / 2)


# classes of each qubit in each shot
CL = np.zeros((N, NQ), np.int8)            # 0 stay, 1 flip, 2 far, 3 off
for j in range(NQ):
    off = np.abs(zA0[:, j]) > 5
    far = (tA[:, j] >= 1.5) & ~off
    flip = np.where(bits[:, j] == 1, tA[:, j] < 0.5, tA[:, j] >= 0.5) & ~off & ~far
    CL[flip, j] = 1; CL[far, j] = 2; CL[off, j] = 3
CLN = ["stay", "flip", "far", "off"]
R["class_rates"] = {f"q{j}_prep{b}": {CLN[c]: rate(((CL[:, j] == c) & (bits[:, j] == b)).sum(), (bits[:, j] == b).sum()) for c in range(4)}
                    for j in range(NQ) for b in (0, 1)}

# ---------------- F1
R["F1_neighbour_event_OE"] = {}
for q in range(NQ):
    for b in (0, 1):
        cand = A[:, q] & (bits[:, q] == b)
        if cand.sum() == 0:
            continue
        e = {}
        for j in range(NQ):
            if j == q:
                continue
            for c in (1, 2, 3):
                ind = CL[:, j] == c
                O = int((cand & ind).sum()); E = 0.0
                for s in np.unique(y[cand]):
                    ms = y == s
                    nc = (cand & ms).sum(); non = ms & ~A[:, q]
                    E += nc * ind[non].mean()
                lo, hi = pois_ci(O)
                e[f"q{j}_{CLN[c]}"] = {"obs": O, "exp": float(E), "OE": (O / E if E > 0 else None),
                                      "OE_ci95": ([lo / E, hi / E] if E > 0 else None), "n_cand": int(cand.sum())}
        R["F1_neighbour_event_OE"][f"q{q}_prep{b}"] = e

# ---------------- F2 transfer (target q, source j prepared |1>, target prepared b)
R["F2_transfer"] = {}
for q in range(NQ):
    for j in range(NQ):
        if j == q:
            continue
        for b in (0, 1):
            m = (bits[:, j] == 1) & (bits[:, q] == b)
            e = {}
            for c in range(4):
                mc = m & (CL[:, j] == c)
                e[CLN[c]] = {"P_cand_q": rate((A[:, q] & mc).sum(), mc.sum()),
                             "median_zA0_perp_q": (float(np.median(zA0[mc, q])) if mc.sum() else None),
                             "mean_zA0_perp_q": (float(np.mean(zA0[mc, q])) if mc.sum() else None)}
            R["F2_transfer"][f"q{q}_prep{b}<-q{j}_prep1"] = e

# ---------------- F3 shot-to-shot memory
R["F3_memory"] = {}
for q in range(NQ):
    for b in (0, 1):
        ac = np.zeros(21); den = 0.0; prev_c = 0; prev_n = 0; tot_c = 0; tot_n = 0
        e_after = []; e_notafter = []
        for bb in range(64):
            if ((bb % 32) >> q) & 1 != b:
                continue
            m = np.flatnonzero(blk == bb)
            m = m[np.argsort(pos[m])]
            c = A[m, q].astype(float); p = c.mean(); d = c - p
            if c.sum() == 0:
                continue
            den += (d * d).sum()
            for k in range(1, 21):
                ac[k] += (d[k:] * d[:-k]).sum()
            prev_n += int(c[:-1].sum()); prev_c += int((c[1:] * c[:-1]).sum())
            tot_c += int(c[1:].sum()); tot_n += len(c) - 1
            idx = np.flatnonzero(c[1:] > 0) + 1
            after = c[idx - 1] > 0
            ze = zA0[m, q]
            # early-window perp z
            ez = D["zA0_early"][m, q]
            e_after += ez[idx[after]].tolist(); e_notafter += ez[idx[~after]].tolist()
        if den == 0:
            continue
        acf = (ac[1:] / den).tolist()
        p_all = tot_c / tot_n
        cond = rate(prev_c, prev_n)
        R["F3_memory"][f"q{q}_prep{b}"] = {
            "acf_lag1_20": acf, "se_iid": float(1 / np.sqrt(16 * (35000 + 15000))),
            "P_cand": float(p_all), "P_cand_given_prev_cand": cond,
            "enhancement": (cond["rate"] / p_all if cond["rate"] is not None else None),
            "early200ns_zperp_median_cand_after_cand": (float(np.median(e_after)) if e_after else None),
            "early200ns_zperp_median_cand_after_noncand": (float(np.median(e_notafter)) if e_notafter else None),
            "early200ns_frac_gt3_after_cand": (float((np.abs(e_after) > 3).mean()) if e_after else None),
            "early200ns_frac_gt3_after_noncand": (float((np.abs(e_notafter) > 3).mean()) if e_notafter else None),
            "n_after_cand": len(e_after), "n_after_noncand": len(e_notafter)}
        # previous-shot location: where was the previous shot (same qubit) when the current one is a candidate
# ---------------- F4 Q4 temporal structure + general per-qubit along/perp ACF (prepared 0 vs 1 blocks)
R["F4_acf"] = {}
Wz = {"along": D["z32a"], "perp": D["z32"]}
for q in range(NQ):
    for comp, Z in Wz.items():
        for b in (0, 1):
            acs = []
            for bb in range(64):
                if ((bb % 32) >> q) & 1 != b:
                    continue
                m = np.flatnonzero(blk == bb); m = m[np.argsort(pos[m])]
                x = Z[m, q].astype(float); x = np.clip(x, -10, 10); d = x - x.mean(); dd = (d * d).sum()
                acs.append([(d[k:] * d[:-k]).sum() / dd for k in (1, 2, 5, 10, 50, 200)])
            acs = np.array(acs)
            R["F4_acf"][f"q{q}_{comp}_prep{b}"] = {"lags": [1, 2, 5, 10, 50, 200], "median_over_blocks": np.median(acs, 0).tolist(),
                                                   "max_over_blocks": acs.max(0).tolist(), "min_over_blocks": acs.min(0).tolist()}
# Q4 prepared-|1> 'blob' indicator (along 0.2..0.7 sep, any perp) time series per 500-row bin
blob = (tA[:, 3] > 0.2) & (tA[:, 3] < 0.7)
ts = {}
for bb in range(64):
    if ((bb % 32) >> 3) & 1 != 1:
        continue
    m = np.flatnonzero(blk == bb); m = m[np.argsort(pos[m])]
    nb = len(m) // 500
    ts[bb] = blob[m][: nb * 500].reshape(nb, 500).mean(1).tolist()
R["F4_q3_prep1_blob_frac_per500"] = {str(k): v for k, v in ts.items()}
allv = np.concatenate([np.array(v) for v in ts.values()])
pm = blob[bits[:, 3] == 1].mean()
R["F4_q3_blob_summary"] = {"blob_frac_prep1": float(pm), "bin_frac_std": float(allv.std()),
                           "binomial_std_expected": float(np.sqrt(pm * (1 - pm) / 500)),
                           "blob_frac_prep0": float(blob[bits[:, 3] == 0].mean())}

# ---------------- F5 extended survival
R["F5_survival_with_nbr_event_screen"] = {}
for q in range(NQ):
    nbr = np.isin(np.delete(CL, q, 1), [2, 3]).any(1)
    # chance probability per state among non-candidates
    for b in (0, 1):
        mb = bits[:, q] == b
        s5 = S[:, q] & mb
        s6 = s5 & ~nbr
        ch = 0.0
        if s5.sum():
            for s in np.unique(y[s5]):
                ms = y == s
                ch += (s5 & ms).sum() * nbr[ms & ~A[:, q]].mean()
            ch /= s5.sum()
        k6 = int(s6.sum()); nb = int(mb.sum())
        R["F5_survival_with_nbr_event_screen"][f"q{q}_prep{b}"] = {
            "candidates": int((A[:, q] & mb).sum()), "after_5_screens": int(s5.sum()), "after_nbr_event_screen": k6,
            "chance_loss": float(ch), "surviving_fraction": rate(k6, int((A[:, q] & mb).sum())),
            "survivor_rate": rate(k6, nb), "survivor_rate_chance_corrected": (float(k6 / (1 - ch) / nb) if nb else None)}


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


json.dump(clean(R), open(f"{OUT}/m4conf_followup.json", "w"), indent=1)
for k, v in R["F5_survival_with_nbr_event_screen"].items():
    print(k, v["candidates"], v["after_5_screens"], v["after_nbr_event_screen"], round(v["chance_loss"], 4), v["survivor_rate"])
print("done")
