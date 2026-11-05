"""m4conf_final.py -- final confounder funnel and leakage-like estimate (agent tag m4conf).

Candidates  A: |zA| > 5 (perp, z per split x qubit x prepared bit, median / 1.4826 MAD), window 1 us
            (feat1us bins 0-9 = raw samples 0-499) and, secondary, 2 us (feat bins 0-19).
Screens (cumulative, in this order):
  S_xtalk : still > 5 when the centre is the median of the shot's FULL 32-state configuration (neighbours'
            prepared bits), same per-bit noise scale. Tests static crosstalk / dispersive cross-shifts only.
            (The earlier variant that also rescales by the per-state MAD is reported as S_xtalk_rescaled:
            it is biased because the per-state MAD is inflated by the very events under test.)
  S_simul : no other qubit with |z32_perp| > 4 in the same shot (z32 = per-32-state centre and MAD).
  S_raw   : no raw-trace glitch flag (|robust z| >= 6 per block for out-of-band power, DC I, DC Q, max jump).
  S_time  : not in a burst (Poisson p < 1e-4 for the count in +-250 rows of the same block) and still > 5
            relative to the median of its own 500-row bin (drift removed).
  -> BASE survivors (upper estimate of single-channel, stationary, glitch-free off-axis events)
  S_nbr   : no NEIGHBOUR EVENT in the same shot: any other qubit j with |zA0_j| > 5 (off-axis, |0>-noise scale);
            or, for j in the high-SNR qubits (Q1, Q3, Q4, Q5): along beyond 1.5 separations, or a state flip
            that is already present in the first 200 ns (prepared 1 read < 0.5 sep in both 0-200 ns and 1 us,
            or prepared 0 read >= 0.5 in both). Q2 flips are not used (SNR 1.2: they are mostly noise).
  -> SINGLE-QUBIT estimate = BASE survivors among shots without a neighbour event, divided by the number of
     prepared-b shots without a neighbour event (conditional rate; lower estimate, because a real event of q
     that also disturbs a neighbour is removed).
Usage: python m4conf_final.py /feat /work/out
"""
import json, sys
import numpy as np
from scipy import stats

FEAT, OUT = sys.argv[1], sys.argv[2]
NQ = 5; SPL = ["train", "test"]; NPER = {"train": 15000, "test": 35000}
D = np.load(f"{OUT}/m4conf_pershot.npz")
y = D["y"]; spi = D["spi"]; pos = D["pos"]; blk = D["blk"]
N = len(y)
bits = ((y[:, None] >> np.arange(NQ)) & 1).astype(np.int8)
P = {}
for wn in ("1us", "2us"):
    arrs = []
    for sp in SPL:
        F = np.load(f"{FEAT}/feat1us_{sp}.npy" if wn == "1us" else f"{FEAT}/feat_{sp}.npy", mmap_mode="r")
        arrs.append(np.asarray(F[..., 1], dtype=np.float32).mean(2, dtype=np.float64))
    P[wn] = np.concatenate(arrs)


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0, c - h)), float(min(1, c + h))]


def rate(k, n):
    return {"k": int(k), "n": int(n), "rate": (float(k / n) if n else None), "ci95": wilson(k, n)}


def robust(x):
    m = np.median(x); return m, 1.4826 * np.median(np.abs(x - m))


def zsets(Pw):
    zA = np.empty((N, NQ)); zS = np.empty((N, NQ)); zR = np.empty((N, NQ)); zL = np.empty((N, NQ))
    for i in range(2):
        ms = spi == i
        for q in range(NQ):
            for b in (0, 1):
                m = ms & (bits[:, q] == b)
                c, s = robust(Pw[m, q]); zA[m, q] = (Pw[m, q] - c) / s
                for st in np.unique(y[m]):
                    mm = m & (y == st)
                    cs, ss = robust(Pw[mm, q])
                    zS[mm, q] = (Pw[mm, q] - cs) / s          # per-state centre, per-bit scale
                    zR[mm, q] = (Pw[mm, q] - cs) / ss         # per-state centre and scale (= z32)
                    # local 500-row bins inside this block
                    rows = np.flatnonzero(mm); pb = pos[rows] // 500
                    for k in np.unique(pb):
                        r = rows[pb == k]
                        zL[r, q] = (Pw[r, q] - np.median(Pw[r, q])) / s
    return zA, zS, zR, zL


def bursts(C):
    B = np.zeros((N, NQ), bool)
    for q in range(NQ):
        for bb in range(64):
            m = np.flatnonzero(blk == bb); m = m[np.argsort(pos[m])]
            c = C[m, q].astype(float)
            if c.sum() < 2:
                continue
            cs = np.concatenate([[0], np.cumsum(c)]); n = len(c); ar = np.arange(n)
            lo = np.clip(ar - 250, 0, n); hi = np.clip(ar + 251, 0, n)
            loc = cs[hi] - cs[lo] - c
            mu = (c.sum() - 1) / (n - 1) * (hi - lo - 1)
            pv = np.where(c > 0, stats.poisson.sf(loc - 1, mu), 1.0)
            B[m, q] = (c > 0) & (pv < 1e-4)
    return B


# neighbour events (1 us classes; for the 2 us funnel the same 1 us + 2 us definitions are applied per window)
def nbr_events(zA0, tA, tE):
    ev = np.zeros((N, NQ), bool)
    for j in range(NQ):
        e = np.abs(zA0[:, j]) > 5
        if j != 1:
            e |= tA[:, j] >= 1.5
            early_flip = np.where(bits[:, j] == 1, (tA[:, j] < 0.5) & (tE[:, j] < 0.5), (tA[:, j] >= 0.5) & (tE[:, j] >= 0.5))
            e |= early_flip
        ev[:, j] = e
    return ev


EV = {"1us": nbr_events(D["zA0"], D["tA"], D["tA_early"]), "2us": nbr_events(D["zA0_2us"], D["tA_2us"], D["tA_early"])}
glitch = D["glitch"]
R = {"definition": __doc__, "funnel": {}}
for wn in ("1us", "2us"):
    zA, zS, zR, zL = zsets(P[wn])
    C = np.abs(zA) > 5
    Bst = bursts(C)
    F4 = np.abs(zR) > 4
    late = D["zA0_late"]
    out = {}
    for q in range(NQ):
        oth = np.delete(F4, q, 1).any(1)
        nev = np.delete(EV[wn], q, 1).any(1)
        for b in (0, 1):
            mb = bits[:, q] == b; n = int(mb.sum())
            c0 = C[:, q] & mb
            s1 = c0 & (np.abs(zS[:, q]) > 5)
            s2 = s1 & ~oth
            s3 = s2 & ~glitch
            s4 = s3 & ~Bst[:, q] & (np.abs(zL[:, q]) > 5)
            quiet = mb & ~nev
            s5 = s4 & ~nev
            kc = int(c0.sum()); k4 = int(s4.sum()); k5 = int(s5.sum()); nq_ = int(quiet.sum())
            r_single = k5 / nq_ if nq_ else None
            e = {"n_prepared": n, "candidates": rate(kc, n),
                 "after_S_xtalk": int(s1.sum()), "after_S_simul": int(s2.sum()), "after_S_raw": int(s3.sum()), "after_S_time": k4,
                 "pass_alone": {"S_xtalk": int((c0 & (np.abs(zS[:, q]) > 5)).sum()), "S_xtalk_rescaled": int((c0 & (np.abs(zR[:, q]) > 5)).sum()),
                                "S_simul": int((c0 & ~oth).sum()), "S_raw": int((c0 & ~glitch).sum()),
                                "S_time": int((c0 & ~Bst[:, q] & (np.abs(zL[:, q]) > 5)).sum()), "S_nbr": int((c0 & ~nev).sum())},
                 "BASE_survivor_rate": rate(k4, n), "BASE_surviving_fraction": rate(k4, kc) if kc else None,
                 "n_no_neighbour_event": nq_, "candidates_no_neighbour_event": rate(int((c0 & ~nev).sum()), nq_),
                 "SINGLE_QUBIT_rate": rate(k5, nq_),
                 "SINGLE_QUBIT_over_candidate_rate": (r_single / (kc / n) if kc and r_single is not None else None),
                 "neighbour_event_rate_in_noncandidates": float(nev[mb & ~C[:, q]].mean()),
                 "neighbour_event_rate_in_BASE_survivors": (float(nev[s4].mean()) if k4 else None),
                 "survivor_sign_positive": (float((zA[s4, q] > 0).mean()) if k4 else None),
                 "BASE_survivors_still_off_axis_in_late_window_1to2us_absz0_gt5": (float((np.abs(late[s4, q]) > 5).mean()) if k4 else None)}
            # CI for the ratio SINGLE/candidate rate (Katz, treating the two rates as independent)
            if kc and k5:
                rr = (k5 / nq_) / (kc / n); se = np.sqrt(1 / k5 - 1 / nq_ + 1 / kc - 1 / n)
                e["SINGLE_QUBIT_over_candidate_rate_ci95"] = [float(rr * np.exp(-1.96 * se)), float(min(1.5, rr * np.exp(1.96 * se)))]
            out[f"q{q}_prep{b}"] = e
            print(wn, f"Q{q + 1}/q{q} p{b}", kc, int(s1.sum()), int(s2.sum()), int(s3.sum()), k4, "|", k5, nq_, e["SINGLE_QUBIT_rate"]["rate"],
                  "evrate", round(e["neighbour_event_rate_in_noncandidates"], 3), e["neighbour_event_rate_in_BASE_survivors"], flush=True)
    R["funnel"][wn] = out


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


json.dump(clean(R), open(f"{OUT}/m4conf_final.json", "w"), indent=1)
print("done")
