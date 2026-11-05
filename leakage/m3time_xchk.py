#!/usr/bin/env python3
"""m3time_xchk.py -- is the "L" of Q1/Q5 transferred from Q3? (agent tag m3time)

Uses the filtered canonical Viterbi paths of m3time_reanalyze.py (TEST). Writes m3time_xchk.json with
  1. lag-resolved locking: for an event E_q of qubit q at step k (k-1 -> k), the probability that qubit r
     makes transition T_r at step k+lag, divided by the probability of T_r at the same absolute step in
     the same shot class without E_q (lift by lag, lag = -6..6);
  2. Q3's |1>->|0> and |1>->L per-step rates (Q3 prepared 1, started in |1>) split by the prepared bits of
     Q1 and Q5: an excitation exchange |1_Q3 1_Qx> <-> |0_Q3 2_Qx> makes Q3 "decay" faster when Qx = |1>;
  3. start-L / ever-L(1 us) of every qubit split by the prepared bits of Q3 and Q4 (Clopper-Pearson CIs).
Run (graham):
  docker run --rm --user $(id -u):$(id -g) -v ~/oraqle-ics-sims/leakage/features:/feat:ro \
    -v ~/oraqle-ics-sims/leakage/m3time:/m3 oraqle-torch:latest python /m3/scripts/m3time_xchk.py --run /m3/out_re/feat_xt1
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import argparse, json, os
import numpy as np
from scipy.stats import beta

NAMES = ["Q1", "Q2", "Q3", "Q4", "Q5"]
L = 2
CODE = {"0": 0, "1": 1, "L": 2}


def cp(k, n):
    if n == 0:
        return [None, None]
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def fr(k, n):
    return {"k": int(k), "n": int(n), "frac": float(k / n) if n else None, "ci95": cp(int(k), int(n))}


def trans(p, s0, s1):
    """(N,K) bool: transition s0->s1 between bin k-1 and k (column k; column 0 False)."""
    t = np.zeros(p.shape, bool)
    t[:, 1:] = (p[:, :-1] == s0) & (p[:, 1:] == s1)
    return t


def lag_lift(Eq, Tr, cls, lags=range(-6, 7), kmin=3):
    """Eq, Tr (N,K) bool; cls (N,) bool shot class. Returns lift[lag] and counts."""
    N, K = Eq.shape
    E = Eq & cls[:, None]
    E[:, :kmin] = False
    noE_shot = cls & ~E.any(1)
    base_k = Tr[noE_shot].mean(0)                  # P(T_r at step j) in class shots with no E_q
    out = {}
    for lag in lags:
        num = 0
        den = 0.0
        for k in range(kmin, K):
            j = k + lag
            if j < 1 or j >= K:
                continue
            ek = E[:, k]
            n = int(ek.sum())
            if n == 0:
                continue
            num += int(Tr[ek, j].sum())
            den += n * base_k[j]
        out[str(lag)] = {"observed": num, "expected": float(den), "lift": float(num / den) if den > 0 else None}
    out["n_events"] = int(E.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    y = np.load(str(LP.FEAT / "y_test.npy")).astype(np.int64)
    bits = (y[:, None] >> np.arange(5)) & 1
    P = [np.load(os.path.join(a.run, f"m3time_re_q{q}_arrays.npz"))["path"].astype(np.int8) for q in range(5)]
    ST = [np.load(os.path.join(a.run, f"m3time_re_q{q}_arrays.npz"))["st"] for q in range(5)]
    res = {"lags": {}}
    pairs = [(4, "1", "L", 2, "1", "0"), (4, "L", "1", 2, "0", "1"), (0, "1", "L", 2, "1", "0"), (0, "L", "1", 2, "0", "1"),
             (4, "1", "L", 2, "1", "L"), (0, "1", "L", 2, "1", "L"),
             (2, "1", "L", 3, "1", "L"), (2, "1", "L", 1, "0", "1"), (2, "1", "L", 4, "1", "L"), (2, "1", "L", 0, "1", "L"),
             (2, "L", "1", 4, "L", "1"), (3, "1", "L", 2, "1", "0"), (1, "1", "L", 2, "1", "0"), (1, "1", "L", 3, "1", "L"),
             (4, "1", "L", 0, "1", "L"), (4, "1", "L", 3, "1", "L")]
    for q, a0, a1, r, b0, b1 in pairs:
        Eq = trans(P[q], CODE[a0], CODE[a1])
        Tr = trans(P[r], CODE[b0], CODE[b1])
        for cname, cls in (("prepq1_prepr1", (bits[:, q] == 1) & (bits[:, r] == 1)),
                           ("prepq1_prepr0", (bits[:, q] == 1) & (bits[:, r] == 0))):
            res["lags"][f"{NAMES[q]} {a0}->{a1} vs {NAMES[r]} {b0}->{b1} | {cname}"] = lag_lift(Eq, Tr, cls)
    # 2. Q3 decay / L entry split by prepared bits of Q1 and Q5
    p3 = P[2]
    st3 = ST[2]
    sel3 = (bits[:, 2] == 1) & (st3 == 1)
    d = {}
    for b0 in (0, 1):
        for b4 in (0, 1):
            m = sel3 & (bits[:, 0] == b0) & (bits[:, 4] == b4)
            Pm = p3[m][:, :20]
            notS = Pm[:, 3:] != 1
            has = notS.any(1)
            e = np.where(has, notS.argmax(1) + 3, 20)
            dest = np.full(len(Pm), -1)
            dest[has] = Pm[np.where(has)[0], e[has]]
            expo = int(np.where(has, e - 2, 17).sum())
            n0, nL = int((dest == 0).sum()), int((dest == L).sum())
            d[f"Q1prep{b0}_Q5prep{b4}"] = {"n_shots": int(m.sum()), "exposure_bin_steps": expo,
                                          "p_1to0_per_100ns": fr(n0, expo), "p_1toL_per_100ns": fr(nL, expo),
                                          "tau_1to0_ns": float(-100 / np.log1p(-n0 / expo)) if n0 else None}
    res["Q3_decay_by_Q1_Q5_prep"] = d
    # same for Q3 prepared 1, split by Q2 and Q4 bits (controls)
    d2 = {}
    for b1 in (0, 1):
        for b3 in (0, 1):
            m = sel3 & (bits[:, 1] == b1) & (bits[:, 3] == b3)
            Pm = p3[m][:, :20]
            notS = Pm[:, 3:] != 1
            has = notS.any(1)
            e = np.where(has, notS.argmax(1) + 3, 20)
            dest = np.full(len(Pm), -1)
            dest[has] = Pm[np.where(has)[0], e[has]]
            expo = int(np.where(has, e - 2, 17).sum())
            n0, nL = int((dest == 0).sum()), int((dest == L).sum())
            d2[f"Q2prep{b1}_Q4prep{b3}"] = {"n_shots": int(m.sum()), "p_1to0_per_100ns": fr(n0, expo),
                                           "p_1toL_per_100ns": fr(nL, expo)}
    res["Q3_decay_by_Q2_Q4_prep"] = d2
    # 3. start-L / ever-L (1 us) by prepared bits of Q3, Q4
    s = {}
    for q in range(5):
        o = {}
        for bq in (0, 1):
            for b2 in (0, 1):
                for b3 in (0, 1):
                    if q in (2, 3):
                        continue
                    m = (bits[:, q] == bq) & (bits[:, 2] == b2) & (bits[:, 3] == b3)
                    o[f"prep{bq}_Q3prep{b2}_Q4prep{b3}"] = {
                        "start_L": fr(((ST[q] == L) & m).sum(), m.sum()),
                        "ever_L_1us": fr(((P[q][:, :10] == L).any(1) & m).sum(), m.sum())}
            for b2 in (0, 1):
                m = (bits[:, q] == bq) & (bits[:, 2] == b2) if q != 2 else (bits[:, q] == bq)
                o[f"prep{bq}_Q3prep{b2}"] = {"start_L": fr(((ST[q] == L) & m).sum(), m.sum()),
                                             "ever_L_1us": fr(((P[q][:, :10] == L).any(1) & m).sum(), m.sum())}
        s[NAMES[q]] = o
    res["startL_by_Q3_Q4_prep"] = s
    with open(os.path.join(a.run, "m3time_xchk.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("[m3time-xchk] done", flush=True)


if __name__ == "__main__":
    main()
