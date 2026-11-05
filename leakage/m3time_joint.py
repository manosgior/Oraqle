#!/usr/bin/env python3
"""m3time_joint.py -- cross-qubit test of the HMM "L" episodes (agent tag m3time).

A genuine |2> of qubit q is a property of qubit q alone. If instead q's L episodes are switched on and
off by what ANOTHER qubit r is doing (amplifier compression, resonator-resonator / state-dependent
crosstalk), q's L occupancy and q's L transitions will be locked to r's state and r's transitions.

Inputs: filtered canonical Viterbi paths m3time_re_q{q}_arrays.npz (TEST, 1.12M shots x 20 bins) from
m3time_reanalyze.py, and y_test. For every ordered pair (q, r):
  * occupancy lift  P(q in L at bin k | r in state s at bin k) / P(q in L at bin k), bins 3..19,
    separately for prepared bit of r = 0/1 (so a prepared-state effect is not confused with dynamics);
  * transition locking: of q's L->1 exits (and 1->L entries) at bin k, the fraction for which r changes
    state at bin k-1, k or k+1, and which change, compared with the same fraction at bins where q is in L
    and does NOT exit (baseline);
  * per prepared bit of r, the fraction of q's shots that start in L (bins 1-2).
Run (graham):
  docker run --rm --user $(id -u):$(id -g) -v ~/oraqle-ics-sims/leakage/features:/feat:ro \
    -v ~/oraqle-ics-sims/leakage/m3time:/m3 oraqle-torch:latest \
    python /m3/scripts/m3time_joint.py --run /m3/out_re/feat_xt1
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import argparse, json, os
import numpy as np

NAMES = ["Q1", "Q2", "Q3", "Q4", "Q5"]
L = 2


def cp(k, n):
    from scipy.stats import beta
    if n == 0:
        return [None, None]
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    y = np.load(str(LP.FEAT / "y_test.npy")).astype(np.int64)
    bits = (y[:, None] >> np.arange(5)) & 1
    P = [np.load(os.path.join(a.run, f"m3time_re_q{q}_arrays.npz"))["path"].astype(np.int8) for q in range(5)]
    st = [np.load(os.path.join(a.run, f"m3time_re_q{q}_arrays.npz"))["st"] for q in range(5)]
    K = P[0].shape[1]
    ks = np.arange(3, K)
    out = {}
    for q in range(5):
        pq = P[q]
        inL = pq[:, ks] == L                                   # (N,17)
        # q's L exits to 1 and entries from 1, at bin k (k in 3..K-1): pq[k-1] -> pq[k]
        prev, cur = pq[:, ks - 1], pq[:, ks]
        exitL1 = (prev == L) & (cur == 1)
        enter1L = (prev == 1) & (cur == L)
        stayL = (prev == L) & (cur == L)
        res = {}
        for r in range(5):
            if r == q:
                continue
            pr = P[r]
            o = {}
            # occupancy lift by r's state, per prepared bit of r and of q
            for bq in (0, 1):
                for br in (0, 1):
                    m = (bits[:, q] == bq) & (bits[:, r] == br)
                    base = inL[m].mean()
                    row = {"P_qL": float(base), "n_shots": int(m.sum())}
                    for s in range(3):
                        sel = pr[m][:, ks] == s
                        n = int(sel.sum())
                        row[f"P_qL_given_r{s}"] = float(inL[m][sel].mean()) if n else None
                        row[f"n_bins_r{s}"] = n
                        row[f"lift_r{s}"] = float(inL[m][sel].mean() / base) if (n and base > 0) else None
                    o[f"prepq{bq}_prepr{br}"] = row
            # transition locking
            chg = np.zeros_like(pr, dtype=bool)
            chg[:, 1:] = pr[:, 1:] != pr[:, :-1]                # r changes state between bin j-1 and j
            # window |j - k| <= 1 around q's transition at bin k
            win = np.zeros((len(pr), len(ks)), bool)
            for dk in (-1, 0, 1):
                j = np.clip(ks + dk, 1, K - 1)
                win |= chg[:, j]
            # what r does at the same step (k-1 -> k)
            rprev, rcur = pr[:, ks - 1], pr[:, ks]
            for nm, ev in (("qL_to_1", exitL1), ("q1_to_L", enter1L), ("q_stays_L", stayL)):
                n = int(ev.sum())
                k_ = int((win & ev).sum())
                d = {"n_events": n, "r_changes_within_1bin": k_, "frac": (k_ / n if n else None), "ci95": cp(k_, n)}
                if n:
                    trans = {}
                    for s0 in range(3):
                        for s1 in range(3):
                            if s0 != s1:
                                c = int(((rprev == s0) & (rcur == s1) & ev).sum())
                                if c:
                                    trans[f"{'01L'[s0]}->{'01L'[s1]}"] = c
                    d["r_same_step_transitions"] = trans
                    d["r_state_before"] = {"01L"[s]: int(((rprev == s) & ev).sum()) for s in range(3)}
                o[nm] = d
            # start-L of q per prepared bit of r and per r's start state
            o["q_startL_by_prepr"] = {}
            for br in (0, 1):
                for bq in (0, 1):
                    m = (bits[:, r] == br) & (bits[:, q] == bq)
                    k_ = int((st[q][m] == L).sum())
                    o["q_startL_by_prepr"][f"prepq{bq}_prepr{br}"] = {"k": k_, "n": int(m.sum()),
                                                                     "frac": k_ / max(m.sum(), 1), "ci95": cp(k_, int(m.sum()))}
            o["q_startL_by_r_start"] = {}
            for s in (-1, 0, 1, 2):
                m = (st[r] == s) & (bits[:, q] == 1)
                k_ = int((st[q][m] == L).sum())
                o["q_startL_by_r_start"][f"prepq1_rstart{s}"] = {"k": k_, "n": int(m.sum()),
                                                                "frac": k_ / max(m.sum(), 1), "ci95": cp(k_, int(m.sum()))}
            res[NAMES[r]] = o
        out[NAMES[q]] = res
        print(f"[m3time-joint] {NAMES[q]} done", flush=True)
    with open(os.path.join(a.run, "m3time_joint.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
