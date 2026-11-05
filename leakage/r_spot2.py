"""r_spot2.py -- where does the third ('L') state go, and where did it come from? (tag r)

Model-light test of the |2> signature on TEST data (560000 shots per prepared bit):
  * references per segment = m3time canonical HMM per-bin state means (|0>, |1>, L; TRAIN fit), averaged
    over the segment's bins;  sigma_seg = 1.4826*MAD(perp) of prepared-|0> shots in that segment;
  * each shot's segment mean is labelled by the nearest reference if it is within 2 sigma_seg of it and
    at least 2 sigma_seg closer to it than to the second-nearest reference (otherwise 'unsure');
  * segments S1 = bins 1-4 (100-500 ns), S2 = bins 5-9 (0.5-1 us), S3 = bins 12-19 (1.2-2 us).
  Cross-tabulations S2 -> S3 (destination) and S1 -> S2 (origin), separately for prepared |0> and |1>.
  A third qubit state above |1> decays to |1> (then to |0>), also when it was entered from |0>;
  a resonator/readout artefact would return to the state it came from.
Also the HMM-path version: first L episode per shot, its origin state and exit state, by prepared bit.

Run on graham:
  docker run -d --name leak-r-spot2 --user $(id -u):$(id -g) \
    -v ~/oraqle-ics-sims/leakage:/lk:ro -v ~/oraqle-ics-sims/leakage/r:/out oraqle-torch:latest \
    python /out/r_spot2.py
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import math

import numpy as np

D = str(LP.FEAT)
F = np.load(D + "/feat_test.npy")
y = np.load(D + "/y_test.npy").astype(np.int64)
N = len(y)
bits = ((y[:, None] >> np.arange(5)) & 1).astype(np.int8)
PAPER = {q: f"Q{q+1}" for q in range(5)}
SEGS = {"S1": (1, 5), "S2": (5, 10), "S3": (12, 20)}
NAMES = ["0", "1", "L", "unsure"]


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [p, max(0.0, c - h), min(1.0, c + h)]


res = {"definitions": __doc__, "per_qubit": {}}
for q in (2, 4, 0, 1):
    mu = np.array(json.load(open(f"{LP.stage('m3time')}/out_re/feat_xt1/m3time_re_q{q}.json"))["mu_canonical"])  # (3,20,2)
    X = F[:, q].astype(np.float64)
    lab = {}
    info = {}
    for k, (lo, hi) in SEGS.items():
        S = X[:, lo:hi].mean(1)
        ref = mu[:, lo:hi].mean(1)  # (3,2)
        p0 = S[bits[:, q] == 0, 1]
        sg = 1.4826 * np.median(np.abs(p0 - np.median(p0)))
        d = np.sqrt(((S[:, None, :] - ref[None]) ** 2).sum(-1)) / sg   # (N,3)
        o = np.sort(d, 1)
        L = d.argmin(1)
        ok = (o[:, 0] < 2) & (o[:, 1] - o[:, 0] > 2)
        lab[k] = np.where(ok, L, 3)
        info[k] = {"ref_0_1_L": ref.round(3).tolist(), "sigma_seg": float(sg),
                   "ref_dist_sigma": {"0-1": float(np.linalg.norm(ref[0] - ref[1]) / sg),
                                      "1-L": float(np.linalg.norm(ref[1] - ref[2]) / sg),
                                      "0-L": float(np.linalg.norm(ref[0] - ref[2]) / sg)}}
    rq = {"segments": info}
    for b in (0, 1):
        m = bits[:, q] == b
        rb = {"label_fractions": {k: {NAMES[i]: float((lab[k][m] == i).mean()) for i in range(4)} for k in SEGS}}
        for k1, k2 in (("S2", "S3"), ("S1", "S2")):
            t = {}
            for i in range(3):
                src = m & (lab[k1] == i)
                n = int(src.sum())
                t[f"from_{NAMES[i]}"] = {"n": n, **{f"to_{NAMES[j]}": wilson(int((lab[k2][src] == j).sum()), n)
                                                    for j in range(4)}}
            rb[f"{k1}->{k2}"] = t
        rq[f"prep{b}"] = rb
    # HMM paths: first L episode
    P = np.load(f"{LP.stage('m3time')}/out_re/feat_xt1/m3time_re_q{q}_arrays.npz")["path"].astype(np.int8)
    isL = P == 2
    first = np.where(isL[:, 1:].any(1), isL[:, 1:].argmax(1) + 1, -1)
    hp = {}
    for b in (0, 1):
        for org in (0, 1):
            sel = (bits[:, q] == b) & (first > 0)
            idx = np.flatnonzero(sel)
            idx = idx[P[idx, first[idx] - 1] == org]
            ex, dw1 = [], []
            for i in idx:
                t = first[i]
                row = P[i]
                j = t
                while j < 20 and row[j] == 2:
                    j += 1
                if j == 20:
                    ex.append(-1)
                    continue
                ex.append(int(row[j]))
                if row[j] == 1:
                    k = j
                    while k < 20 and row[k] == 1:
                        k += 1
                    dw1.append((k - j, k == 20))
            ex = np.array(ex)
            n_exit = int((ex >= 0).sum())
            d1 = np.array([d for d, c in dw1]) if dw1 else np.array([0])
            hp[f"prep{b}_origin{org}"] = {
                "n_episodes": int(len(idx)), "n_exited": n_exit,
                "exit_to_1": wilson(int((ex == 1).sum()), n_exit), "exit_to_0": wilson(int((ex == 0).sum()), n_exit),
                "dwell_in_1_after_exit_bins_median": float(np.median(d1)) if dw1 else None,
                "frac_dwell1_ge3_bins_or_censored": float(np.mean([(d >= 3) or c for d, c in dw1])) if dw1 else None}
    rq["hmm_first_L_episode"] = hp
    res["per_qubit"][PAPER[q]] = rq
    print(PAPER[q], json.dumps({b: rq[b]["S2->S3"]["from_L"] for b in ("prep0", "prep1")}), flush=True)
    print("   origin", json.dumps({b: rq[b]["S1->S2"] for b in ("prep0",)})[:600], flush=True)
    print("   hmm", json.dumps(hp), flush=True)

with open(_os.path.join(LP.outdir(LP.stage("r")), "r_spot2.json"), "w") as f:
    json.dump(res, f, indent=1)
print("done")
