"""r_spotcheck.py -- reconciliation spot checks for the leakage question (tag r).

Independent re-computation on the shared features (TEST split, 560000 shots per prepared bit):

 A. One frame, one statistic, four windows.  For every qubit and every prepared 5-bit state s
    (prepared bit x 16 neighbour configurations):
      sigma0_c = 1.4826 * MAD(perp) of the prepared-|0> state with the same neighbours,
      a_mode_s = smoothed-histogram mode of the along coordinate of state s,
      mu_s     = iterated, symmetrically truncated median of perp for shots of s near a_mode_s,
      z        = (perp - mu_s) / sigma0_c.
    Mirror excess E_k = [N(z > k) - N(z < -k)] / N per prepared bit (one-sided excess that needs only
    a symmetric core), k = 2..8, windows 200 ns (bins 0-1), 400 ns (bins 0-3), 1 us (feat1us 0-9,
    strictly causal = CNN input) and 2 us (bins 0-19).  Also the m4conf-style statistic
    (pooled-over-neighbours median / MAD of the prepared bit, |z| > 5) to show how the
    threshold/scale choice moves the number.  Q1/Q5 conditioned on Q3's prepared bit.
 B. Within-shot switching test for Q4 (and Q3 / Q5 as positive controls): 4-component GMMs on
    segment means S1 = bins 1-4 (100-500 ns), S2 = bins 5-9 (0.5-1 us), S3 = bins 10-19 (1-2 us);
    confident (posterior > 0.9) labels cross-tabulated S1->S2 and S2->S3.
 C. Canonical MCMit-CNN test predictions (preds_MCMitCNN_{pb,tb}.npy, 1 us, file order) cross-tabulated
    with three leakage flags: z(1 us) mirror-corrected, m3time canonical Viterbi paths (ever-L within
    bins 0-9, L at bin 9), and m4conf candidate set A.

Run on graham (read-only mounts, output to ~/oraqle-ics-sims/leakage/r):
  docker run -d --name leak-r-spot --user $(id -u):$(id -g) \
    -v ~/oraqle-ics-sims/leakage:/lk:ro -v ~/oraqle_reports:/rep:ro \
    -v ~/oraqle-ics-sims/leakage/r:/out oraqle-torch:latest python /out/r_spotcheck.py
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import math
import time

import numpy as np

T0 = time.time()
D = str(LP.FEAT)
OUT = LP.outdir(LP.stage("r"))
PAPER = {q: f"Q{q+1}" for q in range(5)}


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [p, max(0.0, c - h), min(1.0, c + h)]


def mirror(npl, nmi, n):
    e = (npl - nmi) / n
    h = 1.959963984540054 * math.sqrt(npl + nmi) / n
    return [e, e - h, e + h]


def amode(aa, s):
    lo, hi = np.percentile(aa, [0.5, 99.5])
    nb = max(10, int((hi - lo) / (s / 4)) + 1)
    h, e = np.histogram(aa, bins=nb, range=(lo, hi))
    k = np.exp(-0.5 * (np.arange(-8, 9) / 2.0) ** 2)
    hs = np.convolve(h, k / k.sum(), mode="same")
    i = int(np.argmax(hs))
    return 0.5 * (e[i] + e[i + 1])


log("loading")
F = np.load(D + "/feat_test.npy")            # (N,5,20,2) float32
F1 = np.load(D + "/feat1us_test.npy")        # (N,5,10,2)
y = np.load(D + "/y_test.npy").astype(np.int64)
N = len(y)
bits = ((y[:, None] >> np.arange(5)) & 1).astype(np.int8)
log("loaded", N)

WINDOWS = {
    "200ns": ("feat bins 0-1 = raw samples 0-99", lambda q: F[:, q, 0:2].mean(1)),
    "400ns": ("feat bins 0-3 = raw samples 0-199 (paper 400 ns)", lambda q: F[:, q, 0:4].mean(1)),
    "1us": ("feat1us bins 0-9 = raw samples 0-499, causal FIR (paper 1 us, CNN input)", lambda q: F1[:, q].mean(1)),
    "2us": ("feat bins 0-19 = raw samples 0-999 (full stored trace)", lambda q: F[:, q].mean(1)),
}
KS = [2, 3, 4, 5, 6, 8]

res = {"tag": "r", "split": "test", "N_per_prepared_bit": int(N // 2),
       "definitions": {
           "z": "(perp - mu_s)/sigma0_c; mu_s = truncated median of perp of prepared 5-bit state s near its along mode; "
                "sigma0_c = 1.4826*MAD(perp) of the prepared-|0> state with the same 16-config neighbours",
           "E_k": "[N(z>k) - N(z<-k)]/N per prepared bit (mirror excess); CI = +-1.96*sqrt(N+ + N-)/N",
           "m4style_A": "|perp - median_b|/(1.4826*MAD_b) > 5 with median/MAD pooled over neighbour configs (m4conf set A, test only)",
       },
       "A": {}, "B": {}, "C": {}}
Z1US = {}

# ----------------------------------------------------------------------------------------------- A
for q in range(5):
    others = y & ~(1 << q)
    _, cfg = np.unique(others, return_inverse=True)
    rq = {}
    for wn, (wdesc, fn) in WINDOWS.items():
        W = np.asarray(fn(q), dtype=np.float64)
        a, p = W[:, 0], W[:, 1]
        z = np.empty(N)
        sig0s, mus = [], {0: [], 1: []}
        for c in range(16):
            m0 = (bits[:, q] == 0) & (cfg == c)
            p0 = p[m0]
            s0 = 1.4826 * np.median(np.abs(p0 - np.median(p0)))
            sig0s.append(s0)
            for b in (0, 1):
                m = (bits[:, q] == b) & (cfg == c)
                aa, pp = a[m], p[m]
                am = amode(aa, s0)
                near = np.abs(aa - am) < s0
                mu = np.median(pp[near])
                for _ in range(5):
                    sel = near & (np.abs(pp - mu) < 2.5 * s0)
                    mu = np.median(pp[sel])
                z[m] = (pp - mu) / s0
                mus[b].append(mu)
        sig0 = float(np.median(sig0s))
        rw = {"window": wdesc, "sigma0_median_adc": sig0}
        for b in (0, 1):
            m = bits[:, q] == b
            zb = z[m]
            n = int(m.sum())
            d = {"N": n}
            for k in KS:
                npl, nmi = int((zb > k).sum()), int((zb < -k).sum())
                d[f"E{k}"] = mirror(npl, nmi, n)
                d[f"counts{k}_plus_minus_gauss"] = [npl, nmi, float(n * 0.5 * math.erfc(k / math.sqrt(2)))]
            # m4conf-style: pooled median / MAD of the prepared bit
            pp = p[m]
            zm = (pp - np.median(pp)) / (1.4826 * np.median(np.abs(pp - np.median(pp))))
            d["m4style_absz_gt5"] = wilson(int((np.abs(zm) > 5).sum()), n)
            d["m4style_z_gt5_plus_minus"] = [int((zm > 5).sum()), int((zm < -5).sum())]
            d["mad_b_over_sigma0"] = float(1.4826 * np.median(np.abs(pp - np.median(pp))) / sig0)
            # conditioned on Q3's prepared bit (for Q1, Q2, Q4, Q5) and on all-other-bits-0
            if q != 2:
                for b3 in (0, 1):
                    mm = m & (bits[:, 2] == b3)
                    zz = z[mm]
                    d[f"E3_given_Q3prep{b3}"] = mirror(int((zz > 3).sum()), int((zz < -3).sum()), int(mm.sum()))
                    d[f"E5_given_Q3prep{b3}"] = mirror(int((zz > 5).sum()), int((zz < -5).sum()), int(mm.sum()))
            mm = m & (others == 0)
            zz = z[mm]
            d["E3_given_all_other_bits_0"] = mirror(int((zz > 3).sum()), int((zz < -3).sum()), int(mm.sum()))
            rw[f"prep{b}"] = d
        rq[wn] = rw
        if wn == "1us":
            Z1US[q] = z.copy()
        log(f"A {PAPER[q]} {wn}: prep1 E3={100*rw['prep1']['E3'][0]:.3f}% E5={100*rw['prep1']['E5'][0]:.3f}% "
            f"prep0 E3={100*rw['prep0']['E3'][0]:.3f}%  m4style p1={100*rw['prep1']['m4style_absz_gt5'][0]:.3f}%")
    res["A"][PAPER[q]] = {"code": f"q{q}", **rq}

# ----------------------------------------------------------------------------------------------- B
from sklearn.mixture import GaussianMixture  # noqa: E402

SEGS = {"S1": (1, 5), "S2": (5, 10), "S3": (10, 20)}
hists = {}
rng = np.random.default_rng(0)
for q in (3, 2, 4):
    X = F[:, q].astype(np.float64)
    rb = {}
    for b in (1, 0):
        m = bits[:, q] == b
        S = {k: X[m, lo:hi].mean(1) for k, (lo, hi) in SEGS.items()}
        lab, post_ok, comps = {}, {}, {}
        for k in SEGS:
            sub = S[k][rng.choice(len(S[k]), 200000, replace=False)]
            g = GaussianMixture(4, covariance_type="full", n_init=3, random_state=0).fit(sub)
            order = np.argsort(g.means_[:, 0])           # sort components by along coordinate
            pr = g.predict_proba(S[k])[:, order]
            lab[k] = pr.argmax(1)
            post_ok[k] = pr.max(1) > 0.9
            comps[k] = {"means_along_perp": g.means_[order].tolist(), "weights": g.weights_[order].tolist(),
                        "sd_along_perp": np.sqrt(np.stack([np.diag(c) for c in g.covariances_])[order]).tolist(),
                        "confident_frac": float(post_ok[k].mean())}
        xt = {}
        for k1, k2 in (("S1", "S2"), ("S2", "S3"), ("S1", "S3")):
            ok = post_ok[k1] & post_ok[k2]
            M = np.zeros((4, 4), dtype=np.int64)
            np.add.at(M, (lab[k1][ok], lab[k2][ok]), 1)
            xt[f"{k1}->{k2}"] = {"matrix_rows_from_cols_to": M.tolist(),
                                 "row_fractions": (M / np.maximum(M.sum(1, keepdims=True), 1)).round(4).tolist(),
                                 "n_confident_both": int(ok.sum())}
        rb[f"prep{b}"] = {"gmm_components_sorted_by_along": comps, "crosstab": xt}
        # histograms for local plotting
        for k in SEGS:
            hists[f"q{q}_p{b}_{k}"] = S[k].astype(np.float32)[rng.choice(len(S[k]), 60000, replace=False)]
        log(f"B {PAPER[q]} prep{b}: S2->S3 row fractions {xt['S2->S3']['row_fractions']}")
    res["B"][PAPER[q]] = {"code": f"q{q}", "segments": {k: f"feat bins {lo}-{hi-1}" for k, (lo, hi) in SEGS.items()},
                          **rb}
np.savez_compressed(OUT + "/r_spot_segments.npz", **hists)

# ----------------------------------------------------------------------------------------------- C
pb = np.load(str(LP.REPORTS / "preds_MCMitCNN_pb.npy")).astype(np.int8)
tb = np.load(str(LP.REPORTS / "preds_MCMitCNN_tb.npy")).astype(np.int8)
res["C"]["aligned_tb_equals_prepared_bits"] = bool(np.array_equal(tb, bits))
res["C"]["source"] = "~/oraqle_reports/preds_MCMitCNN_{pb,tb}.npy (runners/_xtalk_cnn.py, canonical MCMit-CNN m=16, colleague prep, 1 us = raw samples 0-499, full test file)"
err = pb != tb
M4 = np.load(str(LP.stage("m4conf") / "out" / "m4conf_pershot.npz"))
spi, row, A4 = M4["spi"], M4["row"], M4["A"]
tst = spi == 1
A_test = np.zeros((N, 5), bool)
A_test[row[tst]] = A4[tst]
for q in range(5):
    z = Z1US[q]
    P = np.load(f"{LP.stage('m3time')}/out_re/feat_xt1/m3time_re_q{q}_arrays.npz")["path"]
    flags = {"z_gt3": z > 3, "z_lt_m3": z < -3, "z_gt5": z > 5,
             "hmm_everL_bins0_9": (P[:, :10] == 2).any(1), "hmm_L_at_bin9": P[:, 9] == 2,
             "hmm_startL_bins1_2": (P[:, 1] == 2) & (P[:, 2] == 2), "m4_setA": A_test[:, q]}
    rq = {"cnn_error_rate": wilson(int(err[:, q].sum()), N), "n_err": int(err[:, q].sum())}
    for b in (0, 1, "all"):
        m = np.ones(N, bool) if b == "all" else bits[:, q] == b
        e = err[:, q] & m
        ne = int(e.sum())
        d = {"N": int(m.sum()), "n_err": ne, "err_rate": wilson(ne, int(m.sum()))}
        for fn_, fl in flags.items():
            nf = int((fl & m).sum())
            nef = int((fl & e).sum())
            d[fn_] = {"n_flag": nf, "n_err_on_flag": nef,
                      "cnn_err_rate_on_flag": wilson(nef, nf), "share_of_errors": wilson(nef, ne)}
        npl, nmi = d["z_gt3"]["n_err_on_flag"], d["z_lt_m3"]["n_err_on_flag"]
        d["share_of_errors_mirror3"] = mirror(npl, nmi, max(ne, 1))
        d["leak_mirror3_shots"] = mirror(d["z_gt3"]["n_flag"], d["z_lt_m3"]["n_flag"], int(m.sum()))
        rq[f"prep{b}"] = d
    res["C"][PAPER[q]] = {"code": f"q{q}", **rq}
    log(f"C {PAPER[q]}: CNN err {100*rq['cnn_error_rate'][0]:.2f}%  share of errors mirror3 = "
        f"{100*rq['prepall']['share_of_errors_mirror3'][0]:.2f}%  everL share = "
        f"{100*rq['prepall']['hmm_everL_bins0_9']['share_of_errors'][0]:.2f}%  "
        f"prep1 everL cnn err = {100*rq['prep1']['hmm_everL_bins0_9']['cnn_err_rate_on_flag'][0]:.2f}%  "
        f"prep0 everL cnn err = {100*rq['prep0']['hmm_everL_bins0_9']['cnn_err_rate_on_flag'][0]:.2f}%")

# ----------------------------------------------------------------------------------------------- D
# Q3: is L reached only through |1>?  Compare P(ever L | start state) between prepared bits (m3 paths).
P = np.load(str(LP.stage("m3time") / "out_re" / "feat_xt1" / "m3time_re_q2_arrays.npz"))["path"]
dd = {}
for b in (0, 1):
    m = bits[:, 2] == b
    st = P[:, 1]
    for s in (0, 1):
        mm = m & (st == s) & (P[:, 2] == s)
        k = int(((P[mm, 3:10] == 2).any(1)).sum())
        dd[f"prep{b}_start{s}"] = {"n": int(mm.sum()), "enter_L_bins3_9": wilson(k, int(mm.sum()))}
res["D_Q3_entry_by_start_state"] = dd
log("D", json.dumps(dd))

with open(OUT + "/r_spotcheck.json", "w") as f:
    json.dump(res, f, indent=1)
log("done")
