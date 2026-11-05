"""em_e5vstat_lda.py (tag e5vstat): is the late-core overlap reference clean, and is the CNN's overlap really ~300x
the near-Bayes linear filter's?

For every qubit and neighbour configuration a Fisher linear discriminant (LDA) is fit on TRAIN featbox bins 0-9
(1 us, 20 features) using only late-window-stationary shots (late window, featbox bins 10-19, inside the e4recon
late core of the prepared state; that mask is re-derived here with the own e5vstat late frame, radius 1.5).
On TEST it reports
  * the LDA's empirical error on late-core-stationary shots vs its Gaussian prediction Phi(-Delta/2), and
  * the CNN's error on the same shots, and where the CNN's late-core errors sit on the LDA axis
    (deep on the wrong side -> genuine non-stationary/contaminating shots; near/right side -> the CNN boundary).
Also 600 / 400 ns (bins 0-5 / 0-3) with the soft CNN models.

Docker (graham): same mounts as em_e5vstat_check.py; needs /work/out/em_e5vstat_pershot_q{q}.npz from that script.
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.special import ndtr

D, R, S, O = str(LP.FEAT), str(LP.REPORTS), str(LP.SOFT), LP.outdir(LP.stage("em/e5vstat") / "out")
QN = ["Q1", "Q2", "Q3", "Q4", "Q5"]


def bits(y):
    return ((y[:, None] >> np.arange(5)) & 1).astype(np.int8)


def cfg_of(B, q):
    o = [k for k in range(5) if k != q]
    return (B[:, o].astype(np.int64) * (1 << np.arange(4))).sum(1)


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def mode2d(P, s, seed, bw=0.5):
    c = np.asarray(seed, float)
    for _ in range(200):
        d2 = ((P - c) ** 2).sum(1)
        m = d2 < (2 * s) ** 2
        w = np.exp(-d2[m] / (2 * (bw * s) ** 2))
        cn = (w[:, None] * P[m]).sum(0) / w.sum()
        if np.linalg.norm(cn - c) < 1e-6 * s:
            return cn
        c = cn
    return c


G = {}


def worker(q):
    t0 = time.time()
    ytr, yte = G["ytr"], G["yte"]
    Btr, Bte = bits(ytr), bits(yte)
    cft, cfe = cfg_of(Btr, q), cfg_of(Bte, q)
    ptr, pte = Btr[:, q], Bte[:, q]
    Xtr = np.asarray(np.load(f"{D}/featbox_train.npy", mmap_mode="r")[:, q], dtype=np.float64)
    Xte = np.asarray(np.load(f"{D}/featbox_test.npy", mmap_mode="r")[:, q], dtype=np.float64)
    Ltr, Lte = Xtr[:, 10:20].mean(1), Xte[:, 10:20].mean(1)
    ps = np.load(f"{O}/em_e5vstat_pershot_q{q}.npz")
    d0te, d1te = ps["d0l"].astype(np.float64), ps["d1l"].astype(np.float64)
    # TRAIN late-core masks (own frame, same construction as em_e5vstat_check.py: c0 = prep0 late mode,
    # c1 seeded at c0 + steady-state (bins 5-9) separation); median seeds are adequate for TRAIN selection
    lc_tr = np.zeros((len(ytr), 2), bool)
    for c in range(16):
        m0 = (cft == c) & (ptr == 0)
        m1 = (cft == c) & (ptr == 1)
        s = mad(Ltr[m0, 1])
        c0 = mode2d(Ltr[m0], s, np.median(Ltr[m0], 0))
        a0 = mode2d(Xtr[m0, 5:10].mean(1), s, np.median(Xtr[m0, 5:10].mean(1), 0))
        a1 = mode2d(Xtr[m1, 5:10].mean(1), s, np.median(Xtr[m1, 5:10].mean(1), 0))
        c1 = mode2d(Ltr[m1], s, c0 + (a1 - a0))
        lc_tr[m0, 0] = np.linalg.norm(Ltr[m0] - c0, axis=1) < 1.5 * s
        lc_tr[m1, 1] = np.linalg.norm(Ltr[m1] - c1, axis=1) < 1.5 * s
    lc_te = np.c_[d0te < 1.5, d1te < 1.5]
    res = {"q": q, "windows": {}}
    for wn, K, cnn in (("1us", 10, G["pred"]["1us"]), ("600ns", 6, G["pred"]["600ns"]), ("400ns", 4, G["pred"]["400ns"])):
        F_tr = Xtr[:, :K].reshape(len(ytr), -1)
        F_te = Xte[:, :K].reshape(len(yte), -1)
        score = np.empty(len(yte))
        delta = np.empty(16)
        pred_g = np.zeros(2)
        nb = np.zeros(2)
        for c in range(16):
            A0 = F_tr[(cft == c) & (ptr == 0) & lc_tr[:, 0]]
            A1 = F_tr[(cft == c) & (ptr == 1) & lc_tr[:, 1]]
            m0, m1 = A0.mean(0), A1.mean(0)
            C = 0.5 * (np.cov(A0.T) + np.cov(A1.T))
            w = np.linalg.solve(C, m1 - m0)
            dl = float(np.sqrt((m1 - m0) @ w))
            delta[c] = dl
            me = cfe == c
            score[me] = ((F_te[me] - 0.5 * (m0 + m1)) @ w) / dl          # N(-Delta/2, 1) for |0>, N(+Delta/2, 1) for |1>
            for b in (0, 1):
                n_ = int((me & (pte == b)).sum())
                pred_g[b] += n_ * ndtr(-dl / 2)
                nb[b] += n_
        lda = (score > 0).astype(np.int8)
        cp = cnn[:, q]
        out = {"delta_median": float(np.median(delta)), "delta_range": [float(delta.min()), float(delta.max())]}
        for b in (0, 1):
            m = pte == b
            ref = m & lc_te[:, b]
            nr = int(ref.sum())
            e_lda = int((ref & (lda != b)).sum())
            e_cnn = int((ref & (cp != b)).sum())
            sgn = 1 if b == 1 else -1                     # signed LDA score toward the prepared state
            sc_b = sgn * score
            cnn_err_ref = ref & (cp != b)
            hist_edges = [-99, -4, -3, -2, -1, 0, 1, 2, 3, 99]
            out[f"prep{b}"] = {
                "n_all": int(m.sum()), "lda_err_all_pct": float(100 * (lda[m] != b).mean()),
                "cnn_err_all_pct": float(100 * (cp[m] != b).mean()),
                "lda_gauss_pred_pct": float(100 * pred_g[b] / nb[b]),
                "n_latecore": nr, "lda_err_latecore": e_lda, "lda_err_latecore_pct": 100 * e_lda / nr,
                "lda_err_latecore_ci": [100 * x for x in wilson(e_lda, nr)],
                "cnn_err_latecore": e_cnn, "cnn_err_latecore_pct": 100 * e_cnn / nr,
                "cnn_err_latecore_ci": [100 * x for x in wilson(e_cnn, nr)],
                "both_err_latecore": int((ref & (lda != b) & (cp != b)).sum()),
                "cnn_latecore_err_lda_score_hist": {"edges_signed_toward_prepared": hist_edges,
                                                     "counts": np.histogram(sc_b[cnn_err_ref], bins=hist_edges)[0].tolist()},
                "lda_latecore_err_score_hist": np.histogram(sc_b[ref & (lda != b)], bins=hist_edges)[0].tolist(),
                "latecore_all_score_hist": np.histogram(sc_b[ref], bins=hist_edges)[0].tolist(),
                "gauss_expect_per_bin": (nr * np.diff(ndtr(np.array(hist_edges, float) - float(np.median(delta)) / 2))).tolist(),
            }
        res["windows"][wn] = out
    print(time.strftime("%H:%M:%S"), "q", q, round(time.time() - t0), "s", flush=True)
    return res


def main():
    G["ytr"], G["yte"] = np.load(D + "/y_train.npy"), np.load(D + "/y_test.npy")
    soft = {T: np.load(f"{S}/cnn_soft_{T}ns.npz")["prob"] for T in (600, 400)}
    G["pred"] = {"1us": np.load(R + "/preds_MCMitCNN_pb.npy").astype(np.int8),
                 "600ns": (soft[600] > 0.5).astype(np.int8), "400ns": (soft[400] > 0.5).astype(np.int8)}
    with Pool(5) as pool:
        outs = pool.map(worker, range(5))
    json.dump({"note": __doc__, "per_qubit": {QN[o["q"]]: o for o in outs}}, open(f"{O}/em_e5vstat_lda.json", "w"), indent=1, default=float)
    print("done", flush=True)


if __name__ == "__main__":
    main()
