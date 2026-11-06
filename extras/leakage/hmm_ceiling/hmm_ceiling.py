#!/usr/bin/env python3
"""hmm_ceiling.py -- is MCMit-CNN at the model-based ceiling?

Per qubit, fit a hidden Markov model of the 1 us readout trajectory on TRAIN and use it
as a discriminator on TEST: decide the prepared bit b by argmax_b P(x | b), where

    P(x | b) = sum_s0 pi[b, s0] * P(x | s0)      (forward algorithm, equal priors)

pi[b] absorbs preparation errors / decay during ring-up; A (per 100 ns) models decay,
excitation and (3-state) leakage during the window; emissions are per-(state, bin) 2-D
Gaussians with a per-bin covariance tied across states (m3time_hmm model). This is the
Bayes-optimal single-shot discriminator *if the model were true*.

Input: featbox (boxcar 100 ns bins of the demodulated, rotated IQ), bins 0-9 = samples
0-499 = the paper's 1 us window. Boxcar bins use only their own samples, so there is no
information from after 1 us (same window the CNN sees).

Variants per qubit:
  hmm2, hmm3          own channel only, no crosstalk correction
  hmm3_xt_cnn         static crosstalk offsets (fitted on TRAIN with true labels) removed
                      at TEST using the CNN's predicted neighbour bits (deployable)
  hmm3_xt_self        same with the neighbour bits from hmm3 (deployable)
  hmm3_xt_oracle      same with the TRUE neighbour bits (NOT deployable; a bound)
Plus baselines: LDA on the 1 us window mean (2-D) and LDA on the full 10-bin trajectory
(20-D, the optimal linear filter for stationary states), and the canonical MCMit-CNN.
Hybrid: logistic regression on [soft-CNN logit, HMM LLR], 2-fold cross-fitted on TEST.
"""
import argparse, json, os, pathlib, sys, time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "leakage"))
import lkpaths as LP  # noqa: E402  (locations, see leakage/README.md section 2)
import m3time_hmm as M  # noqa: E402  (log_emis, em, init_params, P2, P3)

NAMES = ["Q1", "Q2", "Q3", "Q4", "Q5"]
K = 10                     # 1 us = 10 x 100 ns
SUB = 120_000              # init-selection subsample


def cfg_other(y, q):
    return ((y >> (q + 1)) << q) | (y & ((1 << q) - 1))


def cfg_from_bits(bits, q):
    """bits (N,5) -> 0..15 index of the other four qubits (same packing as cfg_other)."""
    y = (bits.astype(np.int64) << np.arange(5)).sum(1)
    return cfg_other(y, q)


def fwd_ll(logB, A, pi0):
    """log P(x | initial distribution pi0) by the scaled forward pass. pi0 (S,) or (N,S)."""
    N, Kb, S = logB.shape
    mx = logB.max(2, keepdims=True)
    B = np.exp(logB - mx)
    a = np.broadcast_to(pi0, (N, S)) * B[:, 0]
    c = a.sum(1)
    ll = np.log(np.maximum(c, 1e-300))
    a = a / np.maximum(c, 1e-300)[:, None]
    for k in range(1, Kb):
        a = (a @ A) * B[:, k]
        c = a.sum(1)
        ll += np.log(np.maximum(c, 1e-300))
        a = a / np.maximum(c, 1e-300)[:, None]
    return ll + mx[..., 0].sum(1)


def llr(X, P):
    """log P(x|b=1) - log P(x|b=0) under model P."""
    logB = M.log_emis(X, P["mu"], P["Sig"])
    S = P["mu"].shape[0]
    lls = np.stack([fwd_ll(logB, P["A"], np.eye(S)[s]) for s in range(S)], 1)   # (N,S)
    lp = np.log(np.maximum(P["pi"], 1e-300))                                       # (2,S)
    l1 = np.logaddexp.reduce(lls + lp[1], axis=1)
    l0 = np.logaddexp.reduce(lls + lp[0], axis=1)
    return l1 - l0


def fit(X, b, S, q, iters, rng, tag):
    base, inits, geom = M.init_params(X, b, q, S=3)
    if S == 2:
        P, h = M.em(X, b, M.P2(base), iters, q=q, tag=f"{tag}/2")
        return P, {"ll_per_shot": h[-1], "iters": len(h)}
    sub = rng.choice(len(X), min(SUB, len(X)), replace=False)
    best = None
    for name, mu2 in inits:
        P, h = M.em(X[sub], b[sub], M.P3(base, mu2), 60, q=q, tag=f"{tag}/init:{name}")
        if best is None or h[-1] > best[2]:
            best = (name, P, h[-1])
    P, h = M.em(X, b, best[1], iters, q=q, tag=f"{tag}/3:{best[0]}")
    chk = M.state_check(P, base)
    return P, {"ll_per_shot": h[-1], "iters": len(h), "init": best[0], "state_check": chk}


def lda_fit(Z, b):
    m0, m1 = Z[b == 0].mean(0), Z[b == 1].mean(0)
    C = 0.5 * (np.cov((Z[b == 0] - m0).T) + np.cov((Z[b == 1] - m1).T))
    w = np.linalg.solve(np.atleast_2d(C), m1 - m0)
    t = w @ (m0 + m1) / 2
    return w, t


def thr_opt(s, b):
    """threshold maximising accuracy of (s > t) on TRAIN."""
    o = np.argsort(s)
    ss, bb = s[o], b[o]
    # predict 1 for index >= i: errors = (#b==1 below i) + (#b==0 at/above i)
    c1 = np.concatenate([[0], np.cumsum(bb == 1)])
    c0 = np.concatenate([[0], np.cumsum(bb == 0)])
    err = c1 + (c0[-1] - c0)
    i = int(err.argmin())
    if i == 0:
        return ss[0] - 1
    if i == len(ss):
        return ss[-1] + 1
    return 0.5 * (ss[i - 1] + ss[i])


def stage1(args):
    q, a = args
    t0 = time.time()
    rng = np.random.default_rng(7 + q)
    F = np.load(f"{a.feat}/{a.kind}_train.npy", mmap_mode="r")
    Xtr = np.asarray(F[:, q, :K], dtype=np.float64)
    F = np.load(f"{a.feat}/{a.kind}_test.npy", mmap_mode="r")
    Xte = np.asarray(F[:, q, :K], dtype=np.float64)
    ytr = np.load(f"{a.feat}/y_train.npy").astype(np.int64)
    yte = np.load(f"{a.feat}/y_test.npy").astype(np.int64)
    btr, bte = (ytr >> q) & 1, (yte >> q) & 1
    out = {"q": q, "name": NAMES[q]}
    res = {}

    # linear baselines
    Ztr, Zte = Xtr.mean(1), Xte.mean(1)
    w, t = lda_fit(Ztr, btr)
    res["lda_mean"] = Zte @ w - t
    w, t = lda_fit(Xtr.reshape(len(Xtr), -1), btr)
    res["lda_traj"] = Xte.reshape(len(Xte), -1) @ w - t

    # HMMs, no crosstalk correction
    models = {}
    for S in (2, 3):
        P, info = fit(Xtr, btr, S, q, a.iters, rng, f"hmm{S}")
        models[f"hmm{S}"] = P
        out[f"hmm{S}_fit"] = info
        res[f"hmm{S}"] = llr(Xte, P)
        out[f"hmm{S}_thr_train"] = float(thr_opt(llr(Xtr, P), btr))
        M.log(q, f"hmm{S} done ({time.time()-t0:.0f}s)")

    # 3-state HMM on crosstalk-corrected TRAIN (true labels are legitimate on TRAIN)
    ctr = cfg_other(ytr, q)
    m0 = np.median(Xtr[btr == 0], 0)
    offs = np.stack([np.median(Xtr[(btr == 0) & (ctr == c)], 0) - m0 for c in range(16)])
    P, info = fit(Xtr - offs[ctr], btr, 3, q, a.iters, rng, "hmm3xt")
    out["hmm3xt_fit"] = info
    out["xt_offsets_absmax"] = float(np.abs(offs).max())
    res["hmm3_xt_oracle"] = llr(Xte - offs[cfg_other(yte, q)], P)
    models["hmm3xt"] = P
    np.savez(f"{a.out}/model_q{q}.npz", offs=offs,
             **{f"{m}_{k}": v for m, Pm in models.items() for k, v in Pm.items()})
    np.savez(f"{a.out}/scores_q{q}.npz", **{k: v.astype(np.float32) for k, v in res.items()})
    M.log(q, f"stage1 done ({time.time()-t0:.0f}s)")
    return out


def stage2(args):
    """crosstalk correction with *estimated* neighbour bits (CNN, or hmm3 itself)."""
    q, a, nb = args
    F = np.load(f"{a.feat}/{a.kind}_test.npy", mmap_mode="r")
    Xte = np.asarray(F[:, q, :K], dtype=np.float64)
    mz = np.load(f"{a.out}/model_q{q}.npz")
    P = {k: mz[f"hmm3xt_{k}"] for k in ("mu", "Sig", "A", "pi")}
    offs = mz["offs"]
    out = {}
    for name, bits in nb.items():
        out[f"hmm3_xt_{name}"] = llr(Xte - offs[cfg_from_bits(bits, q)], P).astype(np.float32)
    return q, out


def wilson(k, n):
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100 * (c - h), 100 * (c + h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default=str(LP.FEAT))
    ap.add_argument("--kind", default="featbox")
    ap.add_argument("--cnn", default=str(LP.REPORTS / "preds_MCMitCNN_pb.npy"),
                    help="canonical MCMit-CNN predictions, TEST order")
    ap.add_argument("--soft", default=str(LP.SOFT / "cnn_soft_1000ns.npz"), help="soft MCMit-CNN probabilities")
    ap.add_argument("--out", default=str(LP.LK / "hmm_ceiling"))
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    with Pool(a.workers) as pool:
        fits = pool.map(stage1, [(q, a) for q in range(5)])
    S = {q: dict(np.load(f"{a.out}/scores_q{q}.npz")) for q in range(5)}

    yte = np.load(f"{a.feat}/y_test.npy").astype(np.int64)
    bits = ((yte[:, None] >> np.arange(5)) & 1).astype(np.int8)
    cnn = np.load(a.cnn).astype(np.int8)
    sz = np.load(a.soft)
    prob = sz["prob"].astype(np.float64)
    assert (sz["label"] == bits).all(), "soft-CNN label order differs from feature order"
    hmm3_bits = np.stack([(S[q]["hmm3"] > 0) for q in range(5)], 1).astype(np.int8)

    with Pool(a.workers) as pool:
        st2 = pool.map(stage2, [(q, a, {"cnn": cnn, "self": hmm3_bits}) for q in range(5)])
    for q, d in st2:
        S[q].update(d)
        np.savez(f"{a.out}/scores_q{q}.npz", **S[q])

    from sklearn.linear_model import LogisticRegression
    N = len(yte)
    fold = np.arange(N) % 2
    summary = {"conventions": "TEST split (1.12M shots, 560k per prepared bit per qubit). Error rates in % "
                              "of shots, Wilson 95% CI. 'win/loss vs CNN' = shots this method gets right and "
                              "the canonical CNN gets wrong / the reverse. Decision at LLR 0 (equal priors) "
                              "unless 'thr_train'.",
               "input": {"kind": a.kind, "bins": "0-9 (samples 0-499, 1 us)", "iters": a.iters},
               "fits": fits, "per_qubit": {}}
    for q in range(5):
        b = bits[:, q]
        row = {}

        def rate(pred, name):
            err = pred != b
            k = int(err.sum())
            r = {"err_pct": 100 * k / N, "ci95": wilson(k, N),
                 "err_prep0_pct": 100 * err[b == 0].mean(), "err_prep1_pct": 100 * err[b == 1].mean()}
            if name != "cnn":
                ce = cnn[:, q] != b
                r["win_vs_cnn"] = int((~err & ce).sum())
                r["loss_vs_cnn"] = int((err & ~ce).sum())
                r["net_pp_vs_cnn"] = 100 * (r["loss_vs_cnn"] - r["win_vs_cnn"]) / N
            row[name] = r

        rate(cnn[:, q], "cnn")
        rate((prob[:, q] > 0.5).astype(np.int8), "cnn_soft")
        for name, s in S[q].items():
            rate((s > 0).astype(np.int8), name)
        thr = fits[q]["hmm3_thr_train"]
        rate((S[q]["hmm3"] > thr).astype(np.int8), "hmm3_thr_train")

        # hybrid: soft-CNN logit + HMM LLR (2-fold cross-fit on TEST)
        p = np.clip(prob[:, q], 1e-4, 1 - 1e-4)
        lg = np.log(p / (1 - p))
        for hname in ("hmm3", "hmm3_xt_cnn"):
            Xh = np.stack([lg, S[q][hname].astype(np.float64)], 1)
            pred = np.zeros(N, np.int8)
            coefs = []
            for f in (0, 1):
                tr, te = fold != f, fold == f
                m = LogisticRegression(C=1e3, max_iter=1000).fit(Xh[tr], b[tr])
                pred[te] = m.predict(Xh[te])
                coefs.append([float(m.intercept_[0])] + [float(c) for c in m.coef_[0]])
            rate(pred, f"hybrid_cnnsoft+{hname}")
            row[f"hybrid_cnnsoft+{hname}"]["coef_[b0,cnn,hmm]"] = coefs
        summary["per_qubit"][NAMES[q]] = row
        M.log(q, "  ".join(f"{k}={v['err_pct']:.3f}" for k, v in row.items()))

    summary["runtime_s"] = time.time() - t0
    with open(f"{a.out}/hmm_ceiling.json", "w") as f:
        json.dump(M.jsonable(summary), f, indent=1)
    print("wrote", f"{a.out}/hmm_ceiling.json")


if __name__ == "__main__":
    main()
