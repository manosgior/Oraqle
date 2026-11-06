#!/usr/bin/env python3
"""Controls for hmm_ceiling.py: recalibration-only baseline, paired CIs vs the soft CNN,
other-channel information (LDA own channel vs all five), Viterbi profile of win/loss shots."""
import argparse, json, pathlib, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "leakage"))
import lkpaths as LP
import m3time_hmm as M

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(LP.LK / "hmm_ceiling"), help="output directory of hmm_ceiling.py")
OUT = ap.parse_args().out
FEAT = str(LP.FEAT)
K, N_Q = 10, 5
NAMES = ["Q1", "Q2", "Q3", "Q4", "Q5"]
yte = np.load(f"{FEAT}/y_test.npy").astype(np.int64)
ytr = np.load(f"{FEAT}/y_train.npy").astype(np.int64)
bits = ((yte[:, None] >> np.arange(5)) & 1).astype(np.int8)
btr_all = ((ytr[:, None] >> np.arange(5)) & 1).astype(np.int8)
cnn = np.load(LP.REPORTS / "preds_MCMitCNN_pb.npy").astype(np.int8)
sz = np.load(LP.SOFT / "cnn_soft_1000ns.npz")
prob = sz["prob"].astype(np.float64)
N = len(yte)
fold = np.arange(N) % 2
Ftr = np.load(f"{FEAT}/featbox_train.npy", mmap_mode="r")
Fte = np.load(f"{FEAT}/featbox_test.npy", mmap_mode="r")
Xtr_all = np.asarray(Ftr[:, :, :K], dtype=np.float64).reshape(len(ytr), -1)
Xte_all = np.asarray(Fte[:, :, :K], dtype=np.float64).reshape(N, -1)


def crossfit(Z, b):
    pred = np.zeros(N, np.int8)
    for f in (0, 1):
        tr, te = fold != f, fold == f
        pred[te] = LogisticRegression(C=1e3, max_iter=1000).fit(Z[tr], b[tr]).predict(Z[te])
    return pred


def paired(pred, ref, b):
    e, r = pred != b, ref != b
    w, l = int((~e & r).sum()), int((e & ~r).sum())
    net = 100 * (l - w) / N
    se = 100 * np.sqrt(w + l) / N
    return {"err_pct": 100 * e.mean(), "win": w, "loss": l, "net_pp": net, "net_ci95": [net - 1.96 * se, net + 1.96 * se],
            "net_pp_fold": [100 * ((e & ~r)[fold == f].sum() - (~e & r)[fold == f].sum()) / (fold == f).sum() for f in (0, 1)]}


def lda(Ztr, b, Zte):
    m0, m1 = Ztr[b == 0].mean(0), Ztr[b == 1].mean(0)
    C = 0.5 * (np.cov((Ztr[b == 0] - m0).T) + np.cov((Ztr[b == 1] - m1).T))
    w = np.linalg.solve(C, m1 - m0)
    return ((Zte @ w - w @ (m0 + m1) / 2) > 0).astype(np.int8)


out = {}
for q in range(N_Q):
    b = bits[:, q]
    s = np.load(f"{OUT}/scores_q{q}.npz")
    p = np.clip(prob[:, q], 1e-4, 1 - 1e-4)
    lg = np.log(p / (1 - p))
    soft = (prob[:, q] > 0.5).astype(np.int8)
    recal = crossfit(lg[:, None], b)
    hyb = crossfit(np.stack([lg, s["hmm3_xt_self"].astype(np.float64)], 1), b)
    hmm = (s["hmm3_xt_self"] > 0).astype(np.int8)
    r = {"soft_err": 100 * (soft != b).mean(), "canon_err": 100 * (cnn[:, q] != b).mean(),
         "recal_vs_soft": paired(recal, soft, b),
         "hybrid_vs_soft": paired(hyb, soft, b),
         "hybrid_vs_recal": paired(hyb, recal, b),
         "hmm_vs_soft": paired(hmm, soft, b),
         "hmm_vs_canon": paired(hmm, cnn[:, q], b)}
    own = np.arange(q * K * 2, (q + 1) * K * 2)
    pown = lda(Xtr_all[:, own], btr_all[:, q], Xte_all[:, own])
    pall = lda(Xtr_all, btr_all[:, q], Xte_all)
    r["lda_own_err"], r["lda_all_err"] = 100 * (pown != b).mean(), 100 * (pall != b).mean()
    mz = np.load(f"{OUT}/model_q{q}.npz")
    P = {k: mz[f"hmm3_{k}"] for k in ("mu", "Sig", "A", "pi")}
    r["A_per_100ns"], r["pi"] = P["A"], P["pi"]
    r["tau_10_us"] = float(-0.1 / np.log1p(-P["A"][1, 0]))
    r["tau_01_us"] = float(-0.1 / np.log1p(-P["A"][0, 1]))
    X = np.asarray(Fte[:, q, :K], dtype=np.float64)
    bs = (s["hmm3"] > 0).astype(np.int8)
    path = M.viterbi(M.log_emis(X, P["mu"], P["Sig"]), P["A"], P["pi"][b])
    ch = (path[:, 3:] != path[:, 2:-1]).any(1)
    lk = (path == 2).any(1)
    ce, he = cnn[:, q] != b, bs != b
    sets = {"both_right": ~ce & ~he, "both_wrong": ce & he, "hmm_win": ce & ~he, "cnn_win": ~ce & he}
    r["viterbi_profile"] = {n: {"n": int(m.sum()), "pct_shots": 100 * m.mean(),
                                "prep1_share": float(b[m].mean()),
                                "in_window_change_pct": 100 * ch[m].mean(), "any_L_pct": 100 * lk[m].mean()}
                            for n, m in sets.items()}
    out[NAMES[q]] = r
    print(NAMES[q], "done", flush=True)
json.dump(M.jsonable(out), open(f"{OUT}/ceiling_followup.json", "w"), indent=1)
