#!/usr/bin/env python3
"""em_e1hmm_attrib_v6.py (tag e1hmm): HMM-based error-mode attribution of the CANONICAL MCMit-CNN errors.

v6 (2026-11-05) = v5 + one robust quantity and prep-split near-boundary groups.  The full v5 run showed that the
split of detected state changes into "in-window transition" (T_det) and "wrong from window start" (W_det) is
start-prior dependent (Q5: T 27.9% with the fitted label prior vs 2.9% with the agnostic prior) because transitions
in the first ~150-200 ns look like a wrong initial state, and v5's residual (n_err - L - T_det - W_det) absorbed the
deconvolution's degenerate "transition-type errors called wrong" term.  v6 adds
  S_det = (transition + wrong calls among non-leaked errors) - [P(trans|ov) + P(wrong|ov)] * N_ov
i.e. detected STATE CHANGES (any time from before ~150 ns to the window end), FP-corrected only for the calls that
stationary overlap errors would produce (N_ov from the per-config deconvolution, cross-checked against the
pure-Gaussian-overlap expectation), and the robust residual R_S = n_err - L - S_det ("no detected event").
Order swap for S: S on all errors first, then leakage among the remaining (stay/other-called) errors.
Near-boundary groups are additionally split by prepared bit.
(v5 header follows)

em_e1hmm_attrib_v5.py (tag e1hmm): HMM-based error-mode attribution of the CANONICAL MCMit-CNN errors.

v5 (2026-11-05) on top of v4 (per-config emission means, transition-free per-config null, deconvolution).
The v4 smoke test (120k TEST shots, Q3/Q5) showed:
  * the per-config mean refit moves the means by <= 0.3-0.45 window-sigma: v3's "3.5 sigma centroid mismatch" was
    an artefact of its contaminated null B, not an HMM misspecification;
  * the equal-weight in-window LDA misreads 23.8% of Q3 prepared-|1> shots (CNN: 9.5%): mid-window decays.  The
    CNN weights early bins and forgives them, so the CNN's errors are NOT the LDA's errors.  Conditioning the
    synthetic error types (overlap / wrong-throughout / transition-at-uniform-time) on LDA errors gives the wrong
    transition-time distribution among errors, and the 4-category deconvolution fails its goodness of fit;
  * the label-agnostic start prior fits the real call pattern much worse than the fitted label prior (400 ns Q5
    chi2 1380 vs 18 with 32 dof) and moves early (100-300 ns) transitions into "wrong from start".
v5 changes:
  * a per-config LINEAR SURROGATE of the CNN decision on the in-window HMM features (logistic regression of the
    CNN's hard output, IRLS; agreement with the CNN measured on a held-out half of TEST).  All synthetic error types
    are conditioned on SURROGATE errors (exact truncated sampling for the overlap type), so the null's error set is
    selected like the CNN's;
  * near-boundary characterisation against BOTH the optimal stationary LDA and the CNN surrogate boundary, each in
    units of the prepared-state noise width along its normal, with the corresponding pure-Gaussian-overlap
    predictions (expected count and expected fraction within 1 / 2 sigma);
  * PRIMARY detector = the fitted HMM as is (label prior pi_prep(b), per-config means) = variant lab_p50; the
    label-agnostic prior (p50) is a sensitivity variant; closure test run for both.
Everything else as v4 (leakage = line-frame mirror flag reused from full_v3/shared_inputs.npz; Q4 leakage n/m).
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy import sparse
from scipy.stats import norm, truncnorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import em_e1hmm_attrib_v3 as v3  # noqa: E402
import em_e1hmm_attrib_v4 as v4  # noqa: E402

FEAT, NB, Q4 = v3.FEAT, 20, 3
NAMES = v3.NAMES
WINDOWS = v4.WINDOWS
VARIANTS = v4.VARIANTS          # p50 (agnostic), p80, viterbi, fut_p50, lab_p50 (PRIMARY)
PRIMARY = "lab_p50"
TYPES = v4.TYPES
CATS = v4.CATS
OUT = str(LP.stage("em/e1hmm") / "out")
QKEYS6 = v4.QKEYS + ["S_det", "S_det_Aci", "S_det_all", "L_after_S"]


def summarize6(Q, q, nshots):
    o = v4.summarize(Q, q, nshots)
    for x in ("prep0", "prep1", "all"):
        sel = [0] if x == "prep0" else [1] if x == "prep1" else [0, 1]
        g = {k: Q[k][:, sel].sum(1) for k in Q}
        ne, ns = g["n_err"], nshots[x]

        def sh(v):
            return {"count": float(v[0]), "share": float(v[0] / ne[0]) if ne[0] else None,
                    "share_ci95": v3.ci_pct(v[1:] / np.maximum(ne[1:], 1)),
                    "rate_of_shots": float(v[0] / ns), "rate_of_shots_ci95": v3.ci_pct(v[1:] / ns)}
        Lz = g["L"] if q != Q4 else np.zeros_like(g["L"])
        LaS = g["L_after_S"] if q != Q4 else np.zeros_like(g["L"])
        o[x]["order_L_S"] = {"leakage": sh(Lz) if q != Q4 else "n/m", "state_change_detected": sh(g["S_det"]),
                             "of_which_in_window_transition": sh(g["T_det"]),
                             "of_which_wrong_from_start_or_early": sh(g["S_det"] - g["T_det"]),
                             "no_detected_event": sh(ne - Lz - g["S_det"])}
        o[x]["order_S_L"] = {"state_change_detected": sh(g["S_det_all"]),
                             "leakage": sh(LaS) if q != Q4 else "n/m",
                             "no_detected_event": sh(ne - g["S_det_all"] - LaS)}
        o[x]["state_change_detected_nullAci"] = sh(g["S_det_Aci"])
        o[x]["no_detected_event_nullAci"] = sh(ne - Lz - g["S_det_Aci"])
        o[x]["deconv_N_state_change_total_INFERRED"] = sh(g["N_wr"] + g["N_tr"])
    return o


def pool_boot6(results, qs, wname, ename, kind, v, nshot_per_q):
    p = v4.pool_boot(results, qs, wname, ename, kind, v, nshot_per_q)
    if p is None:
        return None
    acc = None
    for r in results:
        if r["kind"] != kind or r["q"] not in qs:
            continue
        B = {k: np.asarray(x).sum(1) for k, x in r["windows"][wname]["errsets"][ename]["boot"][v].items()}
        if r["q"] == Q4:
            for k in ("L", "L_after_T", "L_after_S", "overlap_LT", "Lplus", "Lminus"):
                B[k] = np.zeros_like(B[k])
        acc = B if acc is None else {k: acc[k] + B[k] for k in acc}
    ne = acc["n_err"]
    nsh = nshot_per_q * len(qs)

    def sh(x):
        return {"count": float(x[0]), "share": float(x[0] / ne[0]), "share_ci95": v3.ci_pct(x[1:] / ne[1:]),
                "rate_of_shots": float(x[0] / nsh), "rate_of_shots_ci95": v3.ci_pct(x[1:] / nsh)}
    p["order_L_S"] = {"leakage": sh(acc["L"]), "state_change_detected": sh(acc["S_det"]),
                      "of_which_in_window_transition": sh(acc["T_det"]),
                      "of_which_wrong_from_start_or_early": sh(acc["S_det"] - acc["T_det"]),
                      "no_detected_event": sh(ne - acc["L"] - acc["S_det"])}
    p["order_S_L"] = {"state_change_detected": sh(acc["S_det_all"]), "leakage": sh(acc["L_after_S"]),
                      "no_detected_event": sh(ne - acc["S_det_all"] - acc["L_after_S"])}
    p["state_change_detected_nullAci"] = sh(acc["S_det_Aci"])
    p["deconv_N_state_change_total_INFERRED"] = sh(acc["N_wr"] + acc["N_tr"])
    return p


def fit_logreg(Z, y, l2=1e-2, iters=30):
    """IRLS logistic regression with L2; Z (n,D) is standardised internally. Returns (w, b) in raw units."""
    mu, sd = Z.mean(0), Z.std(0) + 1e-12
    X = np.hstack([(Z - mu) / sd, np.ones((len(Z), 1))])
    w = np.zeros(X.shape[1])
    R = l2 * np.eye(X.shape[1])
    R[-1, -1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + R @ w
        Hs = (X * (p * (1 - p))[:, None]).T @ X + R + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(Hs, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    wr = w[:-1] / sd
    return wr, float(w[-1] - wr @ mu)


class NullS(v4.NullC):
    """v4 per-config transition-free null, with errors defined by a per-config linear CNN surrogate."""

    def set_surrogate(self, K, W, Bc):
        self.sur = {}
        for c in range(16):
            w = np.zeros(2 * NB)
            w[:2 * K] = W[c]
            mid = -Bc[c] * w / (w @ w)                  # w.x + b > 0  <=>  w.(x - mid) > 0
            self.sur[c] = (w, mid)

    def lda(self, c, K, model):                         # used by sample(): surrogate boundary
        return self.sur[c]

    def lda_opt(self, c, K, model="Ac"):
        return v4.NullC.lda(self, c, K, model)

    def oriented_with(self, which, Z, c, b, K, model="Ac"):
        w, mid = self.sur[c] if which == "sur" else self.lda_opt(c, K, model)
        _, S, _ = self.params(c, b, model)
        return (1.0 if b == 1 else -1.0) * ((Z - mid) @ w) / np.sqrt(w @ S @ w)

    def d_with(self, which, c, b, K, model="Ac"):
        w, mid = self.sur[c] if which == "sur" else self.lda_opt(c, K, model)
        m, S, _ = self.params(c, b, model)
        return (1.0 if b == 1 else -1.0) * (w @ (m - mid)) / np.sqrt(w @ S @ w)


def gauss_pred(dg, b, c):
    G, Gb, p1, p2 = 0.0, [0.0, 0.0], 0.0, 0.0
    for cfg in range(16):
        for x in (0, 1):
            ncb = int(((c == cfg) & (b == x)).sum())
            d = dg[cfg, x]
            pe = norm.cdf(-d)
            G += ncb * pe
            Gb[x] += ncb * pe
            p1 += ncb * (pe - norm.cdf(-1 - d))
            p2 += ncb * (pe - norm.cdf(-2 - d))
    return {"expected_overlap_errors": G, "expected_prep0": Gb[0], "expected_prep1": Gb[1],
            "pred_frac_within_1sigma": p1 / G if G > 0 else None, "pred_frac_within_2sigma": p2 / G if G > 0 else None,
            "d_median": float(np.median(dg)), "d_range": [float(dg.min()), float(dg.max())],
            "d_prep0_median": float(np.median(dg[:, 0])), "d_prep1_median": float(np.median(dg[:, 1]))}


def attribute(E, b, c, region, calls, pt, pw, s_opt, s_sur, sw, dgo, dgs, Mv, MvAi, nsim, a, rng, q, K, pcorr):
    ii = np.where(E)[0]
    bb, cc, rr = b[ii], c[ii], region[ii]
    out = {"n_err": int(len(ii)), "n_shots": int(len(E)), "err_rate": len(ii) / len(E),
           "err_rate_ci95": v3.wilson(len(ii), len(E))}
    for x in (0, 1):
        nb_, kb = int((b == x).sum()), int((bb == x).sum())
        out[f"err_rate_prep{x}"] = kb / nb_
        out[f"err_rate_prep{x}_ci95"] = v3.wilson(kb, nb_)
        out[f"n_err_prep{x}"] = kb
    reps = a.nboot
    W = rng.poisson(1.0, size=(reps, len(ii))).astype(np.float32)
    nshots = {"prep0": int((b == 0).sum()), "prep1": int((b == 1).sum()), "all": int(len(b))}
    res_v, boot_v = {}, {}
    for v in VARIANTS:
        cl = calls[v][ii]
        cell = ((cc * 2 + bb) * 3 + rr) * 4 + cl
        oh = sparse.csr_matrix((np.ones(len(ii), np.float32), (np.arange(len(ii)), cell)), shape=(len(ii), 384))
        cnt0 = np.asarray(oh.sum(0)).ravel().reshape(16, 2, 3, 4)
        cntb = np.asarray(oh.T.tocsr() @ W.T).T.reshape(reps, 16, 2, 3, 4)
        allc = np.concatenate([cnt0[None], cntb], 0)
        Q = {k: np.zeros((allc.shape[0], 2)) for k in QKEYS6}
        for r in range(allc.shape[0]):
            for cfg in range(16):
                for x in (0, 1):
                    C = allc[r, cfg, x]
                    Ms = {}
                    for nm, MM in (("Ac", Mv), ("Aci", MvAi)):
                        M = MM[v][cfg, x]
                        if r > 0:
                            M = np.stack([rng.multinomial(nsim, np.clip(M[:, t], 0, 1) / max(M[:, t].sum(), 1e-12))
                                          / nsim for t in range(3)], 1)
                        Ms[nm] = M
                    n_nl = C[0] + 2 * C[1]
                    for nm, M in Ms.items():
                        N, ch = v3.deconv(n_nl, M)
                        sfx = "" if nm == "Ac" else "_Aci"
                        Q["T_det" + sfx][r, x] += n_nl[0] - M[0, 0] * N[0] - M[0, 1] * N[1]
                        Q["W_det" + sfx][r, x] += n_nl[1] - M[1, 0] * N[0] - M[1, 2] * N[2]
                        Q["N_ov" + sfx][r, x] += N[0]
                        Q["N_wr" + sfx][r, x] += N[1]
                        Q["N_tr" + sfx][r, x] += N[2]
                        Q["chi2" + sfx][r, x] += ch
                        Q["S_det" + sfx][r, x] += n_nl[0] + n_nl[1] - (M[0, 0] + M[1, 0]) * N[0]
                    M = Ms["Ac"]
                    n_all = C.sum(0)
                    N2, _ = v3.deconv(n_all, M)
                    Q["S_det_all"][r, x] += n_all[0] + n_all[1] - (M[0, 0] + M[1, 0]) * N2[0]
                    Q["L_after_S"][r, x] += C[2, 2:].sum() - C[1, 2:].sum()
                    Q["n_err"][r, x] += C.sum()
                    Q["L"][r, x] += C[2].sum() - C[1].sum()
                    Q["Lplus"][r, x] += C[2].sum()
                    Q["Lminus"][r, x] += C[1].sum()
                    Q["n_nl"][r, x] += n_nl.sum()
                    Q["n_t"][r, x] += n_nl[0]
                    Q["n_w"][r, x] += n_nl[1]
                    Q["T_lo"][r, x] += n_nl[0] - max(M[0, 0], M[0, 1]) * n_nl.sum()
                    Q["T_det_all"][r, x] += n_all[0] - M[0, 0] * N2[0] - M[0, 1] * N2[1]
                    Q["L_after_T"][r, x] += C[2, 1:].sum() - C[1, 1:].sum()
                    Q["overlap_LT"][r, x] += C[2, 0] - C[1, 0]
        boot_v[v] = Q
        res_v[v] = summarize6(Q, q, nshots)
        res_v[v]["detector_call_probs"] = {
            f"prep{x}": {nm: {f"P_call_given_{t}_cfgmean": MM[v][:, x, :, ti].mean(0) for ti, t in enumerate(TYPES)}
                         for nm, MM in (("Ac", Mv), ("Aci", MvAi))} for x in (0, 1)}
        res_v[v]["detector_call_probs"]["call_order"] = CATS
    out["variants"] = res_v
    out["boot"] = {v: {k: boot_v[v][k].tolist() for k in QKEYS6} for v in ("lab_p50", "p50", "viterbi", "fut_p50")}
    wnl = np.where(rr == 0, 1.0, np.where(rr == 1, 2.0, 0.0)) if q != Q4 else np.ones(len(ii))
    for nm_, val in (("transition_agnostic", pt[ii].astype(np.float64)),
                     ("wrong_from_start_agnostic", pw[ii].astype(np.float64))):
        pm = {}
        for x_ in ("prep0", "prep1", "all"):
            mk = np.ones(len(ii), bool) if x_ == "all" else (bb == (0 if x_ == "prep0" else 1))
            tot0 = float((wnl * val * mk).sum())
            reps_ = (W * (wnl * val * mk)[None]).sum(1) / np.maximum((W * mk[None]).sum(1), 1)
            pm[x_] = {"count": tot0, "share": tot0 / max(mk.sum(), 1), "share_ci95": v3.ci_pct(reps_)}
        out[f"posterior_mass_{nm_}"] = pm
    # ---- independent near-boundary characterisation (mirror-weighted non-leaked errors)
    cl = calls[PRIMARY][ii]
    pc_e = pcorr[ii]

    def dist(mask, s):
        w_ = wnl * mask
        tot = w_.sum()
        if tot <= 0:
            return None
        edges = [-np.inf, -3, -2, -1, 0, 1, np.inf]
        h = [float((w_ * ((s >= lo) & (s < hi))).sum() / tot) for lo, hi in zip(edges[:-1], edges[1:])]
        sel = w_ > 0
        o = np.argsort(s[sel])
        cw = np.cumsum(w_[sel][o]) / tot
        qs = {f"q{int(p * 100)}": float(s[sel][o][min(np.searchsorted(cw, p), sel.sum() - 1)])
              for p in (0.05, 0.25, 0.5, 0.75, 0.95)}
        return {"weighted_n": float(tot), "hist_edges": "[-inf,-3,-2,-1,0,1,inf] sigma (<0: that boundary also wrong)",
                "hist": h, "frac_within_1sigma": float((w_ * (np.abs(s) < 1)).sum() / tot),
                "frac_within_2sigma": float((w_ * (np.abs(s) < 2)).sum() / tot),
                "frac_beyond_3sigma_wrong_side": float((w_ * (s < -3)).sum() / tot),
                "frac_boundary_correct_side": float((w_ * (s > 0)).sum() / tot), "quantiles": qs,
                "softcnn_frac_pcorrect_ge_0.4": float((w_ * (pc_e >= 0.4)).sum() / tot),
                "softcnn_frac_pcorrect_lt_0.1": float((w_ * (pc_e < 0.1)).sum() / tot)}
    groups = {"all_nonleaked_errors": np.ones(len(ii), bool), "no_event_calls_stay_or_other": (cl == 2) | (cl == 3),
              "calls_wrong": cl == 1, "calls_transition": cl == 0}
    for x_ in (0, 1):
        groups[f"all_nonleaked_errors_prep{x_}"] = bb == x_
        groups[f"no_event_calls_stay_or_other_prep{x_}"] = ((cl == 2) | (cl == 3)) & (bb == x_)
    out["near_boundary"] = {
        "cnn_surrogate_boundary": {g: dist(m, s_sur[ii]) for g, m in groups.items()},
        "optimal_lda_boundary": {g: dist(m, s_opt[ii]) for g, m in groups.items()},
        "gaussian_overlap_prediction_cnn_surrogate": gauss_pred(dgs, b, c),
        "gaussian_overlap_prediction_optimal_lda": gauss_pred(dgo, b, c),
        "observed_nonleaked_errors_mirror": float(wnl.sum())}
    out["detected_switch_bin_hist_agnostic_viterbi"] = np.bincount(np.clip(sw[ii][calls["p50"][ii] == 0], 0, K),
                                                                   minlength=K + 1).tolist()
    return out


def run_job(args):
    kind, q, a = args
    t0 = time.time()
    rng = np.random.default_rng(5000 + 17 * q + (0 if kind == "featbox" else 7))
    log = lambda m: print(f"[e1hmm-v6 {kind} q{q}] {m} ({time.time() - t0:.0f}s)", flush=True)
    H = v3.load_hmm(kind, q)
    taps = np.asarray(json.load(open(f"{FEAT}/calib.json"))["fir"]["taps"], dtype=float)
    yte = np.load(f"{FEAT}/y_test.npy").astype(np.int64)
    ytr = np.load(f"{FEAT}/y_train.npy").astype(np.int64)
    sel = np.arange(len(yte))
    if a.smoke:
        sel = np.sort(np.random.default_rng(5).choice(len(yte), a.smoke, replace=False))
    yte = yte[sel]
    bte, cte = (yte >> q) & 1, v3.cfg_other(yte, q)
    btr, ctr = (ytr >> q) & 1, v3.cfg_other(ytr, q)
    Xtr = np.asarray(np.load(f"{FEAT}/{kind}_train.npy", mmap_mode="r")[:, q], dtype=np.float64) - H["offsets"][ctr]
    Xte = np.asarray(np.load(f"{FEAT}/{kind}_test.npy", mmap_mode="r")[sel, q], dtype=np.float64) - H["offsets"][cte]
    res = {"kind": kind, "q": q, "paper": NAMES[q], "perm": H["perm"], "primary_variant": PRIMARY,
           "hmm": {"A": H["A"], "pi_prep0": H["pi"][0], "pi_prep1": H["pi"][1]}}
    mu, ll = v4.refit_cfg_means(Xtr, btr, ctr, H, iters=a.em_iters, n0=a.n0)
    res["cfg_means"] = {"loglik_gain_per_shot_mean": float((ll[:, -1] - ll[:, 0]).mean())}
    log(f"cfg means refit: loglik gain/shot {res['cfg_means']['loglik_gain_per_shot_mean']:.3f}")
    nc = json.load(open(a.noisecheck))[f"q{q}"]
    infl = [max(1.0, float(np.mean(nc[f"prep1_diff_mad_{ax}_perbin"]) / np.mean(nc[f"prep0_diff_mad_{ax}_perbin"])))
            for ax in ("along", "perp")]
    null = NullS(H, mu, kind, taps, infl)
    del Xtr
    shared = np.load(f"{a.shared}/shared_inputs.npz")
    res["windows"] = {}
    store, allcalls = {}, {"row": sel}
    half = (np.random.default_rng(11).random(len(sel)) < 0.5)
    for wname, K in WINDOWS.items():
        # ---- CNN predictions for this window and the per-config linear surrogate
        err_s = shared["err_canon"][sel, q] if wname == "1us" else shared[f"err_soft_{wname}"][sel, q]
        pred = (bte ^ err_s.astype(np.int64)).astype(np.float64)
        Z = Xte[:, :K].reshape(len(Xte), 2 * K)
        Wc, Bc = np.zeros((16, 2 * K)), np.zeros(16)
        agree_ho, agree_all = np.zeros(16), np.zeros(16)
        for cfg in range(16):
            m = cte == cfg
            w_h, b_h = fit_logreg(Z[m & half], pred[m & half])
            mh = m & ~half
            agree_ho[cfg] = (((Z[mh] @ w_h + b_h) > 0) == (pred[mh] > 0.5)).mean()
            Wc[cfg], Bc[cfg] = fit_logreg(Z[m], pred[m])
            agree_all[cfg] = (((Z[m] @ Wc[cfg] + Bc[cfg]) > 0) == (pred[m] > 0.5)).mean()
        null.set_surrogate(K, Wc, Bc)
        sur_pred = np.empty(len(sel), bool)
        for cfg in range(16):
            m = cte == cfg
            sur_pred[m] = (Z[m] @ Wc[cfg] + Bc[cfg]) > 0
        sur_err = sur_pred.astype(np.int64) != bte
        log(f"{wname}: surrogate agreement held-out {agree_ho.mean():.4f}; err cnn {err_s.mean():.4f} "
            f"sur {sur_err.mean():.4f}")
        # ---- detector on TEST
        calls, pt, pw, path = v4.run_detector(Xte, bte, cte, H, mu, K)
        z = shared[f"z_{wname}"][sel, q]
        region = np.where(z > 3, 2, np.where(z < -3, 1, 0)).astype(np.int8)
        if q == Q4:
            region[:] = 0
        s_opt, s_sur = np.empty(len(sel)), np.empty(len(sel))
        dgo, dgs, dgsi = np.zeros((16, 2)), np.zeros((16, 2)), np.zeros((16, 2))
        lda_err = np.zeros((16, 2))
        for cfg in range(16):
            for x in (0, 1):
                m = (cte == cfg) & (bte == x)
                Zm = Xte[m].reshape(-1, 2 * NB)
                s_opt[m] = null.oriented_with("opt", Zm, cfg, x, K)
                s_sur[m] = null.oriented_with("sur", Zm, cfg, x, K)
                dgo[cfg, x] = null.d_with("opt", cfg, x, K)
                dgs[cfg, x] = null.d_with("sur", cfg, x, K)
                dgsi[cfg, x] = null.d_with("sur", cfg, x, K, "Aci")
                lda_err[cfg, x] = (s_opt[m] < 0).mean()
        moved = path[:, 1:] != path[:, [1]]
        sw = np.where(moved.any(1), np.argmax(moved, 1) + 1, -1)
        log(f"{wname}: detector done; simulating null")
        ns = a.nsim

        def mats(model):
            M = {v: np.zeros((16, 2, 4, 3)) for v in VARIANTS}
            fp = {v: np.zeros((16, 2)) for v in VARIANTS}
            for cfg in range(16):
                for x in (0, 1):
                    for ti, typ in enumerate(TYPES):
                        Xs = null.sample(rng, ns, cfg, x, K, typ, model=model)
                        cl, _, _, _ = v4.run_detector(Xs, np.full(len(Xs), x), np.full(len(Xs), cfg), H, mu, K)
                        for v in VARIANTS:
                            M[v][cfg, x, :, ti] = np.bincount(cl[v], minlength=4) / len(Xs)
                    Xs = null.sample(rng, ns, cfg, x, K, "all", model=model)
                    cl, _, _, _ = v4.run_detector(Xs, np.full(len(Xs), x), np.full(len(Xs), cfg), H, mu, K)
                    for v in VARIANTS:
                        fp[v][cfg, x] = (cl[v] == 0).mean()
            return M, fp
        Mv, fpA = mats("Ac")
        MvAi, fpAi = mats("Aci")
        closure = []
        for gen in ("Ac", "Aci"):
            for comp in ((1.0, 1.0, 1.0), (4.0, 2.0, 1.0), (1.0, 6.0, 0.5)):
                for x in (0, 1):
                    Ntrue = (np.asarray(comp) * ns / 8).astype(int)
                    acc = {v: {"That": 0.0, "Ttrue": 0, "What": 0.0, "Wtrue": 0, "chi2": 0.0} for v in ("lab_p50", "p50")}
                    for cfg in range(16):
                        cnt = {v: np.zeros(4) for v in acc}
                        for ti, typ in enumerate(TYPES):
                            Xs = null.sample(rng, int(Ntrue[ti]), cfg, x, K, typ, model=gen)
                            cl, _, _, _ = v4.run_detector(Xs, np.full(len(Xs), x), np.full(len(Xs), cfg), H, mu, K)
                            for v in acc:
                                cnt[v] += np.bincount(cl[v], minlength=4)
                                if typ == "tr":
                                    acc[v]["Ttrue"] += int((cl[v] == 0).sum())
                                if typ == "wr":
                                    acc[v]["Wtrue"] += int((cl[v] == 1).sum())
                        for v in acc:
                            M = Mv[v][cfg, x]
                            Nh, ch = v3.deconv(cnt[v], M)
                            acc[v]["That"] += cnt[v][0] - M[0, 0] * Nh[0] - M[0, 1] * Nh[1]
                            acc[v]["What"] += cnt[v][1] - M[1, 0] * Nh[0] - M[1, 2] * Nh[2]
                            acc[v]["chi2"] += ch
                    closure.append({"generated_with": gen, "deconvolved_with": "Ac", "prep": x,
                                    "N_true_per_cfg_ov_wr_tr": Ntrue.tolist(), "n_err": int(16 * Ntrue.sum()),
                                    **{v: acc[v] for v in acc}})
        inj = {}
        for x in (0, 1):
            rows = []
            for tau in np.arange(0, 100 * K, 25):
                agg = {v: np.zeros(4) for v in VARIANTS}
                n_tot = n_se = 0
                n_te = {v: 0 for v in ("lab_p50", "p50")}
                for cfg in range(16):
                    Xs = null.sample(rng, a.ninj, cfg, x, K, "inj", model="Ac", tau=tau)
                    cl, _, _, _ = v4.run_detector(Xs, np.full(len(Xs), x), np.full(len(Xs), cfg), H, mu, K)
                    e = null.oriented_with("sur", Xs.reshape(-1, 2 * NB), cfg, x, K) < 0
                    for v in VARIANTS:
                        agg[v] += np.bincount(cl[v], minlength=4)
                    n_tot += len(Xs)
                    n_se += int(e.sum())
                    for v in n_te:
                        n_te[v] += int(((cl[v] == 0) & e).sum())
                rows.append({"tau_ns": int(tau), **{f"P_call_{v}": (agg[v] / n_tot).tolist() for v in VARIANTS},
                             "P_surrogate_error": n_se / n_tot,
                             **{f"P_trans_{v}_given_surrogate_error": (n_te[v] / n_se) if n_se else None
                                for v in n_te}, "n": n_tot})
            inj[f"prep{x}"] = rows
        log(f"{wname}: null, closure, injection done")
        esets = {}
        if wname == "1us":
            esets["canon"] = shared["err_canon"][sel, q]
        esets["soft"] = shared[f"err_soft_{wname}"][sel, q]
        p1 = shared[f"p_soft_{wname}"][sel, q].astype(np.float64)
        pcorr = np.where(bte == 1, p1, 1.0 - p1)
        for k_, v_ in (("call_lab_p50", calls["lab_p50"]), ("call_p50", calls["p50"]),
                       ("ptrans_agn", pt.astype(np.float16)), ("pwrong_agn", pw.astype(np.float16)),
                       ("region", region), ("s_opt", s_opt.astype(np.float16)), ("s_sur", s_sur.astype(np.float16))):
            allcalls[f"{wname}_{k_}"] = v_
        Wo = {"K": K, "noise_inflation_Aci": null.infl,
              "surrogate": {"agreement_heldout_mean": float(agree_ho.mean()), "agreement_heldout_min": float(agree_ho.min()),
                            "agreement_all_mean": float(agree_all.mean()), "surrogate_err_rate": float(sur_err.mean()),
                            "cnn_err_rate": float(err_s.mean()),
                            "frac_cnn_errors_also_surrogate_errors": float((sur_err & err_s).sum() / max(err_s.sum(), 1)),
                            "weights_abs_per_bin_cfgmean": np.abs(Wc).reshape(16, K, 2).sum(2).mean(0)},
              "d_opt_per_cfg": dgo, "d_sur_per_cfg": dgs, "d_sur_Aci_per_cfg": dgsi,
              "lda_error_rate_test": {"prep0": float(lda_err[:, 0].mean()), "prep1": float(lda_err[:, 1].mean())},
              "null_fp_per_shot_stationary": {nm: {v: {"prep0": float(f_[v][:, 0].mean()),
                                                       "prep1": float(f_[v][:, 1].mean()),
                                                       "max_cfg": float(f_[v].max())} for v in VARIANTS}
                                              for nm, f_ in (("Ac", fpA), ("Aci", fpAi))},
              "injection": inj, "closure_test": closure, "errsets": {}}
        tr_all = {}
        for x in (0, 1):
            m = bte == x
            for v in ("lab_p50", "p50"):
                k = int(((calls[v] == 0) & m).sum())
                f = float(fpA[v][:, x].mean())
                tr_all[f"prep{x}_{v}"] = {"raw_trans_call_rate": k / m.sum(), "null_fp_rate": f,
                                          "fp_corrected_rate": k / m.sum() - f, "raw_ci95": v3.wilson(k, m.sum())}
        Wo["trans_calls_all_shots"] = tr_all
        for ename, E in esets.items():
            Wo["errsets"][ename] = attribute(E, bte, cte, region, calls, pt, pw, s_opt, s_sur, sw, dgo, dgs, Mv, MvAi,
                                             ns, a, rng, q, K, pcorr)
            store[(wname, ename)] = (E, region, calls[PRIMARY], s_opt, s_sur)
        if q != Q4:
            hid = {}
            for x in (0, 1):
                m = bte == x
                npl, nmi = int(((region == 2) & m).sum()), int(((region == 1) & m).sum())
                hid[f"prep{x}"] = {"n": int(m.sum()), "n_plus": npl, "n_minus": nmi, "E3": (npl - nmi) / m.sum(),
                                   "E3_ci95": v3.ci_mirror(npl, nmi, m.sum())}
            npl, nmi = int((region == 2).sum()), int((region == 1).sum())
            hid["all"] = {"E3": (npl - nmi) / len(region), "E3_ci95": v3.ci_mirror(npl, nmi, len(region))}
            for ename, E in esets.items():
                for x in (0, 1, None):
                    m = np.ones(len(E), bool) if x is None else bte == x
                    p_, m_ = int(((region == 2) & m).sum()), int(((region == 1) & m).sum())
                    ep, em = int((E & (region == 2) & m).sum()), int((E & (region == 1) & m).sum())
                    tag = "all" if x is None else f"prep{x}"
                    den, num = max(p_ - m_, 1), ep - em
                    var = (ep + em) / den ** 2 + (num ** 2) * (p_ + m_) / den ** 4
                    r_ = 1 - num / den
                    hid[f"read_correctly_of_L_{ename}_{tag}"] = r_
                    hid[f"read_correctly_of_L_{ename}_{tag}_ci95"] = [r_ - 1.96 * np.sqrt(var), r_ + 1.96 * np.sqrt(var)]
        else:
            hid = "n/m (Q4 multimodal |1>; line-frame flag not interpretable)"
        Wo["hidden_leakage"] = hid
        res["windows"][wname] = Wo
        log(f"{wname}: done")
    arr = {}
    for (wname, ename), (E, region, cprim, s_opt, s_sur) in store.items():
        ii = np.where(E)[0]
        arr[f"{wname}_{ename}_row"] = sel[ii]
        arr[f"{wname}_{ename}_region"] = region[ii]
        arr[f"{wname}_{ename}_call_primary"] = cprim[ii]
        arr[f"{wname}_{ename}_s_opt"] = s_opt[ii].astype(np.float32)
        arr[f"{wname}_{ename}_s_sur"] = s_sur[ii].astype(np.float32)
    np.savez_compressed(f"{OUT}/em_e1hmm_v6_pershot_{kind}_q{q}.npz", **arr)
    np.savez_compressed(f"{OUT}/em_e1hmm_v6_allshots_{kind}_q{q}.npz", **allcalls)
    res["runtime_s"] = time.time() - t0
    json.dump(v3.jsonable(res), open(f"{OUT}/em_e1hmm_v6_job_{kind}_q{q}.json", "w"))
    log("job done")
    return res


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--nsim", type=int, default=2000)
    ap.add_argument("--ninj", type=int, default=60)
    ap.add_argument("--nboot", type=int, default=300)
    ap.add_argument("--em_iters", type=int, default=4)
    ap.add_argument("--n0", type=float, default=200.0)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--kinds", default="featbox,feat")
    ap.add_argument("--qubits", default="0,1,2,3,4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shared", default=str(LP.stage("em/e1hmm") / "full_v3"))
    ap.add_argument("--noisecheck", default=str(LP.stage("em/e1hmm") / "noisecheck" / "em_e1hmm_noisecheck.json"))
    a = ap.parse_args()
    OUT = a.out
    assert not os.path.exists(f"{OUT}/em_e1hmm_v6_results.json"), "refusing to overwrite"
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    jobs = [(k, q, a) for k in a.kinds.split(",") for q in map(int, a.qubits.split(","))]
    with Pool(min(a.procs, len(jobs))) as pool:
        results = pool.map(run_job, jobs, chunksize=1)
    pooled = {}
    nq = a.smoke if a.smoke else 1120000
    for kind in a.kinds.split(","):
        for wname in WINDOWS:
            for ename in ("canon", "soft"):
                for nm, qs in (("all5", [0, 1, 2, 3, 4]), ("Q1Q3Q4Q5", [0, 2, 3, 4]), ("Q1Q2Q3Q5", [0, 1, 2, 4]),
                               ("Q3Q5", [2, 4])):
                    if not all(any(r["kind"] == kind and r["q"] == k for r in results) for k in qs):
                        continue
                    for v in ("lab_p50", "p50", "viterbi", "fut_p50"):
                        p = pool_boot6(results, qs, wname, ename, kind, v, nq)
                        if p is not None:
                            pooled[f"{kind}|{wname}|{ename}|{nm}|{v}"] = p
    meta = json.load(open(f"{a.shared}/shared_meta.json"))
    out = {"script": "em_e1hmm_attrib_v6.py", "doc": __doc__, "args": vars(a), "primary_variant": PRIMARY,
           "meta_shared_v3": meta, "pooled": pooled, "per_qubit": results, "runtime_s": time.time() - t0}
    json.dump(v3.jsonable(out), open(f"{OUT}/em_e1hmm_v6_results.json", "w"), indent=1)
    print("all done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
