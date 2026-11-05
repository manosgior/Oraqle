#!/usr/bin/env python3
"""em_e1hmm_attrib_v4.py (tag e1hmm): HMM-based error-mode attribution of the CANONICAL MCMit-CNN errors.

v4 (2026-11-05, after the pause).  Critical review of v3 (full_v3 run, which completed) found:
  (1) the m3time HMM emission means are CONFIG-INDEPENDENT after a state-independent crosstalk offset (offsets are
      per-bin medians of own-|0> shots).  The |1> trajectory depends on the neighbour configuration, so null A (the
      HMM's own generative model) was not a realistic null: the null-A LDA misreads 23.5% of Q3 prepared-|1> shots
      (CNN: 9.5%), the per-config |1> centroid is off by up to 3.5 window-sigma (Q3) / median 1.6 (Q4), and the
      deconvolution of the real error calls failed its goodness of fit (chi2 1241 / 896 for Q3 prep 0/1, 9180 for
      Q4, with ~16 dof each) and returned a NEGATIVE FP-corrected transition share for Q3 prepared |0> (-7%).
  (2) the detector used the LABEL-informed start prior pi_prep(b).  For errors (shots that look like 1-b) this
      manufactures transitions: a shot in 1-b for the whole window is called a b->1-b transition 5-28% of the time
      (Q5 prep 0: pi_prep0 = [1, 0, 0] so any wrong-throughout shot "must" have transitioned).
v4 fixes both, keeping the frozen TRAIN HMM (A, pi, tied white Sig[k]) and the canonical relabelling:
  * per-config emission means mu_c[s,k] (s = 0, 1, L) refit on TRAIN by EM on the means only (A, pi, Sig frozen;
    20-bin forward-backward with the label prior, shrinkage n0 pseudo-counts toward the global mu);
  * primary detector uses the LABEL-AGNOSTIC start prior pi_agn = (pi_prep0 + pi_prep1)/2 (label-informed kept as
    variant lab_p50);
  * null (no transitions) = per configuration and state: stationary sequences mu_c[s,k] + white N(0, Sig[k])
    ("Ac"); sensitivity variant "Aci" with the prepared-|1> noise inflated by the measured |1>/|0> ratio
    (em_e1hmm_noisecheck.json).  Null B of v3 (contaminated by real decays) is dropped.
  * per-config deconvolution of the 4 call categories of non-leaked errors into overlap / wrong-throughout /
    transition (chi2 reported as goodness of fit of the null), FP-corrected detected transitions T_det and
    FP-corrected detected wrong-throughout W_det, an assumption-light interval for T, order swap, posterior mass.
Detector (unchanged from v3 apart from prior and means): forward pass over in-window bins 0..K-1 ONLY (featbox:
bin k = samples 50k..50k+49, no look-ahead; 1 us K=10, 600 ns K=6, 400 ns K=4); J(a,e) = P(s_1=a, s_{K-1}=e | x_0..
x_{K-1}); call 'transition' if J(b,1-b) > 0.5 (b = prepared bit); else argmax of wrong J(1-b,1-b) / stay J(b,b) /
other (L-involving).  Bin 0 is ring-up, so the start state is read at bin 1 (100-200 ns).
Leakage: line-frame mirror-corrected flag (z>3 minus z<-3), exactly as f_cnn_leak_xtab.py, reused from
full_v3/shared_inputs.npz (validated: Q3 1 us 3.14% [3.00,3.28] of canonical errors, as in LEAKAGE_REPORT 3.5).
Q4: leakage n/m (multimodal |1>).

Docker (steve, CPU only):
  -v ~/oraqle-ics-sims/leakage/features:/feat:ro -v ~/oraqle-ics-sims/leakage/m3time:/m3:ro
  -v ~/oraqle-ics-sims/leakage/f/in:/soft:ro -v ~/oraqle_reports:/reports:ro
  -v ~/oraqle-ics-sims/leakage/em/e1hmm:/work -v ~/oraqle-ics-sims/leakage/em/e1hmm/scripts:/scripts:ro
  python /scripts/em_e1hmm_attrib_v4.py --out /work/full_v4 --shared /work/full_v3
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
import em_e1hmm_attrib_v3 as v3  # noqa: E402  (validated helpers: HMM loading, forward pass, frames, deconv)

FEAT, NB, L, Q4 = v3.FEAT, 20, 2, 3
NAMES = v3.NAMES
WINDOWS = {"1us": 10, "600ns": 6, "400ns": 4}
VARIANTS = ["p50", "p80", "viterbi", "fut_p50", "lab_p50"]
TYPES = ["ov", "wr", "tr"]
CATS = ["trans", "wrong", "stay", "other"]
OUT = str(LP.stage("em/e1hmm") / "out")


# ------------------------------------------------------------------ per-config emission means (TRAIN EM, means only)
def fb_gamma(B, A, pi0):
    N, K, S = B.shape
    alpha = np.empty_like(B)
    a = pi0 * B[:, 0]
    alpha[:, 0] = a / a.sum(1, keepdims=True)
    for k in range(1, K):
        a = (alpha[:, k - 1] @ A) * B[:, k]
        alpha[:, k] = a / a.sum(1, keepdims=True)
    gam = np.empty_like(B)
    gam[:, K - 1] = alpha[:, K - 1]
    beta = np.ones((N, S))
    for k in range(K - 2, -1, -1):
        beta = (B[:, k + 1] * beta) @ A.T
        beta /= beta.sum(1, keepdims=True)
        g = alpha[:, k] * beta
        gam[:, k] = g / g.sum(1, keepdims=True)
    return gam


def refit_cfg_means(X, b, c, H, iters=4, n0=200.0):
    mu = np.broadcast_to(H["mu"], (16,) + H["mu"].shape).copy()
    ll = np.zeros((16, iters + 1))
    for cc in range(16):
        m = c == cc
        Xc, bc = X[m], b[m]
        for it in range(iters + 1):
            logB = v3.log_emis(Xc, mu[cc], H["Sig"])
            mx = logB.max(2, keepdims=True)
            B = np.exp(logB - mx)
            # log-likelihood (for monitoring) from the forward normalisers
            a = H["pi"][bc] * B[:, 0]
            s = a.sum(1)
            lls = np.log(s) + mx[:, 0, 0]
            a /= s[:, None]
            for k in range(1, NB):
                a = (a @ H["A"]) * B[:, k]
                s = a.sum(1)
                lls += np.log(s) + mx[:, k, 0]
                a /= s[:, None]
            ll[cc, it] = lls.mean()
            if it == iters:
                break
            g = fb_gamma(B, H["A"], H["pi"][bc])
            num = np.einsum("nks,nkd->skd", g, Xc)
            den = g.sum(0).T                                   # (S, K)
            mu[cc] = (num + n0 * H["mu"]) / (den[..., None] + n0)
    return mu, ll


# ------------------------------------------------------------------ detector
def run_detector(X, b, c, H, mu, K):
    """X (n,20,2) xt-corrected; per-config emission means mu (16,3,20,2). Returns calls dict, p_trans, p_wrong,
    in-window Viterbi path (agnostic prior)."""
    n = len(X)
    calls = {v: np.empty(n, np.int8) for v in VARIANTS}
    pt = np.empty(n, np.float32)
    pw = np.empty(n, np.float32)
    path = np.empty((n, K), np.int8)
    pi_agn = 0.5 * (H["pi"][0] + H["pi"][1])
    for cc in np.unique(c):
        m = np.where(c == cc)[0]
        logB = v3.log_emis(X[m], mu[cc], H["Sig"])
        B = v3.emis(logB)
        bm = b[m]
        Ja = v3.joint_start_end(B, H["A"], pi_agn, K)
        beta = v3.backward_to(B, H["A"], K - 1)
        Jf = Ja * beta[:, None, :]
        Jf /= Jf.sum((1, 2), keepdims=True)
        pa = v3.viterbi(logB[:, :K], H["A"], pi_agn)
        ca, pta = v3.calls_from(Ja, Jf, pa, bm, K)
        Jl = v3.joint_start_end(B, H["A"], H["pi"][bm], K)
        cl, _ = v3.calls_from(Jl, Jl, pa, bm, K)
        for v in ("p50", "p80", "viterbi", "fut_p50"):
            calls[v][m] = ca[v]
        calls["lab_p50"][m] = cl["p50"]
        pt[m] = pta
        pw[m] = ca["_pw"]
        path[m] = pa
    return calls, pt, pw, path


# ------------------------------------------------------------------ transition-free null per (config, state)
class NullC:
    def __init__(self, H, mu, kind, taps, infl):
        self.mu = mu[:, :2].reshape(16, 2, 2 * NB)
        S = np.zeros((2 * NB, 2 * NB))
        for k in range(NB):
            S[2 * k:2 * k + 2, 2 * k:2 * k + 2] = H["Sig"][k]
        sc = np.tile(np.asarray(infl, float), NB)
        self.S, self.S1 = S, S * np.outer(sc, sc)
        self.L, self.L1 = np.linalg.cholesky(self.S), np.linalg.cholesky(self.S1)
        self.infl = [float(x) for x in infl]
        self.frac_tab = np.stack([np.repeat(v3.step_fraction(t, kind, taps), 2) for t in np.arange(0, 2001, 25)])

    def params(self, c, s, model):
        if model == "Aci" and s == 1:
            return self.mu[c, s], self.S1, self.L1
        return self.mu[c, s], self.S, self.L

    def lda(self, c, K, model):
        D = 2 * K
        m0, m1 = self.mu[c, 0, :D], self.mu[c, 1, :D]
        St = self.S[:D, :D] if model == "Ac" else 0.5 * (self.S[:D, :D] + self.S1[:D, :D])
        w = np.zeros(2 * NB)
        w[:D] = np.linalg.solve(St, m1 - m0)
        mid = np.zeros(2 * NB)
        mid[:D] = 0.5 * (m0 + m1)
        return w, mid

    def oriented(self, Z, c, b, K, model="Ac"):
        w, mid = self.lda(c, K, model)
        _, S, _ = self.params(c, b, model)
        return (1.0 if b == 1 else -1.0) * ((Z - mid) @ w) / np.sqrt(w @ S @ w)

    def d_gauss(self, c, b, K, model="Ac"):
        w, mid = self.lda(c, K, model)
        m, S, _ = self.params(c, b, model)
        return (1.0 if b == 1 else -1.0) * (w @ (m - mid)) / np.sqrt(w @ S @ w)

    def sample(self, rng, n, c, b, K, typ, model="Ac", tau=None):
        w, mid = self.lda(c, K, model)
        sign = 1.0 if b == 1 else -1.0
        if typ in ("ov", "all"):
            m, S, Lc = self.params(c, b, model)
            eps = rng.standard_normal((n, 2 * NB)) @ Lc.T
            if typ == "ov":                                  # exact truncated sampling: LDA error
                v = w @ S @ w
                g = S @ w / v
                z = eps - np.outer(eps @ w, g)
                d = sign * (w @ (m - mid)) / np.sqrt(v)
                u = truncnorm.rvs(-np.inf, -d, size=n, random_state=rng)
                eps = z + np.outer(sign * u * np.sqrt(v), g)
            return (m + eps).reshape(n, NB, 2)
        out, got, it = [], 0, 0
        while got < n and it < 400:
            it += 1
            nn = max(2 * (n - got), 2000)
            if typ == "wr":
                m, S, Lc = self.params(c, 1 - b, model)
                X = m + rng.standard_normal((nn, 2 * NB)) @ Lc.T
            else:                                             # tr / inj: b -> 1-b at tau
                mb, Sb, _ = self.params(c, b, model)
                mo, So, _ = self.params(c, 1 - b, model)
                Lt = np.linalg.cholesky(0.5 * (Sb + So))
                taus = rng.uniform(0, 100.0 * K, nn) if tau is None else np.full(nn, float(tau))
                fr = self.frac_tab[np.round(taus / 25.0).astype(int)]
                X = (1 - fr) * mb + fr * mo + rng.standard_normal((nn, 2 * NB)) @ Lt.T
            if typ != "inj":
                X = X[sign * ((X - mid) @ w) < 0]
            out.append(X)
            got += len(X)
        X = np.concatenate(out)[:n]
        return X.reshape(len(X), NB, 2)


def ci_pct(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


# ------------------------------------------------------------------ attribution of one error set
QKEYS = ["n_err", "L", "Lplus", "Lminus", "n_nl", "n_t", "n_w", "T_det", "W_det", "N_ov", "N_wr", "N_tr", "chi2",
         "T_lo", "T_det_all", "L_after_T", "overlap_LT", "T_det_Aci", "W_det_Aci", "N_ov_Aci", "N_wr_Aci",
         "N_tr_Aci", "chi2_Aci"]


def attribute(E, b, c, region, calls, pt, pw, s_lda, sw, dg, Mv, MvAi, nsim, a, rng, q, K, pcorr):
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
        nrep = allc.shape[0]
        Q = {k: np.zeros((nrep, 2)) for k in QKEYS}
        for r in range(nrep):
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
                    M = Ms["Ac"]
                    n_all = C.sum(0)
                    N2, _ = v3.deconv(n_all, M)
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
        res_v[v] = summarize(Q, q, nshots)
        res_v[v]["detector_call_probs"] = {
            f"prep{x}": {nm: {f"P_call_given_{t}_cfgmean": MM[v][:, x, :, ti].mean(0) for ti, t in enumerate(TYPES)}
                         for nm, MM in (("Ac", Mv), ("Aci", MvAi))} for x in (0, 1)}
        res_v[v]["detector_call_probs"]["call_order"] = CATS
    out["variants"] = res_v
    out["boot"] = {v: {k: boot_v[v][k].tolist() for k in QKEYS} for v in ("p50", "lab_p50", "viterbi", "fut_p50")}
    # posterior mass (threshold-free; calibrated only if the HMM is right)
    wnl = np.where(rr == 0, 1.0, np.where(rr == 1, 2.0, 0.0)) if q != Q4 else np.ones(len(ii))
    for nm_, val in (("transition", pt[ii].astype(np.float64)), ("wrong_from_start", pw[ii].astype(np.float64))):
        pm = {}
        for x_ in ("prep0", "prep1", "all"):
            mk = np.ones(len(ii), bool) if x_ == "all" else (bb == (0 if x_ == "prep0" else 1))
            tot0 = float((wnl * val * mk).sum())
            reps_ = (W * (wnl * val * mk)[None]).sum(1) / np.maximum((W * mk[None]).sum(1), 1)
            pm[x_] = {"count": tot0, "share": tot0 / max(mk.sum(), 1), "share_ci95": ci_pct(reps_)}
        out[f"posterior_mass_{nm_}"] = pm
    # independent near-boundary characterisation (per-config LDA in noise units; Gaussian-overlap prediction)
    cl = calls["p50"][ii]
    s = s_lda[ii]
    pc_e = pcorr[ii]

    def dist(mask):
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
        return {"weighted_n": float(tot), "hist_edges": "[-inf,-3,-2,-1,0,1,inf] (oriented LDA sigma; <0 LDA wrong)",
                "hist": h, "frac_within_1sigma": float((w_ * (np.abs(s) < 1)).sum() / tot),
                "frac_within_2sigma": float((w_ * (np.abs(s) < 2)).sum() / tot),
                "frac_beyond_3sigma_wrong_side": float((w_ * (s < -3)).sum() / tot),
                "frac_lda_correct": float((w_ * (s > 0)).sum() / tot), "quantiles": qs,
                "softcnn_frac_pcorrect_ge_0.4": float((w_ * (pc_e >= 0.4)).sum() / tot),
                "softcnn_frac_pcorrect_lt_0.1": float((w_ * (pc_e < 0.1)).sum() / tot),
                "softcnn_median_pcorrect": float(np.median(pc_e[sel]))}
    nb = {"no_event_calls_stay_or_other": dist((cl == 2) | (cl == 3)), "calls_wrong": dist(cl == 1),
          "calls_transition": dist(cl == 0), "all_nonleaked_errors": dist(np.ones(len(ii), bool))}
    G, Gb, p1, p2 = 0.0, [0.0, 0.0], 0.0, 0.0
    for cfg in range(16):
        for x in (0, 1):
            ncb = int(((c == cfg) & (b == x)).sum())
            d = dg[cfg, x]
            pe = norm.cdf(-d)
            G += ncb * pe
            Gb[x] += ncb * pe
            if pe > 0:
                p1 += ncb * (pe - norm.cdf(-1 - d))
                p2 += ncb * (pe - norm.cdf(-2 - d))
    nb["gaussian_overlap_prediction"] = {
        "expected_overlap_errors_optimal_stationary": G, "expected_prep0": Gb[0], "expected_prep1": Gb[1],
        "pred_frac_within_1sigma_of_boundary": p1 / G if G > 0 else None,
        "pred_frac_within_2sigma_of_boundary": p2 / G if G > 0 else None,
        "d_gauss_median": float(np.median(dg)), "d_gauss_range": [float(dg.min()), float(dg.max())]}
    out["near_boundary"] = nb
    det = cl == 0
    out["detected_switch_bin_hist"] = np.bincount(np.clip(sw[ii][det], 0, K), minlength=K + 1).tolist()
    return out


def summarize(Q, q, nshots):
    o = {}
    for x in ("prep0", "prep1", "all"):
        sel = [0] if x == "prep0" else [1] if x == "prep1" else [0, 1]
        g = {k: Q[k][:, sel].sum(1) for k in Q}
        ne, ns = g["n_err"], nshots[x]

        def sh(v):
            return {"count": float(v[0]), "share": float(v[0] / ne[0]) if ne[0] else None,
                    "share_ci95": v3.ci_pct(v[1:] / np.maximum(ne[1:], 1)),
                    "rate_of_shots": float(v[0] / ns), "rate_of_shots_ci95": v3.ci_pct(v[1:] / ns)}
        Lz = g["L"] if q != Q4 else np.zeros_like(g["L"])
        LaT = g["L_after_T"] if q != Q4 else np.zeros_like(g["L"])
        o[x] = {"n_err": float(ne[0]), "n_shots": ns,
                "leakage_share_mirror_ci95_analytic": v3.ci_mirror(g["Lplus"][0], g["Lminus"][0], ne[0])
                if q != Q4 else "n/m",
                "order_L_T_W": {"leakage": sh(Lz) if q != Q4 else "n/m", "transition": sh(g["T_det"]),
                                "wrong_from_start": sh(g["W_det"]),
                                "no_detected_event": sh(ne - Lz - g["T_det"] - g["W_det"])},
                "order_T_L_W": {"transition": sh(g["T_det_all"]), "leakage": sh(LaT) if q != Q4 else "n/m",
                                "wrong_from_start": sh(g["W_det"]),
                                "no_detected_event": sh(ne - g["T_det_all"] - LaT - g["W_det"])},
                "overlap_leak_and_transcall": sh(g["overlap_LT"]) if q != Q4 else "n/m",
                "transition_raw_calls_nonleaked": sh(g["n_t"]), "wrong_raw_calls_nonleaked": sh(g["n_w"]),
                "transition_interval_low_assumption_light": sh(g["T_lo"]),
                "transition_nullAci": sh(g["T_det_Aci"]), "wrong_from_start_nullAci": sh(g["W_det_Aci"]),
                "deconv_N_overlap": sh(g["N_ov"]), "deconv_N_wrong_throughout": sh(g["N_wr"]),
                "deconv_N_transition_sens_corrected": sh(g["N_tr"]),
                "deconv_N_overlap_nullAci": sh(g["N_ov_Aci"]), "deconv_N_wrong_nullAci": sh(g["N_wr_Aci"]),
                "deconv_chi2": float(g["chi2"][0]), "deconv_chi2_nullAci": float(g["chi2_Aci"][0]),
                "deconv_dof": 16 * len(sel)}
    return o


# ------------------------------------------------------------------ per (kind, q) job
def run_job(args):
    kind, q, a = args
    t0 = time.time()
    rng = np.random.default_rng(4000 + 17 * q + (0 if kind == "featbox" else 7))
    log = lambda m: print(f"[e1hmm-v4 {kind} q{q}] {m} ({time.time() - t0:.0f}s)", flush=True)
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
    res = {"kind": kind, "q": q, "paper": NAMES[q], "perm": H["perm"],
           "hmm": {"A": H["A"], "pi_prep0": H["pi"][0], "pi_prep1": H["pi"][1]}}
    mu, ll = refit_cfg_means(Xtr, btr, ctr, H, iters=a.em_iters, n0=a.n0)
    res["cfg_means"] = {"train_loglik_per_shot_iter0": ll[:, 0], "train_loglik_per_shot_final": ll[:, -1],
                        "loglik_gain_per_shot_mean": float((ll[:, -1] - ll[:, 0]).mean())}
    log(f"cfg means refit: mean loglik gain/shot {res['cfg_means']['loglik_gain_per_shot_mean']:.3f}")
    nc = json.load(open(a.noisecheck))[f"q{q}"]
    infl = [max(1.0, float(np.mean(nc[f"prep1_diff_mad_{ax}_perbin"]) / np.mean(nc[f"prep0_diff_mad_{ax}_perbin"])))
            for ax in ("along", "perp")]
    null = NullC(H, mu, kind, taps, infl)
    del Xtr
    shared = np.load(f"{a.shared}/shared_inputs.npz")
    res["windows"] = {}
    store, allcalls = {}, {"row": sel}
    for wname, K in WINDOWS.items():
        # shift of the per-config means vs the global HMM means, in window-mean sigma (bins 2..K-1)
        sigw = np.sqrt(np.mean([np.trace(H["Sig"][k]) / 2 for k in range(2, K)]) / (K - 2))
        shift = np.array([[np.linalg.norm((mu[cfg, s_, 2:K] - H["mu"][s_, 2:K]).mean(0)) / sigw
                           for s_ in range(2)] for cfg in range(16)])
        calls, pt, pw, path = run_detector(Xte, bte, cte, H, mu, K)
        z = shared[f"z_{wname}"][sel, q]
        region = np.where(z > 3, 2, np.where(z < -3, 1, 0)).astype(np.int8)
        if q == Q4:
            region[:] = 0
        s_lda = np.empty(len(yte))
        dg, dgi = np.zeros((16, 2)), np.zeros((16, 2))
        lda_err = np.zeros((16, 2))
        for cfg in range(16):
            for x in (0, 1):
                m = (cte == cfg) & (bte == x)
                s_lda[m] = null.oriented(Xte[m].reshape(-1, 2 * NB), cfg, x, K)
                dg[cfg, x] = null.d_gauss(cfg, x, K)
                dgi[cfg, x] = null.d_gauss(cfg, x, K, "Aci")
                lda_err[cfg, x] = (s_lda[m] < 0).mean()
        moved = path[:, 1:] != path[:, [1]]
        sw = np.where(moved.any(1), np.argmax(moved, 1) + 1, -1)
        log(f"{wname}: detector on TEST done; simulating null")
        ns = a.nsim

        def mats(model):
            M = {v: np.zeros((16, 2, 4, 3)) for v in VARIANTS}
            fp = {v: np.zeros((16, 2)) for v in VARIANTS}
            for cfg in range(16):
                for x in (0, 1):
                    for ti, typ in enumerate(TYPES):
                        Xs = null.sample(rng, ns, cfg, x, K, typ, model=model)
                        cl, _, _, _ = run_detector(Xs, np.full(len(Xs), x), np.full(len(Xs), cfg), H, mu, K)
                        for v in VARIANTS:
                            M[v][cfg, x, :, ti] = np.bincount(cl[v], minlength=4) / len(Xs)
                    Xs = null.sample(rng, ns, cfg, x, K, "all", model=model)
                    cl, _, _, _ = run_detector(Xs, np.full(len(Xs), x), np.full(len(Xs), cfg), H, mu, K)
                    for v in VARIANTS:
                        fp[v][cfg, x] = (cl[v] == 0).mean()
            return M, fp
        Mv, fpA = mats("Ac")
        MvAi, fpAi = mats("Aci")
        log(f"{wname}: null matrices done")
        # closure test: synthetic error sets of known composition (all 16 configs), generated under Ac / Aci,
        # deconvolved with the Ac matrices
        closure = []
        for gen in ("Ac", "Aci"):
            for comp in ((1.0, 1.0, 1.0), (4.0, 2.0, 1.0), (1.0, 6.0, 0.5)):
                for x in (0, 1):
                    Ntrue = (np.asarray(comp) * ns / 8).astype(int)
                    That, Ttrue, What, Wtrue, chi = 0.0, 0, 0.0, 0, 0.0
                    for cfg in range(16):
                        cnt = np.zeros(4)
                        for ti, typ in enumerate(TYPES):
                            Xs = null.sample(rng, int(Ntrue[ti]), cfg, x, K, typ, model=gen)
                            cl, _, _, _ = run_detector(Xs, np.full(len(Xs), x), np.full(len(Xs), cfg), H, mu, K)
                            cnt += np.bincount(cl["p50"], minlength=4)
                            if typ == "tr":
                                Ttrue += int((cl["p50"] == 0).sum())
                            if typ == "wr":
                                Wtrue += int((cl["p50"] == 1).sum())
                        M = Mv["p50"][cfg, x]
                        Nh, ch = v3.deconv(cnt, M)
                        That += cnt[0] - M[0, 0] * Nh[0] - M[0, 1] * Nh[1]
                        What += cnt[1] - M[1, 0] * Nh[0] - M[1, 2] * Nh[2]
                        chi += ch
                    closure.append({"generated_with": gen, "prep": x, "N_true_per_cfg_ov_wr_tr": Ntrue.tolist(),
                                    "T_det_true": Ttrue, "T_det_hat": That, "W_det_true": Wtrue, "W_det_hat": What,
                                    "chi2_16cfg": chi, "n_err": int(16 * Ntrue.sum())})
        # injection test: transition b -> 1-b at tau, all configs (null Ac means/noise)
        inj = {}
        for x in (0, 1):
            rows = []
            for tau in np.arange(0, 100 * K, 25):
                agg = {v: np.zeros(4) for v in VARIANTS}
                n_tot = n_err = n_te = 0
                for cfg in range(16):
                    Xs = null.sample(rng, a.ninj, cfg, x, K, "inj", model="Ac", tau=tau)
                    cl, _, _, _ = run_detector(Xs, np.full(len(Xs), x), np.full(len(Xs), cfg), H, mu, K)
                    e = null.oriented(Xs.reshape(-1, 2 * NB), cfg, x, K) < 0
                    for v in VARIANTS:
                        agg[v] += np.bincount(cl[v], minlength=4)
                    n_tot += len(Xs)
                    n_err += int(e.sum())
                    n_te += int(((cl["p50"] == 0) & e).sum())
                rows.append({"tau_ns": int(tau), **{f"P_call_{v}": (agg[v] / n_tot).tolist() for v in VARIANTS},
                             "P_lda_error": n_err / n_tot,
                             "P_trans_given_lda_error": n_te / n_err if n_err else None, "n": n_tot})
            inj[f"prep{x}"] = rows
        log(f"{wname}: closure + injection done")
        esets = {}
        if wname == "1us":
            esets["canon"] = shared["err_canon"][sel, q]
        esets["soft"] = shared[f"err_soft_{wname}"][sel, q]
        p1 = shared[f"p_soft_{wname}"][sel, q].astype(np.float64)
        pcorr = np.where(bte == 1, p1, 1.0 - p1)
        allcalls[f"{wname}_call_p50"] = calls["p50"]
        allcalls[f"{wname}_call_lab_p50"] = calls["lab_p50"]
        allcalls[f"{wname}_ptrans"] = pt.astype(np.float16)
        allcalls[f"{wname}_pwrong"] = pw.astype(np.float16)
        allcalls[f"{wname}_region"] = region
        allcalls[f"{wname}_slda"] = s_lda.astype(np.float16)
        Wo = {"K": K, "noise_inflation_Aci": null.infl,
              "cfg_mean_shift_vs_global_window_sigma": {"prep0_state0": shift[:, 0], "state1": shift[:, 1],
                                                        "max": float(shift.max()), "median": float(np.median(shift))},
              "d_gauss_Ac_per_cfg": dg, "d_gauss_Aci_per_cfg": dgi,
              "lda_error_rate_test_per_cfg": lda_err,
              "lda_error_rate_test": {"prep0": float(lda_err[:, 0].mean()), "prep1": float(lda_err[:, 1].mean())},
              "null_fp_per_shot_stationary": {nm: {v: {"prep0": float(f_[v][:, 0].mean()),
                                                       "prep1": float(f_[v][:, 1].mean()),
                                                       "max_cfg": float(f_[v].max())} for v in VARIANTS}
                                              for nm, f_ in (("Ac", fpA), ("Aci", fpAi))},
              "injection": inj, "closure_test": closure, "errsets": {}}
        tr_all = {}
        for x in (0, 1):
            m = bte == x
            for v in ("p50", "lab_p50"):
                k = int(((calls[v] == 0) & m).sum())
                f = float(fpA[v][:, x].mean())
                tr_all[f"prep{x}_{v}"] = {"raw_trans_call_rate": k / m.sum(), "null_fp_rate": f,
                                          "fp_corrected_rate": k / m.sum() - f, "raw_ci95": v3.wilson(k, m.sum())}
        Wo["trans_calls_all_shots"] = tr_all
        for ename, E in esets.items():
            Wo["errsets"][ename] = attribute(E, bte, cte, region, calls, pt, pw, s_lda, sw, dg, Mv, MvAi, ns, a,
                                             rng, q, K, pcorr)
            store[(wname, ename)] = (E, region, calls["p50"], s_lda, pt, pw)
        hid = {}
        if q != Q4:
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
                    hid[f"read_correctly_of_L_{ename}_{tag}"] = 1 - (ep - em) / max(p_ - m_, 1)
                    # CI: misread count is a mirror difference; propagate both numerator and denominator (delta)
                    den, num = max(p_ - m_, 1), ep - em
                    var = (ep + em) / den ** 2 + (num ** 2) * (p_ + m_) / den ** 4
                    r_ = 1 - num / den
                    hid[f"read_correctly_of_L_{ename}_{tag}_ci95"] = [r_ - 1.96 * np.sqrt(var), r_ + 1.96 * np.sqrt(var)]
        else:
            hid = "n/m (Q4 multimodal |1>; line-frame flag not interpretable)"
        Wo["hidden_leakage"] = hid
        res["windows"][wname] = Wo
        log(f"{wname}: done")
    arr = {}
    for (wname, ename), (E, region, c50, s_lda, pt, pw) in store.items():
        ii = np.where(E)[0]
        arr[f"{wname}_{ename}_row"] = sel[ii]
        arr[f"{wname}_{ename}_region"] = region[ii]
        arr[f"{wname}_{ename}_call"] = c50[ii]
        arr[f"{wname}_{ename}_slda"] = s_lda[ii].astype(np.float32)
        arr[f"{wname}_{ename}_ptrans"] = pt[ii]
        arr[f"{wname}_{ename}_pwrong"] = pw[ii]
    np.savez_compressed(f"{OUT}/em_e1hmm_v4_pershot_{kind}_q{q}.npz", **arr)
    np.savez_compressed(f"{OUT}/em_e1hmm_v4_allshots_{kind}_q{q}.npz", **allcalls)
    np.save(f"{OUT}/em_e1hmm_v4_cfgmeans_{kind}_q{q}.npy", mu)
    res["runtime_s"] = time.time() - t0
    json.dump(v3.jsonable(res), open(f"{OUT}/em_e1hmm_v4_job_{kind}_q{q}.json", "w"))
    log("job done")
    return res


def pool_boot(results, qs, wname, ename, kind, v, nshot_per_q):
    acc = None
    for r in results:
        if r["kind"] != kind or r["q"] not in qs:
            continue
        E = r["windows"][wname]["errsets"].get(ename)
        if E is None:
            return None
        B = {k: np.asarray(x).sum(1) for k, x in E["boot"][v].items()}
        if r["q"] == Q4:
            for k in ("L", "L_after_T", "overlap_LT", "Lplus", "Lminus"):
                B[k] = np.zeros_like(B[k])
        acc = B if acc is None else {k: acc[k] + B[k] for k in acc}
    if acc is None:
        return None
    ne = acc["n_err"]
    nsh = nshot_per_q * len(qs)

    def sh(x):
        return {"count": float(x[0]), "share": float(x[0] / ne[0]), "share_ci95": v3.ci_pct(x[1:] / ne[1:]),
                "rate_of_shots": float(x[0] / nsh), "rate_of_shots_ci95": v3.ci_pct(x[1:] / nsh)}
    return {"qubits": [NAMES[k] for k in qs], "n_err": float(ne[0]), "n_shots": nsh, "err_rate_pooled": ne[0] / nsh,
            "err_rate_pooled_ci95": v3.wilson(int(ne[0]), nsh),
            "leakage_share_mirror_ci95_analytic": v3.ci_mirror(acc["Lplus"][0], acc["Lminus"][0], ne[0]),
            "order_L_T_W": {"leakage": sh(acc["L"]), "transition": sh(acc["T_det"]),
                            "wrong_from_start": sh(acc["W_det"]),
                            "no_detected_event": sh(ne - acc["L"] - acc["T_det"] - acc["W_det"])},
            "order_T_L_W": {"transition": sh(acc["T_det_all"]), "leakage": sh(acc["L_after_T"]),
                            "wrong_from_start": sh(acc["W_det"]),
                            "no_detected_event": sh(ne - acc["T_det_all"] - acc["L_after_T"] - acc["W_det"])},
            "transition_nullAci": sh(acc["T_det_Aci"]), "wrong_from_start_nullAci": sh(acc["W_det_Aci"]),
            "transition_interval_low_assumption_light": sh(acc["T_lo"]),
            "transition_raw_calls_nonleaked": sh(acc["n_t"]),
            "deconv_N_overlap": sh(acc["N_ov"]), "deconv_N_wrong_throughout": sh(acc["N_wr"]),
            "deconv_N_transition_sens_corrected": sh(acc["N_tr"]),
            "note": "Q4 leakage n/m (counted as 0)" if Q4 in qs else ""}


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--nsim", type=int, default=2000, help="synthetic sequences per (config, bit, type)")
    ap.add_argument("--ninj", type=int, default=60, help="injected sequences per (config, bit, tau)")
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
    assert not os.path.exists(f"{OUT}/em_e1hmm_v4_results.json"), "refusing to overwrite"
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
                    for v in ("p50", "lab_p50", "viterbi", "fut_p50"):
                        p = pool_boot(results, qs, wname, ename, kind, v, nq)
                        if p is not None:
                            pooled[f"{kind}|{wname}|{ename}|{nm}|{v}"] = p
    meta = json.load(open(f"{a.shared}/shared_meta.json"))
    out = {"script": "em_e1hmm_attrib_v4.py", "doc": __doc__, "args": vars(a), "meta_shared_v3": meta,
           "pooled": pooled, "per_qubit": results, "runtime_s": time.time() - t0}
    json.dump(v3.jsonable(out), open(f"{OUT}/em_e1hmm_v4_results.json", "w"), indent=1)
    print("all done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
