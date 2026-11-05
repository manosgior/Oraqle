"""em_e2seg_attr.py (tag e2seg): HMM-free, model-light error-mode attribution of the canonical MCMit-CNN errors.

Modes (each error gets exactly one primary mode):
  1. Leakage (non-computational): own line frame per neighbour configuration. Window-mean 2-D cores c0/c1 by
     Gaussian-kernel mean-shift (bandwidth 0.5 sigma) on TRAIN, |0>-|1> line through c0 and c1, z = signed
     perpendicular distance in units of the prepared-|0> perpendicular MAD width. z > 3 = flag, z < -3 = mirror.
     Q4 (q3): not measurable (multimodal |1>), mode skipped and reported n/m.
  2. Mid-readout transition: two-segment change-point likelihood-ratio test along the per-bin core line.
     Per configuration, per-bin core trajectories mu0(k), mu1(k) (featbox 100 ns boxcar bins, TRAIN, shots within
     1 sigma of the window-mean core), normalised coordinate x_k = (r_k - mu0_k).u_k / |mu1_k - mu0_k|
     (0 = |0> core, 1 = |1> core in every bin). Noise covariance C0 = Gnanadesikan-Kettenring robust covariance of
     prepared-|0> TRAIN shots. Step template: the qubit flips at t_s (10 ns grid), the resonator follows with the
     ring-up time constant tau_r fitted per qubit, weighted by the ring-up profile inside each bin.
     LLR = max_ts loglik(step) - max(loglik(const |0>), loglik(const |1>)).
     Relaxation (|1>->|0>) is tested for prepared-|1> errors, excitation (|0>->|1>) for prepared-|0> errors.
     Null (no transitions): per-shot CONDITIONAL stationary Gaussian null. Given the shot's GLS level m (the
     matched-filter statistic), x* = m 1 + (I - 1 w^T) eps, eps ~ N(0, C), w = C^-1 1 / 1^T C^-1 1. This is the exact
     distribution of a stationary (no-transition) Gaussian trace given its level, so it reproduces the selection
     of errors (overlap errors, deep wrong-cloud errors, intermediate stationary sub-clouds) without assuming
     where they sit. FP_i = P(LLR* > lambda | m_i); primary FP = max over C = C0 and C = C1 (prepared-|1> core
     covariance). FP-corrected count = N_flag - sum_i FP_i (conservative: summed over every shot of the pool).
     Data-driven check: wrong-direction flags (excitation-type steps on prepared-|1> shots) versus the null.
     Sensitivity: steps injected into real prepared-|0> core shots at known t_s.
  3. Residual (no detected event). Characterised independently by its matched-filter distance to the boundary
     in noise widths, against the per-configuration Gaussian-overlap prediction.
Attribution order A: 1 -> 2 -> 3 (primary), order B: 2 -> 1 -> 3; overlaps reported (mirror- and FP-corrected).
Windows: 1 us (featbox bins 0-9; leakage on feat1us bins 0-9, the CNN input), 600 ns (bins 0-5), 400 ns (0-3).
Robustness variants: feat1us (FIR) bins at 1 us, tau_r = 0 template, C0-only null, soft-1000 predictions.
Frames/trajectories/covariances fitted on TRAIN, everything evaluated on TEST.
Docker (graham, oraqle-torch): -v features:/feat:ro -v ~/oraqle_reports:/reports:ro -v f/in:/soft:ro -v em/e2seg:/work
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr

D, R, S, O = "/feat", "/reports", "/soft", "/work/out"
os.makedirs(O, exist_ok=True)
LAMS = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
NBANK = 4096
MGRID = np.round(np.arange(-1.5, 2.5001, 0.02), 4)
NCARRIER = 3000            # injection carriers per configuration
SEED = 20261104
DBINS = [-np.inf, 0, 1, 2, 3, np.inf]
NPROC = int(os.environ.get("E2SEG_NPROC", "20"))
FIR = np.array(json.load(open(f"{D}/calib.json"))["fir"]["taps"])


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def bits(y):
    return ((y[:, None] >> np.arange(5)) & 1).astype(np.int8)


def cfg_of(B, q):
    o = [k for k in range(5) if k != q]
    return (B[:, o] * (1 << np.arange(4))).sum(1)


def mad(x, axis=0):
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis)


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def meanshift(P, h, it=300):
    c = np.median(P, 0)
    for _ in range(it):
        w = np.exp(-np.sum((P - c) ** 2, 1) / (2 * h * h))
        cn = (w[:, None] * P).sum(0) / w.sum()
        if np.linalg.norm(cn - c) < 1e-7 * h:
            return cn
        c = cn
    return c


def gk_cov(X):
    s = mad(X, 0)
    Z = X / s
    K = X.shape[1]
    Rm = np.eye(K)
    for i in range(K):
        for j in range(i + 1, K):
            a = mad(Z[:, i] + Z[:, j]) ** 2
            d = mad(Z[:, i] - Z[:, j]) ** 2
            Rm[i, j] = Rm[j, i] = (a - d) / (a + d)
    C = Rm * np.outer(s, s)
    ev, V = np.linalg.eigh(C)
    ev = np.clip(ev, 1e-3 * ev.max(), None)
    return (V * ev) @ V.T


def wmean(src, split, b0, b1):
    F = np.load(f"{D}/{src}_{split}.npy", mmap_mode="r")
    out = np.empty((F.shape[0], 5, 2))
    for s in range(0, F.shape[0], 200000):
        out[s:s + 200000] = np.asarray(F[s:s + 200000, :, b0:b1, :], dtype=np.float64).mean(2)
    return out


# ---------------------------------------------------------------- leakage line frame (own implementation)
def fit_line(A, B):
    fr = {}
    for q in range(5):
        cf = cfg_of(B, q)
        for c in range(16):
            P0 = A[(cf == c) & (B[:, q] == 0), q]
            P1 = A[(cf == c) & (B[:, q] == 1), q]
            s = float(mad(P0[:, 1]))
            c0 = meanshift(P0, 0.5 * s)
            c1 = np.median(P1, 0) if q == 3 else meanshift(P1, 0.5 * s)
            u = (c1 - c0) / np.linalg.norm(c1 - c0)
            n = np.array([-u[1], u[0]])
            if n[1] < 0:
                n = -n
            sig = float(mad((P0 - c0) @ n))
            fr[(q, c)] = (c0, u, n, sig, float(np.linalg.norm(c1 - c0)))
    return fr


def zline(A, B, fr):
    z = np.empty((A.shape[0], 5))
    for q in range(5):
        cf = cfg_of(B, q)
        for c in range(16):
            m = cf == c
            c0, u, n, sig, sep = fr[(q, c)]
            z[m, q] = ((A[m, q] - c0) @ n) / sig
    return z


# ---------------------------------------------------------------- step templates
def ringup_profile(nsamp, d, tau):
    t = np.arange(nsamp) * 2.0 + 1.0
    return np.clip(1 - np.exp(-np.clip(t - d, 0, None) / max(tau, 1e-6)), 0, None)


def conv_same(x, taps):
    return np.convolve(x, taps, mode="same")


def make_templates(K, d, tau_ring, tau_r, fir, step_ns=10.0):
    """Down-step templates a(ts): fraction of '|1>-ness' per bin (1 before ts, resonator-smoothed decay after)."""
    ns = K * 50
    t = np.arange(ns) * 2.0 + 1.0
    Dt = ringup_profile(ns, d, tau_ring)
    den = conv_same(Dt, FIR) if fir else Dt
    den = den.reshape(K, 50).mean(1)
    ts_grid = np.arange(0.0, K * 100.0 + 0.01, step_ns)
    A = np.empty((len(ts_grid), K))
    for i, ts in enumerate(ts_grid):
        if tau_r > 0:
            s = np.where(t < ts, 1.0, np.exp(-(t - ts) / tau_r))
        else:
            s = (t < ts).astype(float)
        num = Dt * s
        if fir:
            num = conv_same(num, FIR)
        A[i] = num.reshape(K, 50).mean(1) / den
    return ts_grid, A


def scores(X, P, A):
    """LLR of the best down-step and best up-step against the best constant (|0> or |1>)."""
    c = P.sum()
    PX = X @ P
    base = np.maximum(0.0, PX.sum(1) - 0.5 * c)
    U = 1.0 - A
    qa = 0.5 * np.einsum("tk,kl,tl->t", A, P, A)
    qu = 0.5 * np.einsum("tk,kl,tl->t", U, P, U)
    Sd = PX @ A.T - qa
    Su = PX @ U.T - qu
    return Sd.max(1) - base, Sd.argmax(1), Su.max(1) - base, Su.argmax(1)


def null_table(P, w, C, A, rng):
    """FP(m, lambda) of the conditional stationary null for down and up steps."""
    K = len(w)
    L = np.linalg.cholesky(C + 1e-12 * np.eye(K))
    eps = rng.standard_normal((NBANK, K)) @ L.T
    E = eps - (eps @ w)[:, None]
    PE = E @ P
    c = P.sum()
    P1 = P.sum(1)
    U = 1.0 - A
    qa = 0.5 * np.einsum("tk,kl,tl->t", A, P, A)
    qu = 0.5 * np.einsum("tk,kl,tl->t", U, P, U)
    GA, HA = PE @ A.T - qa, A @ P1
    GU, HU = PE @ U.T - qu, U @ P1
    s1 = PE.sum(1)                                    # = 0 by construction
    fd = np.empty((len(MGRID), len(LAMS)))
    fu = np.empty((len(MGRID), len(LAMS)))
    lam = np.array(LAMS)
    for i, m in enumerate(MGRID):
        base = np.maximum(0.0, m * c + s1 - 0.5 * c)
        sd = (m * HA[None, :] + GA).max(1) - base
        su = (m * HU[None, :] + GU).max(1) - base
        fd[i] = (sd[:, None] > lam[None, :]).mean(0)
        fu[i] = (su[:, None] > lam[None, :]).mean(0)
    return fd, fu


def interp_fp(m, tab):
    out = np.empty((len(m), tab.shape[1]))
    for j in range(tab.shape[1]):
        out[:, j] = np.interp(m, MGRID, tab[:, j], left=tab[0, j], right=tab[-1, j])
    return out


def fit_ringup(Lk):
    """Bin-averaged A(1-exp(-(t-d)/tau))+ fitted to the raw |mu1-mu0| of bins 0-4."""
    nb = 5

    def model(p):
        A, d, tau = p
        return ringup_profile(nb * 50, d, tau).reshape(nb, 50).mean(1) * A

    p0 = [float(np.max(Lk[:nb])), 20.0, 60.0]
    r = least_squares(lambda p: model(p) - Lk[:nb], p0, bounds=([0, 0, 2.0], [np.inf, 99.0, 2000.0]))
    return {"A": float(r.x[0]), "d_ns": float(r.x[1]), "tau_ns": float(r.x[2]),
            "rms_rel": float(np.sqrt(np.mean((model(r.x) - Lk[:nb]) ** 2)) / r.x[0])}


# ---------------------------------------------------------------- per (variant, qubit) worker
G = {}


def worker(task):
    vname, q = task
    V = G["variants"][vname]
    K, src, win = V["K"], V["src"], V["win"]
    rng = np.random.default_rng(SEED + 100 * q + sum(map(ord, vname)))
    Btr, Bte = G["Btr"], G["Bte"]
    cft, cfe = cfg_of(Btr, q), cfg_of(Bte, q)
    Ftr = np.load(f"{D}/{src}_train.npy", mmap_mode="r")
    Fte = np.load(f"{D}/{src}_test.npy", mmap_mode="r")
    # trajectories always from the 1 us (10-bin) window of the same source
    Xtr = np.asarray(Ftr[:, q, :10, :], dtype=np.float64)
    Xte = np.asarray(Fte[:, q, :K, :], dtype=np.float64)
    zte = G["z"][win][:, q]
    out = {"variant": vname, "q": q, "K": K}
    # ---- core trajectories per configuration
    traj = {}
    Lk_all = []
    sel_frac = []
    for c in range(16):
        mu = {}
        Pw0 = Xtr[(cft == c) & (Btr[:, q] == 0)].mean(1)
        s = float(mad(Pw0[:, 1]))
        for b in (0, 1):
            Xb = Xtr[(cft == c) & (Btr[:, q] == b)]
            Pw = Xb.mean(1)
            cb = meanshift(Pw, 0.5 * s)
            sel = np.linalg.norm(Pw - cb, axis=1) < 1.0 * s
            sel_frac.append(float(sel.mean()))
            mu[b] = np.median(Xb[sel], 0)                # (10,2)
        traj[c] = mu
        Lk_all.append(np.linalg.norm(mu[1] - mu[0], axis=1))
    Lk_mean = np.mean(Lk_all, 0)
    rf = fit_ringup(Lk_mean)
    out["ringup"] = rf
    out["sep_per_bin_raw"] = Lk_mean.tolist()
    out["core_select_frac_median"] = float(np.median(sel_frac))
    tau_r = 0.0 if V.get("tau0") else rf["tau_ns"]
    ts_grid, A = make_templates(K, rf["d_ns"], rf["tau_ns"], tau_r, V["fir"])
    out["tau_r_used_ns"] = tau_r

    def norm(Xr, c):
        mu0, mu1 = traj[c][0][:K], traj[c][1][:K]
        Dl = mu1 - mu0
        Ln = np.linalg.norm(Dl, axis=1)
        u = Dl / Ln[:, None]
        return (((Xr - mu0) * u).sum(-1)) / Ln

    n = Xte.shape[0]
    mhat = np.empty(n)
    Sd = np.empty(n)
    Su = np.empty(n)
    td = np.empty(n)
    tu = np.empty(n)
    FPd = {k: np.empty((n, len(LAMS))) for k in ("C0", "C1")}
    FPu = {k: np.empty((n, len(LAMS))) for k in ("C0", "C1")}
    sig_m = np.empty((16, 2))
    sig_m0 = np.empty((16, 2))
    covinfo = []
    nullinfo = {}
    inj = {"down": {}, "up": {}}
    ts_inj = np.arange(0.0, K * 100.0 + 0.01, 25.0)
    KINDS = {"tau_fit": rf["tau_ns"], "tau2": 2 * rf["tau_ns"], "tau0": 0.0}   # injected response time
    inj_tmpl = {kind: make_templates(K, rf["d_ns"], rf["tau_ns"], tr, V["fir"], step_ns=25.0)[1] for kind, tr in KINDS.items()}
    inj_acc = {mode: {kind: np.zeros((len(ts_inj), 3)) for kind in KINDS}
               for mode in ("down", "up")}           # columns: n, misread, flagged&misread ; plus flagged below
    inj_flag = {mode: {kind: np.zeros(len(ts_inj)) for kind in KINDS} for mode in ("down", "up")}
    carrier_null = np.zeros(4)
    for c in range(16):
        mt0 = (cft == c) & (Btr[:, q] == 0)
        mt1 = (cft == c) & (Btr[:, q] == 1)
        x0 = norm(Xtr[mt0][:, :K], c)
        x1all = norm(Xtr[mt1][:, :K], c)
        C0 = gk_cov(x0)
        P = np.linalg.inv(C0)
        cc = P.sum()
        w = P.sum(1) / cc
        # prepared-|1> core covariance (shots within 1 sigma of the |1> window core)
        Pw0 = Xtr[mt0].mean(1)
        s = float(mad(Pw0[:, 1]))
        Pw1 = Xtr[mt1].mean(1)
        c1w = meanshift(Pw1, 0.5 * s)
        sel1 = np.linalg.norm(Pw1 - c1w, axis=1) < 1.0 * s
        C1 = gk_cov(x1all[sel1])
        m0tr, m1tr = x0 @ w, x1all @ w
        sig_m[c] = [float(mad(m0tr)), float(mad(m1tr))]
        sig_m0[c] = [float(np.sqrt(w @ C0 @ w)), float(np.sqrt(w @ C1 @ w))]
        r0 = x0 - m0tr[:, None]
        r1 = x1all[sel1] - m1tr[sel1][:, None]
        covinfo.append({"sig0_bin": np.sqrt(np.diag(C0)).tolist(), "sig1core_bin": np.sqrt(np.diag(C1)).tolist(),
                        "adjcorr0": (np.diag(C0, 1) / np.sqrt(np.diag(C0)[:-1] * np.diag(C0)[1:])).tolist() if K > 1 else [],
                        "resid_mad_ratio_1core_over_0": (mad(r1, 0) / mad(r0, 0)).tolist()})
        fd0, fu0 = null_table(P, w, C0, A, rng)
        fd1, fu1 = null_table(P, w, C1, A, rng)
        nullinfo[c] = {"fd0": fd0, "fu0": fu0, "fd1": fd1, "fu1": fu1}
        me = cfe == c
        xe = norm(Xte[me], c)
        mhat[me] = xe @ w
        a, b_, d_, e_ = scores(xe, P, A)
        Sd[me], td[me], Su[me], tu[me] = a, ts_grid[b_], d_, ts_grid[e_]
        FPd["C0"][me] = interp_fp(mhat[me], fd0)
        FPd["C1"][me] = interp_fp(mhat[me], fd1)
        FPu["C0"][me] = interp_fp(mhat[me], fu0)
        FPu["C1"][me] = interp_fp(mhat[me], fu1)
        # ---- injection into real prepared-|0> core shots (TEST)
        lam0 = G["lam0"]
        j0 = LAMS.index(lam0)
        cand = np.where(me & (Bte[:, q] == 0))[0]
        mc = mhat[cand]
        ok = (np.abs(mc) < 2 * sig_m[c, 0]) & (np.abs(zte[cand]) < 2)
        cand = cand[ok]
        pick = rng.choice(len(cand), size=min(NCARRIER, len(cand)), replace=False)
        xc = norm(Xte[cand[pick]], c)
        a, _, d_, _ = scores(xc, P, A)
        carrier_null += [len(xc), (a > lam0).sum(), (d_ > lam0).sum(),
                         interp_fp(xc @ w, fd0)[:, j0].sum()]
        for kind, Ai_ in inj_tmpl.items():
            for i, ts in enumerate(ts_inj):
                ai = Ai_[i]
                xd = xc + ai[None, :]                     # |1> -> |0> at ts
                md = xd @ w
                sd_, _, _, _ = scores(xd, P, A)
                mis = md < 0.5
                fl = sd_ > lam0
                inj_acc["down"][kind][i] += [len(xd), mis.sum(), (fl & mis).sum()]
                inj_flag["down"][kind][i] += fl.sum()
                xu = xc + (1.0 - ai)[None, :]             # |0> -> |1> at ts
                mu_ = xu @ w
                _, _, su_, _ = scores(xu, P, A)
                mis = mu_ > 0.5
                fl = su_ > lam0
                inj_acc["up"][kind][i] += [len(xu), mis.sum(), (fl & mis).sum()]
                inj_flag["up"][kind][i] += fl.sum()
    out["sig_m_mad"] = sig_m.tolist()
    out["sig_m_noise"] = sig_m0.tolist()
    out["cov"] = {"sig0_bin_median": np.median([ci["sig0_bin"] for ci in covinfo], 0).tolist(),
                  "sig1core_bin_median": np.median([ci["sig1core_bin"] for ci in covinfo], 0).tolist(),
                  "adjcorr0_median": np.median([ci["adjcorr0"] for ci in covinfo], 0).tolist() if K > 1 else [],
                  "resid_ratio_1core_over_0_median": np.median([ci["resid_mad_ratio_1core_over_0"] for ci in covinfo], 0).tolist()}
    # null summary: FP at selected levels, config-averaged
    lev = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
    ns_ = {}
    for key in ("fd0", "fu0", "fd1", "fu1"):
        T_ = np.mean([nullinfo[c][key] for c in range(16)], 0)
        ns_[key] = {str(l): dict(zip(map(str, LAMS), T_[int(np.argmin(np.abs(MGRID - l)))].round(5).tolist())) for l in lev}
        ns_[key]["max_0.3_0.7"] = dict(zip(map(str, LAMS), T_[(MGRID >= 0.3) & (MGRID <= 0.7)].max(0).round(5).tolist()))
    out["null_fp"] = ns_
    out["carrier_null"] = {"n": int(carrier_null[0]), "flag_down": int(carrier_null[1]), "flag_up": int(carrier_null[2]),
                           "expected_down_C0": float(carrier_null[3])}
    sens = {}
    for mode in ("down", "up"):
        sens[mode] = {}
        for kind in inj_acc[mode]:
            Aac = inj_acc[mode][kind]
            pf = inj_flag[mode][kind] / Aac[:, 0]
            pm = Aac[:, 1] / Aac[:, 0]
            pfm = np.where(Aac[:, 1] > 0, Aac[:, 2] / np.maximum(Aac[:, 1], 1), np.nan)
            inwin = ts_inj < K * 100.0
            eps_bar = float(Aac[inwin, 2].sum() / max(Aac[inwin, 1].sum(), 1))
            sens[mode][kind] = {"ts_ns": ts_inj.tolist(), "p_flag": pf.round(4).tolist(), "p_misread": pm.round(4).tolist(),
                                "p_flag_given_misread": [None if np.isnan(v) else round(float(v), 4) for v in pfm],
                                "eps_bar_error_causing_uniform_ts": eps_bar,
                                "n_per_ts": int(Aac[0, 0])}
    out["sensitivity"] = sens
    # ---- calibration check of the null on real data (all TEST shots)
    cal = {}
    for b, dname, S_, FP_ in ((1, "up_on_prep1", Su, FPu), (0, "down_on_prep0", Sd, FPd),
                              (1, "down_on_prep1", Sd, FPd), (0, "up_on_prep0", Su, FPu)):
        m = Bte[:, q] == b
        cal[dname] = {str(l): {"observed": int((S_[m] > l).sum()), "expected_C0": float(FP_["C0"][m, j].sum()),
                               "expected_C1": float(FP_["C1"][m, j].sum()),
                               "expected_max": float(np.maximum(FP_["C0"][m, j], FP_["C1"][m, j]).sum())}
                      for j, l in enumerate(LAMS)}
    out["calibration_all_shots"] = cal
    # ---- attribution for each prediction set
    out["attr"] = {}
    for pname, pred in G["preds"][V["pred_win"]].items():
        out["attr"][pname] = attribute(q, pred[:, q], Bte[:, q], zte, mhat, Sd, Su, td, tu, FPd, FPu, sig_m, cfe, V)
    # per-shot store for errors of the primary prediction set
    pname0 = list(G["preds"][V["pred_win"]].keys())[0]
    err = G["preds"][V["pred_win"]][pname0][:, q] != Bte[:, q]
    j0 = LAMS.index(G["lam0"])
    np.savez_compressed(f"{O}/em_e2seg_shots_{vname}_q{q}.npz", idx=np.where(err)[0].astype(np.int32),
                        z=zte[err].astype(np.float32), mhat=mhat[err].astype(np.float32),
                        Sd=Sd[err].astype(np.float32), Su=Su[err].astype(np.float32),
                        td=td[err].astype(np.float32), tu=tu[err].astype(np.float32),
                        FPd=np.maximum(FPd["C0"][err, j0], FPd["C1"][err, j0]).astype(np.float32),
                        FPu=np.maximum(FPu["C0"][err, j0], FPu["C1"][err, j0]).astype(np.float32),
                        prep=Bte[err, q].astype(np.int8))
    log("done", vname, q)
    return out


def attribute(q, pred, prep, z, mhat, Sd, Su, td, tu, FPd, FPu, sig_m, cf, V):
    res = {}
    K = V["K"]
    for lam_i, lam in enumerate(LAMS):
        for nullname in ("max", "C0"):
            if nullname == "C0" and lam != G["lam0"]:
                continue
            key = f"lam{lam:g}_{nullname}"
            R_ = {}
            for b in (0, 1):
                m = prep == b
                nb = int(m.sum())
                E = m & (pred != prep)
                N = int(E.sum())
                F = (Sd > lam) if b == 1 else (Su > lam)
                FPt = FPd if b == 1 else FPu
                FP = np.maximum(FPt["C0"][:, lam_i], FPt["C1"][:, lam_i]) if nullname == "max" else FPt["C0"][:, lam_i]
                if q == 3:
                    Pz = np.zeros_like(E)
                    Mz = np.zeros_like(E)
                else:
                    Pz = E & (z > 3)
                    Mz = E & (z < -3)
                nP, nM = int(Pz.sum()), int(Mz.sum())
                notP = E & ~Pz
                fpool = int((F & notP).sum() + (F & Mz).sum())
                fppool = float(FP[notP].sum() + FP[Mz].sum())
                Lc = nP - nM
                Tc = fpool - fppool
                Rc = N - Lc - Tc
                fall = int((F & E).sum())
                fpall = float(FP[E].sum())
                TcB = fall - fpall
                LcB = int((Pz & ~F).sum()) - int((Mz & ~F).sum()) + float(FP[Pz].sum() - FP[Mz].sum())
                RcB = N - TcB - LcB
                Ov = int((Pz & F).sum()) - int((Mz & F).sum()) - float(FP[Pz].sum() - FP[Mz].sum())
                d = {"n_shots": nb, "n_err": N, "err_rate": N / nb, "err_rate_ci95": wilson(N, nb),
                     "leak_plus": nP, "leak_minus": nM, "flag_pool": fpool, "fp_pool": fppool,
                     "flag_all": fall, "fp_all": fpall,
                     "A": {"leak": Lc, "trans": Tc, "resid": Rc},
                     "B": {"trans": TcB, "leak": LcB, "resid": RcB},
                     "overlap_leak_trans": Ov}
                # residual characterisation (order A residual shots: errors, not leak-flagged, not transition-flagged)
                if lam == G["lam0"] and nullname == "max":
                    Rs = notP & ~F
                    sg = sig_m[cf, b]
                    dist = ((0.5 - mhat) if b == 1 else (mhat - 0.5)) / sg
                    hR = np.histogram(dist[Rs], bins=DBINS)[0].tolist()
                    hE = np.histogram(dist[E], bins=DBINS)[0].tolist()
                    # MF (matched filter) errors on all shots of this prepared bit
                    MFerr = m & (((mhat < 0.5) if b == 1 else (mhat > 0.5)))
                    hMF = np.histogram(dist[MFerr], bins=DBINS)[0].tolist()
                    # Gaussian-overlap prediction per configuration
                    pred_bins = np.zeros(len(DBINS) - 1)
                    pred_tot = 0.0
                    for c in range(16):
                        nc = int((m & (cf == c)).sum())
                        a = 0.5 / sig_m[c, b]
                        pred_tot += nc * ndtr(-a)
                        edges = [0, 1, 2, 3, np.inf]
                        for j in range(4):
                            pred_bins[j + 1] += nc * (ndtr(-a - edges[j]) - ndtr(-a - edges[j + 1]))
                    d["resid_char"] = {"dist_bins": ["<0 (MF reads correct)", "[0,1)", "[1,2)", "[2,3)", ">=3"],
                                       "resid_hist": hR, "all_err_hist": hE, "mf_err_hist": hMF,
                                       "gauss_pred_mf_err_hist": pred_bins.round(1).tolist(),
                                       "gauss_pred_mf_err_total": float(pred_tot), "mf_err_total": int(MFerr.sum()),
                                       "n_resid_shots": int(Rs.sum())}
                    tsel = (td if b == 1 else tu)[F & E]
                    d["flag_time_hist_100ns"] = np.histogram(tsel, bins=np.arange(0, K * 100 + 1, 100))[0].tolist()
                    # wrong-direction flags among errors (direction mirror) and their null expectation
                    Fw = (Su > lam) if b == 1 else (Sd > lam)
                    FPw = FPu if b == 1 else FPd
                    d["wrong_direction"] = {"observed": int((Fw & E).sum()),
                                            "expected_max": float(np.maximum(FPw["C0"][E, lam_i], FPw["C1"][E, lam_i]).sum()),
                                            "expected_C0": float(FPw["C0"][E, lam_i].sum())}
                    # hidden leakage
                    if q != 3:
                        Pa, Ma = m & (z > 3), m & (z < -3)
                        d["leak_all"] = {"plus": int(Pa.sum()), "minus": int(Ma.sum()),
                                         "E3": (int(Pa.sum()) - int(Ma.sum())) / nb}
                R_[f"prep{b}"] = d
            res[key] = R_
    return res


def main():
    t0 = time.time()
    ytr, yte = np.load(D + "/y_train.npy"), np.load(D + "/y_test.npy")
    Btr, Bte = bits(ytr), bits(yte)
    G["Btr"], G["Bte"] = Btr, Bte
    # leakage frames for each window
    LW = {"1us": ("feat1us", 0, 10), "600ns": ("featbox", 0, 6), "400ns": ("featbox", 0, 4)}
    G["z"] = {}
    frames = {}
    for wn, (src, b0, b1) in LW.items():
        Atr, Ate = wmean(src, "train", b0, b1), wmean(src, "test", b0, b1)
        fr = fit_line(Atr, Btr)
        G["z"][wn] = zline(Ate, Bte, fr)
        frames[wn] = {f"q{q}": {"sigma_perp_median": float(np.median([fr[(q, c)][3] for c in range(16)])),
                                "angle_deg": [float(np.degrees(np.arctan2(fr[(q, c)][1][1], fr[(q, c)][1][0]))) for c in range(16)]}
                      for q in range(5)}
        log("frame", wn)
    pb = np.load(R + "/preds_MCMitCNN_pb.npy").astype(np.int8)
    tb = np.load(R + "/preds_MCMitCNN_tb.npy").astype(np.int8)
    soft = {T: np.load(f"{S}/cnn_soft_{T}ns.npz") for T in (1000, 600, 400)}
    chk = {"canon_tb_equals_prepared": bool(np.array_equal(tb, Bte))}
    G["preds"] = {"1us": {"canonical": pb, "soft1000": (soft[1000]["prob"] > 0.5).astype(np.int8)},
                  "600ns": {"soft600": (soft[600]["prob"] > 0.5).astype(np.int8)},
                  "400ns": {"soft400": (soft[400]["prob"] > 0.5).astype(np.int8)}}
    for T in (1000, 600, 400):
        chk[f"soft{T}_label_equals_prepared"] = bool(np.array_equal(soft[T]["label"].astype(np.int8), Bte))
        e = ((soft[T]["prob"] > 0.5).astype(np.int8) != Bte).mean(0)
        chk[f"soft{T}_err_per_qubit"] = e.tolist()
        chk[f"soft{T}_err_pooled_Q1Q3Q4Q5"] = float(e[[0, 2, 3, 4]].mean())
    chk["canonical_err_per_qubit"] = (pb != Bte).mean(0).tolist()
    G["variants"] = {
        "box1us": {"K": 10, "src": "featbox", "win": "1us", "pred_win": "1us", "fir": False},
        "box600": {"K": 6, "src": "featbox", "win": "600ns", "pred_win": "600ns", "fir": False},
        "box400": {"K": 4, "src": "featbox", "win": "400ns", "pred_win": "400ns", "fir": False},
        "fir1us": {"K": 10, "src": "feat1us", "win": "1us", "pred_win": "1us", "fir": True},
        "box1us_tau0": {"K": 10, "src": "featbox", "win": "1us", "pred_win": "1us", "fir": False, "tau0": True},
    }
    G["lam0"] = float(os.environ.get("E2SEG_LAMBDA", "4.0"))
    only = os.environ.get("E2SEG_VARIANTS")
    vlist = only.split(",") if only else list(G["variants"])
    tasks = [(v, q) for v in vlist for q in range(5)]
    with Pool(min(NPROC, len(tasks))) as pool:
        outs = pool.map(worker, tasks)
    res = {"note": __doc__, "lam0": G["lam0"], "lams": LAMS, "checks": chk, "frames": frames, "variants": {}}
    for o in outs:
        res["variants"].setdefault(o["variant"], {})[f"q{o['q']}"] = o
    tag = os.environ.get("E2SEG_TAG", "")
    json.dump(res, open(f"{O}/em_e2seg_raw{tag}.json", "w"), indent=1, default=float)
    log("all done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
