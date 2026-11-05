"""em_e2seg_attr_v3.py (tag e2seg, v3): HMM-free, model-light error-mode attribution of the canonical MCMit-CNN errors.

v3 = v2 + one bug fix + diagnostics (v2 file and outputs kept untouched).
  BUG in v2: the per-configuration |b> "core" shots (which define the per-bin trajectories mu_b(k), the noise
  covariances C_b and hence the null) were selected around the densest mode of the LATE window (1-2 us) of prepared-|b>
  TRAIN shots. For Q3 (q2) with Q1 = Q5 = 1 (configs 9, 11, 13, 15) readout-induced decay/leakage is fastest, the
  late-window prepared-|1> mode is not the |1> cloud, mu_1 collapsed (sigma_m1 0.83-1.0 vs 0.15) and the detector
  produced 0 flags in those configs while subtracting their FP.
  FIX: the late-window mean-shift is SEEDED at the late-window median of shots that sit in the |b> core of the
  1 us window (|A_1us - c_b| < 1 sigma, c_b = 1 us mean-shift mode); the selection itself stays on the late window
  (independent noise, no truncation of the analysis-window noise). Per-config diagnostics (seed shift, n_core,
  sigma_m) are written out; a config whose sigma_m exceeds 1.5x the qubit median is reported.
  ADDED: FP under the prepared-state branch only (set "alpha0.01_prepbranch"), a less conservative sensitivity
  variant of the max-over-branches FP.
v2 notes: exact conditional null for the |1>
branch, m-calibrated alpha-level thresholds validated on an independent null bank, MC variance of the FP sum,
selection-aware null with a linear CNN proxy, injection efficiency, Gaussian-overlap characterisation.

Modes (each error gets exactly one primary mode):
  1. Leakage (non-computational). Own line frame per qubit and neighbour configuration (16 per qubit):
     window-mean 2-D cores c0/c1 by Gaussian-kernel mean-shift (bandwidth 0.5 sigma) on TRAIN, |0>-|1> line through
     c0 and c1, z = signed perpendicular distance in units of the prepared-|0> perpendicular MAD width.
     z > 3 = flag, z < -3 = mirror (subtracted). Q4 (q3): not measurable (multimodal |1>), reported n/m.
  2. Mid-readout transition: two-segment change-point likelihood-ratio test along the PER-BIN core line.
     Per configuration, per-bin core trajectories mu0(k), mu1(k) (featbox 100 ns boxcar bins, TRAIN, shots within
     1 sigma of the 1 us window-mean core, per-bin median). Normalised coordinate x_k = (r_k - mu0_k).u_k/|mu1_k - mu0_k|
     (0 = |0> core, 1 = |1> core in every bin; the mean-aligned shared rotation is never used).
     Noise: C0 = Gnanadesikan-Kettenring robust covariance of prepared-|0> TRAIN x, C1 = same for prepared-|1> core
     shots. GLS weights w = C0^-1 1 / 1'C0^-1 1, level m = w'x (matched filter).
     Step template a_ts: the qubit flips at t_s (10 ns grid), the resonator follows with the ring-up time constant
     tau_r fitted per qubit (bin-averaged, weighted by the ring-up profile).
     LLR_down = max_ts l(a_ts) - max(l(0), l(1));  LLR_up uses 1 - a_ts.
     Relaxation (|1>->|0>) is tested on prepared-|1> errors, excitation (|0>->|1>) on prepared-|0> errors.
     Null H0(b): shot stationary in state b with Gaussian noise C_b. Exact conditional given m: x = b1 + (m-b) g_b + r_b,
     g_b = C_b w / w'C_b w, r_b = e - (w'e) g_b, e ~ N(0, C_b). Thresholds lambda_alpha(m) = max_b (1-alpha)-quantile on a
     threshold bank; FP_i = max_b P(LLR > lambda_alpha(m_i) | m_i, H0(b)) measured on an INDEPENDENT validation bank.
     FP-corrected count = N_flag - sum_i FP_i over every shot of the pool (conservative).
     Selection-aware null (check): the same validation bank restricted to null traces that a linear proxy of the CNN
     (least squares of the soft-model logit on x, per configuration) misreads.
     Sensitivity: steps injected into real prepared-|0> core TEST shots at known t_s, three response times.
  3. Residual = no detected event. Characterised independently by the matched-filter distance to the boundary
     (m = 0.5) in noise widths sigma_m = sqrt(w'C_b w), against the per-configuration Gaussian-overlap prediction.
Attribution order A: 1 -> 2 -> 3, order B: 2 -> 1 -> 3; overlaps reported (mirror- and FP-corrected).
Windows: 1 us (featbox bins 0-9; leakage on feat1us bins 0-9 = the CNN input), 600 ns (bins 0-5), 400 ns (0-3).
Frames/trajectories/covariances fitted on TRAIN, everything evaluated on TEST.
Docker: -v features:/feat:ro -v ~/oraqle_reports:/reports:ro -v f/in:/soft:ro -v em/e2seg:/work
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr

D, R, S, O = str(LP.FEAT), str(LP.REPORTS), str(LP.SOFT), str(LP.stage("em/e2seg") / "out")
os.makedirs(O, exist_ok=True)
ALPHAS = [0.05, 0.01, 0.002]
ALPHA0 = 0.01
LAMS = [4.0, 6.0, 8.0]
NB = int(os.environ.get("E2SEG_NB", "16384"))
MGRID = np.round(np.arange(-1.5, 2.5001, 0.02), 4)
MMID = (MGRID[:-1] + MGRID[1:]) / 2
NCARRIER = int(os.environ.get("E2SEG_NCARRIER", "2000"))
SEED = 20261104
DEDGES = [0, 1, 2, 3, np.inf]
NPROC = int(os.environ.get("E2SEG_NPROC", "25"))
LATE_R = 1.5               # late-window core radius in units of the |0> late-window MAD
EPS = 1e-6
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


def meanshift(P, h, it=300, seed=None):
    c = np.median(P, 0) if seed is None else np.asarray(seed, dtype=np.float64)
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
    """Down-step templates a(ts): fraction of |1>-ness per bin (1 before ts, resonator-smoothed decay after)."""
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


def fit_ringup(Lk, fir):
    nb = 5

    def model(p):
        A_, d, tau = p
        prof = ringup_profile(500, d, tau)
        if fir:
            prof = conv_same(prof, FIR)
        return prof[:nb * 50].reshape(nb, 50).mean(1) * A_

    p0 = [float(np.max(Lk[:nb])), 20.0, 60.0]
    r = least_squares(lambda p: model(p) - Lk[:nb], p0, bounds=([0, 0, 2.0], [np.inf, 99.0, 2000.0]))
    return {"A": float(r.x[0]), "d_ns": float(r.x[1]), "tau_ns": float(r.x[2]),
            "rms_rel": float(np.sqrt(np.mean((model(r.x) - Lk[:nb]) ** 2)) / r.x[0])}


class Det:
    """Two-segment LLR detector for one configuration."""

    def __init__(self, P, A):
        self.P, self.A, self.U = P, A, 1.0 - A
        self.c = P.sum()
        self.qa = 0.5 * np.einsum("tk,kl,tl->t", A, P, A)
        self.qu = 0.5 * np.einsum("tk,kl,tl->t", self.U, P, self.U)

    def scores(self, X):
        PX = X @ self.P
        base = np.maximum(0.0, PX.sum(1) - 0.5 * self.c)
        Sd = PX @ self.A.T - self.qa
        Su = PX @ self.U.T - self.qu
        return np.maximum(Sd.max(1) - base, 0.0), Sd.argmax(1), np.maximum(Su.max(1) - base, 0.0), Su.argmax(1)

    def null_llr(self, w, C, b, mvals, rng, n, proxy=None):
        """LLR samples (n, len(mvals)) under H0(b) given m, for down and up; optional proxy scores."""
        K = len(w)
        L = np.linalg.cholesky(C + 1e-12 * np.eye(K))
        e = rng.standard_normal((n, K)) @ L.T
        g = C @ w / (w @ C @ w)
        r = e - (e @ w)[:, None] * g[None, :]
        Pr = r @ self.P
        P1, Pg = self.P.sum(1), self.P @ g
        GA, GU = Pr @ self.A.T - self.qa, Pr @ self.U.T - self.qu
        HA, HGA = self.A @ P1, self.A @ Pg
        HU, HGU = self.U @ P1, self.U @ Pg
        Ld = np.empty((n, len(mvals)), np.float32)
        Lu = np.empty((n, len(mvals)), np.float32)
        Sp = None
        if proxy is not None:
            v, v0 = proxy
            vr = r @ v
            Sp = np.empty((n, len(mvals)), np.float32)
        for i, m in enumerate(mvals):
            base = max(0.0, self.c * (m - 0.5))
            Ld[:, i] = np.maximum((b * HA + (m - b) * HGA + GA).max(1) - base, 0.0)
            Lu[:, i] = np.maximum((b * HU + (m - b) * HGU + GU).max(1) - base, 0.0)
            if proxy is not None:
                Sp[:, i] = b * v.sum() + (m - b) * (v @ g) + vr + v0
        return Ld, Lu, Sp


def interp_cols(m, grid, tab):
    out = np.empty((len(m), tab.shape[1]))
    for j in range(tab.shape[1]):
        out[:, j] = np.interp(m, grid, tab[:, j])
    return out


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
    Xtr = np.asarray(Ftr[:, q, :10, :], dtype=np.float64)
    Xte = np.asarray(Fte[:, q, :K, :], dtype=np.float64)
    zte = G["z"][win][:, q]
    soft = G["soft"][V["soft"]][:, q].astype(np.float64)
    out = {"variant": vname, "q": q, "K": K}
    # ---- core shots selected on the INDEPENDENT late window (featbox bins 10-19 = samples 500-999), so the
    #      analysis-window noise is not truncated by the selection
    Btr_late = np.asarray(np.load(f"{D}/featbox_train.npy", mmap_mode="r")[:, q, 10:20, :], dtype=np.float64).mean(1)
    Bte_late = np.asarray(np.load(f"{D}/featbox_test.npy", mmap_mode="r")[:, q, 10:20, :], dtype=np.float64).mean(1)
    traj, Lk_all, sel_frac, c1info, coresel, latecore = {}, [], [], [], {}, {}
    A1tr = Xtr.mean(1)                                       # 1 us window mean (bins 0-9), TRAIN, used only to SEED
    seeddiag = []
    for c in range(16):
        mu = {}
        s_late = float(mad(Btr_late[(cft == c) & (Btr[:, q] == 0), 1]))
        s_e = float(mad(A1tr[(cft == c) & (Btr[:, q] == 0), 1]))
        for b in (0, 1):
            mb = np.where((cft == c) & (Btr[:, q] == b))[0]
            Pl = Btr_late[mb]
            ce = meanshift(A1tr[mb], 0.5 * s_e)                      # |b> mode of the 1 us window
            ecore = np.linalg.norm(A1tr[mb] - ce, axis=1) < 1.0 * s_e
            seed = np.median(Pl[ecore], 0)
            cb = meanshift(Pl, 0.5 * s_late, seed=seed)
            cb_v2 = meanshift(Pl, 0.5 * s_late)                      # v2 choice, for the diagnostic only
            sel = np.linalg.norm(Pl - cb, axis=1) < LATE_R * s_late
            sel_frac.append(float(sel.mean()))
            coresel[(c, b)] = mb[sel]
            latecore[(c, b)] = (cb, s_late)
            mu[b] = np.median(Xtr[mb[sel]], 0)
            seeddiag.append({"c": c, "b": b, "n_core": int(sel.sum()), "frac_core": float(sel.mean()),
                             "frac_early_core": float(ecore.mean()),
                             "seed_to_core_over_s": float(np.linalg.norm(seed - cb) / s_late),
                             "v2core_to_v3core_over_s": float(np.linalg.norm(cb_v2 - cb) / s_late)})
        traj[c] = mu
        Lk_all.append(np.linalg.norm(mu[1] - mu[0], axis=1))
    out["seed_diag"] = seeddiag
    Lk_mean = np.mean(Lk_all, 0)
    rf = fit_ringup(Lk_mean, V["fir"])
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
    na = len(ALPHAS)
    mhat = np.empty(n)
    sproxy = np.empty(n)
    Sd, Su, td, tu = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    thrd, thru = np.empty((n, na)), np.empty((n, na))
    FPd, FPu = np.empty((n, na)), np.empty((n, na))           # max-branch validated FP at alpha thresholds
    FPdS, FPuS = np.empty(n), np.empty(n)                       # selection-aware FP at ALPHA0
    FPdP, FPuP = np.empty(n), np.empty(n)                       # prepared-branch-only FP at ALPHA0 (v3)
    FPdL, FPuL = np.empty((n, len(LAMS))), np.empty((n, len(LAMS)))
    VARd, VARu = np.empty((n, na)), np.empty((n, na))           # per-shot MC variance of its FP estimate
    VARdL, VARuL = np.empty((n, len(LAMS))), np.empty((n, len(LAMS)))
    cell = np.empty(n, np.int64)
    sig_mb = np.empty((16, 2))
    covinfo, proxyinfo = [], []
    sig_s = np.empty((16, 2))
    s_lev = np.empty((16, 2))
    thr_tab = {}
    ts_inj = np.arange(0.0, K * 100.0 + 0.01, 25.0)
    KINDS = {"tau_fit": rf["tau_ns"], "tau2": 2 * rf["tau_ns"], "tau0": 0.0}
    inj_tmpl = {kind: make_templates(K, rf["d_ns"], rf["tau_ns"], tr, V["fir"], step_ns=25.0)[1] for kind, tr in KINDS.items()}
    j0 = ALPHAS.index(ALPHA0)
    inj = {mode: {kind: np.zeros((len(ts_inj), 6)) for kind in KINDS} for mode in ("down", "up")}
    inj_a = {mode: np.zeros((na, 4)) for mode in ("down", "up")}   # tau_fit, ts uniform in window: n_mf_mis, flag&mf_mis, n_px_mis, flag&px_mis
    # columns: n, flag, MF-misread, flag&MF-misread, proxy-misread, flag&proxy-misread
    carrier = np.zeros(4)
    jj = 0
    for c in range(16):
        x0 = norm(Xtr[coresel[(c, 0)]][:, :K], c)
        x1 = norm(Xtr[coresel[(c, 1)]][:, :K], c)
        C0 = gk_cov(x0)
        P = np.linalg.inv(C0)
        cc = P.sum()
        w = P.sum(1) / cc
        C1 = gk_cov(x1)
        Cb = {0: C0, 1: C1}
        sig_mb[c] = [float(np.sqrt(w @ C0 @ w)), float(np.sqrt(w @ C1 @ w))]
        r0 = x0 - (x0 @ w)[:, None]
        r1 = x1 - (x1 @ w)[:, None]
        covinfo.append({"sig0_bin": np.sqrt(np.diag(C0)).tolist(), "sig1core_bin": np.sqrt(np.diag(C1)).tolist(),
                        "adjcorr0": (np.diag(C0, 1) / np.sqrt(np.diag(C0)[:-1] * np.diag(C0)[1:])).tolist() if K > 1 else [],
                        "resid_mad_ratio_1core_over_0": (mad(r1, 0) / mad(r0, 0)).tolist(),
                        "sig_m": sig_mb[c].tolist(),
                        "m0_train_mad": float(mad(x0 @ w)), "m1core_train_mad": float(mad(x1 @ w)),
                        "n_core": [int(len(x0)), int(len(x1))]})
        det = Det(P, A)
        me = cfe == c
        xe = norm(Xte[me], c)
        mhat[me] = xe @ w
        a_, b_, d_, e_ = det.scores(xe)
        Sd[me], td[me], Su[me], tu[me] = a_, ts_grid[b_], d_, ts_grid[e_]
        # ---- linear proxy of the CNN (soft-model logit on x), fitted on near-boundary TEST shots of this config
        p = np.clip(soft[me], 1e-4, 1 - 1e-4)
        nb_ = (p > 0.02) & (p < 0.98)
        lg = np.log(p / (1 - p))
        Xd = np.c_[xe[nb_], np.ones(nb_.sum())]
        coef = np.linalg.lstsq(Xd, lg[nb_], rcond=None)[0]
        v, v0 = coef[:-1], coef[-1]
        sp = xe @ v + v0
        sproxy[me] = sp
        sig_s[c] = [float(np.sqrt(v @ C0 @ v)), float(np.sqrt(v @ C1 @ v))]
        s_lev[c] = [float(v0), float(v.sum() + v0)]
        agree = float(((sp > 0) == (soft[me] > 0.5)).mean())
        proxyinfo.append({"n_fit": int(nb_.sum()), "agree_with_soft_hard": agree,
                          "v_over_w_norm": (v / v.sum()).tolist() if abs(v.sum()) > 1e-9 else None,
                          "boundary_m_proxy": float((-v0) / v.sum()) if abs(v.sum()) > 1e-9 else None})
        # ---- null: thresholds bank (grid) then independent validation bank (midpoints)
        thr = {}
        for b in (0, 1):
            Ld, Lu, _ = det.null_llr(w, Cb[b], b, MGRID, rng, NB)
            thr[b] = (np.quantile(Ld, [1 - a for a in ALPHAS], axis=0).T, np.quantile(Lu, [1 - a for a in ALPHAS], axis=0).T)
            del Ld, Lu
        tcd = np.maximum(np.maximum(thr[0][0], thr[1][0]), 0.0) + EPS          # (M, na)
        tcu = np.maximum(np.maximum(thr[0][1], thr[1][1]), 0.0) + EPS
        thr_tab[c] = {"down": tcd, "up": tcu}
        tcd_mid = interp_cols(MMID, MGRID, tcd)
        tcu_mid = interp_cols(MMID, MGRID, tcu)
        fpd = np.zeros((2, len(MMID), na))
        fpu = np.zeros((2, len(MMID), na))
        fpdS = np.zeros((2, len(MMID)))
        fpuS = np.zeros((2, len(MMID)))
        fpdL = np.zeros((2, len(MMID), len(LAMS)))
        fpuL = np.zeros((2, len(MMID), len(LAMS)))
        nsel = np.zeros((2, len(MMID), 2))
        for b in (0, 1):
            Ld, Lu, Sp = det.null_llr(w, Cb[b], b, MMID, rng, NB, proxy=(v, v0))
            for k in range(na):
                fpd[b, :, k] = (Ld > tcd_mid[None, :, k]).mean(0)
                fpu[b, :, k] = (Lu > tcu_mid[None, :, k]).mean(0)
            for k, lam in enumerate(LAMS):
                fpdL[b, :, k] = (Ld > lam).mean(0)
                fpuL[b, :, k] = (Lu > lam).mean(0)
            s1 = Sp < 0                                   # proxy reads 0: relevant for prepared-|1> errors (down test)
            s0 = Sp > 0                                   # proxy reads 1: prepared-|0> errors (up test)
            n1, n0 = s1.sum(0), s0.sum(0)
            nsel[b, :, 0], nsel[b, :, 1] = n0, n1
            fd_s = ((Ld > tcd_mid[None, :, j0]) & s1).sum(0) / np.maximum(n1, 1)
            fu_s = ((Lu > tcu_mid[None, :, j0]) & s0).sum(0) / np.maximum(n0, 1)
            fpdS[b] = np.where(n1 >= 100, fd_s, fpd[b, :, j0])
            fpuS[b] = np.where(n0 >= 100, fu_s, fpu[b, :, j0])
            del Ld, Lu, Sp
        mm = mhat[me]
        thrd[me] = interp_cols(mm, MGRID, tcd)
        thru[me] = interp_cols(mm, MGRID, tcu)
        ci_ = np.clip(np.round((mm - MMID[0]) / 0.02).astype(int), 0, len(MMID) - 1)
        cell[me] = c * 1000 + ci_
        for arr_fp, arr_var, tab in ((FPd, VARd, fpd), (FPu, VARu, fpu), (FPdL, VARdL, fpdL), (FPuL, VARuL, fpuL)):
            f0 = interp_cols(mm, MMID, tab[0])
            f1 = interp_cols(mm, MMID, tab[1])
            fm = np.maximum(f0, f1)
            arr_fp[me] = fm
            arr_var[me] = fm * (1 - fm) / NB
        FPdS[me] = np.maximum(np.interp(mm, MMID, fpdS[0]), np.interp(mm, MMID, fpdS[1]))
        FPuS[me] = np.maximum(np.interp(mm, MMID, fpuS[0]), np.interp(mm, MMID, fpuS[1]))
        FPdP[me] = np.interp(mm, MMID, fpd[1, :, j0])            # down test is applied to prepared-|1> errors
        FPuP[me] = np.interp(mm, MMID, fpu[0, :, j0])            # up test is applied to prepared-|0> errors
        if c == 0:
            out["null_example_cfg0"] = {
                "m": MMID[::10].round(3).tolist(),
                "thr_down_alpha0": tcd_mid[::10, j0].round(3).tolist(), "thr_up_alpha0": tcu_mid[::10, j0].round(3).tolist(),
                "fp_down_alpha0_b0": fpd[0, ::10, j0].round(5).tolist(), "fp_down_alpha0_b1": fpd[1, ::10, j0].round(5).tolist(),
                "fp_up_alpha0_b0": fpu[0, ::10, j0].round(5).tolist(), "fp_up_alpha0_b1": fpu[1, ::10, j0].round(5).tolist(),
                "fpsel_down_b0": fpdS[0, ::10].round(5).tolist(), "fpsel_down_b1": fpdS[1, ::10].round(5).tolist(),
                "fpsel_up_b0": fpuS[0, ::10].round(5).tolist(), "fpsel_up_b1": fpuS[1, ::10].round(5).tolist(),
                "nsel_b1_prep1err": nsel[1, ::10, 1].astype(int).tolist(),
                "fp_down_lam_b0": fpdL[0, ::10].round(5).tolist(), "fp_down_lam_b1": fpdL[1, ::10].round(5).tolist()}
        # ---- injection into real prepared-|0> core shots (TEST)
        cand = np.where(me & (Bte[:, q] == 0))[0]
        cb0, s_late = latecore[(c, 0)]
        ok = np.linalg.norm(Bte_late[cand] - cb0, axis=1) < LATE_R * s_late
        cand = cand[ok]
        pick = rng.choice(len(cand), size=min(NCARRIER, len(cand)), replace=False)
        xc = norm(Xte[cand[pick]], c)
        a_, _, d_, _ = det.scores(xc)
        mcar = xc @ w
        carrier += [len(xc), (a_ > np.interp(mcar, MGRID, tcd[:, j0])).sum(),
                    (d_ > np.interp(mcar, MGRID, tcu[:, j0])).sum(), 0]
        for kind, Ai_ in inj_tmpl.items():
            for i, ts in enumerate(ts_inj):
                ai = Ai_[i]
                for mode, xi in (("down", xc + ai[None, :]), ("up", xc + (1.0 - ai)[None, :])):
                    mi_ = xi @ w
                    sd_, _, su_, _ = det.scores(xi)
                    if mode == "down":
                        flA = sd_[:, None] > interp_cols(mi_, MGRID, tcd)
                        mis = mi_ < 0.5
                        pmis = (xi @ v + v0) < 0
                    else:
                        flA = su_[:, None] > interp_cols(mi_, MGRID, tcu)
                        mis = mi_ > 0.5
                        pmis = (xi @ v + v0) > 0
                    fl = flA[:, j0]
                    inj[mode][kind][i] += [len(xi), fl.sum(), mis.sum(), (fl & mis).sum(), pmis.sum(), (fl & pmis).sum()]
                    if kind == "tau_fit" and ts < K * 100.0:
                        inj_a[mode] += np.c_[np.full(na, mis.sum()), (flA & mis[:, None]).sum(0),
                                             np.full(na, pmis.sum()), (flA & pmis[:, None]).sum(0)]
        jj += 1
    out["sig_m"] = sig_mb.tolist()
    out["sig_m_outlier_cfgs"] = {f"b{b_}": [int(c_) for c_ in range(16) if sig_mb[c_, b_] > 1.5 * np.median(sig_mb[:, b_])] for b_ in (0, 1)}
    out["cov"] = {k: np.median([ci[k] for ci in covinfo], 0).tolist()
                  for k in ("sig0_bin", "sig1core_bin", "resid_mad_ratio_1core_over_0", "sig_m")}
    out["cov"]["adjcorr0_median"] = np.median([ci["adjcorr0"] for ci in covinfo], 0).tolist() if K > 1 else []
    out["cov"]["m0_train_mad_over_sig_m0"] = float(np.median([ci["m0_train_mad"] / ci["sig_m"][0] for ci in covinfo]))
    out["cov"]["m1core_train_mad_over_sig_m1"] = float(np.median([ci["m1core_train_mad"] / ci["sig_m"][1] for ci in covinfo]))
    out["proxy"] = {"agree_with_soft_hard_median": float(np.median([p_["agree_with_soft_hard"] for p_ in proxyinfo])),
                    "agree_min": float(np.min([p_["agree_with_soft_hard"] for p_ in proxyinfo])),
                    "boundary_m_median": float(np.median([p_["boundary_m_proxy"] for p_ in proxyinfo if p_["boundary_m_proxy"] is not None])),
                    "weights_rel_cfg0": proxyinfo[0]["v_over_w_norm"],
                    "agree_all_shots": float(((sproxy > 0) == (soft > 0.5)).mean())}
    out["carrier_null_alpha0"] = {"n": int(carrier[0]), "flag_down": int(carrier[1]), "flag_up": int(carrier[2])}
    sens = {}
    for mode in ("down", "up"):
        sens[mode] = {}
        for kind in KINDS:
            Aac = inj[mode][kind]
            inwin = ts_inj < K * 100.0
            sens[mode][kind] = {
                "ts_ns": ts_inj.tolist(), "p_flag": (Aac[:, 1] / Aac[:, 0]).round(4).tolist(),
                "p_mf_misread": (Aac[:, 2] / Aac[:, 0]).round(4).tolist(),
                "p_flag_given_mf_misread": [None if Aac[i, 2] == 0 else round(float(Aac[i, 3] / Aac[i, 2]), 4) for i in range(len(ts_inj))],
                "p_proxy_misread": (Aac[:, 4] / Aac[:, 0]).round(4).tolist(),
                "p_flag_given_proxy_misread": [None if Aac[i, 4] == 0 else round(float(Aac[i, 5] / Aac[i, 4]), 4) for i in range(len(ts_inj))],
                "eff_uniform_ts_given_mf_misread": float(Aac[inwin, 3].sum() / max(Aac[inwin, 2].sum(), 1)),
                "eff_uniform_ts_given_proxy_misread": float(Aac[inwin, 5].sum() / max(Aac[inwin, 4].sum(), 1)),
                "p_flag_uniform_ts": float(Aac[inwin, 1].sum() / Aac[inwin, 0].sum()),
                "n_per_ts": int(Aac[0, 0])}
    out["sensitivity"] = sens
    out["efficiency_by_alpha_tau_fit"] = {mode: {str(a): {"given_mf_misread": float(inj_a[mode][k, 1] / max(inj_a[mode][k, 0], 1)),
                                                         "given_proxy_misread": float(inj_a[mode][k, 3] / max(inj_a[mode][k, 2], 1))}
                                                  for k, a in enumerate(ALPHAS)} for mode in inj_a}
    out["proxy_sig_s"] = sig_s.tolist()
    out["proxy_s_level"] = s_lev.tolist()
    # ---- flags
    Fd = Sd[:, None] > thrd
    Fu = Su[:, None] > thru
    FdL = Sd[:, None] > np.array(LAMS)[None, :]
    FuL = Su[:, None] > np.array(LAMS)[None, :]  # fixed thresholds (robustness only)
    # ---- calibration of the null on all TEST shots
    cal = {}
    for b in (0, 1):
        m = Bte[:, q] == b
        for dname, F_, FP_ in (("down", Fd, FPd), ("up", Fu, FPu)):
            cal[f"{dname}_on_prep{b}"] = {str(a): {"observed": int(F_[m, k].sum()), "expected": float(FP_[m, k].sum()), "n": int(m.sum())}
                                          for k, a in enumerate(ALPHAS)}
            mz = m & (np.abs(zte) < 3) if q != 3 else m
            cal[f"{dname}_on_prep{b}_absz_lt3"] = {str(a): {"observed": int(F_[mz, k].sum()), "expected": float(FP_[mz, k].sum()), "n": int(mz.sum())}
                                                   for k, a in enumerate(ALPHAS)}
            ml = m & (np.linalg.norm(Bte_late - np.array([latecore[(int(cc_), b)][0] for cc_ in range(16)])[cfe], axis=1)
                      < LATE_R * np.array([latecore[(int(cc_), b)][1] for cc_ in range(16)])[cfe])
            cal[f"{dname}_on_prep{b}_latecore"] = {str(a): {"observed": int(F_[ml, k].sum()), "expected": float(FP_[ml, k].sum()), "n": int(ml.sum())}
                                                   for k, a in enumerate(ALPHAS)}
            # in the |b> core only (|m - b| < 1 sigma, |z| < 2): transitions there are rare but not absent
            core = m & (np.abs(mhat - b) < sig_mb[cfe, b]) & (np.abs(zte) < 2)
            cal[f"{dname}_on_prep{b}_core"] = {"observed": int(F_[core, j0].sum()), "expected": float(FP_[core, j0].sum()), "n": int(core.sum())}
    out["calibration_all_shots"] = cal
    # ---- attribution for each prediction set
    out["attr"] = {}
    sets = {}
    for k, a in enumerate(ALPHAS):
        sets[f"alpha{a:g}"] = (Fd[:, k], Fu[:, k], FPd[:, k], FPu[:, k], VARd[:, k], VARu[:, k])
    sets["alpha0.01_selnull"] = (Fd[:, j0], Fu[:, j0], FPdS, FPuS, VARd[:, j0], VARu[:, j0])
    sets["alpha0.01_prepbranch"] = (Fd[:, j0], Fu[:, j0], FPdP, FPuP, FPdP * (1 - FPdP) / NB, FPuP * (1 - FPuP) / NB)
    for k, lam in enumerate(LAMS):
        sets[f"lam{lam:g}"] = (FdL[:, k], FuL[:, k], FPdL[:, k], FPuL[:, k], VARdL[:, k], VARuL[:, k])
    for pname, pred in G["preds"][V["pred_win"]].items():
        pq = pred[:, q]
        out["attr"][pname] = {sname: attribute(q, pq, Bte[:, q], zte, *st, cell) for sname, st in sets.items()}
        out["attr"][pname]["characterisation"] = characterise(q, pq, Bte[:, q], zte, mhat, sig_mb, cfe, Fd[:, j0], Fu[:, j0],
                                                              FPd[:, j0], FPu[:, j0], Su, Sd, td, tu, K, soft,
                                                              sproxy, sig_s, s_lev)
    # per-shot store for errors of the first prediction set
    pname0 = list(G["preds"][V["pred_win"]].keys())[0]
    err = G["preds"][V["pred_win"]][pname0][:, q] != Bte[:, q]
    np.savez_compressed(f"{O}/em_e2seg_v3_shots_{vname}_q{q}.npz", idx=np.where(err)[0].astype(np.int32),
                        z=zte[err].astype(np.float32), mhat=mhat[err].astype(np.float32), sproxy=sproxy[err].astype(np.float32),
                        Sd=Sd[err].astype(np.float32), Su=Su[err].astype(np.float32),
                        td=td[err].astype(np.float32), tu=tu[err].astype(np.float32),
                        thrd=thrd[err].astype(np.float32), thru=thru[err].astype(np.float32),
                        FPd=FPd[err].astype(np.float32), FPu=FPu[err].astype(np.float32),
                        cfg=cfe[err].astype(np.int8), prep=Bte[err, q].astype(np.int8), alphas=np.array(ALPHAS))
    log("done", vname, q)
    return out


def var_fp(S, var, cell):
    """MC variance of sum FP over set S: shots in the same (config, m-cell) share one estimate."""
    if not S.any():
        return 0.0
    keys, inv, cnt = np.unique(cell[S], return_inverse=True, return_counts=True)
    vk = np.bincount(inv, weights=var[S]) / cnt
    return float((cnt.astype(float) ** 2 * vk).sum())


def attribute(q, pred, prep, z, Fd, Fu, FPd, FPu, VARd, VARu, cell):
    res = {}
    for b in (0, 1):
        m = prep == b
        nb = int(m.sum())
        E = m & (pred != prep)
        N = int(E.sum())
        F = Fd if b == 1 else Fu
        FP = FPd if b == 1 else FPu
        VR = VARd if b == 1 else VARu
        if q == 3:
            Pz = np.zeros_like(E)
            Mz = np.zeros_like(E)
        else:
            Pz = E & (z > 3)
            Mz = E & (z < -3)
        nP, nM = int(Pz.sum()), int(Mz.sum())
        notP = E & ~Pz
        d = {"n_shots": nb, "n_err": N, "leak_plus": nP, "leak_minus": nM}
        # order A pool: errors not leak-flagged, plus the mirror set (stand-in for the Gaussian-tail non-leaked shots removed by z > 3)
        pool_flag = int((F & notP).sum() + (F & Mz).sum())
        pool_fp = float(FP[notP].sum() + FP[Mz].sum())
        pool_var_mc = var_fp(notP, VR, cell) + var_fp(Mz, VR, cell)
        all_flag = int((F & E).sum())
        all_fp = float(FP[E].sum())
        all_var_mc = var_fp(E, VR, cell)
        fP, fM = int((F & Pz).sum()), int((F & Mz).sum())
        fpP, fpM = float(FP[Pz].sum()), float(FP[Mz].sum())
        d.update({"flag_pool": pool_flag, "fp_pool": pool_fp, "fp_pool_var_mc": pool_var_mc,
                  "flag_all": all_flag, "fp_all": all_fp, "fp_all_var_mc": all_var_mc,
                  "flag_leakplus": fP, "flag_leakminus": fM, "fp_leakplus": fpP, "fp_leakminus": fpM,
                  "nP_notF": int((Pz & ~F).sum()), "nM_notF": int((Mz & ~F).sum())})
        res[f"prep{b}"] = d
    return res


def characterise(q, pred, prep, z, mhat, sig_mb, cf, Fd, Fu, FPd, FPu, Su, Sd, td, tu, K, soft, sproxy, sig_s, s_lev):
    """Residual characterisation (order A residual = error, not z>3, not transition-flagged at ALPHA0)."""
    res = {}
    for b in (0, 1):
        m = prep == b
        E = m & (pred != prep)
        F = Fd if b == 1 else Fu
        Pz = (E & (z > 3)) if q != 3 else np.zeros_like(E)
        Rs = E & ~Pz & ~F
        sg = sig_mb[cf, b]
        dist = ((0.5 - mhat) if b == 1 else (mhat - 0.5)) / sg
        def hist(sel):
            h = [int((sel & (dist < 0)).sum())]
            for j in range(4):
                h.append(int((sel & (dist >= DEDGES[j]) & (dist < DEDGES[j + 1])).sum()))
            return h
        MFerr = m & (dist > 0)
        pred_bins = np.zeros(4)
        pred_tot = 0.0
        for c in range(16):
            nc = int((m & (cf == c)).sum())
            a = 0.5 / sig_mb[c, b]
            pred_tot += nc * ndtr(-a)
            for j in range(4):
                pred_bins[j] += nc * (ndtr(-a - DEDGES[j]) - ndtr(-a - DEDGES[j + 1]))
        conf_correct = np.where(b == 1, soft, 1 - soft)
        d = {"dist_bins": ["<0 (MF reads correct)", "[0,1)", "[1,2)", "[2,3)", ">=3"],
             "resid_hist": hist(Rs), "all_err_hist": hist(E), "mf_err_hist_all_shots": hist(MFerr),
             "flagged_err_hist": hist(E & F),
             "gauss_pred_mf_err_hist": [None] + pred_bins.round(2).tolist(),
             "gauss_pred_mf_err_total": float(pred_tot), "mf_err_total": int(MFerr.sum()),
             "n_resid_shots": int(Rs.sum()), "n_err": int(E.sum()),
             "resid_soft_pcorrect_lt_0.4_frac": float((conf_correct[Rs] < 0.4).mean()) if Rs.any() else None,
             "resid_soft_pcorrect_median": float(np.median(conf_correct[Rs])) if Rs.any() else None,
             "gauss_pred_frac_within_1sigma_given_err": float(pred_bins[0] / max(pred_bins.sum(), 1e-30)),
             "gauss_pred_frac_within_2sigma_given_err": float(pred_bins[:2].sum() / max(pred_bins.sum(), 1e-30))}
        # same characterisation against the linear CNN proxy boundary (s = 0), distance in proxy noise widths
        sgs = sig_s[cf, b]
        dist = ((-sproxy) if b == 1 else sproxy) / sgs
        Pe = m & (dist > 0)
        pbins = np.zeros(4)
        ptot = 0.0
        for c in range(16):
            nc = int((m & (cf == c)).sum())
            a = (s_lev[c, 1] if b == 1 else -s_lev[c, 0]) / sig_s[c, b]
            ptot += nc * ndtr(-a)
            for j in range(4):
                pbins[j] += nc * (ndtr(-a - DEDGES[j]) - ndtr(-a - DEDGES[j + 1]))
        d["proxy"] = {"resid_hist": hist(Rs), "all_err_hist": hist(E), "proxy_err_hist_all_shots": hist(Pe),
                      "gauss_pred_proxy_err_hist": [None] + pbins.round(2).tolist(), "gauss_pred_proxy_err_total": float(ptot),
                      "proxy_err_total": int(Pe.sum())}
        tsel = (td if b == 1 else tu)[F & E]
        d["flag_time_hist_100ns"] = np.histogram(tsel, bins=np.arange(0, K * 100 + 1, 100))[0].tolist()
        Fw = (Fu if b == 1 else Fd)
        FPw = (FPu if b == 1 else FPd)
        # opposite-direction flags: NOT a clean null (prepared-|0> errors that start in |1> and decay; prepared-|1>
        # errors that start in |0> and get excited), reported as a physical secondary channel
        d["opposite_direction_alpha0"] = {"observed": int((Fw & E).sum()), "expected": float(FPw[E].sum()),
                                          "observed_resid_deep_ge3": int((Fw & Rs & (((0.5 - mhat) if b == 1 else (mhat - 0.5)) / sig_mb[cf, b] >= 3)).sum())}
        # per-configuration flagged errors (relevant direction, alpha0)
        d["per_cfg"] = {"n_shots": [int((m & (cf == c)).sum()) for c in range(16)],
                        "n_err": [int((E & (cf == c)).sum()) for c in range(16)],
                        "flag": [int((E & F & (cf == c)).sum()) for c in range(16)],
                        "fp": [float((FPd if b == 1 else FPu)[E & (cf == c)].sum()) for c in range(16)]}
        if q != 3:
            Pa, Ma = m & (z > 3), m & (z < -3)
            ePa, eMa = Pa & (pred != prep), Ma & (pred != prep)
            d["leak_all"] = {"plus": int(Pa.sum()), "minus": int(Ma.sum()), "n": int(m.sum()),
                             "err_plus": int(ePa.sum()), "err_minus": int(eMa.sum())}
        res[f"prep{b}"] = d
    return res


def main():
    t0 = time.time()
    ytr, yte = np.load(D + "/y_train.npy"), np.load(D + "/y_test.npy")
    Btr, Bte = bits(ytr), bits(yte)
    G["Btr"], G["Bte"] = Btr, Bte
    LW = {"1us": ("feat1us", 0, 10), "600ns": ("featbox", 0, 6), "400ns": ("featbox", 0, 4)}
    G["z"] = {}
    frames = {}
    leakchk = {}
    for wn, (src, b0, b1) in LW.items():
        Atr, Ate = wmean(src, "train", b0, b1), wmean(src, "test", b0, b1)
        fr = fit_line(Atr, Btr)
        z = zline(Ate, Bte, fr)
        G["z"][wn] = z
        frames[wn] = {f"q{q}": {"sigma_perp_median": float(np.median([fr[(q, c)][3] for c in range(16)])),
                                "sep_median": float(np.median([fr[(q, c)][4] for c in range(16)])),
                                "angle_deg_cfg": [round(float(np.degrees(np.arctan2(fr[(q, c)][1][1], fr[(q, c)][1][0]))), 2) for c in range(16)]}
                      for q in range(5)}
        leakchk[wn] = {}
        for q in range(5):
            for b in (0, 1):
                m = Bte[:, q] == b
                npl, nmi = int((z[m, q] > 3).sum()), int((z[m, q] < -3).sum())
                leakchk[wn][f"q{q}_prep{b}"] = {"plus": npl, "minus": nmi, "n": int(m.sum()), "E3": (npl - nmi) / int(m.sum())}
        log("frame", wn)
    pb = np.load(R + "/preds_MCMitCNN_pb.npy").astype(np.int8)
    tb = np.load(R + "/preds_MCMitCNN_tb.npy").astype(np.int8)
    soft = {T: np.load(f"{S}/cnn_soft_{T}ns.npz") for T in (1000, 600, 400)}
    G["soft"] = {T: soft[T]["prob"].astype(np.float32) for T in soft}
    chk = {"canon_tb_equals_prepared": bool(np.array_equal(tb, Bte)),
           "canonical_err_per_qubit": (pb != Bte).mean(0).tolist()}
    for T in (1000, 600, 400):
        chk[f"soft{T}_label_equals_prepared"] = bool(np.array_equal(soft[T]["label"].astype(np.int8), Bte))
        e = ((soft[T]["prob"] > 0.5).astype(np.int8) != Bte).mean(0)
        chk[f"soft{T}_err_per_qubit"] = e.tolist()
    G["preds"] = {"1us": {"canonical": pb, "soft1000": (soft[1000]["prob"] > 0.5).astype(np.int8)},
                  "600ns": {"soft600": (soft[600]["prob"] > 0.5).astype(np.int8)},
                  "400ns": {"soft400": (soft[400]["prob"] > 0.5).astype(np.int8)}}
    G["variants"] = {
        "box1us": {"K": 10, "src": "featbox", "win": "1us", "pred_win": "1us", "fir": False, "soft": 1000},
        "box600": {"K": 6, "src": "featbox", "win": "600ns", "pred_win": "600ns", "fir": False, "soft": 600},
        "box400": {"K": 4, "src": "featbox", "win": "400ns", "pred_win": "400ns", "fir": False, "soft": 400},
        "fir1us": {"K": 10, "src": "feat1us", "win": "1us", "pred_win": "1us", "fir": True, "soft": 1000},
        "box1us_tau0": {"K": 10, "src": "featbox", "win": "1us", "pred_win": "1us", "fir": False, "tau0": True, "soft": 1000},
    }
    only = os.environ.get("E2SEG_VARIANTS")
    vlist = only.split(",") if only else list(G["variants"])
    qs = [int(x) for x in os.environ.get("E2SEG_QUBITS", "0,1,2,3,4").split(",")]
    tasks = [(v, q) for v in vlist for q in qs]
    with Pool(min(NPROC, len(tasks))) as pool:
        outs = pool.map(worker, tasks)
    res = {"note": __doc__, "alpha0": ALPHA0, "alphas": ALPHAS, "lams": LAMS, "NB": NB, "checks": chk,
           "frames": frames, "leak_E3_check": leakchk, "variants": {}}
    for o in outs:
        res["variants"].setdefault(o["variant"], {})[f"q{o['q']}"] = o
    tag = os.environ.get("E2SEG_TAG", "")
    json.dump(res, open(f"{O}/em_e2seg_v3_raw{tag}.json", "w"), indent=1, default=float)
    log("all done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
