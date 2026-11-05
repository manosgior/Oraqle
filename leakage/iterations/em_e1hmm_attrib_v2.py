#!/usr/bin/env python3
"""em_e1hmm_attrib_v2.py (tag e1hmm): HMM-based error-mode attribution of the CANONICAL MCMit-CNN errors.

v2 (2026-11-04, after the laptop restart): v1 (em_e1hmm_attrib.py) never ran (wrong HMM parameter path).
Changes: HMM params read from m3time/out/<kind>_xt1 and checked against out_re/<kind>_xt1 (canonical relabelling);
output dir is an argument (--out) so earlier smoke outputs are never overwritten; added a conservative,
assumption-light FP bound (T_cons = n_trans - max(P(trans|ov), P(trans|wr)) * N_nonleaked), analytic mirror CIs for
the leakage share, mode rates as % of shots, the soft-CNN confidence of residual errors (proxy distance to the CNN
boundary), and per-shot p50 calls / p_trans for ALL test shots (cross-checks with other agents).

Goal: a defensible per-qubit error-mode breakdown of MCMit-CNN errors at 1 us (canonical hard predictions
preds_MCMitCNN_pb vs _tb) and at 600 / 400 ns (soft models cnn_soft_{600,400}ns.npz, hard = prob > 0.5),
replacing the paper's Table tab:error_modes (which was HybridCNN + a near-boundary rule true by construction).

Modes (one primary mode per error):
  1. LEAKAGE (non-computational): line-frame flag exactly as f_cnn_leak_xtab.py: per neighbour configuration,
     z = perpendicular distance from the |0>-|1> line through the two density cores (window means; frames fit on
     TRAIN, applied to TEST), unit = prepared-|0> perp width; flag z > 3, MIRROR-corrected by z < -3.
     Q4 (q3): not measurable (multimodal |1>), leakage = n/m and all its errors go to modes 2/3.
  2. MID-READOUT TRANSITION (relaxation 1->0 for prepared-|1> errors, excitation 0->1 for prepared-|0> errors):
     the m3time 3-state HMM (states 0, 1, L; canonical relabelling as in m3time_reanalyze.py), parameters FROZEN
     from the TRAIN fit, applied to TEST using ONLY the in-window bins 0..K-1 (1 us K=10, 600 ns K=6, 400 ns K=4).
     Primary features = `featbox` (boxcar 100 ns bins: bin k uses only samples 50k..50k+49, so no information from
     after the window enters), with the featbox_xt1 HMM. Detector: J(a,e) = P(s_1=a, s_{K-1}=e | x_0..x_{K-1})
     by a forward pass; call "transition" if J(b, 1-b) > 0.5 (b = prepared bit). Bin 0 is ring-up, so the start
     state is taken at bin 1 (100-200 ns): a change before ~200 ns is indistinguishable from a wrong initial state.
     Calls (4 categories): 0 trans | 1 wrong-throughout J(1-b,1-b) | 2 stay J(b,b) | 3 other (L-involving etc.).
  3. RESIDUAL = everything else (no detected event).
FP correction of mode 2 (null without transitions): per neighbour configuration c and prepared bit b, a Gaussian
null fitted on TRAIN (robust core mean and full 40x40 bin/IQ covariance of the xt-corrected 20-bin vector, per
state): synthetic stationary shots. Three synthetic error types are pushed through the identical detector:
  ov = stationary in b, conditioned EXACTLY (truncated sampling) on being misclassified by the per-config LDA on
       the in-window vector  (pure Gaussian-overlap errors);
  wr = stationary in 1-b with label b (wrong state for the whole trace: preparation error / decay before readout);
  tr = transition b->1-b at a uniform time in the window (linear bin mixing), conditioned on an LDA error.
The observed call counts of non-leaked errors are deconvolved (weighted NNLS, per c,b) into N_ov, N_wr, N_tr;
FP-corrected detected transitions T_det = n_trans - P(trans|ov) N_ov - P(trans|wr) N_wr. CIs from a joint
Poisson bootstrap over error shots plus multinomial resampling of the null matrices.
Also: null A (the HMM's own generative model with all transition probabilities 0: mu[s], Sig[k], white bins),
an injection test (sensitivity vs transition time), detector variants (p>0.8, argmax, Viterbi, future-informed
p>0.5 using all 20 bins), the FIR `feat` features (feat_xt1 HMM; 100 ns FIR look-ahead), order swap (2 before 1),
and an independent near-boundary characterisation of the residual (oriented per-config LDA distance in noise
units, compared with the pure-Gaussian-overlap prediction).

Docker (graham, oraqle-torch):
  -v ~/oraqle-ics-sims/leakage/features:/feat:ro -v ~/oraqle-ics-sims/leakage/m3time:/m3:ro
  -v ~/oraqle-ics-sims/leakage/f/in:/soft:ro -v ~/oraqle_reports:/reports:ro -v ~/oraqle-ics-sims/leakage/em/e1hmm:/work
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.optimize import nnls
from scipy.stats import chi2, norm, truncnorm

FEAT, M3, SOFT, REP = "/feat", "/m3", "/soft", "/reports"
OUT = "/work/out"          # overridden by --out in main()
NB = 20
L = 2
NAMES = ["Q1", "Q2", "Q3", "Q4", "Q5"]
WINDOWS = {"1us": {"K": 10, "T": 1000, "leak": ("feat1us", 0, 10)},
           "600ns": {"K": 6, "T": 600, "leak": ("featbox", 0, 6)},
           "400ns": {"K": 4, "T": 400, "leak": ("featbox", 0, 4)}}
VARIANTS = ["p50", "p80", "argmax", "viterbi", "fut_p50"]
CATS = ["trans", "wrong", "stay", "other"]
TYPES = ["ov", "wr", "tr"]
Q4 = 3


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def ci_mirror(npl, nmi, n):
    if n == 0:
        return [None, None]
    return [float((npl - nmi) / n - 1.96 * np.sqrt(npl + nmi) / n),
            float((npl - nmi) / n + 1.96 * np.sqrt(npl + nmi) / n)]


def bits(y):
    return ((y[:, None] >> np.arange(5)) & 1).astype(np.int8)


def cfg_other(y, q):
    return ((y >> (q + 1)) << q) | (y & ((1 << q) - 1))


# ============================================================ leakage frame (replica of f_cnn_leak_xtab.py)
def wmean(split, src, b0, b1, idx=None):
    F = np.load(f"{FEAT}/{src}_{split}.npy", mmap_mode="r")
    n = F.shape[0] if idx is None else len(idx)
    out = np.empty((n, 5, 2))
    for s in range(0, n, 100000):
        rows = slice(s, s + 100000) if idx is None else idx[s:s + 100000]
        out[s:s + 100000] = np.asarray(F[rows, :, b0:b1, :], dtype=np.float64).mean(2)
    return out


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def mode2d(P, s):
    lo = np.percentile(P, 0.5, axis=0) - s
    hi = np.percentile(P, 99.5, axis=0) + s
    nb = np.minimum(np.ceil((hi - lo) / (s / 3)).astype(int), 600)
    H, ex, ey = np.histogram2d(P[:, 0], P[:, 1], bins=nb, range=[[lo[0], hi[0]], [lo[1], hi[1]]])
    H = gaussian_filter(H, 1.5)
    i, j = np.unravel_index(np.argmax(H), H.shape)
    c = np.array([(ex[i] + ex[i + 1]) / 2, (ey[j] + ey[j + 1]) / 2])
    for _ in range(8):
        m = np.sum((P - c) ** 2, 1) < s ** 2
        c = P[m].mean(0)
    return c


def cfg_of(B, q):
    o = [k for k in range(5) if k != q]
    return (B[:, o] * (1 << np.arange(4))).sum(1)


def fit_frames(A, B):
    fr = {}
    for q in range(5):
        cf = cfg_of(B, q)
        for c in range(16):
            m0 = (cf == c) & (B[:, q] == 0)
            m1 = (cf == c) & (B[:, q] == 1)
            P0, P1 = A[m0, q], A[m1, q]
            s = mad(P0[:, 1])
            c0 = mode2d(P0, s)
            c1 = np.median(P1, 0) if q == 3 else mode2d(P1, s)
            u = (c1 - c0) / np.linalg.norm(c1 - c0)
            n = np.array([-u[1], u[0]])
            if n[1] < 0:
                n = -n
            sig = mad((P0 - c0) @ n)
            fr[(q, c)] = (c0, u, n, sig, np.linalg.norm(c1 - c0))
    return fr


def apply_frames(A, B, fr):
    z = np.empty((A.shape[0], 5))
    for q in range(5):
        cf = cfg_of(B, q)
        for c in range(16):
            m = cf == c
            c0, u, n, sig, sep = fr[(q, c)]
            z[m, q] = ((A[m, q] - c0) @ n) / sig
    return z


# ============================================================ HMM machinery (as m3time_hmm.py / m3time_reanalyze.py)
def canon_perm(P):
    pi0, pi1 = np.asarray(P["pi_prep0"]), np.asarray(P["pi_prep1"])
    s0 = int(np.argmax(pi0))
    rest = [s for s in range(3) if s != s0]
    s1 = rest[int(np.argmax(pi1[rest]))]
    sL = [s for s in rest if s != s1][0]
    return [s0, s1, sL]


def load_hmm(kind, q):
    d = json.load(open(f"{M3}/out/{kind}_xt1/m3time_q{q}.json"))
    assert d["kind"] == kind and bool(d["xt_correction"]), (d["kind"], d["xt_correction"])
    P = d["params_train"]
    perm = canon_perm(P)
    dre = json.load(open(f"{M3}/out_re/{kind}_xt1/m3time_re_q{q}.json"))
    assert perm == dre["perm_old_index_of_0_1_L"], (kind, q, perm, dre["perm_old_index_of_0_1_L"])
    Are = np.asarray(dre["params_train_canonical"]["A"])
    assert np.allclose(np.asarray(P["A"])[np.ix_(perm, perm)], Are), "canonical A mismatch vs out_re"
    H = {"mu": np.asarray(P["mu"])[perm], "Sig": np.asarray(P["Sig"]),
         "A": np.asarray(P["A"])[np.ix_(perm, perm)],
         "pi": np.stack([np.asarray(P["pi_prep0"])[perm], np.asarray(P["pi_prep1"])[perm]]),
         "offsets": np.asarray(d["offsets_xt"]), "xt": bool(d["xt_correction"]), "perm": perm,
         "re_occ": dre["soft_occupancy_test"]}
    assert H["xt"]
    return H


def log_emis(X, mu, Sig):
    N, K, _ = X.shape
    S = mu.shape[0]
    inv = np.linalg.inv(Sig[:K])
    logdet = np.log(np.linalg.det(Sig[:K]))
    a, b, c = inv[:, 0, 0], inv[:, 0, 1], inv[:, 1, 1]
    out = np.empty((N, K, S))
    for s in range(S):
        d0 = X[:, :, 0] - mu[s, :K, 0]
        d1 = X[:, :, 1] - mu[s, :K, 1]
        out[:, :, s] = -0.5 * (a * d0 * d0 + 2 * b * d0 * d1 + c * d1 * d1) - 0.5 * logdet - np.log(2 * np.pi)
    return out


def emis(logB):
    return np.exp(logB - logB.max(2, keepdims=True))


def joint_start_end(B, A, pi0, K):
    """J[n,a,e] = P(s_1=a, s_{K-1}=e | x_0..x_{K-1}); B = scaled emissions (N,>=K,S)."""
    N, _, S = B.shape
    a0 = pi0 * B[:, 0]
    a0 /= a0.sum(1, keepdims=True)
    a1 = (a0 @ A) * B[:, 1]
    a1 /= a1.sum(1, keepdims=True)
    F = np.zeros((N, S, S))
    F[:, np.arange(S), np.arange(S)] = a1
    for k in range(2, K):
        F = (F @ A) * B[:, k][:, None, :]
        F /= F.sum((1, 2), keepdims=True)
    return F


def backward_to(B, A, k_end):
    """beta_{k_end}(e) proportional to P(x_{k_end+1}..x_{19} | s_{k_end}=e)."""
    N, K, S = B.shape
    beta = np.ones((N, S))
    for k in range(K - 2, k_end - 1, -1):
        beta = (B[:, k + 1] * beta) @ A.T
        beta /= beta.sum(1, keepdims=True)
    return beta


def viterbi(logB, A, pi0):
    N, K, S = logB.shape
    logA = np.log(np.maximum(A, 1e-300))
    d = np.log(np.maximum(pi0, 1e-300)) + logB[:, 0]
    bp = np.empty((N, K, S), np.int8)
    for k in range(1, K):
        cand = d[:, :, None] + logA[None]
        bp[:, k] = cand.argmax(1)
        d = cand.max(1) + logB[:, k]
    path = np.empty((N, K), np.int8)
    path[:, K - 1] = d.argmax(1)
    ar = np.arange(N)
    for k in range(K - 1, 0, -1):
        path[:, k - 1] = bp[ar, k, path[:, k]]
    return path


def fb_gamma_mean(B, A, pi0, groups):
    """mean posterior occupancy per bin for each group (validation against m3time_reanalyze outputs)."""
    N, K, S = B.shape
    alpha = np.empty_like(B)
    a = pi0 * B[:, 0]
    alpha[:, 0] = a / a.sum(1, keepdims=True)
    for k in range(1, K):
        a = (alpha[:, k - 1] @ A) * B[:, k]
        alpha[:, k] = a / a.sum(1, keepdims=True)
    beta = np.ones((N, S))
    out = [np.zeros((K, S)) for _ in groups]
    for k in range(K - 1, -1, -1):
        if k < K - 1:
            beta = (B[:, k + 1] * beta) @ A.T
            beta /= beta.sum(1, keepdims=True)
        g = alpha[:, k] * beta
        g /= g.sum(1, keepdims=True)
        for i, m in enumerate(groups):
            out[i][k] = g[m].mean(0)
    return out


def calls_from(J, Jf, path, b, K):
    """4-way calls (0 trans, 1 wrong, 2 stay, 3 other) per detector variant."""
    n = len(b)
    ar = np.arange(n)
    nb = 1 - b
    def parts(JJ):
        pt = JJ[ar, b, nb]
        pw = JJ[ar, nb, nb]
        ps = JJ[ar, b, b]
        po = np.clip(1.0 - pt - pw - ps, 0, 1)
        return pt, pw, ps, po
    out = {}
    pt, pw, ps, po = parts(J)
    rest = np.argmax(np.stack([pw, ps, po], 1), 1) + 1
    out["p50"] = np.where(pt > 0.5, 0, rest).astype(np.int8)
    out["p80"] = np.where(pt > 0.8, 0, rest).astype(np.int8)
    out["argmax"] = np.argmax(np.stack([pt, pw, ps, po], 1), 1).astype(np.int8)
    s1, se = path[:, 1], path[:, K - 1]
    v = np.full(n, 3, np.int8)
    v[(s1 == b) & (se == nb)] = 0
    v[(s1 == nb) & (se == nb)] = 1
    v[(s1 == b) & (se == b)] = 2
    out["viterbi"] = v
    ptf, pwf, psf, pof = parts(Jf)
    restf = np.argmax(np.stack([pwf, psf, pof], 1), 1) + 1
    out["fut_p50"] = np.where(ptf > 0.5, 0, restf).astype(np.int8)
    return out, pt


def run_detector(X20, b, H, K):
    """X20: (n,20,2) xt-corrected. Returns calls per variant, p_trans, in-window Viterbi path."""
    logB = log_emis(X20, H["mu"], H["Sig"])
    B = emis(logB)
    pi0 = H["pi"][b]
    J = joint_start_end(B, H["A"], pi0, K)
    beta = backward_to(B, H["A"], K - 1)
    Jf = J * beta[:, None, :]
    Jf /= Jf.sum((1, 2), keepdims=True)
    path = viterbi(logB[:, :K], H["A"], pi0)
    calls, pt = calls_from(J, Jf, path, b, K)
    return calls, pt, path


# ============================================================ Gaussian null per configuration
def robust_mc(Z, m0, S0, q=0.99, iters=10):
    p = Z.shape[1]
    c2 = chi2.ppf(q, p)
    cons = chi2.cdf(c2, p + 2) / q
    m, S = m0.copy(), S0.copy()
    for _ in range(iters):
        Si = np.linalg.inv(S)
        D = Z - m
        d2 = np.einsum("ni,ij,nj->n", D, Si, D)
        sel = d2 < c2
        m = Z[sel].mean(0)
        S = np.cov((Z[sel] - m).T) / cons
    return m, S, float(sel.mean())


def step_fraction(tau_ns, kind, taps):
    """fraction of each 100 ns bin (bin value relative to a constant signal) after a step at tau_ns."""
    ts = int(np.ceil(tau_ns / 2.0))
    s = (np.arange(1000) >= ts).astype(float)
    if kind == "featbox":
        return s.reshape(NB, 50).mean(1)
    ys = np.convolve(s, taps, mode="same").reshape(NB, 50).mean(1)
    yc = np.convolve(np.ones(1000), taps, mode="same").reshape(NB, 50).mean(1)
    return ys / yc


class Null:
    """per (config, state): mean m (40,), covariance S (40,40) of the xt-corrected 20-bin vector."""

    def __init__(self, Xtr, btr, ctr, H, kind, taps):
        self.kind, self.taps = kind, taps
        m_init = [H["mu"][s].reshape(-1) for s in (0, 1)]
        S_init = np.zeros((40, 40))
        for k in range(NB):
            S_init[2 * k:2 * k + 2, 2 * k:2 * k + 2] = H["Sig"][k]
        self.m = np.zeros((16, 2, 40))
        self.S = np.zeros((16, 2, 40, 40))
        self.keep = np.zeros((16, 2))
        Z = Xtr.reshape(len(Xtr), 40)
        for c in range(16):
            for s in (0, 1):
                sel = (ctr == c) & (btr == s)
                self.m[c, s], self.S[c, s], self.keep[c, s] = robust_mc(Z[sel], m_init[s], S_init)
        # HMM-model null (null A): mu[s], block-diagonal Sig[k], same for all configs
        self.mA = np.stack(m_init)
        self.SA = S_init
        self.chol = np.linalg.cholesky(self.S)
        self.cholA = np.linalg.cholesky(self.SA)
        # mixing fractions on a 25 ns grid (0 .. 2000 ns), repeated for (along, perp)
        self.frac_tab = np.stack([np.repeat(step_fraction(t, kind, taps), 2) for t in np.arange(0, 2001, 25)])

    def lda(self, c, K, model="B"):
        D = 2 * K
        if model == "B":
            m0, m1 = self.m[c, 0, :D], self.m[c, 1, :D]
            St = 0.5 * (self.S[c, 0, :D, :D] + self.S[c, 1, :D, :D])
        else:
            m0, m1 = self.mA[0, :D], self.mA[1, :D]
            St = self.SA[:D, :D]
        w = np.linalg.solve(St, m1 - m0)
        wp = np.zeros(40)
        wp[:D] = w
        mid = np.zeros(40)
        mid[:D] = 0.5 * (m0 + m1)
        return wp, mid

    def params(self, c, s, model):
        if model == "B":
            return self.m[c, s], self.S[c, s], self.chol[c, s]
        return self.mA[s], self.SA, self.cholA

    def oriented(self, X, c, b, K, model="B"):
        """oriented LDA statistic in units of the prepared-state noise width along w (positive = toward b)."""
        w, mid = self.lda(c, K, model)
        _, S, _ = self.params(c, b, model)
        sd = np.sqrt(w @ S @ w)
        sign = 1.0 if b == 1 else -1.0
        return sign * ((X - mid) @ w) / sd

    def d_gauss(self, c, b, K, model="B"):
        w, mid = self.lda(c, K, model)
        m, S, _ = self.params(c, b, model)
        sign = 1.0 if b == 1 else -1.0
        return sign * (w @ (m - mid)) / np.sqrt(w @ S @ w)

    def sample(self, rng, n, c, b, K, typ, model="B", tau=None):
        """synthetic 20-bin vectors (n,20,2) for label b of config c."""
        w, mid = self.lda(c, K, model)
        sign = 1.0 if b == 1 else -1.0
        if typ == "ov" or typ == "all":
            m, S, Lc = self.params(c, b, model)
            eps = rng.standard_normal((n, 40)) @ Lc.T
            if typ == "ov":
                v = w @ S @ w
                g = S @ w / v
                y = eps @ w
                z = eps - np.outer(y, g)
                d = sign * (w @ (m - mid)) / np.sqrt(v)
                u = truncnorm.rvs(-np.inf, -d, size=n, random_state=rng)
                eps = z + np.outer(sign * u * np.sqrt(v), g)
            return (m + eps).reshape(n, NB, 2)
        out = []
        got = 0
        it = 0
        while got < n and it < 200:
            it += 1
            nn = max(2 * (n - got), 1000)
            if typ == "wr":
                m, S, Lc = self.params(c, 1 - b, model)
                X = m + rng.standard_normal((nn, 40)) @ Lc.T
            else:  # tr / inj: transition b -> 1-b at tau (uniform in window if tau is None)
                mb, Sb, _ = self.params(c, b, model)
                mo, So, _ = self.params(c, 1 - b, model)
                Lt = np.linalg.cholesky(0.5 * (Sb + So))
                if tau is None:
                    taus = rng.uniform(0, 100.0 * K, nn)
                else:
                    taus = np.full(nn, float(tau))
                fr = self.frac_tab[np.round(taus / 25.0).astype(int)]   # 25 ns grid for the mixing fractions
                X = (1 - fr) * mb + fr * mo + rng.standard_normal((nn, 40)) @ Lt.T
            if typ == "inj":
                out.append(X)
                got += nn
                continue
            s = sign * ((X - mid) @ w)
            X = X[s < 0]
            out.append(X)
            got += len(X)
        X = np.concatenate(out)[:n]
        return X.reshape(len(X), NB, 2)


# ============================================================ deconvolution
def deconv(n, M):
    """n: (4,) counts; M: (4,3) call probabilities per type. Returns N (3,), chi2."""
    sig = np.sqrt(np.maximum(n, 1.0))
    N, _ = nnls(M / sig[:, None], n / sig)
    r = (n - M @ N) / sig
    return N, float((r * r).sum())


# ============================================================ per (kind, q) job
def run_job(args):
    kind, q, a = args
    t0 = time.time()
    rng = np.random.default_rng(1000 + 17 * q + (0 if kind == "featbox" else 7))
    log = lambda m: print(f"[e1hmm {kind} q{q}] {m} ({time.time()-t0:.0f}s)", flush=True)
    H = load_hmm(kind, q)
    taps = np.asarray(json.load(open(f"{FEAT}/calib.json"))["fir"]["taps"], dtype=float)
    yte = np.load(f"{FEAT}/y_test.npy").astype(np.int64)
    ytr = np.load(f"{FEAT}/y_train.npy").astype(np.int64)
    sel_te = np.load(f"{OUT}/sel_test.npy") if a.smoke else np.arange(len(yte))
    yte = yte[sel_te]
    bte, cte = (yte >> q) & 1, cfg_other(yte, q)
    btr, ctr = (ytr >> q) & 1, cfg_other(ytr, q)
    F = np.load(f"{FEAT}/{kind}_test.npy", mmap_mode="r")
    Xte = np.asarray(F[sel_te, q], dtype=np.float64) - H["offsets"][cte]
    Ftr = np.load(f"{FEAT}/{kind}_train.npy", mmap_mode="r")
    Xtr = np.asarray(Ftr[:, q], dtype=np.float64) - H["offsets"][ctr]
    del F, Ftr
    res = {"kind": kind, "q": q, "paper": NAMES[q], "perm": H["perm"],
           "hmm": {"A": H["A"], "pi_prep0": H["pi"][0], "pi_prep1": H["pi"][1]}}
    # --- validation: full-trace posterior occupancy vs m3time_reanalyze (out_re) values
    if not a.smoke:
        B = emis(log_emis(Xte, H["mu"], H["Sig"]))
        g0, g1 = fb_gamma_mean(B, H["A"], H["pi"][bte], [bte == 0, bte == 1])
        ref1 = np.asarray(H["re_occ"]["prep1"]["P_L_per_bin"])
        ref0 = np.asarray(H["re_occ"]["prep0"]["P_L_per_bin"])
        res["validation_PL_maxabsdiff_vs_out_re"] = {"prep1": float(np.abs(g1[:, L] - ref1).max()),
                                                     "prep0": float(np.abs(g0[:, L] - ref0).max())}
        del B
        log(f"validation {res['validation_PL_maxabsdiff_vs_out_re']}")
    # --- Gaussian null per configuration (TRAIN)
    null = Null(Xtr, btr, ctr, H, kind, taps)
    res["null_core_keep_fraction"] = null.keep
    del Xtr
    log("null fitted")
    # --- error sets and leakage regions
    shared = np.load(f"{OUT}/shared_inputs.npz")
    res["windows"] = {}
    store = {}
    allcalls = {"row": sel_te}
    for wname, W in WINDOWS.items():
        K = W["K"]
        calls, pt, path = run_detector(Xte, bte, H, K)
        z = shared[f"z_{wname}"][:, q]
        region = np.where(z > 3, 2, np.where(z < -3, 1, 0)).astype(np.int8)
        if q == Q4:
            region[:] = 0
        # oriented LDA statistic per shot + per-config Gaussian prediction
        s_lda = np.empty(len(yte))
        dg = np.zeros((16, 2))
        for c in range(16):
            for b in (0, 1):
                m = (cte == c) & (bte == b)
                s_lda[m] = null.oriented(Xte[m].reshape(-1, 40), c, b, K)
                dg[c, b] = null.d_gauss(c, b, K)
        # first switch bin of the in-window Viterbi path (for transition timing)
        sw = np.argmax(path[:, 1:] != path[:, [1]], 1) + 1
        sw[~(path[:, 1:] != path[:, [1]]).any(1)] = -1
        # ---- null simulations
        log(f"{wname}: simulating null")
        Mv = {v: np.zeros((16, 2, 4, 3)) for v in VARIANTS}
        MvA = {v: np.zeros((16, 2, 4)) for v in VARIANTS}       # null A, type ov
        fp_all = {v: np.zeros((16, 2)) for v in VARIANTS}       # per-shot FP on unconditioned stationary shots
        ov_s = []                                               # oriented stat of ov sims (check)
        ns = a.nsim
        for c in range(16):
            for b in (0, 1):
                for ti, typ in enumerate(TYPES):
                    Xs = null.sample(rng, ns, c, b, K, typ)
                    cl, _, _ = run_detector(Xs, np.full(len(Xs), b), H, K)
                    for v in VARIANTS:
                        Mv[v][c, b, :, ti] = np.bincount(cl[v], minlength=4) / len(Xs)
                    if typ == "ov" and c == 0:
                        ov_s.append(null.oriented(Xs.reshape(-1, 40), c, b, K).max())
                Xs = null.sample(rng, ns, c, b, K, "all")
                cl, _, _ = run_detector(Xs, np.full(ns, b), H, K)
                for v in VARIANTS:
                    fp_all[v][c, b] = (cl[v] == 0).mean()
                Xs = null.sample(rng, ns, c, b, K, "ov", model="A")
                cl, _, _ = run_detector(Xs, np.full(ns, b), H, K)
                for v in VARIANTS:
                    MvA[v][c, b] = np.bincount(cl[v], minlength=4) / ns
        # ---- injection test (pooled over configs), primary variant + viterbi
        inj = {}
        taus = np.arange(0, 100 * K, 25)
        for b in (0, 1):
            rows = []
            for tau in taus:
                det = np.zeros(4)
                n_tot = n_err = n_det_err = 0
                for c in range(16):
                    Xs = null.sample(rng, a.ninj, c, b, K, "inj", tau=tau)
                    cl, _, _ = run_detector(Xs, np.full(len(Xs), b), H, K)
                    e = null.oriented(Xs.reshape(-1, 40), c, b, K) < 0
                    det += np.bincount(cl["p50"], minlength=4)
                    n_tot += len(Xs)
                    n_err += int(e.sum())
                    n_det_err += int(((cl["p50"] == 0) & e).sum())
                rows.append({"tau_ns": int(tau), "P_call": (det / n_tot).tolist(), "P_lda_error": n_err / n_tot,
                             "P_trans_given_lda_error": (n_det_err / n_err) if n_err else None, "n": int(n_tot)})
            inj[f"prep{b}"] = rows
        # ---- error sets
        esets = {}
        if wname == "1us":
            esets["canon"] = shared["err_canon"][:, q]
        esets["soft"] = shared[f"err_soft_{wname}"][:, q]
        p1 = shared[f"p_soft_{wname}"][:, q].astype(np.float64)
        pcorr = np.where(bte == 1, p1, 1.0 - p1)          # soft-CNN P(correct) (proxy for canon at 1 us)
        allcalls[f"{wname}_call_all"] = calls["p50"].astype(np.int8)
        allcalls[f"{wname}_ptrans_all"] = pt.astype(np.float16)
        allcalls[f"{wname}_region_all"] = region
        W_out = {"K": K, "d_gauss_per_config": dg, "injection": inj,
                 "null_fp_per_shot_stationary": {v: {"prep0": float(fp_all[v][:, 0].mean()),
                                                     "prep1": float(fp_all[v][:, 1].mean())} for v in VARIANTS},
                 "check_ov_sims_max_oriented_stat": float(max(ov_s)), "errsets": {}}
        for ename, E in esets.items():
            W_out["errsets"][ename] = attribute(E, bte, cte, region, calls, pt, s_lda, sw, dg, Mv, MvA, a, rng,
                                                q, K, pcorr)
            store[(wname, ename)] = (E, region, calls["p50"], s_lda, pt)
        # hidden leakage (all shots)
        hid = {}
        for b in (0, 1):
            m = bte == b
            npl, nmi = int(((region == 2) & m).sum()), int(((region == 1) & m).sum())
            hid[f"prep{b}"] = {"n": int(m.sum()), "n_plus": npl, "n_minus": nmi,
                               "E3": (npl - nmi) / m.sum(), "E3_ci95": ci_mirror(npl, nmi, m.sum())}
        npl, nmi = int((region == 2).sum()), int((region == 1).sum())
        hid["all"] = {"E3": (npl - nmi) / len(region), "E3_ci95": ci_mirror(npl, nmi, len(region))}
        for ename, E in esets.items():
            epl, emi = int((E & (region == 2)).sum()), int((E & (region == 1)).sum())
            hid[f"misread_of_L_{ename}"] = (epl - emi) / max(npl - nmi, 1)
            hid[f"read_correctly_of_L_{ename}"] = 1 - (epl - emi) / max(npl - nmi, 1)
            hid[f"read_correctly_of_L_{ename}_ci95"] = [1 - x for x in ci_mirror(epl, emi, max(npl - nmi, 1))[::-1]]
            for b in (0, 1):
                m = bte == b
                p_, m_ = int(((region == 2) & m).sum()), int(((region == 1) & m).sum())
                ep, em = int((E & (region == 2) & m).sum()), int((E & (region == 1) & m).sum())
                hid[f"read_correctly_of_L_{ename}_prep{b}"] = 1 - (ep - em) / max(p_ - m_, 1)
        W_out["hidden_leakage"] = hid if q != Q4 else "n/m (Q4 multimodal |1>; line-frame flag not interpretable)"
        # raw calls among ALL shots (per-shot transition rates; FP-corrected with the per-shot stationary null)
        tr_all = {}
        for b in (0, 1):
            m = bte == b
            k = int(((calls["p50"] == 0) & m).sum())
            f = float(fp_all["p50"][:, b].mean())
            tr_all[f"prep{b}"] = {"raw_trans_call_rate": k / m.sum(), "null_fp_rate": f,
                                  "fp_corrected_rate": k / m.sum() - f, "raw_ci95": wilson(k, m.sum())}
        W_out["trans_calls_all_shots"] = tr_all
        res["windows"][wname] = W_out
        log(f"{wname}: done")
    # per-shot arrays for errors (cluster only)
    arr = {}
    for (wname, ename), (E, region, c50, s_lda, pt) in store.items():
        ii = np.where(E)[0]
        arr[f"{wname}_{ename}_row"] = sel_te[ii]
        arr[f"{wname}_{ename}_region"] = region[ii]
        arr[f"{wname}_{ename}_call"] = c50[ii]
        arr[f"{wname}_{ename}_slda"] = s_lda[ii].astype(np.float32)
        arr[f"{wname}_{ename}_ptrans"] = pt[ii].astype(np.float32)
    np.savez_compressed(f"{OUT}/em_e1hmm_pershot_{kind}_q{q}.npz", **arr)
    np.savez_compressed(f"{OUT}/em_e1hmm_allshots_{kind}_q{q}.npz", **allcalls)
    res["runtime_s"] = time.time() - t0
    json.dump(jsonable(res), open(f"{OUT}/em_e1hmm_job_{kind}_q{q}.json", "w"))
    log("job done")
    return res


def attribute(E, b, c, region, calls, pt, s_lda, sw, dg, Mv, MvA, a, rng, q, K, pcorr):
    """Attribution of one error set for one qubit/window. Returns summary + bootstrap replicate arrays."""
    ii = np.where(E)[0]
    bb, cc, rr = b[ii], c[ii], region[ii]
    out = {"n_err": int(len(ii)), "n_shots": int(len(E)), "err_rate": len(ii) / len(E),
           "err_rate_ci95": wilson(len(ii), len(E))}
    for x in (0, 1):
        n_b = int((b == x).sum())
        k_b = int((bb == x).sum())
        out[f"err_rate_prep{x}"] = k_b / n_b
        out[f"err_rate_prep{x}_ci95"] = wilson(k_b, n_b)
    reps = a.nboot
    W = rng.poisson(1.0, size=(reps, len(ii))).astype(np.float32)
    nsim = a.nsim
    # cell index (c, b, region, call)
    res_v = {}
    boot_v = {}
    for v in VARIANTS:
        cl = calls[v][ii]
        cell = ((cc * 2 + bb) * 3 + rr) * 4 + cl
        oh = sparse.csr_matrix((np.ones(len(ii), np.float32), (np.arange(len(ii)), cell)), shape=(len(ii), 384))
        cnt0 = np.asarray(oh.sum(0)).ravel().reshape(16, 2, 3, 4)
        cntb = np.asarray(oh.T.tocsr() @ W.T).T.reshape(reps, 16, 2, 3, 4)
        allc = np.concatenate([cnt0[None], cntb], 0)            # rep 0 = point estimate
        nrep = allc.shape[0]
        # quantities per rep and prepared bit
        Q = {k: np.zeros((nrep, 2)) for k in ("n_err", "L", "T_det", "N_ov", "N_wr", "N_tr", "false_trans",
                                               "n_trans_nonleak", "chi2", "T_det_all", "L_after_T", "overlap_LT",
                                               "T_det_nullA", "n_trans_all", "T_cons", "Lplus", "Lminus",
                                               "n_wrong_nonleak")}
        for r in range(nrep):
            for cfg in range(16):
                for x in (0, 1):
                    C = allc[r, cfg, x]                            # (3 regions, 4 calls)
                    M = Mv[v][cfg, x]
                    MA = M.copy()
                    MA[:, 0] = MvA[v][cfg, x]
                    if r > 0:
                        M = np.stack([rng.multinomial(nsim, np.clip(M[:, t], 0, 1) / max(M[:, t].sum(), 1e-12))
                                      / nsim for t in range(3)], 1)
                    n_err = C.sum()
                    Lc = C[2].sum() - C[1].sum()
                    n_nl = C[0] + 2 * C[1]                          # non-leaked calls (mirror replaces plus)
                    N, ch = deconv(n_nl, M)
                    false = M[0, 0] * N[0] + M[0, 1] * N[1]
                    NA, _ = deconv(n_nl, MA)
                    falseA = MA[0, 0] * NA[0] + MA[0, 1] * NA[1]
                    # order swap: transitions first on all errors
                    n_all = C.sum(0)
                    N2, _ = deconv(n_all, M)
                    false2 = M[0, 0] * N2[0] + M[0, 1] * N2[1]
                    Lafter = (C[2, 1:].sum() - C[1, 1:].sum())       # leakage among non-trans-called errors
                    Q["n_err"][r, x] += n_err
                    Q["L"][r, x] += Lc
                    Q["T_det"][r, x] += n_nl[0] - false
                    Q["N_ov"][r, x] += N[0]
                    Q["N_wr"][r, x] += N[1]
                    Q["N_tr"][r, x] += N[2]
                    Q["false_trans"][r, x] += false
                    Q["n_trans_nonleak"][r, x] += n_nl[0]
                    Q["chi2"][r, x] += ch
                    Q["T_det_all"][r, x] += n_all[0] - false2
                    Q["L_after_T"][r, x] += Lafter
                    Q["overlap_LT"][r, x] += C[2, 0] - C[1, 0]
                    Q["T_det_nullA"][r, x] += n_nl[0] - falseA
                    Q["n_trans_all"][r, x] += n_all[0]
                    # assumption-light bound: every non-leaked error treated as a potential false call
                    Q["T_cons"][r, x] += n_nl[0] - max(M[0, 0], M[0, 1]) * n_nl.sum()
                    Q["Lplus"][r, x] += C[2].sum()
                    Q["Lminus"][r, x] += C[1].sum()
                    Q["n_wrong_nonleak"][r, x] += n_nl[1]
        boot_v[v] = {k: Q[k] for k in Q}
        res_v[v] = summarize(Q, q, {"prep0": int((b == 0).sum()), "prep1": int((b == 1).sum()), "all": len(b)})
        # FP rates of the detector per prepared bit (weights: Gaussian-predicted overlap errors per config)
        fp = {}
        for x in (0, 1):
            wov = norm.cdf(-dg[:, x])
            wov = wov / wov.sum() if wov.sum() > 0 else np.full(16, 1 / 16)
            fp[f"prep{x}"] = {"P_trans_given_ov": float((Mv[v][:, x, 0, 0] * wov).sum()),
                              "P_trans_given_ov_nullA": float((MvA[v][:, x, 0] * wov).sum()),
                              "P_trans_given_wr": float(Mv[v][:, x, 0, 1].mean()),
                              "P_trans_given_tr": float(Mv[v][:, x, 0, 2].mean()),
                              "P_call_given_ov": (Mv[v][:, x, :, 0] * wov[:, None]).sum(0),
                              "P_call_given_wr": Mv[v][:, x, :, 1].mean(0),
                              "P_call_given_tr": Mv[v][:, x, :, 2].mean(0)}
        res_v[v]["detector_rates"] = fp
    out["variants"] = res_v
    out["boot"] = {v: {k: boot_v[v][k].tolist() for k in ("n_err", "L", "T_det", "N_ov", "N_wr", "N_tr",
                                                          "T_det_all", "L_after_T", "overlap_LT", "T_det_nullA",
                                                          "T_cons", "Lplus", "Lminus", "n_trans_nonleak",
                                                          "n_wrong_nonleak")}
                   for v in ("p50", "fut_p50", "viterbi")}
    # ---- independent near-boundary characterisation of the residual (primary variant p50, order 1)
    cl = calls["p50"][ii]
    wgt = np.where(rr == 0, 1.0, np.where(rr == 1, 2.0, 0.0))     # mirror-corrected non-leaked weights
    s = s_lda[ii]
    pc_e = pcorr[ii]
    def dist(mask):
        w_ = wgt * mask
        tot = w_.sum()
        if tot <= 0:
            return None
        edges = [-np.inf, -3, -2, -1, 0, 1, np.inf]
        h = [float((w_ * ((s >= lo) & (s < hi))).sum() / tot) for lo, hi in zip(edges[:-1], edges[1:])]
        o = np.argsort(s[w_ > 0])
        sv, wv = s[w_ > 0][o], w_[w_ > 0][o]
        cw = np.cumsum(wv) / wv.sum()
        qs = {f"q{int(p*100)}": float(sv[min(np.searchsorted(cw, p), len(sv) - 1)])
              for p in (0.05, 0.25, 0.5, 0.75, 0.95)}
        o2 = np.argsort(pc_e[w_ > 0])
        pc, wp = pc_e[w_ > 0][o2], w_[w_ > 0][o2]
        cwp = np.cumsum(wp) / wp.sum()
        qpc = {f"q{int(p*100)}": float(pc[min(np.searchsorted(cwp, p), len(pc) - 1)])
               for p in (0.05, 0.25, 0.5, 0.75, 0.95)}
        return {"weighted_n": float(tot), "hist_edges": "[-inf,-3,-2,-1,0,1,inf] (oriented LDA sigma; <0 = LDA also wrong)",
                "hist": h, "frac_abs_lt1": float((w_ * (np.abs(s) < 1)).sum() / tot),
                "frac_abs_lt2": float((w_ * (np.abs(s) < 2)).sum() / tot),
                "frac_lda_correct": float((w_ * (s > 0)).sum() / tot), "quantiles": qs,
                "softcnn_pcorrect_weighted_quantiles": qpc,
                "softcnn_frac_pcorrect_ge_0.4": float((w_ * (pc_e >= 0.4)).sum() / tot),
                "softcnn_frac_pcorrect_ge_0.5": float((w_ * (pc_e >= 0.5)).sum() / tot),
                "softcnn_frac_pcorrect_lt_0.1": float((w_ * (pc_e < 0.1)).sum() / tot),
                "softcnn_median_pcorrect": float(np.median(pc_e[w_ > 0]))}
    nb_out = {"residual_all": dist(cl != 0), "residual_call_stay": dist(cl == 2),
              "residual_call_wrong": dist(cl == 1), "residual_call_other": dist(cl == 3),
              "detected_transitions": dist(cl == 0), "all_nonleaked_errors": dist(np.ones(len(ii), bool))}
    # Gaussian-overlap prediction
    G = 0.0
    G_b = [0.0, 0.0]
    p1 = p2 = 0.0
    for cfg in range(16):
        for x in (0, 1):
            ncb = int(((c == cfg) & (b == x)).sum())
            d = dg[cfg, x]
            g = ncb * norm.cdf(-d)
            G += g
            G_b[x] += g
            if norm.cdf(-d) > 0:
                p1 += g * (norm.cdf(-d) - norm.cdf(-1 - d)) / norm.cdf(-d)
                p2 += g * (norm.cdf(-d) - norm.cdf(-2 - d)) / norm.cdf(-d)
    lda_err = int((s_lda < 0).sum())
    nb_out["gaussian_overlap"] = {
        "expected_overlap_errors": G, "expected_prep0": G_b[0], "expected_prep1": G_b[1],
        "expected_overlap_error_rate": G / len(E),
        "pred_frac_of_overlap_errors_within_1sigma": p1 / G if G > 0 else None,
        "pred_frac_of_overlap_errors_within_2sigma": p2 / G if G > 0 else None,
        "observed_errors": int(len(ii)),
        "observed_residual_weighted": float((wgt * (cl != 0)).sum()) if q != Q4 else float((cl != 0).sum()),
        "observed_lda_errors_all_shots": lda_err, "observed_lda_error_rate": lda_err / len(E),
        "d_gauss_median": float(np.median(dg))}
    if q == Q4:
        for k_, v_ in nb_out.items():
            pass
    out["near_boundary"] = nb_out
    # transition timing among detected (first switch bin of the in-window Viterbi path, bins)
    det = cl == 0
    out["detected_switch_bin_hist"] = np.bincount(np.clip(sw[ii][det], 0, K), minlength=K + 1).tolist()
    out["p_trans_quantiles_errors"] = np.quantile(pt[ii], [0.5, 0.9, 0.99]).tolist()
    return out


def ci_pct(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def summarize(Q, q, n_shots=None):
    """shares of errors per mode (point = rep 0; CI = bootstrap percentiles over reps 1..)."""
    o = {}
    for x in ("prep0", "prep1", "all"):
        sel = [0] if x == "prep0" else [1] if x == "prep1" else [0, 1]
        g = {k: Q[k][:, sel].sum(1) for k in Q}
        ne = g["n_err"]
        ns = None if n_shots is None else n_shots[x]
        def sh(v):
            d = {"count": float(v[0]), "share": float(v[0] / ne[0]) if ne[0] else None,
                 "share_ci95": ci_pct(v[1:] / np.maximum(ne[1:], 1))}
            if ns:
                d["rate_of_shots"] = float(v[0] / ns)
                d["rate_of_shots_ci95"] = ci_pct(v[1:] / ns)
            return d
        resid = ne - g["L"] - g["T_det"]
        resid2 = ne - g["T_det_all"] - g["L_after_T"]
        lp, lm = float(g["Lplus"][0]), float(g["Lminus"][0])
        o[x] = {"n_err": float(ne[0]), "n_shots": ns,
                "leakage_share_mirror_ci95": ci_mirror(lp, lm, float(ne[0])) if q != Q4 else "n/m",
                "leak_n_plus": lp, "leak_n_minus": lm,
                "order_L_then_T": {"leakage": sh(g["L"]) if q != Q4 else "n/m",
                                   "transition_fp_corrected": sh(g["T_det"]),
                                   "residual": sh(resid)},
                "transition_conservative_bound": sh(g["T_cons"]),
                "residual_if_conservative_T": sh(ne - g["L"] - g["T_cons"]),
                "wrong_from_start_raw_calls_nonleaked": sh(g["n_wrong_nonleak"]),
                "order_T_then_L": {"transition_fp_corrected": sh(g["T_det_all"]),
                                   "leakage": sh(g["L_after_T"]) if q != Q4 else "n/m",
                                   "residual": sh(resid2)},
                "overlap_leak_and_transcall_mirror": sh(g["overlap_LT"]) if q != Q4 else "n/m",
                "transition_raw_calls_nonleaked": sh(g["n_trans_nonleak"]),
                "transition_false_calls_est": sh(g["false_trans"]),
                "transition_fp_corrected_nullA": sh(g["T_det_nullA"]),
                "deconv_N_overlap": sh(g["N_ov"]), "deconv_N_wrongstate": sh(g["N_wr"]),
                "deconv_N_transition_sens_corrected": sh(g["N_tr"]),
                "deconv_chi2_sum_over_configs": float(g["chi2"][0])}
    return o


def prepare(a):
    """shared inputs: leakage z per window (TEST), error masks."""
    os.makedirs(OUT, exist_ok=True)
    ytr, yte = np.load(FEAT + "/y_train.npy"), np.load(FEAT + "/y_test.npy")
    sel = np.arange(len(yte))
    if a.smoke:
        rng = np.random.default_rng(5)
        sel = np.sort(rng.choice(len(yte), 120000, replace=False))
        np.save(f"{OUT}/sel_test.npy", sel)
    Btr, Bte = bits(ytr), bits(yte)
    sh = {}
    meta = {}
    for wname, W in WINDOWS.items():
        src, b0, b1 = W["leak"]
        Atr, Ate = wmean("train", src, b0, b1), wmean("test", src, b0, b1)
        fr = fit_frames(Atr, Btr)
        z = apply_frames(Ate, Bte, fr)
        sh[f"z_{wname}"] = z[sel]
        soft = np.load(f"{SOFT}/cnn_soft_{W['T']}ns.npz")
        prob, lab = soft["prob"].astype(np.float32), soft["label"].astype(np.int8)
        meta[f"soft_{wname}_label_equals_prepared"] = bool(np.array_equal(lab, Bte))
        sh[f"err_soft_{wname}"] = ((prob > 0.5).astype(np.int8) != Bte)[sel]
        sh[f"p_soft_{wname}"] = prob[sel]
        meta[f"soft_{wname}_err_ci95"] = [wilson(int(k), len(Bte)) for k in ((prob > 0.5).astype(np.int8) != Bte).sum(0)]
        if wname == "1us":
            meta["soft_1us_vs_canon_disagreement_rate"] = None   # filled below
        meta[f"soft_{wname}_err_rate"] = ((prob > 0.5).astype(np.int8) != Bte).mean(0).tolist()
        print("prepared", wname, flush=True)
    pb = np.load(REP + "/preds_MCMitCNN_pb.npy").astype(np.int8)
    tb = np.load(REP + "/preds_MCMitCNN_tb.npy").astype(np.int8)
    meta["canon_tb_equals_prepared"] = bool(np.array_equal(tb, Bte))
    meta["canon_err_rate"] = (pb != Bte).mean(0).tolist()
    meta["canon_err_ci95"] = [wilson(int(k), len(Bte)) for k in (pb != Bte).sum(0)]
    s1 = np.load(f"{SOFT}/cnn_soft_1000ns.npz")["prob"]
    meta["soft_1us_vs_canon_disagreement_rate"] = ((s1 > 0.5).astype(np.int8) != pb).mean(0).tolist()
    sh["err_canon"] = (pb != Bte)[sel]
    paper = {}
    for T in (1000, 800, 600, 400, 200):
        f = f"{REP}/cnn_length_{T}ns.csv"
        if os.path.exists(f):
            ln = open(f).read().strip().splitlines()
            hdr, val = ln[0].split(","), ln[1].split(",")
            paper[str(T)] = {h: float(v) for h, v in zip(hdr, val)}
    meta["paper_cnn_length_sweep_accuracy"] = paper
    np.savez(f"{OUT}/shared_inputs.npz", **sh)
    json.dump(jsonable(meta), open(f"{OUT}/shared_meta.json", "w"), indent=1)
    return meta


def pool_boot(results, qs, wname, ename, kind="featbox", v="p50", a_smoke=None):
    """pooled shares over qubits qs using rep-wise sums of independent per-qubit bootstraps."""
    acc = None
    for r in results:
        if r["kind"] != kind or r["q"] not in qs:
            continue
        E = r["windows"][wname]["errsets"].get(ename)
        if E is None:
            return None
        B = {k: np.asarray(x).sum(1) for k, x in E["boot"][v].items()}   # (nrep,) summed over prep bits
        if r["q"] == Q4:
            for k in ("L", "L_after_T", "overlap_LT", "Lplus", "Lminus"):
                B[k] = np.zeros_like(B[k])
        acc = B if acc is None else {k: acc[k] + B[k] for k in acc}
    if acc is None:
        return None
    ne = acc["n_err"]
    nsh = 1120000 * len(qs) if a_smoke is None else a_smoke * len(qs)
    def sh(x):
        return {"count": float(x[0]), "share": float(x[0] / ne[0]), "share_ci95": ci_pct(x[1:] / ne[1:]),
                "rate_of_shots": float(x[0] / nsh), "rate_of_shots_ci95": ci_pct(x[1:] / nsh)}
    return {"qubits": [NAMES[k] for k in qs], "n_err": float(ne[0]), "n_shots": nsh,
            "err_rate_pooled": float(ne[0] / nsh),
            "leakage_share_mirror_ci95": ci_mirror(float(acc["Lplus"][0]), float(acc["Lminus"][0]), float(ne[0])),
            "order_L_then_T": {"leakage": sh(acc["L"]), "transition_fp_corrected": sh(acc["T_det"]),
                               "residual": sh(ne - acc["L"] - acc["T_det"])},
            "transition_conservative_bound": sh(acc["T_cons"]),
            "residual_if_conservative_T": sh(ne - acc["L"] - acc["T_cons"]),
            "order_T_then_L": {"transition_fp_corrected": sh(acc["T_det_all"]), "leakage": sh(acc["L_after_T"]),
                               "residual": sh(ne - acc["T_det_all"] - acc["L_after_T"])},
            "deconv_N_overlap": sh(acc["N_ov"]), "deconv_N_wrongstate": sh(acc["N_wr"]),
            "deconv_N_transition_sens_corrected": sh(acc["N_tr"]),
            "note": "Q4 leakage n/m (counted as 0 in the pooled leakage share)" if Q4 in qs else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nsim", type=int, default=6000)
    ap.add_argument("--ninj", type=int, default=250)
    ap.add_argument("--nboot", type=int, default=300)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--kinds", default="featbox,feat")
    ap.add_argument("--qubits", default="0,1,2,3,4")
    ap.add_argument("--out", default="/work/out")
    a = ap.parse_args()
    global OUT
    OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    meta = prepare(a)
    print("shared inputs ready", round(time.time() - t0), flush=True)
    jobs = [(k, q, a) for k in a.kinds.split(",") for q in map(int, a.qubits.split(","))]
    with Pool(min(a.procs, len(jobs))) as pool:
        results = pool.map(run_job, jobs, chunksize=1)
    pooled = {}
    for kind in a.kinds.split(","):
        for wname in WINDOWS:
            for ename in ("canon", "soft"):
                for nm, qs in (("all5", [0, 1, 2, 3, 4]), ("Q1Q3Q4Q5", [0, 2, 3, 4]), ("Q1Q2Q3Q5", [0, 1, 2, 4])):
                    if not all(any(r["kind"] == kind and r["q"] == k for r in results) for k in qs):
                        continue
                    for v in ("p50", "fut_p50", "viterbi"):
                        p = pool_boot(results, qs, wname, ename, kind, v, 120000 if a.smoke else None)
                        if p is not None:
                            pooled[f"{kind}|{wname}|{ename}|{nm}|{v}"] = p
    out = {"script": "em_e1hmm_attrib_v2.py", "doc": __doc__, "args": vars(a), "meta": meta, "pooled": pooled,
           "per_qubit": results, "runtime_s": time.time() - t0}
    json.dump(jsonable(out), open(f"{OUT}/em_e1hmm_results_full.json", "w"), indent=1)
    print("all done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
