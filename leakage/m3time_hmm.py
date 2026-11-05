#!/usr/bin/env python3
"""m3time_hmm.py -- TEMPORAL SIGNATURE of leakage (agent tag m3time).

Per qubit q (code q0..q4 = paper Q1..Q5) and per shot we use the 20 x 100 ns binned,
demodulated, rotated IQ trajectory (feat_<split>.npy, [...,0]=along, [...,1]=perp).

Model (one per qubit): hidden Markov model with
  * states 0=|0>, 1=|1>, 2=L ("off-line" third state whose per-bin mean is FREE),
  * per-(state, bin) 2-D emission means mu[s,k] (absorbs ring-up in bins 0-1, the
    zero-padding gain in bin 19 and slow common-mode drifts such as Q2's 0.63 MHz
    mirror-image beat),
  * per-bin 2x2 emission covariance Sig[k], TIED across states (amplifier noise does
    not depend on the qubit state; untying lets a broad state swallow all tails),
  * a per-prepared-bit initial distribution pi[b] (b = prepared bit of q),
  * one time-homogeneous 3x3 transition matrix A per 100 ns, shared by both prepared bits.
Fitted by Baum-Welch on TRAIN (several L-initialisations on a subsample, the best refined
on all of TRAIN) and then applied FROZEN to TEST (held out). A 2-state model (0,1) is fitted
the same way to give a held-out likelihood comparison.

Static crosstalk: before fitting, each shot's trajectory has the per-bin offset of its
neighbour configuration (the 16 prepared states of the other four qubits) subtracted. The
offsets are per-bin medians of own-prepared-|0> TRAIN shots of that configuration minus the
pooled own-|0> median (--xt 0 disables this).

On TEST we report: soft occupancies, Viterbi paths (raw and with 1-bin blips removed),
start-state counts, exit destinations/times, survival curves, censored-geometric lifetimes,
HMM-matrix lifetimes with delta-method CIs, mid-readout entries into L, the location of L
per prepared bit and per neighbour configuration, cross-qubit co-occurrence flags, and the
strongest example shots (indices) for raw-trace inspection.

Run (graham):
  docker run --rm --user $(id -u):$(id -g) -e OMP_NUM_THREADS=6 \
    -v ~/oraqle-ics-sims/leakage/features:/feat:ro -v ~/oraqle-ics-sims/leakage/m3time:/m3 \
    oraqle-torch:latest python /m3/scripts/m3time_hmm.py --kind feat --out /m3/out/feat_xt1
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import argparse, json, os, sys, time
import numpy as np
from multiprocessing import Pool
from scipy.stats import beta as _beta, norm as _norm

FEAT = str(LP.FEAT)
NB = 20
DT_NS = 100.0
L = 2                      # index of the third state
NAMES = ["Q1", "Q2", "Q3", "Q4", "Q5"]


# ----------------------------------------------------------------------------- utils
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def cp_ci(k, n, a=0.05):
    """Clopper-Pearson interval for k successes out of n."""
    k = int(k); n = int(n)
    if n == 0:
        return [None, None]
    lo = 0.0 if k == 0 else float(_beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1 - a / 2, k + 1, n - k))
    return [lo, hi]


def frac(k, n):
    return {"k": int(k), "n": int(n), "frac": (k / n if n else None), "ci95": cp_ci(k, n)}


def tau_from_p(p):
    """mean dwell (ns) for a per-100ns exit probability p (geometric -> exponential)."""
    if p is None or p <= 0:
        return float("inf")
    if p >= 1:
        return 0.0
    return -DT_NS / np.log1p(-p)


def cfg_other(y, q):
    """index 0..15 of the prepared bits of the other four qubits."""
    return ((y >> (q + 1)) << q) | (y & ((1 << q) - 1))


def log(q, msg):
    print(f"[m3time q{q}/{NAMES[q]}] {msg}", flush=True)


# ----------------------------------------------------------------------------- HMM core
def log_emis(X, mu, Sig):
    """X (N,K,2), mu (S,K,2), Sig (K,2,2) -> (N,K,S) log N(x; mu, Sig)."""
    N, K, _ = X.shape
    S = mu.shape[0]
    inv = np.linalg.inv(Sig)
    logdet = np.log(np.linalg.det(Sig))
    a, b, c = inv[:, 0, 0], inv[:, 0, 1], inv[:, 1, 1]
    out = np.empty((N, K, S))
    for s in range(S):
        d0 = X[:, :, 0] - mu[s, :, 0]
        d1 = X[:, :, 1] - mu[s, :, 1]
        out[:, :, s] = -0.5 * (a * d0 * d0 + 2 * b * d0 * d1 + c * d1 * d1) - 0.5 * logdet - np.log(2 * np.pi)
    return out


def fb(logB, A, pi0, want_xi_shot=False):
    """Scaled forward-backward. logB (N,K,S), A (S,S), pi0 (N,S).
    Returns gamma (N,K,S), xi_sum (S,S), ll (N,), [xi_shot (N,S,S)]."""
    N, K, S = logB.shape
    mx = logB.max(2, keepdims=True)
    B = np.exp(logB - mx)
    alpha = np.empty_like(B)
    c = np.empty((N, K))
    a = pi0 * B[:, 0]
    c[:, 0] = a.sum(1)
    alpha[:, 0] = a / c[:, 0, None]
    for k in range(1, K):
        a = (alpha[:, k - 1] @ A) * B[:, k]
        c[:, k] = a.sum(1)
        alpha[:, k] = a / c[:, k, None]
    beta = np.empty_like(B)
    beta[:, K - 1] = 1.0
    xi = np.zeros((S, S))
    xis = np.zeros((N, S, S)) if want_xi_shot else None
    for k in range(K - 2, -1, -1):
        bb = B[:, k + 1] * beta[:, k + 1] / c[:, k + 1, None]
        beta[:, k] = bb @ A.T
        if want_xi_shot:
            t = alpha[:, k, :, None] * A[None] * bb[:, None, :]
            xis += t
            xi += t.sum(0)
        else:
            xi += A * (alpha[:, k].T @ bb)
    gamma = alpha * beta
    gamma /= gamma.sum(2, keepdims=True)
    ll = np.log(c).sum(1) + mx[..., 0].sum(1)
    if want_xi_shot:
        return gamma, xi, ll, xis
    return gamma, xi, ll


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


def em(X, bp, P, iters, tol=1e-8, q=0, tag=""):
    """Baum-Welch. P = dict(mu (S,K,2), Sig (K,2,2), A (S,S), pi (2,S))."""
    mu, Sig, A, pi = P["mu"].copy(), P["Sig"].copy(), P["A"].copy(), P["pi"].copy()
    N, K, _ = X.shape
    S = mu.shape[0]
    hist = []
    t0 = time.time()
    for it in range(iters):
        logB = log_emis(X, mu, Sig)
        gamma, xi, ll = fb(logB, A, pi[bp])
        hist.append(float(ll.mean()))
        for b in (0, 1):
            pi[b] = np.maximum(gamma[bp == b, 0].mean(0), 1e-12)
            pi[b] /= pi[b].sum()
        A = (xi + 1e-9) / (xi + 1e-9).sum(1, keepdims=True)
        w = gamma.sum(0) + 1e-12                                     # (K,S)
        mu = np.einsum("nks,nkd->skd", gamma, X) / w.T[:, :, None]
        Sig = np.zeros((K, 2, 2))
        for s in range(S):
            d0 = X[:, :, 0] - mu[s, :, 0]
            d1 = X[:, :, 1] - mu[s, :, 1]
            g = gamma[:, :, s]
            Sig[:, 0, 0] += (g * d0 * d0).sum(0)
            Sig[:, 0, 1] += (g * d0 * d1).sum(0)
            Sig[:, 1, 1] += (g * d1 * d1).sum(0)
        Sig[:, 1, 0] = Sig[:, 0, 1]
        Sig /= N
        if it % 25 == 0:
            log(q, f"EM{tag} it {it} ll/N {hist[-1]:.6f} ({time.time()-t0:.0f}s)")
        if it > 5 and abs(hist[-1] - hist[-2]) < tol:
            break
    return {"mu": mu, "Sig": Sig, "A": A, "pi": pi}, hist


# ----------------------------------------------------------------------------- setup
def load(split, q, kind):
    F = np.load(f"{FEAT}/{kind}_{split}.npy", mmap_mode="r")
    X = np.asarray(F[:, q], dtype=np.float64)
    y = np.load(f"{FEAT}/y_{split}.npy").astype(np.int64)
    idx = np.load(f"{FEAT}/idx_{split}.npy")
    return X, y, idx


def robust_cov(Z):
    """Z (n,2) -> 2x2 covariance from the core (iterative 3-sigma clipping)."""
    m = np.median(Z, 0)
    C = np.cov((Z - m).T)
    for _ in range(5):
        Ci = np.linalg.inv(C)
        d = np.einsum("ni,ij,nj->n", Z - m, Ci, Z - m)
        sel = d < 9.0
        m = Z[sel].mean(0)
        C = np.cov((Z[sel] - m).T) / 0.9494   # per-dim variance retained by a Mahalanobis^2<9 clip of a 2-D Gaussian
    return m, C


def init_params(X, bp, q, S=3):
    """Initial means from prepared-bit medians; L candidates from window-mean outliers."""
    K = X.shape[1]
    mu0 = np.median(X[bp == 0], 0)            # (K,2)
    mu1 = np.median(X[bp == 1], 0)
    Sig = np.stack([robust_cov(X[bp == 0, k])[1] for k in range(K)])
    w = X[:, 2:].mean(1)
    c0, c1 = np.median(w[bp == 0], 0), np.median(w[bp == 1], 0)
    sep = np.linalg.norm(c1 - c0)
    u = (c1 - c0) / sep
    v = np.array([-u[1], u[0]])
    sw = np.median(np.abs(w[bp == 0] - c0), 0).mean() * 1.4826
    t = (w - c0) @ u / sep
    dp = (w - c0) @ v / sw
    out = np.where(t < 0, -t, np.where(t > 1, t - 1, 0.0)) * sep / sw
    dist = np.sqrt(dp ** 2 + out ** 2)
    inits = []
    for thr in (6.0, 4.5, 3.5, 3.0):
        cand = dist > thr
        if cand.sum() >= 300:
            break
    W = w[cand]
    lo, hi = np.percentile(W, 0.5, 0), np.percentile(W, 99.5, 0)
    W = np.clip(W, lo, hi)
    modes = []
    rem = np.ones(len(W), bool)
    for _ in range(2):
        if rem.sum() < 100:
            break
        H, ex, ey = np.histogram2d(W[rem, 0], W[rem, 1], bins=[np.arange(W[:, 0].min() - sw, W[:, 0].max() + 2 * sw, sw),
                                                               np.arange(W[:, 1].min() - sw, W[:, 1].max() + 2 * sw, sw)])
        i, j = np.unravel_index(H.argmax(), H.shape)
        m = np.array([(ex[i] + ex[i + 1]) / 2, (ey[j] + ey[j + 1]) / 2])
        near = np.linalg.norm(W - m, axis=1) < 2 * sw
        sel = np.where(cand)[0][near & rem]
        if len(sel) >= 30:
            inits.append(("mode%d" % len(modes), np.median(X[sel], 0)))
            modes.append(m)
        rem &= np.linalg.norm(W - m, axis=1) > 4 * sw
    sb = np.sqrt(np.mean([np.trace(S_) / 2 for S_ in Sig[2:]]))
    off = np.zeros((K, 2))
    off[2:] = 1.0
    inits.append(("beyond1", mu1 + (mu1 - mu0) * 0.8))
    inits.append(("perp+", mu1 + off * v[None] * 4 * sb))
    inits.append(("perp-", mu1 - off * v[None] * 4 * sb))
    A = np.full((S, S), 0.004)
    np.fill_diagonal(A, 0.992)
    pi = np.array([[0.96, 0.03, 0.01], [0.04, 0.95, 0.01]])
    base = {"mu0": mu0, "mu1": mu1, "Sig": Sig, "A": A, "pi": pi}
    geom = {"c0_w": c0, "c1_w": c1, "sep_w": sep, "sigma_w": sw, "u": u, "v": v, "thr_used": thr,
            "n_cand": int(cand.sum()), "modes_w": modes}
    return base, inits, geom


def P3(base, mu2):
    return {"mu": np.stack([base["mu0"], base["mu1"], mu2]), "Sig": base["Sig"].copy(),
            "A": base["A"].copy(), "pi": base["pi"].copy()}


def P2(base):
    A = np.array([[0.995, 0.005], [0.005, 0.995]])
    pi = np.array([[0.97, 0.03], [0.04, 0.96]])
    return {"mu": np.stack([base["mu0"], base["mu1"]]), "Sig": base["Sig"].copy(), "A": A, "pi": pi}


def state_check(P, base):
    """distance (bins 3..19) of each fitted state mean to the |0>/|1> inits: EM could swap labels."""
    m = P["mu"][:, 3:].mean(1)
    a0, a1 = base["mu0"][3:].mean(0), base["mu1"][3:].mean(0)
    return [[float(np.linalg.norm(m[s] - a0)), float(np.linalg.norm(m[s] - a1))] for s in range(len(m))]


# ----------------------------------------------------------------------------- analysis helpers
def blip_filter(path):
    """remove isolated 1-bin runs (incl. a lone final bin): they are replaced by the previous state."""
    p = path.copy()
    K = p.shape[1]
    for k in range(1, K - 1):
        iso = (p[:, k] != p[:, k - 1]) & (p[:, k] != p[:, k + 1])
        p[iso, k] = p[iso, k - 1]
    iso = p[:, K - 1] != p[:, K - 2]
    p[iso, K - 1] = p[iso, K - 2]
    return p


def start_state(p):
    return np.where(p[:, 1] == p[:, 2], p[:, 1], -1)


def exit_stats(p, sel, s, Kw):
    """shots in sel that START in s (bins 1-2): first exit at bin e in [3, Kw-1]."""
    P = p[sel][:, :Kw]
    n = len(P)
    if n == 0:
        return None
    notS = P[:, 3:] != s
    has = notS.any(1)
    e = np.where(has, notS.argmax(1) + 3, Kw)
    dest = np.full(n, -1)
    dest[has] = P[np.where(has)[0], e[has]]
    exposure = int(np.where(has, e - 2, Kw - 3).sum())
    nev = int(has.sum())
    res = {"n_start": n, "exposure_bin_steps": exposure, "n_exit": nev,
           "n_exit_to": {str(d): int((dest == d).sum()) for d in range(3) if d != s},
           "survival_bins": list(range(2, Kw)),
           "survival": [float((e > k).mean()) for k in range(2, Kw)],
           "exit_bin_hist": [int(((e == k) & has).sum()) for k in range(Kw)]}
    p_all = nev / exposure if exposure else None
    ci = cp_ci(nev, exposure)
    res["p_exit_per_100ns"] = p_all
    res["p_exit_ci95"] = ci
    res["tau_ns"] = tau_from_p(p_all)
    res["tau_ns_ci95"] = [tau_from_p(ci[1]), tau_from_p(ci[0])]
    for d in range(3):
        if d == s:
            continue
        k_ = int((dest == d).sum())
        cid = cp_ci(k_, exposure)
        res[f"p_to{d}_per_100ns"] = k_ / exposure if exposure else None
        res[f"p_to{d}_ci95"] = cid
        res[f"tau_to{d}_ns"] = tau_from_p(k_ / exposure if exposure else None)
        res[f"tau_to{d}_ns_ci95"] = [tau_from_p(cid[1]), tau_from_p(cid[0])]
        res[f"branch_to{d}"] = frac(k_, nev)
    return res, e, dest


def mid_entries(p, sel_b, st, Kw):
    """shots of sel_b that start in 0 or 1 and later (bin >=3, < Kw) enter L."""
    P = p[:, :Kw]
    base = sel_b & ((st == 0) | (st == 1))
    isL = P[:, 3:] == L
    has = isL.any(1) & base
    k = np.where(isL.any(1), isL.argmax(1) + 3, -1)
    frm = np.full(len(P), -1)
    ii = np.where(has)[0]
    frm[ii] = P[ii, k[ii] - 1]
    # dwell after entry
    dwell = np.zeros(len(P), int)
    after = np.full(len(P), -1)
    # vectorised dwell: for each entry shot, count consecutive L from k
    if len(ii):
        sub = P[ii]
        kk = k[ii]
        cols = np.arange(Kw)[None]
        afterL = (cols >= kk[:, None]) & (sub != L)
        has_end = afterL.any(1)
        end = np.where(has_end, afterL.argmax(1), Kw)
        dwell[ii] = end - kk
        after[ii] = np.where(has_end, sub[np.arange(len(ii)), np.minimum(end, Kw - 1)], -1)
    n_base = int(base.sum())
    res = {"n_start_0or1": n_base, "n_enter_L": int(has.sum()), "enter_L": frac(has.sum(), n_base),
           "from0": frac(((frm == 0) & has).sum(), n_base), "from1": frac(((frm == 1) & has).sum(), n_base),
           "entry_bin_hist": [int(((k == j) & has).sum()) for j in range(Kw)],
           "dwell_hist_bins": [int(((dwell == j) & has).sum()) for j in range(Kw + 1)],
           "after_L_state": {str(d): int(((after == d) & has).sum()) for d in (-1, 0, 1)}}
    for s0 in (0, 1):
        b2 = sel_b & (st == s0)
        res[f"start{s0}_enter_L"] = frac((has & (st == s0)).sum(), b2.sum())
    return res, has, k, frm


def delta_ratio_ci(num_shot, den_shot):
    """ratio R = sum num / sum den with delta-method SE (shots i.i.d.)."""
    Sn, Sd = num_shot.sum(), den_shot.sum()
    R = Sn / Sd
    z = num_shot - R * den_shot
    se = np.sqrt((z * z).sum()) / Sd
    return float(R), float(se), [float(R - 1.96 * se), float(R + 1.96 * se)]


# ----------------------------------------------------------------------------- per-qubit job
def run_qubit(args):
    q, a = args
    t0 = time.time()
    rng = np.random.default_rng(1234 + q)
    out = {"qubit_code": f"q{q}", "qubit_paper": NAMES[q], "kind": a.kind, "xt_correction": bool(a.xt),
           "bin_ns": DT_NS, "n_bins": NB}
    Xtr, ytr, idxtr = load("train", q, a.kind)
    Xte, yte, idxte = load("test", q, a.kind)
    btr, bte = (ytr >> q) & 1, (yte >> q) & 1
    ctr, cte = cfg_other(ytr, q), cfg_other(yte, q)
    # --- static crosstalk offsets (TRAIN, own=0)
    m0 = np.median(Xtr[btr == 0], 0)
    offs = np.zeros((16, NB, 2))
    for c in range(16):
        offs[c] = np.median(Xtr[(btr == 0) & (ctr == c)], 0) - m0
    out["xt_offsets_absmax"] = float(np.abs(offs).max())
    if a.xt:
        Xtr = Xtr - offs[ctr]
        Xte = Xte - offs[cte]
    # --- noise correlation between adjacent bins (prepared |0> core, bins 2..18)
    Z = Xte[bte == 0][:200000]
    Zc = Z - np.median(Z, 0)
    core = (np.abs(Zc[:, 2:]) < 3 * 1.4826 * np.median(np.abs(Zc[:, 2:]), 0)).all((1, 2))
    Zc = Zc[core]
    out["noise_lag1_corr_prep0_core"] = {
        "along": float(np.mean([np.corrcoef(Zc[:, k, 0], Zc[:, k + 1, 0])[0, 1] for k in range(2, 18)])),
        "perp": float(np.mean([np.corrcoef(Zc[:, k, 1], Zc[:, k + 1, 1])[0, 1] for k in range(2, 18)])),
        "lag2_along": float(np.mean([np.corrcoef(Zc[:, k, 0], Zc[:, k + 2, 0])[0, 1] for k in range(2, 17)]))}
    # --- init + EM restarts on a TRAIN subsample
    base, inits, geom = init_params(Xtr, btr, q)
    out["init_geom"] = geom
    sub = rng.choice(len(Xtr), size=min(a.nsub, len(Xtr)), replace=False)
    restarts = []
    best = None
    for name, mu2 in inits:
        P, h = em(Xtr[sub], btr[sub], P3(base, mu2), a.iters_init, q=q, tag=f"[{name}]")
        restarts.append({"init": name, "ll_per_shot": h[-1], "n_iter": len(h),
                         "mu2_mean_bins3_19": P["mu"][2, 3:].mean(0), "pi": P["pi"], "A": P["A"]})
        log(q, f"restart {name}: ll/N {h[-1]:.6f}  mu2 {P['mu'][2,3:].mean(0)}  A_LL {P['A'][2,2]:.4f}")
        if best is None or h[-1] > best[1]:
            best = (P, h[-1], name)
    out["restarts_subsample"] = restarts
    out["best_init"] = best[2]
    # --- refine on all TRAIN
    Ptr, htr = em(Xtr, btr, best[0], a.iters, q=q, tag="[train3]")
    out["train_fit3"] = {"ll_per_shot": htr[-1], "n_iter": len(htr), "ll_hist_tail": htr[-5:],
                         "state_dist_to_inits_0_1": state_check(Ptr, base)}
    P2tr, h2 = em(Xtr, btr, P2(base), a.iters, q=q, tag="[train2]")
    out["train_fit2"] = {"ll_per_shot": h2[-1], "n_iter": len(h2)}
    # --- TEST refit (replication), warm start from the train solution
    Pte, hte = em(Xte, bte, Ptr, a.iters_test, q=q, tag="[test3]")
    out["test_refit3"] = {"ll_per_shot": hte[-1], "n_iter": len(hte)}
    for nm, P in (("train", Ptr), ("test", Pte)):
        A = P["A"]
        o = {"A": A, "pi_prep0": P["pi"][0], "pi_prep1": P["pi"][1],
             "mu": P["mu"], "Sig": P["Sig"]}
        for s in range(3):
            o[f"tau_state{s}_ns"] = tau_from_p(1 - A[s, s])
            o[f"branch_from{s}"] = (A[s] / (1 - A[s, s])).tolist()
        out[f"params_{nm}"] = o
    # --- held-out analysis on TEST with frozen TRAIN params
    P = Ptr
    logB = log_emis(Xte, P["mu"], P["Sig"])
    gamma, xi, ll3, xis = fb(logB, P["A"], P["pi"][bte], want_xi_shot=True)
    logB2 = log_emis(Xte, P2tr["mu"], P2tr["Sig"])
    _, _, ll2 = fb(logB2, P2tr["A"], P2tr["pi"][bte])
    dll = ll3 - ll2
    nparam3 = 3 * NB * 2 + NB * 3 + 6 + 4 * 2
    nparam2 = 2 * NB * 2 + NB * 3 + 2 + 2 * 1
    out["heldout_ll"] = {"ll3_per_shot": float(ll3.mean()), "ll2_per_shot": float(ll2.mean()),
                         "delta_total": float(dll.sum()), "delta_per_shot": float(dll.mean()),
                         "frac_shots_dll_gt_5": float((dll > 5).mean()),
                         "frac_shots_dll_gt_5_prep": [float((dll[bte == b] > 5).mean()) for b in (0, 1)],
                         "n_param_3": nparam3, "n_param_2": nparam2}
    # --- geometry of L in physical units (TEST-fitted means, bins 3..9 and 3..19)
    geo = {}
    for wn, (k0, k1) in {"1us": (3, 10), "2us": (3, 20)}.items():
        mu = P["mu"][:, k0:k1].mean(1)          # (3,2)
        sep = mu[1] - mu[0]
        sepn = np.linalg.norm(sep)
        u = sep / sepn
        v = np.array([-u[1], u[0]])
        sb = np.sqrt(np.mean([np.trace(S_) / 2 for S_ in P["Sig"][k0:k1]]))
        d2 = mu[2] - mu[0]
        geo[wn] = {"mu0": mu[0], "mu1": mu[1], "muL": mu[2], "sep": sepn, "sigma_bin": sb,
                   "L_along_in_sep": float(d2 @ u / sepn), "L_perp_in_sep": float(d2 @ v / sepn),
                   "L_perp_in_sigma_bin": float(d2 @ v / sb),
                   "L_dist_to_1_in_sigma_bin": float(np.linalg.norm(mu[2] - mu[1]) / sb),
                   "L_dist_to_0_in_sigma_bin": float(np.linalg.norm(mu[2] - mu[0]) / sb),
                   "sep_in_sigma_bin": float(sepn / sb)}
    geo["muL_per_bin"] = P["mu"][2]
    geo["mu0_per_bin"] = P["mu"][0]
    geo["mu1_per_bin"] = P["mu"][1]
    out["geometry_train_params"] = geo
    # --- soft occupancies
    occ = {}
    for b in (0, 1):
        g = gamma[bte == b]
        occ[f"prep{b}"] = {"P_L_per_bin": g[:, :, L].mean(0), "P_1_per_bin": g[:, :, 1].mean(0),
                           "P_0_per_bin": g[:, :, 0].mean(0),
                           "P_L_bin1_se": float(g[:, 1, L].std() / np.sqrt(len(g))),
                           "n": int(len(g))}
        # posterior initial-state probability = one-step M-step estimate of pi
        occ[f"prep{b}"]["pi_L_onestep"] = float(g[:, 0, L].mean())
        occ[f"prep{b}"]["pi_L_onestep_ci95"] = [float(g[:, 0, L].mean() - 1.96 * g[:, 0, L].std() / np.sqrt(len(g))),
                                                float(g[:, 0, L].mean() + 1.96 * g[:, 0, L].std() / np.sqrt(len(g)))]
    out["soft_occupancy_test"] = occ
    # --- HMM transition rates on TEST with delta-method CIs (posteriors from frozen TRAIN params)
    gnl = gamma[:, :-1].sum(1)                      # (N,S) expected visits that have a successor
    rates = {}
    for s in range(3):
        out_s = xis[:, s, :].sum(1) - xis[:, s, s]
        R, se, ci = delta_ratio_ci(out_s, gnl[:, s])
        rates[f"from{s}"] = {"p_exit_per_100ns": R, "ci95": ci, "tau_ns": tau_from_p(R),
                             "tau_ns_ci95": [tau_from_p(ci[1]), tau_from_p(ci[0])]}
        for d in range(3):
            if d == s:
                continue
            R2, se2, ci2 = delta_ratio_ci(xis[:, s, d], gnl[:, s])
            Rb, seb, cib = delta_ratio_ci(xis[:, s, d], out_s)
            rates[f"from{s}"][f"p_to{d}"] = R2
            rates[f"from{s}"][f"p_to{d}_ci95"] = ci2
            rates[f"from{s}"][f"branch_to{d}"] = Rb
            rates[f"from{s}"][f"branch_to{d}_ci95"] = cib
        for b in (0, 1):
            m = bte == b
            out_sb = out_s[m]
            R3, _, ci3 = delta_ratio_ci(out_sb, gnl[m, s])
            rates[f"from{s}"][f"prep{b}_p_exit"] = R3
            rates[f"from{s}"][f"prep{b}_p_exit_ci95"] = ci3
            for d in range(3):
                if d == s:
                    continue
                R4, _, ci4 = delta_ratio_ci(xis[m, s, d], gnl[m, s])
                rates[f"from{s}"][f"prep{b}_p_to{d}"] = R4
                rates[f"from{s}"][f"prep{b}_p_to{d}_ci95"] = ci4
    out["hmm_rates_test_onestep"] = rates
    del xis
    # --- Viterbi (hard) analysis
    path_raw = viterbi(logB, P["A"], P["pi"][bte])
    path = blip_filter(path_raw)
    hard = {}
    for pname, pp in (("viterbi_raw", path_raw), ("viterbi_filtered", path)):
        st = start_state(pp)
        H = {}
        for wn, Kw in (("1us", 10), ("2us", 20)):
            W = {}
            for b in (0, 1):
                sb_ = bte == b
                nb = int(sb_.sum())
                W[f"prep{b}"] = {"start_L": frac(((st == L) & sb_).sum(), nb),
                                 "start_0": frac(((st == 0) & sb_).sum(), nb),
                                 "start_1": frac(((st == 1) & sb_).sum(), nb),
                                 "start_ambiguous": frac(((st == -1) & sb_).sum(), nb),
                                 "ever_L": frac(((pp[:, :Kw] == L).any(1) & sb_).sum(), nb),
                                 "L_at_bin": [int(((pp[:, k] == L) & sb_).sum()) for k in range(Kw)]}
                for s in range(3):
                    r = exit_stats(pp, sb_ & (st == s), s, Kw)
                    W[f"prep{b}"][f"exit_from{s}"] = r[0] if r else None
                me, _, _, _ = mid_entries(pp, sb_, st, Kw)
                W[f"prep{b}"]["mid_entry_L"] = me
            for s in range(3):
                r = exit_stats(pp, st == s, s, Kw)
                W[f"pooled_exit_from{s}"] = r[0] if r else None
            H[wn] = W
        hard[pname] = H
    out["hard_test"] = hard
    # --- location of L per prepared bit / neighbour config (filtered Viterbi, bins 3..19 in L)
    st = start_state(path)
    mask = (path[:, 3:] == L)
    Xl = Xte[:, 3:]
    geo2 = geo["2us"]
    u = (geo2["mu1"] - geo2["mu0"]) / geo2["sep"]
    v = np.array([-u[1], u[0]])
    loc = {}

    def locate(sel, add_off=False):
        mm = mask[sel]
        n = int(mm.sum())
        if n < 30:
            return {"n_bins": n}
        Z = Xl[sel] + (offs[cte[sel]][:, 3:] if (add_off and a.xt) else 0)
        z = Z[mm]
        d = z.mean(0) - geo2["mu0"]
        return {"n_bins": n, "n_shots": int(mm.any(1).sum()), "mean": z.mean(0),
                "along_in_sep": float(d @ u / geo2["sep"]), "perp_in_sep": float(d @ v / geo2["sep"]),
                "perp_in_sigma_bin": float(d @ v / geo2["sigma_bin"])}
    for b in (0, 1):
        loc[f"prep{b}"] = locate(bte == b)
        loc[f"prep{b}_startL"] = locate((bte == b) & (st == L))
        loc[f"prep{b}_midentry"] = locate((bte == b) & (st != L))
    loc["per_config_prep1"] = {str(c): locate((bte == 1) & (cte == c)) for c in range(16)}
    loc["per_config_prep0"] = {str(c): locate((bte == 0) & (cte == c)) for c in range(16)}
    loc["per_config_prep1_uncorrected"] = {str(c): locate((bte == 1) & (cte == c), True) for c in range(16)}
    out["L_location_test"] = loc
    # --- per-config start-L fractions (prepared 1 / 0) to check neighbour dependence
    pc = {}
    for b in (0, 1):
        pc[f"prep{b}"] = {str(c): frac(((st == L) & (bte == b) & (cte == c)).sum(), ((bte == b) & (cte == c)).sum())
                          for c in range(16)}
    out["start_L_per_config"] = pc
    # --- examples (TEST rows)
    ex = []
    gL, g1, g0 = gamma[:, :, L], gamma[:, :, 1], gamma[:, :, 0]
    rr = exit_stats(path, st == L, L, NB)
    iL = np.where(st == L)[0]
    e, dest = (rr[1], rr[2]) if rr else (np.zeros(0, int), np.zeros(0, int))

    def add(cat, rows, score, n):
        if len(rows) == 0:
            return
        o = np.argsort(-score)[:n]
        for j in o:
            i = rows[j]
            ex.append({"cat": cat, "test_row": int(i), "raw_idx": int(idxte[i]), "y": int(yte[i]),
                       "prep_bit": int(bte[i]), "cfg_other": int(cte[i]),
                       "path_raw": "".join("01L"[s] for s in path_raw[i]),
                       "path": "".join("01L"[s] for s in path[i]),
                       "postL": np.round(gL[i], 3), "score": float(score[j])})
    # A: start L -> 1
    m = (dest == 1) & (e >= 4)
    rows = iL[m]
    sc = np.array([gL[i, 1:ee].mean() * (ee - 1) + g1[i, ee:].mean() * (NB - ee) for i, ee in zip(rows, e[m])])
    add("startL_to1", rows, sc, 7)
    m = (dest == 0) & (e >= 4)
    rows = iL[m]
    sc = np.array([gL[i, 1:ee].mean() * (ee - 1) + g0[i, ee:].mean() * (NB - ee) for i, ee in zip(rows, e[m])])
    add("startL_to0", rows, sc, 2)
    m = dest == -1
    rows = iL[m]
    add("startL_stays", rows, gL[rows, 1:].sum(1), 5)
    _, has, kk, frm = mid_entries(path, np.ones(len(path), bool), st, NB)
    for s0, n in ((1, 3), (0, 2)):
        rows = np.where(has & (frm == s0) & (kk >= 4))[0]
        g_s = g1 if s0 == 1 else g0
        sc = np.array([g_s[i, 1:k].mean() * (k - 1) + gL[i, k:].mean() * (NB - k) for i, k in zip(rows, kk[rows])])
        add(f"mid_{s0}toL", rows, sc, n)
    rr1 = exit_stats(path, st == 1, 1, NB)
    if rr1:
        _, e1, d1 = rr1
        i1 = np.where(st == 1)[0]
        m = (d1 == 0) & (e1 >= 6) & (e1 <= 14)
        rows = i1[m]
        sc = np.array([g1[i, 1:ee].mean() * (ee - 1) + g0[i, ee:].mean() * (NB - ee) for i, ee in zip(rows, e1[m])])
        add("T1_1to0_ref", rows, sc, 3)
    out["examples"] = ex
    out["offsets_xt"] = offs
    # arrays for co-occurrence / later use (cluster only)
    np.savez_compressed(os.path.join(a.out, f"m3time_q{q}_arrays.npz"),
                        path=path, path_raw=path_raw, gL=gL.astype(np.float16), ll3=ll3.astype(np.float32),
                        ll2=ll2.astype(np.float32), st=st.astype(np.int8))
    out["runtime_s"] = time.time() - t0
    with open(os.path.join(a.out, f"m3time_q{q}.json"), "w") as f:
        json.dump(jsonable(out), f, indent=1)
    log(q, f"done in {time.time()-t0:.0f}s")
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="feat", choices=["feat", "featbox"])
    ap.add_argument("--xt", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qubits", default="0,1,2,3,4")
    ap.add_argument("--nsub", type=int, default=120000)
    ap.add_argument("--iters_init", type=int, default=60)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--iters_test", type=int, default=150)
    ap.add_argument("--procs", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    qs = [int(x) for x in a.qubits.split(",")]
    with Pool(min(a.procs, len(qs))) as pool:
        for q in pool.imap_unordered(run_qubit, [(q, a) for q in qs]):
            print(f"[m3time] finished q{q}", flush=True)
    # cross-qubit co-occurrence of start-L / ever-L (TEST)
    if len(qs) == 5:
        y = np.load(f"{FEAT}/y_test.npy").astype(np.int64)
        flags = {}
        for q in qs:
            z = np.load(os.path.join(a.out, f"m3time_q{q}_arrays.npz"))
            flags[q] = {"startL": z["st"] == L, "everL": (z["path"] == L).any(1)}
        co = {}
        for f in ("startL", "everL"):
            M = {}
            for q in qs:
                for r in qs:
                    if r == q:
                        continue
                    a_, b_ = flags[q][f], flags[r][f]
                    pa, pb, pab = a_.mean(), b_.mean(), (a_ & b_).mean()
                    M[f"{NAMES[q]}&{NAMES[r]}"] = {"P_q": float(pa), "P_r": float(pb), "P_both": float(pab),
                                                  "lift": float(pab / (pa * pb)) if pa * pb > 0 else None,
                                                  "share_of_q_also_r": float(pab / pa) if pa > 0 else None}
            co[f] = M
            # share of each qubit's flagged shots flagged on ANY other qubit
            for q in qs:
                other = np.zeros(len(y), bool)
                for r in qs:
                    if r != q:
                        other |= flags[r][f]
                co[f][f"{NAMES[q]}_share_any_other"] = float((flags[q][f] & other).sum() / max(flags[q][f].sum(), 1))
        with open(os.path.join(a.out, "m3time_cooccurrence.json"), "w") as fh:
            json.dump(jsonable(co), fh, indent=1)
    print("[m3time] all done", flush=True)


if __name__ == "__main__":
    main()
