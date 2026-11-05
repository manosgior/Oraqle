#!/usr/bin/env python
"""m1tail_analysis.py  (agent tag m1tail)

OFF-AXIS TAIL STATISTICS AGAINST AN EXPLICIT NULL, for the Lienhard et al. 5-qubit
readout dataset, using the shared feature dataset built by the "x" agent
(~/oraqle-ics-sims/leakage/features, read-only).

Null hypothesis H0 (per qubit q, window w):
    perp = mu_s + sigma * eps,  eps ~ N(0,1),
    mu_s depends on the full prepared 5-bit state s (crosstalk shifts from the
    other 4 qubits' prepared bits), sigma = amplifier noise, state-independent.
Standardisations:
    z_a : centre AND width fitted per prepared 5-bit state s  (task variant (a))
    z_b : centre per state s, width = noise sigma of the prep-|0> state with the
          same neighbour configuration (state-independent noise, as H0 says)
Robust per-state centre/width: self-consistent truncated-Gaussian ML fit on
|x - mu| < 2.5 sigma, initialised at the histogram mode (resistant to one-sided
contamination and to multi-modal |1> distributions).

2-D "two-state segment" statistic: in the noise-normalised frame of the
neighbour configuration (x = along/sigma_A0, y = perp/sigma_P0, origin at c0),
d_seg = distance to the segment c0-c1.  Every two-state outcome (|0>, |1>, or a
straight |1>->|0> / |0>->|1> transition during the window) has its noiseless
window mean ON the segment, so under H0 + isotropic noise P(d_seg > k) <=
exp(-k^2/2) (Rayleigh bound).  This also catches clouds displaced along the axis
(Q3), which a perp-only cut misses.

Usage (inside docker oraqle-torch:latest):
    python m1tail_analysis.py --feat /feat --out /work/out --stage main
"""
import argparse
import warnings
import json
import os
import time

import numpy as np
from scipy import ndimage, optimize, stats
from scipy.special import ndtr

NQ = 5
WINDOWS = {  # name: (array, first bin, last bin exclusive)
    "early": ("feat", 0, 2),        # raw samples 0-99 (FIR sees up to 149): first 200 ns
    "1us": ("feat1us", 0, 10),      # raw samples 0-499, strictly causal FIR = CNN "1 us" input
    "2us": ("feat", 0, 20),         # raw samples 0-999, full stored trace
}
ZCUTS = [3, 4, 5, 6, 8]
DCUTS = [4, 5, 6, 8]
PAPER = ["Q1", "Q2", "Q3", "Q4", "Q5"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- robust fits
def mode1d(x, s):
    lo, hi = np.percentile(x, [0.05, 99.95])
    bw = s / 10.0
    nb = int(min(max((hi - lo) / bw, 50), 20000))
    h, e = np.histogram(x, bins=nb, range=(lo, hi))
    hs = ndimage.gaussian_filter1d(h.astype(float), max(1.0, 3.0 * (s / 10.0) / ((hi - lo) / nb)))
    i = int(np.argmax(hs))
    return 0.5 * (e[i] + e[i + 1])


def tfit(x, mu0, s0, k=2.5, iters=40):
    """Self-consistent truncated-normal ML fit on |x-mu|<k*s.  Returns (mu, s, n_in)."""
    x = np.asarray(x, np.float64)
    mu, s = float(mu0), float(s0)
    n = 0
    for _ in range(iters):
        lo, hi = mu - k * s, mu + k * s
        xi = x[(x > lo) & (x < hi)]
        n = len(xi)
        if n < 50:
            break
        c = xi.mean()
        d = xi - c
        S1 = d.sum()
        S2 = (d * d).sum()
        lo_c, hi_c = lo - c, hi - c

        def nll(p):
            m, ls = p
            sd = np.exp(ls)
            Z = ndtr((hi_c - m) / sd) - ndtr((lo_c - m) / sd)
            q = (S2 - 2 * m * S1 + n * m * m) / (sd * sd)
            return 0.5 * q + n * ls + n * np.log(max(Z, 1e-300))

        r = optimize.minimize(nll, [mu - c, np.log(s)], method="Nelder-Mead",
                              options=dict(xatol=1e-8, fatol=1e-10, maxiter=4000))
        m2, s2 = float(r.x[0] + c), float(np.exp(r.x[1]))
        conv = abs(m2 - mu) < 1e-5 * s and abs(s2 - s) < 1e-5 * s
        mu, s = m2, s2
        if conv:
            break
    return mu, s, n


def garwood(n, alpha=0.05):
    lo = 0.0 if n == 0 else stats.chi2.ppf(alpha / 2, 2 * n) / 2
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * n + 2) / 2
    return float(lo), float(hi)


def wilson(k, n, z=1.959964):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(cen - hw), float(cen + hw)]


def seg_dist(x, y, S, D):
    L2 = S * S + D * D
    t = np.clip((x * S + y * D) / L2, 0.0, 1.0)
    return np.hypot(x - t * S, y - t * D), t


# ----------------------------------------------------------------------------- data
def load(featdir, split, stride=1):
    y = np.load(f"{featdir}/y_{split}.npy")[::stride]
    F = np.load(f"{featdir}/feat_{split}.npy", mmap_mode="r")
    F = np.ascontiguousarray(F[::stride])                  # (N,5,20,2) float32, in RAM
    F1 = np.load(f"{featdir}/feat1us_{split}.npy", mmap_mode="r")[::stride]
    wm = {
        "early": F[:, :, 0:2].astype(np.float64).mean(2),
        "1us": np.asarray(F1, dtype=np.float64).mean(2),
        "2us": F.astype(np.float64).mean(2),
    }
    return y, F, wm


# ----------------------------------------------------------------------------- frame
def build_frame(A, P, y, q, sig0_hint=None):
    """Per-window frame for qubit q.  Returns dict with rotation, per-state fits,
    rotated A,P and standardised coordinates."""
    b = (y >> q) & 1
    # noise hint from prep0 perp MAD
    if sig0_hint is None:
        p0 = P[b == 0]
        sig0_hint = 1.4826 * np.median(np.abs(p0 - np.median(p0)))
    # rotation from pooled core centres
    cen = {}
    for bb in (0, 1):
        m = b == bb
        a0 = mode1d(A[m], sig0_hint)
        aa = tfit(A[m], a0, sig0_hint)
        p0_ = mode1d(P[m], sig0_hint)
        pp = tfit(P[m], p0_, sig0_hint)
        cen[bb] = (aa[0], pp[0])
    phi = float(np.arctan2(cen[1][1] - cen[0][1], cen[1][0] - cen[0][0]))
    # mean-based angle for comparison
    mA0, mP0 = A[b == 0].mean(), P[b == 0].mean()
    mA1, mP1 = A[b == 1].mean(), P[b == 1].mean()
    phi_mean = float(np.arctan2(mP1 - mP0, mA1 - mA0))
    c, s_ = np.cos(phi), np.sin(phi)
    Ar = A * c + P * s_
    Pr = -A * s_ + P * c
    # per-state fits
    muA = np.zeros(32); sA = np.zeros(32); muP = np.zeros(32); sP = np.zeros(32)
    for st in range(32):
        m = y == st
        a0 = mode1d(Ar[m], sig0_hint)
        muA[st], sA[st], _ = tfit(Ar[m], a0, sig0_hint)
        p0_ = mode1d(Pr[m], sig0_hint)
        muP[st], sP[st], _ = tfit(Pr[m], p0_, sig0_hint)
    st0 = y & ~(1 << q)            # matching prep-0 state (same neighbour config)
    st1 = y | (1 << q)
    za = (Pr - muP[y]) / sP[y]
    zb = (Pr - muP[y]) / sP[st0]
    # 2-D noise-normalised frame, origin c0 of the same neighbour config
    x = (Ar - muA[st0]) / sA[st0]
    yy = (Pr - muP[st0]) / sP[st0]
    S = (muA[st1] - muA[st0]) / sA[st0]
    D = (muP[st1] - muP[st0]) / sP[st0]
    sep = muA[st1] - muA[st0]
    u = (Ar - muA[st0]) / sep
    v = (Pr - muP[st0]) / sep
    d, t = seg_dist(x, yy, S, D)
    return dict(phi=phi, phi_mean=phi_mean, muA=muA, sA=sA, muP=muP, sP=sP,
                Ar=Ar, Pr=Pr, za=za, zb=zb, x=x, y=yy, S=S, D=D, u=u, v=v, d=d, t=t,
                sep=sep, b=b, sig0_hint=float(sig0_hint))


def frame_summary(fr, q):
    st = np.arange(32)
    b = (st >> q) & 1
    out = dict(rotation_deg_core=np.degrees(fr["phi"]), rotation_deg_mean=np.degrees(fr["phi_mean"]))
    for bb in (0, 1):
        m = b == bb
        out[f"prep{bb}"] = dict(
            muA=fr["muA"][m].tolist(), sA=fr["sA"][m].tolist(),
            muP=fr["muP"][m].tolist(), sP=fr["sP"][m].tolist(),
            sP_median=float(np.median(fr["sP"][m])), sA_median=float(np.median(fr["sA"][m])),
            muP_spread_over_configs_in_sigma=float((fr["muP"][m].max() - fr["muP"][m].min()) / np.median(fr["sP"][m])),
        )
    s0 = fr["sP"][b == 0]; s1 = fr["sP"][b == 1]
    out["sP_ratio_prep1_over_prep0_median"] = float(np.median(s1 / s0))
    out["sA0_over_sP0_median"] = float(np.median(fr["sA"][b == 0] / fr["sP"][b == 0]))
    sepcfg = fr["muA"][b == 1] - fr["muA"][b == 0]
    out["sep_core_median"] = float(np.median(sepcfg))
    out["S_sigma_units_median"] = float(np.median(sepcfg / fr["sA"][b == 0]))
    out["D_perp_offset_c1_minus_c0_sigma_median"] = float(np.median((fr["muP"][b == 1] - fr["muP"][b == 0]) / fr["sP"][b == 0]))
    return out


# ----------------------------------------------------------------------------- 1-D tails
def tail_table(z, N):
    rows = {}
    for k in ZCUTS:
        e = N * float(ndtr(-k))
        r = {}
        for side, cnt in (("plus", int((z > k).sum())), ("minus", int((z < -k).sum()))):
            lo, hi = garwood(cnt)
            r[side] = dict(obs=cnt, exp_gauss=e, ratio=(cnt / e if e > 0 else None),
                           excess=cnt - e, excess_frac=(cnt - e) / N,
                           excess_frac_ci95=[(lo - e) / N, (hi - e) / N])
        r["asym_plus_minus_frac"] = (r["plus"]["obs"] - r["minus"]["obs"]) / N
        rows[str(k)] = r
    return rows


def fit_models(z, seed=0):
    from sklearn.mixture import GaussianMixture
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(seed)
    zz = z.astype(np.float64)
    N = len(zz)
    res = {}
    mu, sd = zz.mean(), zz.std()
    ll_g = float(stats.norm.logpdf(zz, mu, sd).sum())
    res["gauss"] = dict(mu=float(mu), sd=float(sd), loglik=ll_g, bic=-2 * ll_g + 2 * np.log(N))
    sub = zz if N <= 250000 else rng.choice(zz, 250000, replace=False)
    df, loc, sc = stats.t.fit(sub)
    ll_t = float(stats.t.logpdf(zz, df, loc, sc).sum())
    res["student_t"] = dict(df=float(df), loc=float(loc), scale=float(sc), loglik=ll_t,
                            bic=-2 * ll_t + 3 * np.log(N), fit_subsample=len(sub))
    gm = {}
    for kc in (2, 3, 4):
        g = GaussianMixture(kc, n_init=1, random_state=seed, max_iter=1000, tol=1e-7).fit(sub[:, None])
        w = g.weights_; m = g.means_.ravel(); s = np.sqrt(g.covariances_.ravel())
        o = np.argsort(-w)
        core = o[0]
        noncore = [i for i in range(kc) if i != core]
        gm[str(kc)] = dict(
            weights=w[o].tolist(), means=m[o].tolist(), sds=s[o].tolist(),
            loglik=float(g.score(zz[:, None]) * N), bic=float(g.bic(zz[:, None])), fit_subsample=len(sub),
            noncore_weight=float(w[noncore].sum()),
            noncore_weight_plus=float(sum(w[i] for i in noncore if m[i] > m[core])),
            noncore_weight_minus=float(sum(w[i] for i in noncore if m[i] <= m[core])),
        )
    res["gmm"] = gm
    best = min(gm, key=lambda k: gm[k]["bic"])
    res["gmm_best_k_by_bic"] = int(best)
    return res


# ----------------------------------------------------------------------------- 2-D clouds
def find_peaks_2d(xs, ys, grid=0.1, smooth=0.5, nmin=25, maxpk=6):
    if len(xs) < nmin:
        return []
    x0, x1 = np.percentile(xs, [0.1, 99.9]); y0, y1 = np.percentile(ys, [0.1, 99.9])
    x0 -= 2; x1 += 2; y0 -= 2; y1 += 2
    nx = int(np.ceil((x1 - x0) / grid)); ny = int(np.ceil((y1 - y0) / grid))
    H, ex, ey = np.histogram2d(xs, ys, bins=[nx, ny], range=[[x0, x1], [y0, y1]])
    Hs = ndimage.gaussian_filter(H, smooth / grid)
    mx = ndimage.maximum_filter(Hs, size=int(round(2.0 / grid)) | 1)
    cand = np.argwhere((Hs == mx) & (Hs > 0))
    vals = Hs[cand[:, 0], cand[:, 1]]
    order = np.argsort(-vals)
    peaks = []
    for i in order[:40]:
        cx = ex[cand[i, 0]] + grid / 2; cy = ey[cand[i, 1]] + grid / 2
        n15 = int(((xs - cx) ** 2 + (ys - cy) ** 2 < 1.5 ** 2).sum())
        if n15 < nmin:
            continue
        if any((cx - p[0]) ** 2 + (cy - p[1]) ** 2 < 2.0 ** 2 for p in peaks):
            continue
        peaks.append((float(cx), float(cy), n15))
        if len(peaks) >= maxpk:
            break
    return peaks


def meanshift(x, y, cx, cy, r=1.5, w=None, it=100):
    for _ in range(it):
        m = (x - cx) ** 2 + (y - cy) ** 2 < r * r
        if m.sum() < 5:
            break
        if w is None:
            nx, ny = x[m].mean(), y[m].mean()
        else:
            ww = w[m]
            if ww.sum() == 0:
                break
            nx, ny = (x[m] * ww).sum() / ww.sum(), (y[m] * ww).sum() / ww.sum()
        if (nx - cx) ** 2 + (ny - cy) ** 2 < 1e-8:
            cx, cy = nx, ny
            break
        cx, cy = nx, ny
    return float(cx), float(cy)


# Gaussian (unit sigma, isotropic) mass inside r<2 and 2<r<3; uniform area ratio annulus/disc
PG_D = 1 - np.exp(-2.0)                 # 0.8647
PG_A = np.exp(-2.0) - np.exp(-4.5)      # 0.1242
AR = (9.0 - 4.0) / 4.0                  # 1.25


def disc_decomp(nd, na):
    """n_disc = PG_D*C + B ; n_ann = PG_A*C + AR*B  ->  C (compact unit-sigma cloud), B (flat bkg in disc)."""
    det = PG_D * AR - PG_A
    C = (AR * nd - na) / det
    B = (PG_D * na - PG_A * nd) / det
    varC = (AR * AR * nd + na) / det ** 2
    return float(C), float(B), float(np.sqrt(max(varC, 0)))


def cloud_stats(x, y, d, t, cfg, cx, cy, S, D, N, rng, nboot=300, dex=5.0):
    """Population and shape of a candidate cloud seeded at (cx,cy).
    The centre is refined by mean-shift on the EXCESS shots only (d_seg > dex), so it is
    not dragged into the |0>/|1> cores; populations are then counted on ALL shots."""
    ex = d > dex
    cx, cy = meanshift(x[ex], y[ex], cx, cy)
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    ind = r2 < 4.0
    ann = (r2 >= 4.0) & (r2 < 9.0)
    nd, na = int(ind.sum()), int(ann.sum())
    C, B, sC = disc_decomp(nd, na)
    dc, tc = seg_dist(np.array([cx]), np.array([cy]), S, D)
    null_disc = N * float(np.exp(-max(dc[0] - 2.0, 0) ** 2 / 2)) if dc[0] > 2 else float("nan")
    if nd > 10:
        cov = np.cov(np.vstack([x[ind] - cx, y[ind] - cy]))
        ev, evec = np.linalg.eigh(cov)
        major_dir_deg = float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])))
    else:
        ev = np.array([np.nan, np.nan]); major_dir_deg = float("nan")
    # trimmed (r<3) covariance of the excess shots around the centre: cloud size in noise-sigma units
    m3 = ex & (r2 < 9.0)
    if m3.sum() > 10:
        ev3 = np.linalg.eigvalsh(np.cov(np.vstack([x[m3] - cx, y[m3] - cy])))
    else:
        ev3 = np.array([np.nan, np.nan])
    ncfg = np.bincount(cfg[ind], minlength=16)
    ncfg_tot = np.bincount(cfg, minlength=16)
    exp_cfg = ncfg_tot * nd / max(len(cfg), 1)
    chi2 = float((((ncfg - exp_cfg) ** 2) / np.maximum(exp_cfg, 1e-9)).sum())
    pchi = float(stats.chi2.sf(chi2, 15)) if nd > 0 else float("nan")
    cfg_centres = []
    for c in range(16):
        mm = ind & (cfg == c)
        if mm.sum() >= 20:
            cfg_centres.append([float(x[mm].mean()), float(y[mm].mean())])
    cfg_centres = np.array(cfg_centres) if cfg_centres else np.zeros((0, 2))
    loc = r2 < 16.0
    xl, yl, exl = x[loc], y[loc], ex[loc]
    bc = []
    for _ in range(nboot):
        w = rng.poisson(1.0, size=len(xl)).astype(float)
        bx, by = meanshift(xl, yl, cx, cy, w=w * exl, it=30)
        rr = (xl - bx) ** 2 + (yl - by) ** 2
        bnd = float(w[rr < 4].sum()); bna = float(w[(rr >= 4) & (rr < 9)].sum())
        Cb, _, _ = disc_decomp(bnd, bna)
        bc.append((bx, by, bnd, Cb))
    bc = np.array(bc)
    return dict(
        centre_sigma=[cx, cy],
        centre_sigma_ci95=[np.percentile(bc[:, 0], [2.5, 97.5]).tolist(), np.percentile(bc[:, 1], [2.5, 97.5]).tolist()],
        d_seg_of_centre_sigma=float(dc[0]), t_proj_of_centre=float(tc[0]),
        n_disc_r2=nd, n_annulus_r2_3=na, n_disc_excess_share=float((ind & ex).sum() / nd) if nd else None,
        frac_disc=nd / N, frac_disc_wilson95=wilson(nd, N),
        null_bound_in_disc=null_disc,
        C_compact=C, C_compact_se_poisson=sC, B_flat_in_disc=B,
        C_frac=C / N, C_frac_ci95_poisson=[(C - 1.96 * sC) / N, (C + 1.96 * sC) / N],
        C_frac_ci95_boot=(np.percentile(bc[:, 3], [2.5, 97.5]) / N).tolist(),
        n_disc_ci95_boot=np.percentile(bc[:, 2], [2.5, 97.5]).tolist(),
        annulus_over_disc=(na / nd if nd else None),
        annulus_over_disc_ref=dict(compact_unit_gauss=PG_A / PG_D, flat=AR),
        cov_eig_in_disc=ev.tolist(), cov_eig_ref_unit_gauss_trunc2=0.687, cov_eig_ref_flat_disc=1.0,
        excess_sd_minor_major_r3=np.sqrt(np.maximum(ev3, 0)).tolist(), excess_sd_ref_unit_gauss_trunc3=0.974,
        major_axis_dir_deg=major_dir_deg,
        per_config_counts=ncfg.tolist(), per_config_chi2=chi2, per_config_chi2_p=pchi,
        per_config_centre_sd_sigma=(cfg_centres.std(0).tolist() if len(cfg_centres) > 2 else None),
        n_configs_with_20=int(len(cfg_centres)),
    )


def dbscan_summary(xs, ys, eps=0.5, min_samples=15):
    from sklearn.cluster import DBSCAN
    if len(xs) < min_samples:
        return dict(n_points=int(len(xs)), clusters=[])
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(np.c_[xs, ys])
    out = []
    for l in sorted(set(lab) - {-1}):
        m = lab == l
        cov = np.cov(np.vstack([xs[m], ys[m]])) if m.sum() > 2 else np.eye(2) * np.nan
        ev = np.linalg.eigvalsh(cov)
        out.append(dict(n=int(m.sum()), centroid_sigma=[float(xs[m].mean()), float(ys[m].mean())],
                        sd_minor=float(np.sqrt(max(ev[0], 0))), sd_major=float(np.sqrt(max(ev[1], 0))),
                        elongation=float(np.sqrt(max(ev[1], 0) / max(ev[0], 1e-12)))))
    out.sort(key=lambda r: -r["n"])
    return dict(n_points=int(len(xs)), n_noise=int((lab == -1).sum()), clusters=out[:8])


# ----------------------------------------------------------------------------- main
def run_split(featdir, split, outdir, models=True, peaks_from=None, rng_seed=1, stride=1):
    log("loading", split)
    yv, F, wm = load(featdir, split, stride)
    N_all = len(yv)
    res = dict(split=split, N=N_all, windows={})
    hist_store = {}
    excess_store = {}
    frames = {}
    rng = np.random.default_rng(rng_seed)
    zedges = np.arange(-15, 30.0001, 0.1)
    for w in WINDOWS:
        res["windows"][w] = {}
        for q in range(NQ):
            t0 = time.time()
            fr = build_frame(wm[w][:, q, 0], wm[w][:, q, 1], yv, q)
            frames[(w, q)] = fr
            b = fr["b"]
            cfgs = np.zeros(N_all, np.int64)
            o = 0
            for j in range(NQ):
                if j == q:
                    continue
                cfgs |= ((yv >> j) & 1).astype(np.int64) << o
                o += 1
            R = dict(paper=PAPER[q], code=f"q{q}", frame=frame_summary(fr, q))
            for bb in (0, 1):
                m = b == bb
                N = int(m.sum())
                za, zb = fr["za"][m], fr["zb"][m]
                d = fr["d"][m]
                RB = dict(N=N, tails_za=tail_table(za, N), tails_zb=tail_table(zb, N))
                # 2-D segment-distance excess
                RB["dseg"] = {}
                for k in DCUTS:
                    cnt = int((d > k).sum()); bound = N * float(np.exp(-k * k / 2))
                    lo, hi = garwood(cnt)
                    RB["dseg"][str(k)] = dict(obs=cnt, null_upper_bound=bound, frac=cnt / N,
                                              frac_ci95=[lo / N, hi / N],
                                              excess_frac_lower=max(cnt - bound, 0) / N,
                                              excess_frac_ci95=[(lo - bound) / N, (hi - bound) / N])
                # perp-side breakdown of d>5 shots
                m5 = d > 5
                RB["dseg5_side"] = dict(perp_plus=int((m5 & (fr["y"][m] > 0)).sum()),
                                        perp_minus=int((m5 & (fr["y"][m] <= 0)).sum()),
                                        beyond_c1_along=int((m5 & (fr["t"][m] >= 1)).sum()),
                                        before_c0_along=int((m5 & (fr["t"][m] <= 0)).sum()))
                hist_store[f"{w}_q{q}_b{bb}_za"] = np.histogram(za, zedges)[0]
                hist_store[f"{w}_q{q}_b{bb}_zb"] = np.histogram(zb, zedges)[0]
                # 2-D clouds among excess shots (d_seg > 5)
                xs, ys = fr["x"][m], fr["y"][m]
                ex = d > 5
                excess_store[f"{w}_q{q}_b{bb}"] = np.c_[xs[ex], ys[ex], fr["u"][m][ex], fr["v"][m][ex],
                                                        cfgs[m][ex]].astype(np.float32)
                # 2-D histogram of all shots for plotting (sigma units)
                hist_store[f"{w}_q{q}_b{bb}_h2"] = np.histogram2d(
                    xs, ys, bins=[240, 200], range=[[-8, 40], [-15, 25]])[0].astype(np.int32)
                pk = find_peaks_2d(xs[ex], ys[ex])
                RB["dbscan_excess_d5"] = dbscan_summary(xs[ex], ys[ex])
                RB["peaks"] = []
                for (cx, cy, n15) in pk[:4]:
                    cs = cloud_stats(xs, ys, d, fr["t"][m], cfgs[m], cx, cy,
                                     float(np.median(fr["S"][m])), float(np.median(fr["D"][m])), N, rng)
                    st0mask = ((np.arange(32) >> q) & 1) == 0
                    sepm = float(np.median(fr["sep"][m]))
                    sP0med = float(np.median(fr["sP"][st0mask]))
                    # convert centre to separation units (median config scale)
                    Smed = float(np.median(fr["S"][m]))
                    cs["centre_sep_units"] = [cs["centre_sigma"][0] / Smed,
                                              cs["centre_sigma"][1] * sP0med / sepm]
                    cs["n_excess_within_1p5_at_seed"] = n15
                    RB["peaks"].append(cs)
                R[f"prep{bb}"] = RB
            res["windows"][w][f"q{q}"] = R
            log(split, w, f"q{q}", f"{time.time() - t0:.1f}s")
    # cross-evaluation: peaks of one prepared bit evaluated in the other bit
    for w in WINDOWS:
        for q in range(NQ):
            fr = frames[(w, q)]
            R = res["windows"][w][f"q{q}"]
            for src in (0, 1):
                for pk in R[f"prep{src}"]["peaks"][:2]:
                    cx, cy = pk["centre_sigma"]
                    pk["cross_eval"] = {}
                    for tgt in (0, 1):
                        m = fr["b"] == tgt
                        r2 = (fr["x"][m] - cx) ** 2 + (fr["y"][m] - cy) ** 2
                        nd = int((r2 < 4).sum()); na = int(((r2 >= 4) & (r2 < 9)).sum())
                        C, B, sC = disc_decomp(nd, na)
                        pk["cross_eval"][f"prep{tgt}"] = dict(n_disc=nd, n_ann=na, C=C, C_se=sC,
                                                              C_frac=C / m.sum(), N=int(m.sum()))
                    # are the disc members also off-segment on another qubit (shot-level crosstalk)?
                    m = fr["b"] == src
                    memb = m & ((fr["x"] - cx) ** 2 + (fr["y"] - cy) ** 2 < 4)
                    co = {}
                    for j in range(NQ):
                        if j == q:
                            continue
                        fj = frames[(w, j)]["d"] > 5
                        nm = int(memb.sum())
                        base = float(fj[m].mean())
                        co[f"q{j}"] = dict(frac_members_flagged=(float(fj[memb].mean()) if nm else None),
                                           base_rate_same_prep=base,
                                           lift=(float(fj[memb].mean() / base) if nm and base > 0 else None))
                    pk["members_other_qubit_dseg5"] = co
    # shot-level co-occurrence of d_seg>5 across qubits
    res["cooccur_dseg5"] = {}
    for w in ("1us", "2us"):
        flags = np.stack([frames[(w, q)]["d"] > 5 for q in range(NQ)], 1)
        co = {}
        for q in range(NQ):
            fq = flags[:, q]
            n = int(fq.sum())
            other = flags[:, [j for j in range(NQ) if j != q]].any(1)
            co[f"q{q}"] = dict(n_flag=n, frac_also_other=(float((fq & other).sum() / n) if n else None),
                               base_rate_other=float(other.mean()),
                               lift=(float((fq & other).sum() / n / other.mean()) if n and other.mean() > 0 else None))
        res["cooccur_dseg5"][w] = co
    # drift check: chunked perp medians inside each prepared-state block (shots stored in state order)
    res["drift"] = {}
    for w in ("1us", "2us"):
        dd = {}
        for q in range(NQ):
            fr = frames[(w, q)]
            worst = 0.0; worst_st = None; sl = []
            for st in range(32):
                idx = np.where(yv == st)[0]
                zz = fr["za"][idx]
                nch = len(idx) // 1000
                med = np.array([np.median(zz[i * 1000:(i + 1) * 1000]) for i in range(nch)])
                rng_ = float(med.max() - med.min())
                # first-vs-second-half median shift
                sl.append(float(np.median(zz[len(zz) // 2:]) - np.median(zz[:len(zz) // 2])))
                if rng_ > worst:
                    worst, worst_st = rng_, st
            dd[f"q{q}"] = dict(max_range_of_1000shot_medians_sigma=worst, state=worst_st,
                               expected_range_pure_noise_sigma_approx=float(2 * 2.2 * 1.2533 / np.sqrt(1000)),
                               half_split_median_shift_sigma_max_abs=float(np.max(np.abs(sl))))
        res["drift"][w] = dd
    np.savez_compressed(f"{outdir}/m1tail_hist_{split}.npz", zedges=zedges, **hist_store)
    np.savez_compressed(f"{outdir}/m1tail_excess_{split}.npz", **excess_store)
    return res, frames, F, yv


def sliding(frames_unused, F, yv, outdir, split):
    """Off-segment excess vs time, sliding 500 ns windows (5 feat bins), per qubit/bit."""
    res = {}
    starts = list(range(0, 16))
    for q in range(NQ):
        rq = {"start_bin": starts, "prep0": {}, "prep1": {}}
        for k in starts:
            Wm = F[:, q, k:k + 5].astype(np.float64).mean(1)
            fr = build_frame(Wm[:, 0], Wm[:, 1], yv, q)
            for bb in (0, 1):
                m = fr["b"] == bb
                N = int(m.sum())
                zb = fr["zb"][m]; d = fr["d"][m]
                for key, val in (("zb_plus4", int((zb > 4).sum())), ("zb_minus4", int((zb < -4).sum())),
                                 ("dseg4", int((d > 4).sum())), ("dseg5", int((d > 5).sum())),
                                 ("S_sigma", float(np.median(fr["S"][m])))):
                    rq[f"prep{bb}"].setdefault(key, []).append(val)
                rq[f"prep{bb}"]["N"] = N
        res[f"q{q}"] = rq
        log("sliding", split, f"q{q}")
    return res


def trajectories(frames, F, yv, peaks_res, outdir, split):
    """Per-bin median trajectory of shots in the strongest 2us cloud vs |0>/|1> core shots,
    in the 2us frame (rotation + per-config centres of the 2us window), sigma units of the 2us window."""
    out = {}
    for q in range(NQ):
        fr = frames[("2us", q)]
        c, s_ = np.cos(fr["phi"]), np.sin(fr["phi"])
        A = F[:, q, :, 0].astype(np.float64); P = F[:, q, :, 1].astype(np.float64)
        Ar = A * c + P * s_; Pr = -A * s_ + P * c
        st0 = yv & ~(1 << q)
        # per-bin coordinates relative to the 2us c0 of the config, scaled by the 2us separation
        sep = (fr["muA"][yv | (1 << q)] - fr["muA"][st0])
        U = (Ar - fr["muA"][st0][:, None]) / sep[:, None]
        V = (Pr - fr["muP"][st0][:, None]) / sep[:, None]
        rq = {}
        for bb in (0, 1):
            pks = peaks_res["windows"]["2us"][f"q{q}"][f"prep{bb}"]["peaks"]
            m = fr["b"] == bb
            groups = {}
            core0 = m & (fr["d"] < 1.5) & (fr["t"] < 0.1)
            core1 = m & (fr["d"] < 1.5) & (fr["t"] > 0.9)
            groups["core_c0"] = core0; groups["core_c1"] = core1
            for i, pk in enumerate(pks[:3]):
                cx, cy = pk["centre_sigma"]
                groups[f"peak{i}"] = m & ((fr["x"] - cx) ** 2 + (fr["y"] - cy) ** 2 < 4)
            rb = {}
            for g, mm in groups.items():
                n = int(mm.sum())
                if n < 5:
                    continue
                rb[g] = dict(n=n, U_median=np.median(U[mm], 0).tolist(), V_median=np.median(V[mm], 0).tolist(),
                             U_q25=np.percentile(U[mm], 25, 0).tolist(), U_q75=np.percentile(U[mm], 75, 0).tolist(),
                             V_q25=np.percentile(V[mm], 25, 0).tolist(), V_q75=np.percentile(V[mm], 75, 0).tolist())
            rq[f"prep{bb}"] = rb
        out[f"q{q}"] = rq
    return out


def _do_models(task):
    from sklearn.mixture import GaussianMixture
    warnings.filterwarnings("ignore")
    key, za, xy, Smed, ysep = task
    out = dict(models_za=fit_models(za))
    if xy is not None:
        rng = np.random.default_rng(0)
        sub = xy if len(xy) <= 200000 else xy[rng.choice(len(xy), 200000, replace=False)]
        census = {}
        for kc in range(2, 9):
            g = GaussianMixture(kc, covariance_type="full", n_init=2, random_state=0, max_iter=1000,
                                tol=1e-6).fit(sub)
            comps = []
            for i in np.argsort(-g.weights_):
                mu = g.means_[i]; ev, evec = np.linalg.eigh(g.covariances_[i])
                dd, tt = seg_dist(np.array([mu[0]]), np.array([mu[1]]), Smed, 0.0)
                comps.append(dict(w=float(g.weights_[i]), mean_sigma=mu.tolist(),
                                  mean_sep_units=[float(mu[0] / Smed), float(mu[1] * ysep)],
                                  sd_minor=float(np.sqrt(ev[0])), sd_major=float(np.sqrt(ev[1])),
                                  major_dir_deg=float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))),
                                  d_seg_sigma=float(dd[0]), t_proj=float(tt[0])))
            census[str(kc)] = dict(bic=float(g.bic(sub)), comps=comps)
        best = min(census, key=lambda k: census[k]["bic"])
        out["gmm2d"] = dict(best_k=int(best), n_fit=int(len(sub)), census=census)
    return key, out


def models_stage(featdir, outdir, split="test", stride=1, procs=30):
    from multiprocessing import Pool
    yv, F, wm = load(featdir, split, stride)
    tasks = []
    for w in WINDOWS:
        for q in range(NQ):
            fr = build_frame(wm[w][:, q, 0], wm[w][:, q, 1], yv, q)
            st0mask = ((np.arange(32) >> q) & 1) == 0
            for bb in (0, 1):
                m = fr["b"] == bb
                Smed = float(np.median(fr["S"][m]))
                ysep = float(np.median(fr["sP"][st0mask])) / float(np.median(fr["sep"][m]))
                xy = np.c_[fr["x"][m], fr["y"][m]] if w in ("1us", "2us") else None
                tasks.append((f"{w}_q{q}_b{bb}", fr["za"][m].copy(), xy, Smed, ysep))
            log("models: frame", w, q)
    del F, wm
    with Pool(procs) as pool:
        res = dict(pool.map(_do_models, tasks, chunksize=1))
    with open(f"{outdir}/m1tail_models_{split}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    log("models written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="/feat")
    ap.add_argument("--out", default="/work/out")
    ap.add_argument("--splits", default="test,train")
    ap.add_argument("--no-models", action="store_true")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--stage", default="main", choices=["main", "models"])
    ap.add_argument("--procs", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.stage == "models":
        models_stage(a.feat, a.out, "test", a.stride, a.procs)
        return
    for split in a.splits.split(","):
        res, frames, F, yv = run_split(a.feat, split, a.out, models=not a.no_models, stride=a.stride)
        res["sliding_500ns"] = sliding(frames, F, yv, a.out, split)
        res["trajectories_2us_peaks"] = trajectories(frames, F, yv, res, a.out, split)
        with open(f"{a.out}/m1tail_results_{split}.json", "w") as fh:
            json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        log("wrote", split)
        del frames, F


if __name__ == "__main__":
    main()
