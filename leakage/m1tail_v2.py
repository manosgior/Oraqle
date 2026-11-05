#!/usr/bin/env python
"""m1tail_v2.py  (agent tag m1tail) -- second pass of the off-axis tail analysis.

Why a v2 (the v1 script m1tail_analysis.py is kept unchanged):
  * v1 located the |1> centre with a truncated-Gaussian fit whose width was free; for Q3/Q4 the
    |1> along-axis distribution is multi-component and the fit converged to different solutions in
    different neighbour configurations (Q3 1 us: sigma_A1/sigma_A0 jumping 1.15 <-> 3.9, separation
    3.25 <-> 4.84), which also tilted the rotation (artefact jump in the Q3 sliding window).
    v2 uses a FIXED-bandwidth 2-D mean-shift mode (bandwidth = amplifier-noise sigma) per prepared
    5-bit state for c0 and c1, and fits the perpendicular centre/width with the truncated fit.
  * v1 seeded "clouds" on the d_seg = 5 boundary (excess-only density is always highest at the cut).
    v2 looks for local maxima of the FULL 2-D density (all shots of a prepared bit, per-config
    centred and noise-normalised), with persistence merging (dip significance + ratio), and reports
    each non-core mode, its basin population, and the null expectation in that basin.
  * v1's 1-D GMM never tried k = 1; v2 compares Gaussian, Student-t and GMM k = 1..3 by BIC.

Null hypothesis H0 (task): perp = mu_s + sigma * eps, eps ~ N(0,1), mu_s = crosstalk-shifted centre of
the full prepared state s (= prepared bit x 16 neighbour configurations), sigma = amplifier noise
(state independent). Two nulls are evaluated for 2-D regions, both by resampling only the perp
coordinate of every observed shot (along kept as observed):
    strict  : perp_null = mu_s + sigma_P(prep-0 state, same neighbours) * N(0,1)      (= H0)
    lenient : perp_null = mu_s + sigma_P(own state s)              * N(0,1)      (allows a
              state-dependent Gaussian width, e.g. the broad Q4 |1>)
1-D tails use z_b = (perp - mu_s)/sigma_P(prep-0 state, same neighbours)  (H0 standardisation)
             z_a = (perp - mu_s)/sigma_P(s)                                (own-state width)
and the mirror statistic E(k) = [N(z>k) - N(z<-k)]/N, which needs only a symmetric core.

Usage (docker oraqle-torch:latest):
    python m1tail_v2.py --feat /feat --out /work/out_v2 --split test [--sliding] [--traj]
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import argparse
import json
import os
import time
import warnings

import numpy as np
from scipy import ndimage, optimize, stats
from scipy.special import ndtr

NQ = 5
PAPER = ["Q1", "Q2", "Q3", "Q4", "Q5"]
ZCUTS = [2, 3, 4, 5, 6, 8]
DCUTS = [4, 5, 6, 8]
GRID = 0.2            # sigma units
SMOOTH = 0.7          # sigma units (kernel for mode finding)
XR = (-10.0, 50.0)
YR = (-20.0, 35.0)
NX = int(round((XR[1] - XR[0]) / GRID))
NY = int(round((YR[1] - YR[0]) / GRID))
ZEDGES = np.arange(-15, 40.0001, 0.1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- small stats helpers
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


def seg_dist(x, y, S, D):
    L2 = S * S + D * D
    t = np.clip((x * S + y * D) / L2, 0.0, 1.0)
    return np.hypot(x - t * S, y - t * D), t


def meanshift2d(x, y, cx, cy, h, it=200, tol=1e-6):
    for _ in range(it):
        dx = x - cx
        dy = y - cy
        m = dx * dx + dy * dy < (3 * h) ** 2
        if m.sum() < 5:
            break
        w = np.exp(-0.5 * (dx[m] ** 2 + dy[m] ** 2) / (h * h))
        nx_ = cx + (w * dx[m]).sum() / w.sum()
        ny_ = cy + (w * dy[m]).sum() / w.sum()
        done = (nx_ - cx) ** 2 + (ny_ - cy) ** 2 < (tol * h) ** 2
        cx, cy = float(nx_), float(ny_)
        if done:
            break
    return cx, cy


def mode2d(x, y, h):
    """Highest-density point of the 2-D KDE with bandwidth h (histogram seed + mean-shift)."""
    x0, x1 = np.percentile(x, [0.1, 99.9])
    y0, y1 = np.percentile(y, [0.1, 99.9])
    g = h / 4.0
    nx_ = int(min(max((x1 - x0) / g, 10), 4000))
    ny_ = int(min(max((y1 - y0) / g, 10), 4000))
    H, ex, ey = np.histogram2d(x, y, bins=[nx_, ny_], range=[[x0, x1], [y0, y1]])
    Hs = ndimage.gaussian_filter(H, [h / ((x1 - x0) / nx_), h / ((y1 - y0) / ny_)])
    i, j = np.unravel_index(np.argmax(Hs), Hs.shape)
    return meanshift2d(x, y, 0.5 * (ex[i] + ex[i + 1]), 0.5 * (ey[j] + ey[j + 1]), h)


def c1_em(x, y, c0x, c0y, sg, iters=200, init=None):
    """c1 of a prepared-|1> state: EM for w0 N(c0, sg^2 I) + w1 N(c1, sg^2 I) + wb Uniform(box),
    c0 and sg fixed (state-independent noise), c1/weights free. Robust to the |0> (decay / prep
    error) population, to multi-component |1> distributions and to off-axis outliers."""
    d0 = np.hypot(x - c0x, y - c0y)
    far = d0 > 2.5 * sg
    if init is not None:
        c1x, c1y = init
    elif far.sum() > 1000:
        c1x, c1y = mode2d(x[far], y[far], sg)
    else:
        c1x, c1y = float(np.median(x)), float(np.median(y))
    x0, x1 = np.percentile(x, [0.05, 99.95]); y0, y1 = np.percentile(y, [0.05, 99.95])
    area = max((x1 - x0 + 10 * sg) * (y1 - y0 + 10 * sg), 1e-9)
    w = np.array([0.3, 0.6, 0.1])
    g0 = np.exp(-0.5 * d0 ** 2 / sg ** 2) / (2 * np.pi * sg * sg)
    for _ in range(iters):
        g1 = np.exp(-0.5 * ((x - c1x) ** 2 + (y - c1y) ** 2) / sg ** 2) / (2 * np.pi * sg * sg)
        L = np.vstack([w[0] * g0, w[1] * g1, np.full(len(x), w[2] / area)])
        tot = L.sum(0) + 1e-300
        r = L / tot
        w = r.mean(1)
        nx_ = float((r[1] * x).sum() / r[1].sum()); ny_ = float((r[1] * y).sum() / r[1].sum())
        done = np.hypot(nx_ - c1x, ny_ - c1y) < 1e-6 * sg
        c1x, c1y = nx_, ny_
        if done:
            break
    return c1x, c1y, dict(w0=float(w[0]), w1=float(w[1]), wbg=float(w[2]))


# ----------------------------------------------------------------------------- frame
def build_frame(A, P, y, q):
    b = ((y >> q) & 1).astype(np.int8)
    p0 = P[b == 0]
    sg = float(1.4826 * np.median(np.abs(p0 - np.median(p0))))
    cA = np.zeros(32)
    cP = np.zeros(32)
    s0 = np.array([s for s in range(32) if not (s >> q) & 1])
    s1 = s0 | (1 << q)
    em_info = {}
    for s in s0:                      # c0: KDE mode of the prepared-|0> state (|0> dominates)
        m = y == s
        cA[s], cP[s] = mode2d(A[m], P[m], sg)
    for s_0, s_1 in zip(s0, s1):      # c1: fixed-width EM (c0 fixed, isotropic noise sg, flat background)
        m = y == s_1
        cA[s_1], cP[s_1], em_info[int(s_1)] = c1_em(A[m], P[m], cA[s_0], cP[s_0], sg)
    # second pass: re-seed every configuration at c0 + median(c1 - c0) so that all 16 configurations
    # converge to the same physical |1> response (prevents locking onto a sub-cloud in a few configs)
    vA = float(np.median(cA[s1] - cA[s0])); vP = float(np.median(cP[s1] - cP[s0]))
    for s_0, s_1 in zip(s0, s1):
        m = y == s_1
        cA[s_1], cP[s_1], em_info[int(s_1)] = c1_em(A[m], P[m], cA[s_0], cP[s_0], sg,
                                                    init=(cA[s_0] + vA, cP[s_0] + vP))
    ang = np.arctan2(cP[s1] - cP[s0], cA[s1] - cA[s0])
    phi = float(np.median(ang))
    c, sn = np.cos(phi), np.sin(phi)
    Ar = A * c + P * sn
    Pr = -A * sn + P * c
    cAr = cA * c + cP * sn
    cPr = -cA * sn + cP * c
    muP = np.zeros(32); sP = np.zeros(32); muA = np.zeros(32); sA = np.zeros(32)
    for s in range(32):
        m = y == s
        muP[s], sP[s], _ = tfit(Pr[m], cPr[s], sg)
        muA[s], sA[s], _ = tfit(Ar[m], cAr[s], sg)
    st0 = y & ~(1 << q)
    st1 = y | (1 << q)
    sep_state = np.zeros(32)
    sep_state[s0] = cAr[s1] - cAr[s0]
    sep_state[s1] = cAr[s1] - cAr[s0]
    sepv = sep_state[y]
    x = (Ar - cAr[st0]) / sA[st0]
    yy = (Pr - muP[st0]) / sP[st0]
    S = (cAr[st1] - cAr[st0]) / sA[st0]
    D = (muP[st1] - muP[st0]) / sP[st0]
    d, t = seg_dist(x, yy, S, D)
    za = (Pr - muP[y]) / sP[y]
    zb = (Pr - muP[y]) / sP[st0]
    u = (Ar - cAr[st0]) / sepv
    v = (Pr - muP[st0]) / sepv
    cfg = np.zeros(len(y), np.int64)
    o = 0
    for j in range(NQ):
        if j == q:
            continue
        cfg |= ((y >> j) & 1).astype(np.int64) << o
        o += 1
    return dict(phi=phi, phi_cfg_deg=np.degrees(ang).tolist(), sg=sg, em_info=em_info, cA=cAr, cP=cPr, muP=muP, sP=sP,
                muA=muA, sA=sA, sep_state=sep_state, Ar=Ar, Pr=Pr, x=x, y=yy, S=S, D=D, d=d, t=t,
                za=za, zb=zb, u=u, v=v, b=b, cfg=cfg, st0=st0, st1=st1)


def frame_summary(fr, q):
    st = np.arange(32)
    bb = (st >> q) & 1
    s0 = st[bb == 0]; s1 = s0 | (1 << q)
    sep = fr["cA"][s1] - fr["cA"][s0]
    return dict(
        rotation_extra_deg=float(np.degrees(fr["phi"])),
        rotation_cfg_spread_deg=float(np.std(fr["phi_cfg_deg"])),
        noise_sigma_perp_prep0_median=float(np.median(fr["sP"][s0])),
        noise_sigma_along_prep0_median=float(np.median(fr["sA"][s0])),
        sigma_perp_prep1_over_prep0=(fr["sP"][s1] / fr["sP"][s0]).tolist(),
        sigma_along_prep1_over_prep0_truncfit=(fr["sA"][s1] / fr["sA"][s0]).tolist(),
        sep_mode_per_cfg=sep.tolist(), sep_mode_median=float(np.median(sep)),
        sep_over_sigma_median=float(np.median(sep / fr["sA"][s0])),
        c1_perp_offset_sigma=((fr["muP"][s1] - fr["muP"][s0]) / fr["sP"][s0]).tolist(),
        c0_mode=[fr["cA"][s0].tolist(), fr["cP"][s0].tolist()],
        c1_mode=[fr["cA"][s1].tolist(), fr["cP"][s1].tolist()],
        muP=fr["muP"].tolist(), sP=fr["sP"].tolist(),
        c1_em_weights={str(k): v for k, v in fr["em_info"].items()},
    )


# ----------------------------------------------------------------------------- 1-D tails
def tail_table(z, N):
    rows = {}
    for k in ZCUTS:
        e = N * float(ndtr(-k))
        npl, nmi = int((z > k).sum()), int((z < -k).sum())
        r = {}
        for side, cnt in (("plus", npl), ("minus", nmi)):
            lo, hi = garwood(cnt)
            r[side] = dict(obs=cnt, exp_gauss=e, excess_frac=(cnt - e) / N,
                           excess_frac_ci95=[(lo - e) / N, (hi - e) / N])
        se = np.sqrt(npl + nmi)
        r["mirror_excess_frac"] = (npl - nmi) / N
        r["mirror_excess_frac_ci95"] = [(npl - nmi - 1.96 * se) / N, (npl - nmi + 1.96 * se) / N]
        rows[str(k)] = r
    return rows


def fit_models_1d(z, seed=0):
    from sklearn.mixture import GaussianMixture
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(seed)
    zz = np.asarray(z, np.float64)
    sub = zz if len(zz) <= 200000 else rng.choice(zz, 200000, replace=False)
    n = len(sub)
    res = {}
    mu, sd = sub.mean(), sub.std()
    llg = float(stats.norm.logpdf(sub, mu, sd).sum())
    res["gauss"] = dict(mu=float(mu), sd=float(sd), bic=-2 * llg + 2 * np.log(n))
    df, loc, sc = stats.t.fit(sub)
    llt = float(stats.t.logpdf(sub, df, loc, sc).sum())
    res["student_t"] = dict(df=float(df), loc=float(loc), scale=float(sc), bic=-2 * llt + 3 * np.log(n))
    for kc in (1, 2, 3):
        g = GaussianMixture(kc, n_init=2, random_state=seed, max_iter=1000, tol=1e-7).fit(sub[:, None])
        w = g.weights_; m = g.means_.ravel(); s = np.sqrt(g.covariances_.ravel())
        o = np.argsort(-w)
        res[f"gmm{kc}"] = dict(weights=w[o].tolist(), means=m[o].tolist(), sds=s[o].tolist(),
                               bic=float(g.bic(sub[:, None])))
    keys = ["gauss", "student_t", "gmm1", "gmm2", "gmm3"]
    res["best_by_bic"] = min(keys, key=lambda k: res[k]["bic"])
    res["n_fit"] = n
    return res


# ----------------------------------------------------------------------------- 2-D modes
def cell_index(xs, ys):
    i = np.floor((xs - XR[0]) / GRID).astype(np.int64)
    j = np.floor((ys - YR[0]) / GRID).astype(np.int64)
    ok = (i >= 0) & (i < NX) & (j >= 0) & (j < NY)
    return i, j, ok


def find_modes(xs, ys, neff_floor=10.0, zmin=5.0, rmin=1.15, nmin=30):
    """Local maxima of the smoothed 2-D density with persistence merging.
    Returns (modes list, label map (NX,NY) of surviving roots (-1 = sparse/unassigned), H, Hs)."""
    i, j, ok = cell_index(xs, ys)
    H = np.zeros((NX, NY))
    np.add.at(H, (i[ok], j[ok]), 1.0)
    s = SMOOTH / GRID
    Hs = ndimage.gaussian_filter(H, s, mode="constant")
    kfac = 4 * np.pi * s * s                    # N_eff = Hs * kfac (Poisson-equivalent counts)
    valid = Hs * kfac >= neff_floor
    Pp = np.pad(np.where(valid, Hs, -1.0), 1, constant_values=-1.0)
    shifts = [(0, 0)] + [(a, c) for a in (-1, 0, 1) for c in (-1, 0, 1) if (a, c) != (0, 0)]
    stack = np.stack([Pp[1 + a:1 + a + NX, 1 + c:1 + c + NY] for a, c in shifts], 0)
    k = np.argmax(stack, 0)
    da = np.array([sh[0] for sh in shifts])[k]
    dc = np.array([sh[1] for sh in shifts])[k]
    I, J = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    ptr = np.where(valid, (I + da) * NY + (J + dc), -1).ravel()
    for _ in range(200):
        nxt = np.where(ptr >= 0, ptr[np.maximum(ptr, 0)], -1)
        if np.array_equal(nxt, ptr):
            break
        ptr = nxt
    lab = ptr.reshape(NX, NY)
    Hf = Hs.ravel()
    # saddles between adjacent basins
    pairs = {}
    for a, c in ((1, 0), (0, 1), (1, 1), (1, -1)):
        ra, rb = slice(0, NX - a), slice(a, NX)
        if c >= 0:
            ca, cb = slice(0, NY - c), slice(c, NY)
        else:
            ca, cb = slice(-c, NY), slice(0, NY + c)
        A_, B_ = lab[ra, ca], lab[rb, cb]
        HA, HB = Hs[ra, ca], Hs[rb, cb]
        m = (A_ >= 0) & (B_ >= 0) & (A_ != B_)
        la, lb = A_[m], B_[m]
        val = np.minimum(HA[m], HB[m])
        lo_, hi_ = np.minimum(la, lb), np.maximum(la, lb)
        for p_, q_, v_ in zip(lo_.tolist(), hi_.tolist(), val.tolist()):
            key = (p_, q_)
            if v_ > pairs.get(key, -1.0):
                pairs[key] = v_
    peaks = np.unique(lab[lab >= 0])
    height = {int(p): float(Hf[p]) for p in peaks}
    # shot counts per basin
    sl = np.full(len(xs), -1, np.int64)
    sl[ok] = lab[i[ok], j[ok]]
    cnt = {int(p): 0 for p in peaks}
    u_, c_ = np.unique(sl[sl >= 0], return_counts=True)
    for p, cc in zip(u_.tolist(), c_.tolist()):
        cnt[p] = cc
    adj = {int(p): {} for p in peaks}
    for (p_, q_), v_ in pairs.items():
        adj[p_][q_] = v_
        adj[q_][p_] = v_
    parent = {int(p): int(p) for p in peaks}
    info = {}
    for p in sorted(height, key=lambda kk: height[kk]):
        higher = {r: sv for r, sv in adj[p].items() if height[r] > height[p]}
        if not higher:
            info[p] = dict(z_dip=None, ratio=None, saddle_to=None)
            continue
        r, sv = max(higher.items(), key=lambda kv: kv[1])
        Np = height[p] * kfac
        Ns = sv * kfac
        zdip = (Np - Ns) / np.sqrt(Np + Ns)
        ratio = height[p] / sv if sv > 0 else np.inf
        if zdip < zmin or ratio < rmin or cnt[p] < nmin:
            parent[p] = r
            cnt[r] += cnt[p]
            for c2, sv2 in adj[p].items():
                if c2 == r:
                    continue
                if sv2 > adj[r].get(c2, -1.0):
                    adj[r][c2] = sv2
                    adj[c2][r] = sv2
                adj[c2].pop(p, None)
            adj[r].pop(p, None)
            adj[p] = {}
        else:
            info[p] = dict(z_dip=float(zdip), ratio=float(ratio), saddle_to=int(r))

    def root(p):
        while parent[p] != p:
            p = parent[p]
        return p

    roots = sorted({root(int(p)) for p in peaks})
    rmap = {int(p): root(int(p)) for p in peaks}
    lab2 = np.full(NX * NY, -1, np.int64)
    flat = lab.ravel()
    okc = flat >= 0
    lab2[okc] = np.array([rmap[v_] for v_ in flat[okc].tolist()], np.int64) if okc.any() else lab2[okc]
    lab2 = lab2.reshape(NX, NY)
    modes = []
    for r in roots:
        if cnt[r] < nmin:
            lab2[lab2 == r] = -1
            continue
        ii, jj = divmod(r, NY)
        modes.append(dict(root=int(r), x=float(XR[0] + (ii + 0.5) * GRID), y=float(YR[0] + (jj + 0.5) * GRID),
                          neff_peak=float(height[r] * kfac), basin_count=int(cnt[r]), **info.get(r, {})))
    modes.sort(key=lambda mm: -mm["basin_count"])
    return modes, lab2, H, Hs


def lookup(lab2, xs, ys):
    i, j, ok = cell_index(xs, ys)
    out = np.full(len(xs), -1, np.int64)
    out[ok] = lab2[i[ok], j[ok]]
    return out


def analyse_modes(fr, bb, other_frame_b, rng, nboot=500):
    """Mode census of prepared bit bb in frame fr."""
    m = fr["b"] == bb
    N = int(m.sum())
    xs, ys = fr["x"][m], fr["y"][m]
    Sm = float(np.median(fr["S"][m])); Dm = float(np.median(fr["D"][m]))
    modes, lab2, H, Hs = find_modes(xs, ys)
    # nulls (perp resampled, along kept)
    sl_obs = lookup(lab2, xs, ys)
    st = fr["st0"][m]
    # own-state centre/width in the noise-normalised frame of the neighbour configuration
    st_own = fr["st1"][m] if bb == 1 else fr["st0"][m]
    cen_own = (fr["muP"][st_own] - fr["muP"][st]) / fr["sP"][st]
    wid_own = fr["sP"][st_own] / fr["sP"][st]
    null_strict = []; null_len = []
    for rep in range(4):
        e = rng.standard_normal(N)
        null_strict.append(lookup(lab2, xs, cen_own + e))
        null_len.append(lookup(lab2, xs, cen_own + wid_own * e))
    cfg = fr["cfg"][m]
    ncfg_tot = np.bincount(cfg, minlength=16)
    u, v = fr["u"][m], fr["v"][m]
    for md in modes:
        r = md["root"]
        inb = sl_obs == r
        nb = int(inb.sum())
        md["basin_count"] = nb
        md["frac"] = nb / N
        md["frac_wilson95"] = wilson(nb, N)
        ns = float(np.mean([(a == r).sum() for a in null_strict]))
        nl = float(np.mean([(a == r).sum() for a in null_len]))
        md["null_strict"] = ns
        md["null_lenient"] = nl
        lo, hi = wilson(nb, N)
        md["excess_frac_strict"] = (nb - ns) / N
        md["excess_frac_lenient"] = (nb - nl) / N
        md["excess_frac_lenient_ci95"] = [lo - nl / N, hi - nl / N]
        # refine location
        cx, cy = meanshift2d(xs, ys, md["x"], md["y"], SMOOTH)
        md["x_ms"], md["y_ms"] = cx, cy
        near = (xs - cx) ** 2 + (ys - cy) ** 2 < 1.0
        md["n_within_1sigma"] = int(near.sum())
        md["n_within_2sigma"] = int(((xs - cx) ** 2 + (ys - cy) ** 2 < 4.0).sum())
        md["uv_sep_units"] = [float(np.mean(u[near])), float(np.mean(v[near]))] if near.sum() else [None, None]
        dd, tt = seg_dist(np.array([cx]), np.array([cy]), Sm, Dm)
        md["d_seg_sigma"] = float(dd[0]); md["t_proj"] = float(tt[0])
        md["is_core0"] = bool(np.hypot(cx, cy) < 2.0)
        md["is_core1"] = bool(np.hypot(cx - Sm, cy - Dm) < 2.0)
        # per-config
        nc = np.bincount(cfg[inb], minlength=16)
        fc = nc / np.maximum(ncfg_tot, 1)
        expc = ncfg_tot * nb / N
        chi2 = float((((nc - expc) ** 2) / np.maximum(expc, 1e-9)).sum())
        md["per_cfg_frac"] = fc.tolist()
        md["per_cfg_chi2_p"] = float(stats.chi2.sf(chi2, 15)) if nb > 0 else None
        md["per_cfg_frac_rel_sd"] = float(fc.std() / fc.mean()) if fc.mean() > 0 else None
        bs = []
        for _ in range(nboot):
            pick = rng.integers(0, 16, 16)
            bs.append(nc[pick].sum() / ncfg_tot[pick].sum())
        md["frac_cfg_bootstrap95"] = np.percentile(bs, [2.5, 97.5]).tolist()
        locs = []
        for c in range(16):
            mc = (cfg == c) & ((xs - cx) ** 2 + (ys - cy) ** 2 < 9.0)
            if mc.sum() >= 30:
                lx, ly = meanshift2d(xs[mc], ys[mc], cx, cy, SMOOTH)
                nn = (cfg == c) & ((xs - lx) ** 2 + (ys - ly) ** 2 < 1.0)
                if nn.sum() >= 10:
                    locs.append([lx, ly, float(np.mean(u[nn])), float(np.mean(v[nn]))])
        locs = np.array(locs) if locs else np.zeros((0, 4))
        md["per_cfg_loc_n"] = int(len(locs))
        if len(locs) >= 3:
            md["per_cfg_loc_sd_sigma"] = locs[:, :2].std(0).tolist()
            md["per_cfg_loc_sd_sep"] = locs[:, 2:].std(0).tolist()
            md["per_cfg_loc_range_sep"] = [[float(locs[:, 2].min()), float(locs[:, 2].max())],
                                           [float(locs[:, 3].min()), float(locs[:, 3].max())]]
        # the same basin evaluated on the other prepared bit (its own frame coordinates)
        if other_frame_b is not None:
            mo = fr["b"] == (1 - bb)
            slo = lookup(lab2, fr["x"][mo], fr["y"][mo])
            no = int((slo == r).sum())
            st_o = fr["st1"][mo] if bb == 0 else fr["st0"][mo]
            st0_o = fr["st0"][mo]
            ceno = (fr["muP"][st_o] - fr["muP"][st0_o]) / fr["sP"][st0_o]
            wido = fr["sP"][st_o] / fr["sP"][st0_o]
            nlo = float(np.mean([(lookup(lab2, fr["x"][mo], ceno + wido * rng.standard_normal(int(mo.sum()))) == r).sum()
                                 for _ in range(2)]))
            md["other_prep"] = dict(count=no, N=int(mo.sum()), frac=no / int(mo.sum()), null_lenient=nlo,
                                    excess_frac_lenient=(no - nlo) / int(mo.sum()),
                                    wilson95=wilson(no, int(mo.sum())))
    # far-excess mode vs cut: a compact cloud keeps its location as the d_seg cut moves outward,
    # a monotone tail/halo makes the excess mode follow the cut.
    far = {}
    dd = fr["d"][m]
    for kc in (5, 6, 7, 8, 10):
        mk = dd > kc
        nk = int(mk.sum())
        if nk < 50:
            far[str(kc)] = dict(n=nk)
            continue
        fx, fy = mode2d(xs[mk], ys[mk], SMOOTH)
        dmode, tmode = seg_dist(np.array([fx]), np.array([fy]), Sm, Dm)
        near = mk & ((xs - fx) ** 2 + (ys - fy) ** 2 < 1.0)
        sub = np.vstack([xs[mk] - xs[mk].mean(), ys[mk] - ys[mk].mean()])
        ev = np.linalg.eigvalsh(np.cov(sub))
        far[str(kc)] = dict(n=nk, mode_xy=[fx, fy], mode_dseg=float(dmode[0]), mode_t=float(tmode[0]),
                            mode_minus_cut=float(dmode[0] - kc),
                            mode_uv=([float(np.mean(u[near])), float(np.mean(v[near]))] if near.sum() else None),
                            centroid_xy=[float(xs[mk].mean()), float(ys[mk].mean())],
                            centroid_uv=[float(np.mean(u[mk])), float(np.mean(v[mk]))],
                            sd_minor_major_sigma=np.sqrt(np.maximum(ev, 0)).tolist(),
                            frac_perp_plus=float((ys[mk] > 0).mean()))
    return dict(N=N, S_median=Sm, D_median=Dm, modes=modes, far_excess=far), lab2, H


# ----------------------------------------------------------------------------- per window/qubit
def analyse_window(fr, q, rng, want_modes=True, want_models=False):
    R = dict(paper=PAPER[q], code=f"q{q}", frame=frame_summary(fr, q))
    store = {}
    for bb in (0, 1):
        m = fr["b"] == bb
        N = int(m.sum())
        za, zb, d = fr["za"][m], fr["zb"][m], fr["d"][m]
        RB = dict(N=N, tails_zb=tail_table(zb, N), tails_za=tail_table(za, N))
        RB["dseg"] = {}
        # perp-resampled null for d_seg (along kept)
        st0 = fr["st0"][m]
        st_own = fr["st1"][m] if bb == 1 else st0
        cen = (fr["muP"][st_own] - fr["muP"][st0]) / fr["sP"][st0]
        wid = fr["sP"][st_own] / fr["sP"][st0]
        e = rng.standard_normal(N)
        Sv, Dv = fr["S"][m], fr["D"][m]
        dn_s, _ = seg_dist(fr["x"][m], cen + e, Sv, Dv)
        dn_l, _ = seg_dist(fr["x"][m], cen + wid * e, Sv, Dv)
        for k in DCUTS:
            cnt = int((d > k).sum())
            lo, hi = garwood(cnt)
            bound = N * float(np.exp(-k * k / 2))
            ns, nl = int((dn_s > k).sum()), int((dn_l > k).sum())
            RB["dseg"][str(k)] = dict(obs=cnt, frac=cnt / N, frac_ci95=[lo / N, hi / N], rayleigh_bound=bound,
                                      null_strict_resampled=ns, null_lenient_resampled=nl,
                                      excess_frac_lenient=(cnt - nl) / N,
                                      excess_frac_lenient_ci95=[(lo - nl) / N, (hi - nl) / N])
        m5 = d > 5
        RB["dseg5_side"] = dict(perp_plus=int((m5 & (fr["y"][m] > 0)).sum()),
                                perp_minus=int((m5 & (fr["y"][m] <= 0)).sum()),
                                beyond_c1_along=int((m5 & (fr["t"][m] >= 1)).sum()),
                                before_c0_along=int((m5 & (fr["t"][m] <= 0)).sum()))
        # share of (mid-point threshold) errors that are off-axis
        uu = fr["u"][m]
        err = (uu > 0.5) if bb == 0 else (uu < 0.5)
        ne = int(err.sum())
        RB["midpoint_threshold_errors"] = dict(
            n_err=ne, err_rate=ne / N, err_rate_wilson95=wilson(ne, N),
            n_err_zb_gt3=int((err & (zb > 3)).sum()), n_err_zb_lt_m3=int((err & (zb < -3)).sum()),
            n_err_dseg_gt5=int((err & (d > 5)).sum()),
            share_err_zb_gt3=(float((err & (zb > 3)).sum() / ne) if ne else None),
            share_err_mirror3=(float(((err & (zb > 3)).sum() - (err & (zb < -3)).sum()) / ne) if ne else None),
            share_err_dseg_gt5=(float((err & (d > 5)).sum() / ne) if ne else None),
            gauss_expect_share_zb_gt3=float(ndtr(-3)))
        store[f"b{bb}_zb"] = np.histogram(zb, ZEDGES)[0].astype(np.int32)
        store[f"b{bb}_za"] = np.histogram(za, ZEDGES)[0].astype(np.int32)
        if want_models:
            RB["models_zb"] = fit_models_1d(zb)
        R[f"prep{bb}"] = RB
    if want_modes:
        for bb in (0, 1):
            MR, lab2, H = analyse_modes(fr, bb, True, rng)
            R[f"prep{bb}"]["mode_census"] = MR
            store[f"b{bb}_H"] = H.astype(np.int32)
            mm_ = fr["b"] == bb
            store[f"b{bb}_Huv"] = np.histogram2d(fr["u"][mm_], fr["v"][mm_], bins=[250, 250],
                                                 range=[[-1.0, 4.0], [-1.5, 3.5]])[0].astype(np.int32)
            lab_small = np.full(lab2.shape, -1, np.int16)
            for ii, md in enumerate(MR["modes"][:30]):
                lab_small[lab2 == md["root"]] = ii
            store[f"b{bb}_lab"] = lab_small
    return R, store


def load(featdir, split, stride=1):
    y = np.load(f"{featdir}/y_{split}.npy")[::stride]
    F = np.ascontiguousarray(np.load(f"{featdir}/feat_{split}.npy", mmap_mode="r")[::stride])
    F1 = np.ascontiguousarray(np.load(f"{featdir}/feat1us_{split}.npy", mmap_mode="r")[::stride])
    return y, F, F1


def window_means(F, F1):
    return {"early": F[:, :, 0:2].astype(np.float64).mean(2),
            "1us": F1.astype(np.float64).mean(2),
            "2us": F.astype(np.float64).mean(2)}


def trajectories(fr, R, Fbins, q):
    """Per-bin coordinates (rotated, relative to the window c0 of the shot's neighbour config, raw
    units) for shots near each mode, plus core groups."""
    c, sn = np.cos(fr["phi"]), np.sin(fr["phi"])
    A = Fbins[:, q, :, 0].astype(np.float64)
    P = Fbins[:, q, :, 1].astype(np.float64)
    Ar = A * c + P * sn
    Pr = -A * sn + P * c
    Ar -= fr["cA"][fr["st0"]][:, None]
    Pr -= fr["muP"][fr["st0"]][:, None]
    out = {}
    for bb in (0, 1):
        m = fr["b"] == bb
        groups = {"core0": m & (np.hypot(fr["x"], fr["y"]) < 1.5),
                  "core1": m & (np.hypot(fr["x"] - fr["S"], fr["y"] - fr["D"]) < 1.5)}
        for ii, md in enumerate(R[f"prep{bb}"]["mode_census"]["modes"][:8]):
            if md["is_core0"] or md["is_core1"]:
                continue
            groups[f"mode{ii}"] = m & (np.hypot(fr["x"] - md["x_ms"], fr["y"] - md["y_ms"]) < 1.5)
        for kc in ("5", "7"):
            fe = R[f"prep{bb}"]["mode_census"]["far_excess"].get(kc, {})
            if "mode_xy" in fe:
                groups[f"far{kc}"] = m & (fr["d"] > float(kc)) & (np.hypot(fr["x"] - fe["mode_xy"][0], fr["y"] - fe["mode_xy"][1]) < 1.5)
        ex5 = m & (fr["d"] > 5)
        if ex5.sum() >= 50:
            q90 = float(np.quantile(fr["d"][ex5], 0.9))
            groups["streak_5to7"] = ex5 & (fr["d"] < 7)
            groups["streak_end_top10pct"] = ex5 & (fr["d"] >= q90)
        rb = {}
        for g, mm in groups.items():
            n = int(mm.sum())
            if n < 5:
                continue
            rb[g] = dict(n=n, A_med=np.median(Ar[mm], 0).tolist(), P_med=np.median(Pr[mm], 0).tolist(),
                         A_q25=np.percentile(Ar[mm], 25, 0).tolist(), A_q75=np.percentile(Ar[mm], 75, 0).tolist(),
                         P_q25=np.percentile(Pr[mm], 25, 0).tolist(), P_q75=np.percentile(Pr[mm], 75, 0).tolist())
        out[f"prep{bb}"] = rb
    out["sep_median_raw"] = float(np.median(fr["sep_state"]))
    return out


def early_link(fr_w, fr_early, R, q):
    """For shots near each non-core mode of window w: fraction already off-axis in the first 200 ns."""
    out = {}
    for bb in (0, 1):
        m = fr_w["b"] == bb
        ro = {}
        groups = {"core1": m & (np.hypot(fr_w["x"] - fr_w["S"], fr_w["y"] - fr_w["D"]) < 1.5),
                  "core0": m & (np.hypot(fr_w["x"], fr_w["y"]) < 1.5)}
        for ii, md in enumerate(R[f"prep{bb}"]["mode_census"]["modes"][:8]):
            if md["is_core0"] or md["is_core1"]:
                continue
            groups[f"mode{ii}"] = m & (np.hypot(fr_w["x"] - md["x_ms"], fr_w["y"] - md["y_ms"]) < 1.5)
        for kc in ("5", "7"):
            fe = R[f"prep{bb}"]["mode_census"]["far_excess"].get(kc, {})
            if "mode_xy" in fe:
                groups[f"far{kc}"] = m & (fr_w["d"] > float(kc)) & (np.hypot(fr_w["x"] - fe["mode_xy"][0], fr_w["y"] - fe["mode_xy"][1]) < 1.5)
        ex5 = m & (fr_w["d"] > 5)
        if ex5.sum() >= 50:
            q90 = float(np.quantile(fr_w["d"][ex5], 0.9))
            groups["streak_5to7"] = ex5 & (fr_w["d"] < 7)
            groups["streak_end_top10pct"] = ex5 & (fr_w["d"] >= q90)
        for g, mm in groups.items():
            n = int(mm.sum())
            if n < 5:
                continue
            ze = fr_early["zb"][mm]
            ro[g] = dict(n=n, early_zb_median=float(np.median(ze)), early_frac_zb_gt3=float((ze > 3).mean()),
                         early_frac_zb_lt_m3=float((ze < -3).mean()),
                         early_x_median=float(np.median(fr_early["x"][mm])),
                         early_S_median=float(np.median(fr_early["S"][mm])))
        out[f"prep{bb}"] = ro
    return out


def sliding(F, y, rng):
    res = {}
    for q in range(NQ):
        rq = {"start_bin": list(range(16)), "prep0": {}, "prep1": {}}
        for k0 in range(16):
            W = F[:, q, k0:k0 + 5].astype(np.float64).mean(1)
            fr = build_frame(W[:, 0], W[:, 1], y, q)
            R, _ = analyse_window(fr, q, rng, want_modes=True)
            for bb in (0, 1):
                RB = R[f"prep{bb}"]
                P = rq[f"prep{bb}"]
                P["N"] = RB["N"]
                for kk in ("3", "4"):
                    P.setdefault(f"mirror{kk}", []).append(RB["tails_zb"][kk]["mirror_excess_frac"])
                    P.setdefault(f"plus{kk}", []).append(RB["tails_zb"][kk]["plus"]["obs"])
                    P.setdefault(f"minus{kk}", []).append(RB["tails_zb"][kk]["minus"]["obs"])
                P.setdefault("dseg5", []).append(RB["dseg"]["5"]["obs"])
                P.setdefault("dseg5_null_len", []).append(RB["dseg"]["5"]["null_lenient_resampled"])
                P.setdefault("S_sigma", []).append(RB["mode_census"]["S_median"])
                P.setdefault("sep_raw", []).append(R["frame"]["sep_mode_median"])
                P.setdefault("far7", []).append(RB["mode_census"]["far_excess"].get("7", {}))
                P.setdefault("far5", []).append(RB["mode_census"]["far_excess"].get("5", {}))
                P.setdefault("modes", []).append([
                    dict(uv=md["uv_sep_units"], xy=[md["x_ms"], md["y_ms"]], n=md["basin_count"],
                         excess_len=md["excess_frac_lenient"], core0=md["is_core0"], core1=md["is_core1"],
                         d_seg=md["d_seg_sigma"])
                    for md in RB["mode_census"]["modes"][:8]])
            log("sliding", f"q{q}", k0)
        res[f"q{q}"] = rq
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default=str(LP.FEAT))
    ap.add_argument("--out", default=str(LP.stage("m1tail") / "out_v2"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--sliding", action="store_true")
    ap.add_argument("--traj", action="store_true")
    ap.add_argument("--models", action="store_true")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--qubits", default="0,1,2,3,4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(12345)
    log("load", a.split)
    y, F, F1 = load(a.feat, a.split, a.stride)
    wm = window_means(F, F1)
    res = dict(split=a.split, N=int(len(y)), windows={}, grid=dict(GRID=GRID, SMOOTH=SMOOTH, XR=XR, YR=YR))
    store = {"zedges": ZEDGES}
    frames = {}
    QS = [int(v) for v in a.qubits.split(",")]
    for q in QS:
        for w in ("early", "1us", "2us"):
            t0 = time.time()
            fr = build_frame(wm[w][:, q, 0], wm[w][:, q, 1], y, q)
            R, st = analyse_window(fr, q, rng, want_modes=True, want_models=a.models)
            res["windows"].setdefault(w, {})[f"q{q}"] = R
            for kk, vv in st.items():
                store[f"{w}_q{q}_{kk}"] = vv
            frames[w] = fr
            log(a.split, w, f"q{q}", f"{time.time() - t0:.1f}s")
        if a.traj:
            for w, Fb in (("1us", F1), ("2us", F)):
                res["windows"][w][f"q{q}"]["trajectories"] = trajectories(frames[w], res["windows"][w][f"q{q}"], Fb, q)
                res["windows"][w][f"q{q}"]["early_link"] = early_link(frames[w], frames["early"],
                                                                      res["windows"][w][f"q{q}"], q)
        # shot-level co-occurrence: off-segment (d>5) on q vs on any other qubit is done below
        frames[f"flags_q{q}"] = {w: frames[w]["d"] > 5 for w in ("1us", "2us")}
        frames[f"zb_q{q}"] = {w: frames[w]["zb"] > 4 for w in ("1us", "2us")}
        for w in ("early", "1us", "2us"):
            frames.pop(w, None)
    res["cooccur"] = {}
    for w in (("1us", "2us") if len(QS) == NQ else ()):
        co = {}
        for key in ("flags", "zb"):
            Fl = np.stack([frames[f"{key}_q{q}"][w] for q in range(NQ)], 1)
            for q in range(NQ):
                fq = Fl[:, q]
                other = Fl[:, [j for j in range(NQ) if j != q]].any(1)
                n = int(fq.sum())
                co[f"{key}_q{q}"] = dict(n=n, frac_also_other=(float((fq & other).sum() / n) if n else None),
                                         base_rate_other=float(other.mean()),
                                         lift=(float((fq & other).sum() / n / other.mean()) if n and other.mean() else None))
        res["cooccur"][w] = co
    np.savez_compressed(f"{a.out}/m1tail_v2_store_{a.split}.npz", **store)
    with open(f"{a.out}/m1tail_v2_results_{a.split}.json", "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log("wrote main", a.split)
    if a.sliding:
        sl = sliding(F, y, rng)
        with open(f"{a.out}/m1tail_v2_sliding_{a.split}.json", "w") as fh:
            json.dump(sl, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        log("wrote sliding", a.split)


if __name__ == "__main__":
    main()
