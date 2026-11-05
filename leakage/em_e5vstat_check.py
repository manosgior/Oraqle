"""em_e5vstat_check.py (tag e5vstat, v2: |1> steady-state seed fix, see C.): ADVERSARIAL re-computation of the e4recon error-mode attribution, own code.

Independent of the e2seg/e4recon code paths except where stated (the step-detector flags/FP are re-used only to test
their null on the sub-population that matters, i.e. misread shots that stayed in their state).

What is computed (TEST split, frames/cores fit on TRAIN):
  A. error rates (Wilson) for canonical MCMit-CNN (1 us), soft CNN hard decisions (1000/600/400 ns), per-trace HERQULES.
  B. own line-frame leakage z per window (1 us = feat1us bins 0-9; 600/400 ns = featbox 0-5/0-3):
     per (qubit, neighbour cfg) cores by a smoothed-2-D-histogram global mode + local Gaussian refinement (own method),
     line through the two cores, perpendicular in prepared-|0> MAD units. Mirror excess at k = 1..5 among errors.
  C. own late-window (featbox bins 10-19 = 1-2 us, never seen by any model) state classification:
     |0> late core = mode of prepared-|0> late points; |1> late core = mean-shift seeded at
     c0_late + (c1 - c0) of the steady-state bins 5-9 (independent of the e2seg 1-us-core seeding).
     Reference sets: late core radius r in {1.0, 1.5, 2.0, 2.5} sigma and a wide half-plane set.
     O(r) = sum_cfg N(b,cfg) * p_ov(b,cfg,r);  state change SC = N - L3 - O.
  D. noise-correlation (truncation) tests of the late-core reference: GK robust correlation early-vs-late window,
     and model error rate vs late-window displacement on the clean "away" side of the |b> core.
  E. model-free identity: fraction of errors whose late window is in the OTHER state's half-plane vs correct shots
     -> s = (f_err - f_ref) / (1 - f_ref), an estimate of the state-change share that uses no overlap model.
  F. step-detector null on stationary misread shots (e4recon per-shot flags/FP at alpha = 0.01, read-only).
  G. co-failure of per-trace HERQULES and CNN errors.

Docker (graham):
  docker run -d --name leak-em-e5vstat-check --user $(id -u):$(id -g) \
    -v ~/oraqle-ics-sims/leakage/features:/feat:ro -v ~/oraqle_reports:/reports:ro -v ~/oraqle-ics-sims/leakage/f/in:/soft:ro \
    -v ~/oraqle-ics-sims/leakage/em/e4recon:/e4:ro -v ~/oraqle-ics-sims/leakage/em/e5vstat:/work \
    oraqle-torch:latest python /work/scripts/em_e5vstat_check.py
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import gaussian_filter

D, R, S, E4, O = str(LP.FEAT), str(LP.REPORTS), str(LP.SOFT), str(LP.stage("em/e4recon")), str(LP.stage("em/e5vstat") / "out")
os.makedirs(O, exist_ok=True)
SUB = int(os.environ.get("E5_SUB", "0"))          # >0: smoke test on the first SUB shots of each prepared state block
RADII = [1.0, 1.5, 2.0, 2.5]
KS = [1, 2, 3, 4, 5]
QN = ["Q1", "Q2", "Q3", "Q4", "Q5"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def bits(y):
    return ((y[:, None] >> np.arange(5)) & 1).astype(np.int8)


def cfg_of(B, q):
    o = [k for k in range(5) if k != q]
    return (B[:, o].astype(np.int64) * (1 << np.arange(4))).sum(1)


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def mode2d(P, s, seed=None, bw=0.5):
    """Own core estimator: global maximum of a Gaussian-smoothed 2-D histogram (bin 0.1 s, smoothing bw*s),
    then local Gaussian-weighted refinement (bandwidth bw*s) restricted to points within 2 s of the current estimate.
    If seed is given, the histogram step is skipped and refinement starts from seed."""
    if seed is None:
        lo = np.quantile(P, 0.002, axis=0)
        hi = np.quantile(P, 0.998, axis=0)
        nb = np.clip(((hi - lo) / (0.1 * s)).astype(int) + 1, 20, 600)
        H, ex, ey = np.histogram2d(P[:, 0], P[:, 1], bins=nb, range=[[lo[0], hi[0]], [lo[1], hi[1]]])
        Hs = gaussian_filter(H, sigma=[bw * s / (ex[1] - ex[0]), bw * s / (ey[1] - ey[0])])
        i, j = np.unravel_index(np.argmax(Hs), Hs.shape)
        c = np.array([(ex[i] + ex[i + 1]) / 2, (ey[j] + ey[j + 1]) / 2])
    else:
        c = np.asarray(seed, float)
    for _ in range(200):
        d2 = ((P - c) ** 2).sum(1)
        m = d2 < (2 * s) ** 2
        w = np.exp(-d2[m] / (2 * (bw * s) ** 2))
        cn = (w[:, None] * P[m]).sum(0) / w.sum()
        if np.linalg.norm(cn - c) < 1e-6 * s:
            return cn
        c = cn
    return c


def gk_corr(x, y):
    sx, sy = mad(x), mad(y)
    a, b = x / sx, y / sy
    u, v = mad(a + b) ** 2, mad(a - b) ** 2
    return float((u - v) / (u + v))


def wmean(F, q, b0, b1):
    return np.asarray(F[:, q, b0:b1, :], dtype=np.float64).mean(1)


def subidx(y):
    if SUB <= 0:
        return np.arange(len(y))
    out = []
    for s in range(32):
        ii = np.where(y == s)[0]
        out.append(ii[:SUB])
    return np.sort(np.concatenate(out))


G = {}


def worker(q):
    t0 = time.time()
    ytr, yte = G["ytr"], G["yte"]
    itr, ite = G["itr"], G["ite"]
    Btr, Bte = bits(ytr)[itr], bits(yte)[ite]
    cft, cfe = cfg_of(Btr, q), cfg_of(Bte, q)
    ptr, pte = Btr[:, q], Bte[:, q]
    res = {"q": q}
    Fb_tr = np.load(f"{D}/featbox_train.npy", mmap_mode="r")
    Fb_te = np.load(f"{D}/featbox_test.npy", mmap_mode="r")
    F1_tr = np.load(f"{D}/feat1us_train.npy", mmap_mode="r")
    F1_te = np.load(f"{D}/feat1us_test.npy", mmap_mode="r")
    Xb_tr = np.asarray(Fb_tr[:, q], dtype=np.float64)[itr]          # (N,20,2)
    Xb_te = np.asarray(Fb_te[:, q], dtype=np.float64)[ite]
    W = {"1us": (np.asarray(F1_tr[:, q], dtype=np.float64)[itr].mean(1), np.asarray(F1_te[:, q], dtype=np.float64)[ite].mean(1)),
         "600ns": (Xb_tr[:, 0:6].mean(1), Xb_te[:, 0:6].mean(1)),
         "400ns": (Xb_tr[:, 0:4].mean(1), Xb_te[:, 0:4].mean(1))}
    # ---------------- B. own line frame per window
    z = {}
    frinfo = {}
    for wn, (Atr, Ate) in W.items():
        zz = np.empty(len(ite))
        tilt = []
        for c in range(16):
            P0 = Atr[(cft == c) & (ptr == 0)]
            P1 = Atr[(cft == c) & (ptr == 1)]
            s = mad(P0[:, 1])
            c0 = mode2d(P0, s)
            c1 = mode2d(P1, s)
            u = (c1 - c0) / np.linalg.norm(c1 - c0)
            n = np.array([-u[1], u[0]])
            if n[1] < 0:
                n = -n
            sig = mad((P0 - c0) @ n)
            me = cfe == c
            zz[me] = ((Ate[me] - c0) @ n) / sig
            # tilt of the core line vs the mean line (mean-aligned rotation erratum check)
            mu = (P1.mean(0) - P0.mean(0))
            tilt.append(float(np.degrees(np.arctan2(u[1], u[0]) - np.arctan2(mu[1], mu[0]))))
        z[wn] = zz
        frinfo[wn] = {"core_vs_mean_line_tilt_deg_median": float(np.median(tilt)), "range": [float(min(tilt)), float(max(tilt))]}
    res["frame"] = frinfo
    # ---------------- C. late-window state classification (own)
    Ltr, Lte = Xb_tr[:, 10:20].mean(1), Xb_te[:, 10:20].mean(1)
    S59tr = Xb_tr[:, 5:10].mean(1)
    late = {k: np.zeros(len(ite), bool) for k in ["hp0", "hp1"] + [f"c{b}_{r}" for b in (0, 1) for r in RADII]}
    ul, vl, d0l, d1l, sepl = (np.empty(len(ite)) for _ in range(5))
    lateinfo = []
    for c in range(16):
        P0 = Ltr[(cft == c) & (ptr == 0)]
        P1 = Ltr[(cft == c) & (ptr == 1)]
        s = mad(P0[:, 1])
        c0 = mode2d(P0, s)
        a0 = mode2d(S59tr[(cft == c) & (ptr == 0)], s)
        # v2 fix: in the fastest-decay Q3 configuration the global density maximum of prepared-|1> bins 5-9 is the
        # |0> cloud, so the |1> steady-state mode is searched only among points > 3 sigma from the |0> mode
        P1s = S59tr[(cft == c) & (ptr == 1)]
        P1s = P1s[np.linalg.norm(P1s - a0, axis=1) > 3 * s]
        a1 = mode2d(P1s, s)
        seed = c0 + (a1 - a0)
        c1 = mode2d(P1, s, seed=seed)
        sep = np.linalg.norm(c1 - c0)
        u = (c1 - c0) / sep
        n = np.array([-u[1], u[0]])
        me = cfe == c
        X = Lte[me]
        ul[me] = ((X - c0) @ u) / sep
        vl[me] = ((X - c0) @ n) / s
        d0l[me] = np.linalg.norm(X - c0, axis=1) / s
        d1l[me] = np.linalg.norm(X - c1, axis=1) / s
        sepl[me] = sep / s
        lateinfo.append({"c": c, "sep_over_s": float(sep / s), "seed_to_c1_over_s": float(np.linalg.norm(seed - c1) / s),
                         "c1_minus_c0_raw_over_s": ((c1 - c0) / s).tolist(),
                         "s_par_over_s_perp_prep0": float(mad((P0 - c0) @ u) / s)})
    late["hp0"] = (ul < 0.5) & (np.abs(vl) < 3)
    late["hp1"] = (ul >= 0.5) & (np.abs(vl) < 3)
    for r in RADII:
        late[f"c0_{r}"] = d0l < r
        late[f"c1_{r}"] = d1l < r
    res["late_info"] = {"sep_over_s_median": float(np.median([x["sep_over_s"] for x in lateinfo])),
                        "seed_to_c1_over_s_max": float(max(x["seed_to_c1_over_s"] for x in lateinfo)),
                        "sep_over_s_min": float(min(x["sep_over_s"] for x in lateinfo)),
                        "s_par_over_s_perp_median": float(np.median([x["s_par_over_s_perp_prep0"] for x in lateinfo])),
                        "frac_prep0_in_c0_1.5": float(late["c0_1.5"][pte == 0].mean()),
                        "frac_prep1_in_c1_1.5": float(late["c1_1.5"][pte == 1].mean()),
                        "frac_prep0_hp0": float(late["hp0"][pte == 0].mean()), "frac_prep1_hp1": float(late["hp1"][pte == 1].mean())}
    # e4recon late core (read-only), for an arithmetic reproduction of their O
    lc4 = np.load(f"{E4}/out/em_e4recon_latecore.npz")[f"latecore_q{q}"][ite]
    res["late_agreement_with_e4recon"] = {f"b{b}": {"e4_core_and_mine_c{b}_1.5": float((lc4[:, b] & late[f"c{b}_1.5"]).sum() / max(lc4[:, b].sum(), 1)),
                                                     "mine_and_e4": float((lc4[:, b] & late[f"c{b}_1.5"]).sum() / max(late[f"c{b}_1.5"].sum(), 1))}
                                           for b in (0, 1)}
    # ---------------- D. noise correlation early vs late (prepared |0>, cfg-centred, in u/v units of the late frame)
    E10 = Xb_te[:, 0:10].mean(1)
    corr = {}
    for b in (0, 1):
        sel = (pte == b)
        eu, ev, lu, lv = [], [], [], []
        for c in range(16):
            m = sel & (cfe == c)
            Pe, Pl = E10[m], Lte[m]
            ce, cl = np.median(Pe, 0), np.median(Pl, 0)
            eu.append(Pe[:, 0] - ce[0]); ev.append(Pe[:, 1] - ce[1]); lu.append(Pl[:, 0] - cl[0]); lv.append(Pl[:, 1] - cl[1])
        eu, ev, lu, lv = map(np.concatenate, (eu, ev, lu, lv))
        corr[f"prep{b}"] = {"gk_par_early_late": gk_corr(eu, lu), "gk_perp_early_late": gk_corr(ev, lv)}
    # lag structure within the window for prep0 (par component), cfg-centred, GK
    sel = pte == 0
    Xc = Xb_te[sel][:, :, 0].copy()
    for c in range(16):
        m = cfe[sel] == c
        Xc[m] -= np.median(Xc[m], 0)
    corr["prep0_par_lagcorr_from_bin5"] = [gk_corr(Xc[:, 5], Xc[:, 5 + L]) for L in range(1, 15)]
    corr["prep0_par_lagcorr_from_bin2"] = [gk_corr(Xc[:, 2], Xc[:, 2 + L]) for L in range(1, 18)]
    res["noise_corr"] = corr
    # ---------------- per model/window attribution
    models = G["models"]
    res["models"] = {}
    e4fl = {}
    for wn, mdict in models.items():
        for mname, (P, P1) in mdict.items():
            pred = P[ite, q].astype(np.int8)
            out = {}
            for b in (0, 1):
                m = pte == b
                nb = int(m.sum())
                err = m & (pred != pte)
                corr_ = m & (pred == pte)
                N = int(err.sum())
                d = {"n": nb, "N": N, "err_pct": 100 * N / nb, "err_ci": [100 * x for x in wilson(N, nb)]}
                zw = z[wn]
                for k in KS:
                    nP, nM = int((err & (zw > k)).sum()), int((err & (zw < -k)).sum())
                    d[f"L{k}"] = nP - nM
                    d[f"L{k}_var"] = nP + nM
                    hP, hM = int((m & (zw > k)).sum()), int((m & (zw < -k)).sum())
                    d[f"leaked_shots{k}"] = hP - hM
                    d[f"leaked_shots{k}_var"] = hP + hM
                # overlap reference variants
                refs = {f"r{r}": late[f"c{b}_{r}"] for r in RADII}
                refs["hp"] = late[f"hp{b}"]
                refs["e4_lc"] = lc4[:, b]
                for rn, ref in refs.items():
                    Oexp, Ovar, nref, neref = 0.0, 0.0, 0, 0
                    for c in range(16):
                        mc = m & (cfe == c)
                        lr = mc & ref
                        nl = int(lr.sum())
                        ne = int((lr & err).sum())
                        nref += nl
                        neref += ne
                        if nl == 0:
                            continue
                        p = ne / nl
                        Nc = int(mc.sum())
                        Oexp += Nc * p
                        Ovar += Nc * Nc * max(p * (1 - p), 1.0 / nl ** 2) / nl
                    d[f"O_{rn}"] = Oexp
                    d[f"O_{rn}_var"] = Ovar
                    d[f"pov_{rn}_pct"] = 100 * neref / max(nref, 1)
                    d[f"nref_{rn}"] = nref
                # late-window noise-correlation test: error rate vs displacement on the "away" side of the |b> core
                ub = (ul if b == 0 else 1 - ul) * sepl      # sigma units: 0 at own core, increasing toward the other state
                awaysel = m & (ub <= 0) & (np.abs(vl) < 1.0)       # displaced AWAY from the other state (clean of transitions)
                prof = []
                for lo, hi in [(-0.5, 0.0), (-1.0, -0.5), (-1.5, -1.0), (-2.0, -1.5), (-3.0, -2.0), (-9.0, -3.0)]:
                    mm = awaysel & (ub > lo) & (ub <= hi)
                    nn = int(mm.sum())
                    ee = int((mm & err).sum())
                    prof.append({"u_away_sigma": [lo, hi], "n": nn, "err": ee, "rate_pct": 100 * ee / max(nn, 1),
                                 "ci": [100 * x for x in wilson(ee, nn)] if nn else None})
                d["away_profile"] = prof
                pprof = []
                for lo, hi in [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.5)]:
                    mm = m & (np.abs(vl) >= lo) & (np.abs(vl) < hi) & (np.abs(ub) < 0.5)
                    nn, ee = int(mm.sum()), int((mm & err).sum())
                    pprof.append({"abs_v_sigma": [lo, hi], "n": nn, "err": ee, "rate_pct": 100 * ee / max(nn, 1),
                                  "ci": [100 * x for x in wilson(ee, nn)] if nn else None})
                d["perp_profile"] = pprof
                # E. model-free identity from the late window
                other = late[f"hp{1 - b}"]
                fe = float(other[err].mean()) if N else None
                fc = float(other[corr_].mean())
                d["late_other_frac_err"] = fe
                d["late_other_frac_correct"] = fc
                d["late_own_frac_err"] = float(late[f"hp{b}"][err].mean()) if N else None
                d["late_neither_frac_err"] = float((~late["hp0"] & ~late["hp1"])[err].mean()) if N else None
                d["s_identity_vs_correct"] = (fe - fc) / (1 - fc) if N else None
                # errors excluding k=3 leaked shots
                errL = err & ~(zw > 3)
                d["late_other_frac_err_noL"] = float(other[errL].mean()) if errL.any() else None
                out[f"prep{b}"] = d
            res["models"][f"{wn}|{mname}"] = out
            # stash per-shot error vectors for F (only 1 us canonical)
            if wn == "1us" and mname == "cnn_canonical":
                e4fl["err"] = pred != pte
    # ---------------- F. step-detector null on stationary misread shots (e4recon flags, read-only)
    try:
        A = np.load(f"{E4}/out/em_e4recon_allshots_box1us_q{q}.npz")
        Fd = (A["Sd"] > A["thrd"][:, 1])[ite]
        Fu = (A["Su"] > A["thru"][:, 1])[ite]
        FPd = A["FPd"][:, 1].astype(np.float64)[ite]
        FPu = A["FPu"][:, 1].astype(np.float64)[ite]
        err = e4fl["err"]
        fnull = {}
        for b, F, FP, test in ((1, Fd, FPd, "down"), (0, Fu, FPu, "up")):
            m = pte == b
            for rn, ref in (("e4_lc", lc4[:, b]), ("my_r1.5", late[f"c{b}_1.5"]), ("my_r1.0", late[f"c{b}_1.0"]), ("my_hp", late[f"hp{b}"])):
                for en, sel in (("misread", m & ref & err), ("correct", m & ref & ~err)):
                    nn = int(sel.sum())
                    obs = int(F[sel].sum())
                    exp = float(FP[sel].sum())
                    fnull[f"{test}_prep{b}_{rn}_{en}"] = {"n": nn, "obs": obs, "exp": exp, "ratio": obs / exp if exp > 0 else None,
                                                         "obs_ci95": [obs - 1.96 * np.sqrt(max(obs, 1)), obs + 1.96 * np.sqrt(max(obs, 1))]}
            # T as e4recon computes it (order A, among non-leaked errors) for an arithmetic check
            E = m & err
            Pz = E & (A["z"][ite] > 3) if q != 3 else np.zeros_like(E)
            Mz = E & (A["z"][ite] < -3) if q != 3 else np.zeros_like(E)
            notP = E & ~Pz
            fnull[f"T_orderA_prep{b}"] = {"flags": int((F & notP).sum() + (F & Mz).sum()), "fp": float(FP[notP].sum() + FP[Mz].sum())}
            # flags among errors whose late window is in the misread state's half-plane (state changed) vs own core
            for rn, ref in (("late_other_hp", late[f"hp{1 - b}"]), ("late_own_hp", late[f"hp{b}"])):
                sel = E & ref
                fnull[f"{test}_prep{b}_err_{rn}"] = {"n": int(sel.sum()), "obs": int(F[sel].sum()), "exp": float(FP[sel].sum())}
        res["step_null"] = fnull
    except Exception as ex:  # noqa: BLE001
        res["step_null_error"] = repr(ex)
    # keep per-shot late classes for the co-failure / pooled work in main
    np.savez_compressed(f"{O}/em_e5vstat_pershot_q{q}{'_sub' if SUB else ''}.npz", z1us=z["1us"].astype(np.float32),
                        z600=z["600ns"].astype(np.float32), z400=z["400ns"].astype(np.float32),
                        ul=ul.astype(np.float32), vl=vl.astype(np.float32), d0l=d0l.astype(np.float32), d1l=d1l.astype(np.float32))
    # compare my z with e4recon z
    Z4 = np.load(f"{E4}/out/em_e4recon_z.npz")
    res["z_vs_e4recon"] = {}
    for wn, key in (("1us", "z_1us"), ("600ns", "z_600ns"), ("400ns", "z_400ns")):
        z4 = Z4[key][ite, q].astype(np.float64)
        a, b_ = z[wn] > 3, z4 > 3
        res["z_vs_e4recon"][wn] = {"pearson": float(np.corrcoef(z[wn], z4)[0, 1]), "flag_jaccard": float((a & b_).sum() / max((a | b_).sum(), 1)),
                                   "n_flag_mine": int(a.sum()), "n_flag_e4": int(b_.sum()), "median_abs_diff": float(np.median(np.abs(z[wn] - z4)))}
    log("q", q, "done", round(time.time() - t0), "s")
    return res


def main():
    t0 = time.time()
    ytr, yte = np.load(D + "/y_train.npy"), np.load(D + "/y_test.npy")
    G["ytr"], G["yte"] = ytr, yte
    G["itr"], G["ite"] = subidx(ytr), subidx(yte)
    Bte = bits(yte)
    pb = np.load(R + "/preds_MCMitCNN_pb.npy").astype(np.int8)
    tb = np.load(R + "/preds_MCMitCNN_tb.npy").astype(np.int8)
    chk = {"tb_equals_prepared": bool(np.array_equal(tb, Bte))}
    soft = {T: np.load(f"{S}/cnn_soft_{T}ns.npz")["prob"].astype(np.float32) for T in (1000, 600, 400)}
    H = np.load(f"{E4}/in/em_e4recon_herq_in.npz")
    chk["herq_y_1us_eq"] = bool(np.array_equal(H["y_1us"].astype(np.int64), yte.astype(np.int64)))
    chk["herq_y_400_eq"] = bool(np.array_equal(H["y_400ns"].astype(np.int64), yte.astype(np.int64)))
    G["models"] = {
        "1us": {"cnn_canonical": (pb, None), "cnn_soft1000": ((soft[1000] > 0.5).astype(np.int8), None),
                "herq_pertrace": (H["pt_pred_1us"].astype(np.int8), None)},
        "600ns": {"cnn_soft600": ((soft[600] > 0.5).astype(np.int8), None)},
        "400ns": {"cnn_soft400": ((soft[400] > 0.5).astype(np.int8), None),
                  "herq_pertrace": (H["pt_pred_400ns"].astype(np.int8), None)},
    }
    # G. co-failure (full TEST)
    cof = {}
    for wn, cnnp, hp in (("1us", pb, H["pt_pred_1us"].astype(np.int8)), ("400ns", (soft[400] > 0.5).astype(np.int8), H["pt_pred_400ns"].astype(np.int8))):
        for q in range(5):
            ec, eh = cnnp[:, q] != Bte[:, q], hp[:, q] != Bte[:, q]
            k, n = int((ec & eh).sum()), int(eh.sum())
            cof[f"{wn}_{QN[q]}"] = {"herq_err": n, "cnn_err": int(ec.sum()), "both": k, "frac_herq_err_also_cnn": k / n,
                                    "ci": wilson(k, n), "frac_cnn_err_also_herq": k / int(ec.sum())}
    # soft vs canonical agreement
    chk["canon_vs_soft1000_disagree_pct"] = [float(100 * (pb[:, q] != (soft[1000][:, q] > 0.5)).mean()) for q in range(5)]
    with Pool(int(os.environ.get("E5_NPROC", "5"))) as pool:
        outs = pool.map(worker, range(5))
    res = {"note": __doc__, "sub": SUB, "checks": chk, "cofailure": cof, "per_qubit": {QN[o["q"]]: o for o in outs}}
    json.dump(res, open(f"{O}/em_e5vstat_check{'_sub' if SUB else ''}.json", "w"), indent=1, default=float)
    log("all done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
