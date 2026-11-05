# e5vrepro: independent reproduction of the MCMit-CNN error-mode attribution (L -> T -> O -> S)
# from my own demodulated raw traces (em_e5vrepro_demod.py, bins2 = 20-ns boxcar bins, own IF, DC removed).
# Nothing from the shared feature arrays or other agents' code is used.  Only model predictions are inputs:
#   1 us : canonical MCMit-CNN hard predictions (preds_MCMitCNN_pb.npy, TEST order)
#   400 ns: soft MCMit-CNN (cnn_soft_400ns.npz, prob>=0.5)
#
# v2 (after v1 diagnostics, before any comparison with reconciled shares): late-window cores anchored (see loop).
# Own choices (all fixed before looking at the attribution numbers):
#  * Frame: per qubit, per neighbour configuration (other 4 prepared bits, 16 cfgs), window-mean IQ points on TRAIN,
#    density cores by Gaussian mean-shift (bandwidth 0.7 sigma), line through the two cores, perpendicular unit
#    = robust (MAD) width of prepared-|0> perpendicular coordinate.  Applied unchanged to TEST.
#    Sign of the perpendicular: per qubit/window, the side with the larger TRAIN prepared-|1> |z|>3 count.
#  * Leakage: z>3 flagged, z<-3 mirror subtracted.
#  * Late core (overlap reference): late window = samples 500-999 (20-ns bins 50-99); per-cfg late frame from TRAIN;
#    shot "in |b> late core" if |p - c_b| < r sigma_late (r=2 primary) and on the |b> side of the midpoint.
#  * Step detector: KNOWN-LEVEL change-point GLRT on the 1-D projection of each 20-ns bin onto the per-bin
#    |0>->|1> mean-trajectory difference (trajectories from TRAIN late-core stationary shots, per cfg), full noise
#    covariance (TRAIN residuals), whitened.  Lambda = max_tau l(step at tau) - max(l(stay b), l(stay 1-b)).
#    Threshold: alpha (0.01 primary) under "stay b" by Monte Carlo per cfg.
#    FP correction: exact Gaussian null conditioned on the shot's matched-filter level t (MC on a grid of t).
#    Because both stationary means lie on the same whitened line, the conditional null is the same for
#    "stayed in b" and "was in 1-b all window".
#  * Gate: null passes on real quasi-null TEST shots (prepared b, late core b, |z|<=3): obs/exp in [0.6,1.3],
#    and efficiency given a (matched-filter) misread >= 5%.
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import numpy as np, json, os, sys, time

OUT = str(LP.stage('em/e5vrepro') / 'out')
t0 = time.time()
def log(*a):
    print('[%7.1fs]' % (time.time() - t0), *a, flush=True)

WIN = {'1us': (0, 50), '400ns': (0, 20), 'late': (50, 100)}  # 20-ns bin ranges (dict order matters: 1us first)
ALPHAS = [0.05, 0.01, 0.002]
ALPHA0 = 0.01
RADII = [1.5, 2.0, 2.5]
R0 = 2.0
rng = np.random.default_rng(20261105)

Ytr = np.load(OUT + '/em_e5vrepro_y_train.npy').astype(np.int64)
Yte = np.load(OUT + '/em_e5vrepro_y_test.npy').astype(np.int64)
log('loading bins')
Btr = np.load(OUT + '/em_e5vrepro_bins2_train.npy')
Bte = np.load(OUT + '/em_e5vrepro_bins2_test.npy')
log('loaded', Btr.shape, Bte.shape)

# ---------------- model predictions ----------------
pb = np.load(str(LP.REPORTS / 'preds_MCMitCNN_pb.npy')).astype(np.int64)
tb = np.load(str(LP.REPORTS / 'preds_MCMitCNN_tb.npy')).astype(np.int64)
soft4 = np.load(str(LP.SOFT / 'cnn_soft_400ns.npz'))  # was /work/in (byte copy)
soft10 = np.load(str(LP.SOFT / 'cnn_soft_1000ns.npz'))  # was /work/in (byte copy)
bits_te = np.stack([(Yte >> q) & 1 for q in range(5)], 1)
bits_tr = np.stack([(Ytr >> q) & 1 for q in range(5)], 1)
checks = dict(tb_matches_y=bool((tb == bits_te).all()),
              soft400_label_matches_y=bool((soft4['label'].astype(np.int64) == bits_te).all()),
              soft1000_label_matches_y=bool((soft10['label'].astype(np.int64) == bits_te).all()))
log('alignment checks', checks)
PRED = {'1us': pb, '400ns': (soft4['prob'].astype(np.float32) >= 0.5).astype(np.int64)}
ERR = {w: (PRED[w] != bits_te) for w in PRED}
soft1us_err = ((soft10['prob'].astype(np.float32) >= 0.5).astype(np.int64) != bits_te)

def cfg_of(y, q):
    lo = y & ((1 << q) - 1); hi = y >> (q + 1)
    return lo | (hi << q)

def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))

def meanshift(P, h, c, it=300):
    for _ in range(it):
        w = np.exp(-((P - c) ** 2).sum(1) / (2 * h * h))
        cn = (w[:, None] * P).sum(0) / w.sum()
        if np.hypot(*(cn - c)) < 1e-5 * h:
            return cn
        c = cn
    return c

def c2r(z):
    return np.stack([z.real, z.imag], -1).astype(np.float64)

def wilson(k, n, zc=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + zc * zc / n
    c = (p + zc * zc / (2 * n)) / d; h = zc * np.sqrt(p * (1 - p) / n + zc * zc / (4 * n * n)) / d
    return [100 * (c - h), 100 * (c + h)]

def ci(val, var, n):
    s = 1.96 * np.sqrt(max(var, 0.0))
    return [100 * (val - s) / n, 100 * (val + s) / n]

# ---------------- frames ----------------
def fit_frames(Ptr, btr, ctr, sel0=None, sel1=None):
    """per-cfg core-line frames from TRAIN window-mean points (n,2).
    sel0/sel1 (optional): boolean masks of shots used ONLY to initialise the mean-shift (anchoring);
    the mode search itself always runs on all prepared-|b> shots of the configuration."""
    fr = {}
    for c in range(16):
        m0 = (btr == 0) & (ctr == c); m1 = (btr == 1) & (ctr == c)
        P0 = Ptr[m0]; P1 = Ptr[m1]
        s_iso = np.median([mad(P0[:, 0]), mad(P0[:, 1])])
        i0 = np.median(Ptr[m0 & sel0], 0) if sel0 is not None else np.median(P0, 0)
        i1 = np.median(Ptr[m1 & sel1], 0) if sel1 is not None else np.median(P1, 0)
        c0 = meanshift(P0, 0.7 * s_iso, i0)
        c1 = meanshift(P1, 0.7 * s_iso, i1)
        u = (c1 - c0) / np.hypot(*(c1 - c0)); n = np.array([-u[1], u[0]])
        sig = mad((P0 - c0) @ n)
        fr[c] = dict(c0=c0, c1=c1, u=u, n=n, sig=sig, sep=float(np.hypot(*(c1 - c0)) / sig),
                     sig_iso=float(s_iso))
    return fr

def apply_frame(P, cf, fr):
    z = np.empty(len(P)); a = np.empty(len(P)); d0 = np.empty(len(P)); d1 = np.empty(len(P))
    for c in range(16):
        m = cf == c; f = fr[c]
        D = P[m] - f['c0']
        z[m] = (D @ f['n']) / f['sig']
        a[m] = (D @ f['u']) / np.hypot(*(f['c1'] - f['c0']))
        d0[m] = np.hypot(*(P[m] - f['c0']).T) / f['sig']
        d1[m] = np.hypot(*(P[m] - f['c1']).T) / f['sig']
    return z, a, d0, d1

res = dict(meta=dict(tag='e5vrepro', checks=checks,
                     if_json=json.load(open(OUT + '/em_e5vrepro_if2.json')),
                     windows_20ns_bins=WIN, alpha_primary=ALPHA0, late_core_radius_primary=R0,
                     n_test=int(len(Yte)), n_train=int(len(Ytr))),
           error_rates={}, leak_E3={}, frames={}, qubits={})
pershot = {}

for q in range(5):
    log('=== qubit', q)
    btr, bte = bits_tr[:, q], bits_te[:, q]
    ctr, cte = cfg_of(Ytr, q), cfg_of(Yte, q)
    Xtr = Btr[:, q, :]; Xte = Bte[:, q, :]
    Z = {}
    for w, (k0, k1) in WIN.items():
        Ptr = c2r(Xtr[:, k0:k1].mean(1)); Pte = c2r(Xte[:, k0:k1].mean(1))
        if w == 'late':
            # v2 fix: anchor the late-window mode search.  v1 started at the median of all prepared-|1> late points,
            # which for Q3 (most prepared-|1> shots decayed or leaked by 1-2 us) and Q2 collapsed onto the |0> mode.
            # Anchor: shots whose 800-1000 ns window mean (in the 1-us frame) is clearly in |b>, and |z_1us|<=3.
            aend = apply_frame(c2r(Xtr[:, 40:50].mean(1)), ctr, FR1)[1]
            fr = fit_frames(Ptr, btr, ctr, sel0=(aend < 0.4), sel1=(aend > 0.6) & (np.abs(Z['1us']['ztr']) <= 3))
        else:
            fr = fit_frames(Ptr, btr, ctr)
        if w == '1us':
            FR1 = fr
        ztr, atr, d0tr, d1tr = apply_frame(Ptr, ctr, fr)
        zte, ate, d0te, d1te = apply_frame(Pte, cte, fr)
        m1 = btr == 1
        exc = int((ztr[m1] > 3).sum()) - int((ztr[m1] < -3).sum())
        sgn = 1.0 if exc >= 0 else -1.0
        ztr *= sgn; zte *= sgn
        Z[w] = dict(ztr=ztr, zte=zte, atr=atr, ate=ate, d0tr=d0tr, d1tr=d1tr, d0te=d0te, d1te=d1te)
        res['frames'].setdefault(str(q), {})[w] = dict(
            sign=sgn, sep_median=float(np.median([fr[c]['sep'] for c in fr])),
            sep_range=[float(min(fr[c]['sep'] for c in fr)), float(max(fr[c]['sep'] for c in fr))],
            tilt_deg_vs_cfg0=[float(np.degrees(np.arctan2(fr[c]['u'][1], fr[c]['u'][0]) -
                                               np.arctan2(fr[0]['u'][1], fr[0]['u'][0]))) for c in range(16)])
        if w != 'late':
            e3 = {}
            for b in (0, 1):
                for nm, zz, bb, N in (('test', zte, bte, None), ('train', ztr, btr, None)):
                    mm = bb == b; Np = int((zz[mm] > 3).sum()); Nm = int((zz[mm] < -3).sum()); NN = int(mm.sum())
                    e3['%s_prep%d' % (nm, b)] = dict(E3=100 * (Np - Nm) / NN, ci=ci(Np - Nm, Np + Nm, NN),
                                                     Nplus=Np, Nminus=Nm, N=NN)
            Np = int((zte > 3).sum()); Nm = int((zte < -3).sum())
            e3['test_all'] = dict(E3=100 * (Np - Nm) / len(zte), ci=ci(Np - Nm, Np + Nm, len(zte)))
            res['leak_E3'].setdefault(str(q), {})[w] = e3
            log(q, w, 'E3 prep1 %.3f prep0 %.3f all %.3f  sep %.2f' % (
                e3['test_prep1']['E3'], e3['test_prep0']['E3'], e3['test_all']['E3'],
                res['frames'][str(q)][w]['sep_median']))
    # late cores
    L = Z['late']
    def latecore(d0, d1, a, bsel, r):
        return np.where(bsel == 1, (d1 < r) & (a > 0.5), (d0 < r) & (a < 0.5))
    LC_te = {r: latecore(L['d0te'], L['d1te'], L['ate'], bte, r) for r in RADII}
    LC_tr = latecore(L['d0tr'], L['d1tr'], L['atr'], btr, R0)

    qres = {}
    for w in ('1us', '400ns'):
        k0, k1 = WIN[w]; K = k1 - k0
        err = ERR[w][:, q]
        zte = Z[w]['zte']; ztr = Z[w]['ztr']
        nerr = int(err.sum())
        res['error_rates'].setdefault(str(q), {})[w] = dict(rate=100 * nerr / len(err), ci=wilson(nerr, len(err)),
                                                            n_err=nerr)
        if w == '1us':
            res['error_rates'][str(q)]['1us_soft_model'] = 100 * float(soft1us_err[:, q].mean())
        # ---- reference trajectories & noise (TRAIN, stationary late-core, |z|<=3) ----
        stat_tr = LC_tr & (np.abs(ztr) <= 3)
        Xw_tr = Xtr[:, k0:k1]; Xw_te = Xte[:, k0:k1]
        M = {}
        for c in range(16):
            for b in (0, 1):
                sel = stat_tr & (btr == b) & (ctr == c)
                M[(c, b)] = Xw_tr[sel].mean(0)
        def project(Xw, cf):
            Yp = np.empty(Xw.shape, np.float64)
            for c in range(16):
                m = cf == c
                dl = M[(c, 1)] - M[(c, 0)]; e = dl / np.abs(dl)
                Yp[m] = ((Xw[m] - M[(c, 0)]) * np.conj(e)).real
            return Yp
        Ytr_p = project(Xw_tr, ctr)
        lev = {c: np.abs(M[(c, 1)] - M[(c, 0)]) for c in range(16)}
        resid = []; resid_b = {0: [], 1: []}
        for c in range(16):
            for b in (0, 1):
                sel = stat_tr & (btr == b) & (ctr == c)
                rr = Ytr_p[sel] - (lev[c] if b == 1 else 0.0)
                resid.append(rr); resid_b[b].append(rr)
        resid = np.concatenate(resid)
        Sig = np.cov(resid.T)
        Sig_b = {b: np.cov(np.concatenate(resid_b[b]).T) for b in (0, 1)}
        Lch = np.linalg.cholesky(Sig); Linv = np.linalg.inv(Lch)
        noise_info = dict(diag_mean=float(np.mean(np.diag(Sig))),
                          var_ratio_prep1_over_prep0=float(np.mean(np.diag(Sig_b[1])) / np.mean(np.diag(Sig_b[0]))),
                          lag1_corr=float(np.mean([Sig[i, i + 1] / np.sqrt(Sig[i, i] * Sig[i + 1, i + 1])
                                                   for i in range(K - 1)])),
                          n_resid=int(len(resid)))
        Yte_p = project(Xw_te, cte)
        Yte_w = Yte_p @ Linv.T
        wq = {}
        for bb in (1, 0):   # bb = prepared (start) state; test bb -> 1-bb
            dname = 'relax' if bb == 1 else 'excite'
            mask_b = bte == bb
            Lam = np.full(len(Yte), np.nan); Tlev = np.full(len(Yte), np.nan)
            PFP = {a: np.full(len(Yte), np.nan) for a in ALPHAS}
            FLAG = {a: np.zeros(len(Yte), bool) for a in ALPHAS}
            mcvar_cell = {a: {} for a in ALPHAS}
            eff = {a: [] for a in ALPHAS}
            pflag_tau = {a: np.zeros(K - 1) for a in ALPHAS}
            lam_thr = {a: [] for a in ALPHAS}
            grid = np.linspace(-0.6, 1.6, 89)
            NG = 6000; NL = 100000; NE = 3000
            for c in range(16):
                A = lev[c] if bb == 1 else np.zeros(K)
                C = np.zeros(K) if bb == 1 else lev[c]
                S = np.array([np.where(np.arange(K) < tau, A, C) for tau in range(1, K)])
                At, Ct, St = A @ Linv.T, C @ Linv.T, S @ Linv.T
                hA, hC, hS = 0.5 * At @ At, 0.5 * Ct @ Ct, 0.5 * (St ** 2).sum(1)
                D = np.sqrt(((Ct - At) ** 2).sum()); wv = (Ct - At) / D
                def stat(Yw):
                    lS = Yw @ St.T - hS
                    l0 = np.maximum(Yw @ At - hA, Yw @ Ct - hC)
                    return lS.max(1) - l0, ((Yw - At) @ wv) / D
                # threshold under "stay bb"
                E = rng.standard_normal((NL, K))
                l_null, _ = stat(At + E)
                thr = {a: float(np.quantile(l_null, 1 - a)) for a in ALPHAS}
                for a in ALPHAS: lam_thr[a].append(thr[a])
                # conditional FP on level grid
                pg = {a: np.zeros(len(grid)) for a in ALPHAS}
                for gi, tg in enumerate(grid):
                    E = rng.standard_normal((NG, K)); E -= np.outer(E @ wv, wv)
                    lg, _ = stat(At + tg * D * wv + E)
                    for a in ALPHAS: pg[a][gi] = (lg > thr[a]).mean()
                # efficiency vs true step time (uniform over interior bins)
                pm_tot = 0.0; pfm_tot = {a: 0.0 for a in ALPHAS}
                for ti in range(K - 1):
                    E = rng.standard_normal((NE, K))
                    le, te = stat(St[ti] + E)
                    mis = te > 0.5
                    pm_tot += mis.mean()
                    for a in ALPHAS:
                        pflag_tau[a][ti] += (le > thr[a]).mean() / 16
                        pfm_tot[a] += ((le > thr[a]) & mis).mean()
                for a in ALPHAS: eff[a].append(pfm_tot[a] / max(pm_tot, 1e-12))
                # apply to TEST shots of prepared bb in cfg c
                m = mask_b & (cte == c)
                lt, tt = stat(Yte_w[m])
                Lam[m] = lt; Tlev[m] = tt
                for a in ALPHAS:
                    FLAG[a][m] = lt > thr[a]
                    PFP[a][m] = np.interp(tt, grid, pg[a])
                    # MC variance: shots sharing a grid cell share the MC error of that cell (conservative)
                    idx = np.clip(np.searchsorted(grid, tt), 0, len(grid) - 1)
                    mcvar_cell[a][c] = (idx, pg[a])
            # ----- attribution pieces for this direction -----
            d = dict(noise=noise_info, K=K)
            for a in ALPHAS:
                pf = PFP[a]; fl = FLAG[a]
                def mcvar(sel):
                    v = 0.0
                    for c in range(16):
                        idx, pgc = mcvar_cell[a][c]
                        mm = (mask_b & (cte == c))
                        ss = sel[mm]
                        cnt = np.bincount(idx[ss], minlength=len(grid)).astype(float)
                        v += float((cnt ** 2 * pgc * (1 - pgc) / NG).sum())
                    return v
                E_b = err & mask_b
                zp = zte > 3; zm = zte < -3
                def X(sel):
                    return float(fl[sel].sum() - pf[sel].sum())
                Tval = X(E_b) - (X(E_b & zp) - X(E_b & zm))
                F = int(fl[E_b].sum())
                varT = F + mcvar(E_b)
                QN = mask_b & LC_te[R0] & (np.abs(zte) <= 3)
                qn = dict(obs=int(fl[QN].sum()), exp=float(pf[QN].sum()), n=int(QN.sum()),
                          obs_err=int(fl[QN & err].sum()), exp_err=float(pf[QN & err].sum()), n_err=int((QN & err).sum()),
                          obs_ok=int(fl[QN & ~err].sum()), exp_ok=float(pf[QN & ~err].sum()))
                qn['ratio'] = qn['obs'] / max(qn['exp'], 1e-9)
                qn['ratio_err'] = qn['obs_err'] / max(qn['exp_err'], 1e-9)
                eff_m = float(np.mean(eff[a]))
                pft = pflag_tau[a]
                hi = np.where(pft >= 0.5)[0]
                gate = (0.6 <= qn['ratio'] <= 1.3) and eff_m >= 0.05
                d['alpha_%g' % a] = dict(T=Tval, varT=varT, flags_in_errors=F, fp_sum_errors=float(pf[E_b].sum()),
                                         quasi_null=qn, eff_given_mf_misread=eff_m,
                                         pflag_ge_0p5_ns=([int(20 * (hi[0] + 1)), int(20 * (hi[-1] + 1))]
                                                          if len(hi) else None),
                                         pflag_max=float(pft.max()), lam_thr_median=float(np.median(lam_thr[a])),
                                         gate_pass=bool(gate))
            wq[dname] = d
            pershot['%s_q%d_%s_lam' % (w, q, dname)] = Lam.astype(np.float32)
            pershot['%s_q%d_%s_t' % (w, q, dname)] = Tlev.astype(np.float32)
            pershot['%s_q%d_%s_pfp' % (w, q, dname)] = PFP[ALPHA0].astype(np.float32)
            pershot['%s_q%d_%s_flag' % (w, q, dname)] = FLAG[ALPHA0]
            log(q, w, dname, json.dumps({k: (v['T'], v['quasi_null']['ratio'], v['quasi_null']['ratio_err'],
                                             v['eff_given_mf_misread'], v['gate_pass'])
                                         for k, v in d.items() if k.startswith('alpha')}))
        # ---- L, O, S per prepared bit ----
        per_b = {}
        for bb in (0, 1):
            mask_b = bte == bb; E_b = err & mask_b
            N = int(E_b.sum())
            Lp = int((E_b & (zte > 3)).sum()); Lm = int((E_b & (zte < -3)).sum())
            Ov = {}
            for r in RADII:
                Oexp = 0.0; varO = 0.0; Oexp_nl = 0.0; pov = []
                for c in range(16):
                    mc = mask_b & (cte == c)
                    core = mc & LC_te[r]
                    ncore = int(core.sum()); k = int((core & err).sum())
                    p = k / ncore; Ncfg = int(mc.sum())
                    Oexp += Ncfg * p; varO += Ncfg ** 2 * p * (1 - p) / ncore
                    core2 = core & (np.abs(zte) <= 3); p2 = (core2 & err).sum() / core2.sum()
                    Oexp_nl += Ncfg * p2
                    pov.append(dict(cfg=c, n_core=ncore, frac_core=ncore / Ncfg, k_err=k, p_ov=p))
                Ov[r] = dict(Oexp=Oexp, varO=varO, Oexp_excl_absz3=Oexp_nl, per_cfg=pov,
                             core_frac=float(np.mean([x['frac_core'] for x in pov])))
            # late-window model-free check: errors / correct shots in the OTHER late core
            other_core = np.where(bb == 1, (L['d0te'] < R0) & (L['ate'] < 0.5), (L['d1te'] < R0) & (L['ate'] > 0.5))
            lw = dict(err_in_other_core=float(other_core[E_b].mean()) * 100,
                      ok_in_other_core=float(other_core[mask_b & ~err].mean()) * 100,
                      err_in_other_core_ci=wilson(int(other_core[E_b].sum()), N))
            per_b[bb] = dict(N=N, n_shots=int(mask_b.sum()), Lp=Lp, Lm=Lm, L=Lp - Lm, overlap=Ov, latewin=lw)
        qres[w] = dict(steps=wq, per_bit=per_b)
    res['qubits'][str(q)] = qres
    pershot['q%d_z1us' % q] = Z['1us']['zte'].astype(np.float32)
    pershot['q%d_z400' % q] = Z['400ns']['zte'].astype(np.float32)
    pershot['q%d_late_core_r2' % q] = LC_te[R0]
    json.dump(res, open(OUT + '/em_e5vrepro_raw_v2.json', 'w'), indent=1, default=float)

np.savez_compressed(OUT + '/em_e5vrepro_pershot_v2.npz', **pershot)
json.dump(res, open(OUT + '/em_e5vrepro_raw_v2.json', 'w'), indent=1, default=float)
log('done')
