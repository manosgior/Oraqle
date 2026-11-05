"""vstatistics_main.py -- adversarial STATISTICS check of the reconciled leakage headline (tag vstatistics).

Independent re-computation with methods that differ from m1tail / r_spotcheck:
  * Frame fitted on TRAIN, applied to TEST (out of sample).  Per prepared 5-bit state s:
      centre mu_s and width by a truncated-normal ML fit with ASYMMETRIC truncation [mu-4s, mu+1s]
      (uses the minus side + the top of the core; the + side, where the excess lives, is excluded).
      sigma_c = width of the prepared-|0> state with the same neighbours (same estimator).
      Prepared-|1> centre fitted with sigma fixed to sigma_c (state-independent-noise null);
      own width also fitted freely and reported.
  * A second, fully independent noise scale from WITHIN-shot bin-to-bin fluctuations of the
    boxcar features (featbox), which never looks at the across-shot window-mean distribution.
  * Gaussian-null excess G(k) = N(z>k)/N - Q(k) next to the mirror excess E(k); minus-side
    calibration ratio N(z<-k)/(N Q(k)).
  * CIs: binomial/mirror, stratified CHUNK bootstrap (500 consecutive rows within each state block),
    random-effects over the 16 neighbour configurations, and in-sample vs out-of-sample frame.
  * Constant-length windows (0-500, 500-1000, 1000-1500, 1500-2000 ns) to separate creation during
    readout from the SNR growth with window length.
  * MCMit-CNN cross-tab with my own flags; check of the error-subpopulation centre (mirror bias).
  * Train/test acquisition overlap: drift pattern vs fractional position in block.
Run on graham:
  docker run -d --name leak-vstat-main --user $(id -u):$(id -g) -v ~/oraqle-ics-sims/leakage:/lk:ro \
     -v ~/oraqle_reports:/rep:ro -v ~/oraqle-ics-sims/leakage/vstatistics:/out oraqle-torch:latest \
     python /out/vstatistics_main.py
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import math
import time

import numpy as np
from scipy import optimize
from scipy.special import ndtr

T0 = time.time()
D = str(LP.FEAT)
OUT = LP.outdir(LP.stage("vstatistics"))
PAPER = [f"Q{q+1}" for q in range(5)]
KS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
Z95 = 1.959963984540054


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def Qtail(k):
    return 0.5 * math.erfc(k / math.sqrt(2))


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def mirror(npl, nmi, n):
    e = (npl - nmi) / n
    h = Z95 * math.sqrt(npl + nmi) / n
    return [e, e - h, e + h]


def wilson(k, n):
    if n == 0:
        return [None, None, None]
    p = k / n
    den = 1 + Z95 ** 2 / n
    c = (p + Z95 ** 2 / (2 * n)) / den
    h = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n)) / den
    return [p, c - h, c + h]


def tnfit(x, mu, s, lo_k=4.0, hi_k=1.0, fix_s=None, iters=40):
    """Truncated-normal ML on [mu-lo_k*s, mu+hi_k*s] (window re-centred each iteration)."""
    x = np.asarray(x, np.float64)
    if fix_s is not None:
        s = fix_s
    for _ in range(iters):
        a, b = mu - lo_k * s, mu + hi_k * s
        xi = x[(x > a) & (x < b)]
        n = len(xi)
        c = xi.mean()
        d = xi - c
        S1, S2 = d.sum(), (d * d).sum()
        ac, bc = a - c, b - c
        if fix_s is None:
            def nll(p):
                m, ls = p
                sd = math.exp(ls)
                Z = ndtr((bc - m) / sd) - ndtr((ac - m) / sd)
                return 0.5 * (S2 - 2 * m * S1 + n * m * m) / sd ** 2 + n * ls + n * math.log(max(Z, 1e-300))
            r = optimize.minimize(nll, [mu - c, math.log(s)], method="Nelder-Mead",
                                  options=dict(xatol=1e-9, fatol=1e-9, maxiter=4000))
            m2, s2 = r.x[0] + c, math.exp(r.x[1])
        else:
            sd = fix_s
            def nll1(p):
                m = p[0]
                Z = ndtr((bc - m) / sd) - ndtr((ac - m) / sd)
                return 0.5 * (S2 - 2 * m * S1 + n * m * m) / sd ** 2 + n * math.log(max(Z, 1e-300))
            r = optimize.minimize(nll1, [mu - c], method="Nelder-Mead",
                                  options=dict(xatol=1e-10, fatol=1e-10, maxiter=4000))
            m2, s2 = r.x[0] + c, sd
        conv = abs(m2 - mu) < 1e-6 * s and abs(s2 - s) < 1e-6 * s
        mu, s = float(m2), float(s2)
        if conv:
            break
    return mu, s


log("loading")
DAT = {}
for sp in ("train", "test"):
    DAT[sp] = dict(
        F1=np.load(f"{D}/feat1us_{sp}.npy"),
        Fb=np.load(f"{D}/featbox_{sp}.npy"),
        F=np.load(f"{D}/feat_{sp}.npy"),
        y=np.load(f"{D}/y_{sp}.npy").astype(np.int64))
    DAT[sp]["bits"] = ((DAT[sp]["y"][:, None] >> np.arange(5)) & 1).astype(np.int8)
log("loaded")

WINDOWS = {
    "fir1us": ("feat1us bins 0-9 = raw 0-499 (paper 1 us, CNN input)", lambda d, q: d["F1"][:, q].mean(1)),
    "box1us": ("featbox bins 0-9 = raw 0-499 boxcar", lambda d, q: d["Fb"][:, q, 0:10].mean(1)),
    "fir400": ("feat bins 0-3 = raw 0-199 (paper 400 ns)", lambda d, q: d["F"][:, q, 0:4].mean(1)),
    "fir2us": ("feat bins 0-19 = raw 0-999", lambda d, q: d["F"][:, q].mean(1)),
    "box_0_500": ("featbox bins 0-4 = 0-500 ns", lambda d, q: d["Fb"][:, q, 0:5].mean(1)),
    "box_500_1000": ("featbox bins 5-9 = 500-1000 ns", lambda d, q: d["Fb"][:, q, 5:10].mean(1)),
    "box_1000_1500": ("featbox bins 10-14", lambda d, q: d["Fb"][:, q, 10:15].mean(1)),
    "box_1500_2000": ("featbox bins 15-19", lambda d, q: d["Fb"][:, q, 15:20].mean(1)),
}
QUBITS = [0, 1, 2, 3, 4]


def frame(Wtr, ytr, q, hi_k=1.0):
    """Per prepared 5-bit state: centre (sigma fixed to the |0> width of the same neighbours),
    own free width, and sigma_c of the |0> state.  Fitted on the given split."""
    P = Wtr[:, 1]
    fr = {}
    for s in range(32):
        if (s >> q) & 1:
            continue
        s1 = s | (1 << q)
        p0 = P[ytr == s]
        med = float(np.median(p0)); mad = float(1.4826 * np.median(np.abs(p0 - med)))
        mu0, sg0 = tnfit(p0, med, mad, hi_k=hi_k)
        p1 = P[ytr == s1]
        med1 = float(np.median(p1))
        mu1_fix, _ = tnfit(p1, med1, sg0, hi_k=hi_k, fix_s=sg0)
        mu1_free, sg1 = tnfit(p1, mu1_fix, sg0, hi_k=hi_k)
        fr[s] = dict(mu=mu0, sig=sg0, sig_own=sg0, mad=mad)
        fr[s1] = dict(mu=mu1_fix, sig=sg0, sig_own=sg1, mu_free=mu1_free, mad=float(1.4826 * np.median(np.abs(p1 - med1))))
    return fr


def zscore(W, y, fr):
    z = np.empty(len(y))
    for s, f in fr.items():
        m = y == s
        z[m] = (W[m, 1] - f["mu"]) / f["sig"]
    return z


def tail_table(z, n):
    t = {}
    for k in KS:
        npl, nmi = int((z > k).sum()), int((z < -k).sum())
        t[f"k{k}"] = dict(n_plus=npl, n_minus=nmi, gauss=n * Qtail(k), E=mirror(npl, nmi, n),
                          G=[npl / n - Qtail(k)] + [x - Qtail(k) for x in wilson(npl, n)[1:]],
                          minus_over_gauss=nmi / (n * Qtail(k)))
    return t


res = {"tag": "vstatistics", "definitions": {
    "frame": "TRAIN-fitted per 5-bit-state truncated-normal ML, truncation [mu-4 sigma, mu+1 sigma]; prepared-|1> centre with sigma fixed to the |0> width of the same neighbours",
    "z": "(perp - mu_s[train]) / sigma_c[train]",
    "E_k": "mirror excess [N(z>k)-N(z<-k)]/N, CI +-1.96 sqrt(N+ + N-)/N",
    "G_k": "Gaussian-null one-sided excess N(z>k)/N - Q(k), Wilson CI shifted"}, "windows": {}}

ZSTORE = {}
for wn, (wdesc, fn) in WINDOWS.items():
    rw = {"desc": wdesc}
    for q in QUBITS:
        Wtr = np.asarray(fn(DAT["train"], q), np.float64)
        Wte = np.asarray(fn(DAT["test"], q), np.float64)
        ytr, yte = DAT["train"]["y"], DAT["test"]["y"]
        fr_tr = frame(Wtr, ytr, q)
        fr_te = frame(Wte, yte, q)            # in-sample frame for comparison
        z_oos = zscore(Wte, yte, fr_tr)
        z_ins = zscore(Wte, yte, fr_te)
        z_trn = zscore(Wtr, ytr, fr_tr)
        bte = DAT["test"]["bits"][:, q]
        btr = DAT["train"]["bits"][:, q]
        rq = {}
        for b in (0, 1):
            m = bte == b
            n = int(m.sum())
            d = {"N_test": n,
                 "oos": tail_table(z_oos[m], n),
                 "insample_E3": mirror(int((z_ins[m] > 3).sum()), int((z_ins[m] < -3).sum()), n),
                 "train_insample_E3": mirror(int((z_trn[btr == b] > 3).sum()), int((z_trn[btr == b] < -3).sum()),
                                             int((btr == b).sum()))}
            sts = [s for s in fr_tr if ((s >> q) & 1) == b]
            d["sig_own_over_sig0_median"] = float(np.median([fr_tr[s]["sig_own"] / fr_tr[s]["sig"] for s in sts]))
            d["mad_over_sig0_median"] = float(np.median([fr_tr[s]["mad"] / fr_tr[s]["sig"] for s in sts]))
            d["sig0_median_adc"] = float(np.median([fr_tr[s]["sig"] for s in sts]))
            if b == 1:
                d["mu_free_minus_mu_fix_over_sig0_median"] = float(np.median(
                    [(fr_tr[s]["mu_free"] - fr_tr[s]["mu"]) / fr_tr[s]["sig"] for s in sts]))
            rq[f"prep{b}"] = d
        rw[PAPER[q]] = rq
        if wn in ("fir1us", "box1us"):
            ZSTORE[(wn, q)] = z_oos
        log(f"{wn} {PAPER[q]}: E3 p1 oos {100*rq['prep1']['oos']['k3.0']['E'][0]:.3f}% "
            f"(ins {100*rq['prep1']['insample_E3'][0]:.3f}, train {100*rq['prep1']['train_insample_E3'][0]:.3f}) "
            f"p0 {100*rq['prep0']['oos']['k3.0']['E'][0]:.3f}%  minus/gauss p1 {rq['prep1']['oos']['k3.0']['minus_over_gauss']:.2f} "
            f"sig_own/sig0 p1 {rq['prep1']['sig_own_over_sig0_median']:.3f}")
    res["windows"][wn] = rw

# ---------------------------------------------------------------- within-shot noise scale (boxcar)
log("within-shot noise")
wsn = {}
for q in QUBITS:
    rq = {}
    for sp in ("train",):
        d = DAT[sp]
        X = d["Fb"][:, q, 0:10, 1].astype(np.float64)     # perp per bin
        y = d["y"]
        out = []
        for s in range(32):
            Xs = X[y == s]
            R = Xs - np.median(Xs, 0)
            Dd = np.diff(R, axis=1)
            s_diff = math.sqrt(np.median(Dd[:, 1:] ** 2) / 0.4549364 / 2)   # bins 1..9
            core = (np.abs(R) < 3.5 * s_diff).all(1)
            Rc = R[core]
            C = np.cov(Rc.T)
            rho = [float(np.mean([C[i, i + l] / math.sqrt(C[i, i] * C[i + l, i + l]) for i in range(1, 10 - l)])) for l in (1, 2, 3)]
            # predicted sd of the 10-bin mean from bin sd + measured lag correlations (lags 1-3)
            fac = 1 + 2 * sum((1 - l / 10) * rho[l - 1] for l in (1, 2, 3))
            s_pred = s_diff / math.sqrt(1 - rho[0]) * math.sqrt(fac / 10)
            Wm = Xs.mean(1)
            mu, sg = tnfit(Wm, float(np.median(Wm)), s_pred, hi_k=1.0)
            out.append(dict(state=s, prep=int((s >> q) & 1), s_bin_diff=s_diff, rho123=rho, s_pred_mean=s_pred,
                            s_across_tn=sg, ratio_across_over_within=sg / s_pred))
        rq[sp] = out
    r0 = [o["ratio_across_over_within"] for o in rq["train"] if o["prep"] == 0]
    r1 = [o["ratio_across_over_within"] for o in rq["train"] if o["prep"] == 1]
    rq["summary"] = dict(ratio_prep0_median=float(np.median(r0)), ratio_prep0_range=[float(min(r0)), float(max(r0))],
                         ratio_prep1_median=float(np.median(r1)), ratio_prep1_range=[float(min(r1)), float(max(r1))],
                         rho1_median=float(np.median([o["rho123"][0] for o in rq["train"]])))
    wsn[PAPER[q]] = rq
    log(f"within {PAPER[q]}: across/within p0 {rq['summary']['ratio_prep0_median']:.3f} p1 {rq['summary']['ratio_prep1_median']:.3f} rho1 {rq['summary']['rho1_median']:.3f}")
res["within_shot_noise_box1us"] = wsn

# Gaussian-null excess with the WITHIN-shot sigma (box1us window, TEST, train-fitted centres)
log("G with within-shot sigma")
gw = {}
for q in QUBITS:
    d = DAT["test"]; y = d["y"]
    W = d["Fb"][:, q, 0:10].mean(1).astype(np.float64)
    Wtr = DAT["train"]["Fb"][:, q, 0:10].mean(1).astype(np.float64)
    fr = frame(Wtr, DAT["train"]["y"], q)
    smap = {o["state"]: o["s_pred_mean"] for o in wsn[PAPER[q]]["train"]}
    z = np.empty(len(y))
    for s in range(32):
        s0 = s & ~(1 << q)
        m = y == s
        z[m] = (W[m, 1] - fr[s]["mu"]) / smap[s0]
    rq = {}
    for b in (0, 1):
        m = d["bits"][:, q] == b
        rq[f"prep{b}"] = tail_table(z[m], int(m.sum()))
    gw[PAPER[q]] = rq
    log(f"Gwithin {PAPER[q]}: p1 G3 {100*rq['prep1']['k3.0']['G'][0]:.3f}% E3 {100*rq['prep1']['k3.0']['E'][0]:.3f}% minus/gauss {rq['prep1']['k3.0']['minus_over_gauss']:.2f}; "
        f"p0 G3 {100*rq['prep0']['k3.0']['G'][0]:.3f}% minus/gauss {rq['prep0']['k3.0']['minus_over_gauss']:.2f}")
res["gauss_null_within_sigma_box1us"] = gw

# ---------------------------------------------------------------- CI robustness (fir1us, TEST)
log("CI robustness")
rng = np.random.default_rng(12345)
cir = {}
yte = DAT["test"]["y"]
for q in QUBITS:
    z = ZSTORE[("fir1us", q)]
    rq = {}
    for b in (0, 1):
        sts = [s for s in range(32) if ((s >> q) & 1) == b]
        npl_c, nmi_c, n_c = [], [], []
        chunk_pl, chunk_mi = [], []
        for s in sts:
            idx = np.where(yte == s)[0]          # file order = acquisition order within block (assumed)
            zs = z[idx]
            npl_c.append(int((zs > 3).sum())); nmi_c.append(int((zs < -3).sum())); n_c.append(len(zs))
            nch = len(zs) // 500
            cp = (zs[:nch * 500].reshape(nch, 500) > 3).sum(1)
            cm = (zs[:nch * 500].reshape(nch, 500) < -3).sum(1)
            chunk_pl.append(cp); chunk_mi.append(cm)
        npl_c, nmi_c, n_c = map(np.array, (npl_c, nmi_c, n_c))
        Ec = (npl_c - nmi_c) / n_c
        N = n_c.sum()
        E = float((npl_c - nmi_c).sum() / N)
        # chunk bootstrap stratified by state
        B = 4000
        boots = np.empty(B)
        for i in range(B):
            tot = 0
            for cp, cm in zip(chunk_pl, chunk_mi):
                j = rng.integers(0, len(cp), len(cp))
                tot += (cp[j] - cm[j]).sum()
            boots[i] = tot / (500 * sum(len(c) for c in chunk_pl))
        # dispersion index of the plus-count per chunk (pooled within-state)
        disp = []
        for cp in chunk_pl:
            if cp.mean() > 0:
                disp.append(cp.var(ddof=1) / (cp.mean() * (1 - cp.mean() / 500)))
        # random effects over the 16 configurations (DerSimonian-Laird)
        v = (npl_c + nmi_c) / n_c ** 2
        w = 1 / v
        Efix = (w * Ec).sum() / w.sum()
        Qh = (w * (Ec - Efix) ** 2).sum()
        tau2 = max(0.0, (Qh - (len(Ec) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
        wr = 1 / (v + tau2)
        Ere = (wr * Ec).sum() / wr.sum()
        se_re = math.sqrt(1 / wr.sum())
        rq[f"prep{b}"] = dict(E3=mirror(int(npl_c.sum()), int(nmi_c.sum()), int(N)),
                              chunk_boot_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                              chunk_dispersion_index_median=float(np.median(disp)) if disp else None,
                              per_config_E3=[float(x) for x in Ec], config_range=[float(Ec.min()), float(Ec.max())],
                              cochran_Q=float(Qh), df=len(Ec) - 1, tau=float(math.sqrt(tau2)),
                              random_effects=[float(Ere), float(Ere - Z95 * se_re), float(Ere + Z95 * se_re)],
                              unweighted_mean_sd_over_sqrt16=[float(Ec.mean()), float(Ec.std(ddof=1) / 4)])
        log(f"CI {PAPER[q]} p{b}: E3 {100*E:.3f}% boot [{100*rq[f'prep{b}']['chunk_boot_ci'][0]:.3f},{100*rq[f'prep{b}']['chunk_boot_ci'][1]:.3f}] "
            f"RE [{100*rq[f'prep{b}']['random_effects'][1]:.3f},{100*rq[f'prep{b}']['random_effects'][2]:.3f}] tau {100*rq[f'prep{b}']['tau']:.3f}pp disp {rq[f'prep{b}']['chunk_dispersion_index_median']}")
    cir[PAPER[q]] = rq
res["ci_robustness_fir1us"] = cir

# Q5 and Q1 by (Q3, Q4) prepared bits; Q3 by (Q1,Q5)
cond = {}
bte = DAT["test"]["bits"]
for q in (0, 1, 2, 4):
    z = ZSTORE[("fir1us", q)]
    cq = {}
    pairs = [(2, 3)] if q != 2 else [(0, 4)]
    for (a, c) in pairs:
        for ba in (0, 1):
            for bc in (0, 1):
                m = (bte[:, q] == 1) & (bte[:, a] == ba) & (bte[:, c] == bc)
                cq[f"prep1_{PAPER[a]}={ba}_{PAPER[c]}={bc}"] = mirror(int((z[m] > 3).sum()), int((z[m] < -3).sum()), int(m.sum()))
    cond[PAPER[q]] = cq
res["conditional_E3_fir1us"] = cond

# ---------------------------------------------------------------- CNN cross-tab
log("CNN")
pb = np.load(str(LP.REPORTS / "preds_MCMitCNN_pb.npy")).astype(np.int8)
tb = np.load(str(LP.REPORTS / "preds_MCMitCNN_tb.npy")).astype(np.int8)
cnn = {"tb_equals_prepared_bits": bool(np.array_equal(tb, bte))}
err = pb != tb
tot_err, tot_leak = 0, 0.0
tot_pl, tot_mi = 0, 0
for q in (0, 1, 2, 3, 4):
    z = ZSTORE[("fir1us", q)]
    e = err[:, q]
    ne = int(e.sum())
    zp, zm = int((e & (z > 3)).sum()), int((e & (z < -3)).sum())
    core = e & (np.abs(z) < 2.5)
    delta = float(np.median(z[core]))
    # mirror among errors, bias from an off-centre error subpopulation: n_err*(Q(3-d)-Q(3+d))
    bias = ne * (Qtail(3 - delta) - Qtail(3 + delta)) if delta > 0 else -ne * (Qtail(3 + delta) - Qtail(3 - delta))
    # bias-corrected: recentre the error subpopulation
    zc = z[e] - delta
    zpc, zmc = int((zc > 3).sum()), int((zc < -3).sum())
    d = {"n_err": ne, "err_rate": wilson(ne, len(z)), "share_mirror3": mirror(zp, zm, ne),
         "err_core_median_z": delta, "mirror_bias_shots": float(bias),
         "share_mirror3_recentred": mirror(zpc, zmc, ne)}
    for b in (0, 1):
        m = bte[:, q] == b
        ebm = e & m
        nb = int(ebm.sum())
        d[f"prep{b}_share_mirror3"] = mirror(int((ebm & (z > 3)).sum()), int((ebm & (z < -3)).sum()), max(nb, 1))
        lp, lm = int((m & (z > 3)).sum()), int((m & (z < -3)).sum())
        ep, em = int((m & e & (z > 3)).sum()), int((m & e & (z < -3)).sum())
        d[f"prep{b}_misread_on_leaked_mirror"] = (ep - em) / max(lp - lm, 1)
        d[f"prep{b}_misread_on_zgt3_raw"] = ep / max(lp, 1)
    cnn[PAPER[q]] = d
    if q != 3:
        tot_err += ne; tot_pl += zp; tot_mi += zm
    log(f"CNN {PAPER[q]}: err {100*ne/len(z):.2f}% share {100*d['share_mirror3'][0]:.3f}% recentred {100*d['share_mirror3_recentred'][0]:.3f}% "
        f"delta {delta:+.3f} misread leaked p1 {100*d['prep1_misread_on_leaked_mirror']:.2f}% p0 {100*d['prep0_misread_on_leaked_mirror']:.2f}%")
cnn["pooled_Q1Q2Q3Q5"] = dict(n_err=tot_err, leaked_err_mirror=tot_pl - tot_mi, share=mirror(tot_pl, tot_mi, tot_err))
res["cnn_crosstab_fir1us"] = cnn

# ---------------------------------------------------------------- train/test acquisition overlap
log("train/test overlap")
ov = {}
for q in QUBITS:
    cs = []
    for s in range(32):
        if (s >> q) & 1:
            continue
        pats = []
        for sp in ("train", "test"):
            d = DAT[sp]
            idx = np.where(d["y"] == s)[0]
            A = d["F1"][idx, q].mean(1)[:, 0].astype(np.float64)
            sl = np.array_split(A, 20)
            med = np.array([np.median(x) for x in sl])
            pats.append(med - med.mean())
        cs.append(float(np.corrcoef(pats[0], pats[1])[0, 1]))
    ov[PAPER[q]] = dict(corr_train_test_drift_pattern_per_state=cs, median=float(np.median(cs)))
    log(f"overlap {PAPER[q]}: median corr {np.median(cs):.3f}")
res["train_test_drift_pattern_corr"] = ov

with open(OUT + "/vstatistics_main.json", "w") as f:
    json.dump(res, f, indent=1)
log("done")
