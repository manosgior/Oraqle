#!/usr/bin/env python3
"""vreproduction follow-up: (A) IF-frequency and centre-estimator sensitivity of E3,
(B) CNN error cross-tab by prepared bit / frame, (C) model-light Q3 L-lifetime check with a
fixed clean per-bin reference frame. Own code only (re-implements the frame of
vreproduction_estimate.py)."""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from scipy.optimize import curve_fit

INP = str(LP.stage('vreproduction') / 'out')
S = {sp: dict(np.load(f'{INP}/vreproduction_sub_{sp}.npz')) for sp in ('train', 'test')}
cal = json.load(open(f'{INP}/vreproduction_calib.json'))
y = np.concatenate([S['train']['y'], S['test']['y']])
split_id = np.concatenate([np.zeros(len(S['train']['y']), int), np.ones(len(S['test']['y']), int)])
N = len(y)
bits = (y[:, None] >> np.arange(5)) & 1
BOX = np.concatenate([S['train']['box'], S['test']['box']])
BW = np.concatenate([S['train']['bw'], S['test']['bw']])
tk = (np.arange(20) * 50 + 24.5) / 500e6          # bin centre times
TC = 2.5
FTR = 1 - 2 * TC * norm.pdf(TC) / (2 * norm.cdf(TC) - 1)


def mode2d(P, s):
    re, im = P.real, P.imag
    lr, hr = np.percentile(re, [0.5, 99.5]); li, hi = np.percentile(im, [0.5, 99.5])
    h = s / 3
    nr = int(np.clip((hr - lr) / h, 10, 400)); ni = int(np.clip((hi - li) / h, 10, 400))
    H, er, ei = np.histogram2d(re, im, bins=[nr, ni], range=[[lr, hr], [li, hi]])
    H = gaussian_filter(H, 1.5)
    i, j = np.unravel_index(H.argmax(), H.shape)
    return complex((er[i] + er[i + 1]) / 2, (ei[j] + ei[j + 1]) / 2)


def centre(P, s, rad=2.0, how='mode'):
    if how == 'mean':
        return P.mean()
    if how == 'median':
        return complex(np.median(P.real), np.median(P.imag))
    m = mode2d(P, s)
    if how == 'modeonly':
        return m
    for _ in range(50):
        sel = np.abs(P - m) < rad * s
        mn = P[sel].mean()
        if abs(mn - m) < 1e-5 * s:
            return mn
        m = mn
    return m


def trunc(v):
    m = np.median(v); s = 1.4826 * np.median(np.abs(v - m))
    for _ in range(50):
        sel = np.abs(v - m) < TC * s
        m2 = v[sel].mean(); s2 = np.sqrt(((v[sel] - m2) ** 2).mean() / FTR)
        if abs(m2 - m) < 1e-6 * s and abs(s2 - s) < 1e-6 * s:
            return m2, s2
        m, s = m2, s2
    return m, s


def zframe(X, q, rad=2.0, how='mode'):
    z = np.full(N, np.nan); a = np.full(N, np.nan)
    nbr = y & ~(1 << q)
    for sp in (0, 1):
        for c in np.unique(nbr):
            g0 = np.flatnonzero((nbr == c) & (bits[:, q] == 0) & (split_id == sp))
            g1 = np.flatnonzero((nbr == c) & (bits[:, q] == 1) & (split_id == sp))
            X0, X1 = X[g0], X[g1]
            s_ini = 0.5 * 1.4826 * (np.median(np.abs(X0.real - np.median(X0.real))) +
                                    np.median(np.abs(X0.imag - np.median(X0.imag))))
            c0 = centre(X0, s_ini, rad, how); c1 = centre(X1, s_ini, rad, how)
            u = (c1 - c0) / abs(c1 - c0)
            g = np.concatenate([g0, g1])
            w = (X[g] - c0) * np.conj(u)
            _, s0 = trunc(w.imag[:len(g0)])
            z[g] = w.imag / s0; a[g] = w.real / abs(c1 - c0)
    return z, a


def est(zp, zm, n):
    e = (zp - zm) / n; h = 1.96 * np.sqrt(zp + zm) / n
    return [e, e - h, e + h, int(zp), int(zm), int(n)]


def E3(z, sel, k=3):
    return est(np.sum(z[sel] > k), np.sum(z[sel] < -k), sel.sum())


R = {}
f_full = np.array(cal['f_used'])
FV = {'full': f_full,
      'late': np.array([c['f_peak']['late'] for c in cal['calib']]),
      '1us': np.array([c['f_peak']['1us'] for c in cal['calib']]),
      'herqules': np.array(cal['herqules_freq'])}
# ---------------- (A) frequency + centre-estimator sensitivity ----------------
A = {}
zstore = {}
for fname, fv in FV.items():
    for q in range(5):
        ph = np.exp(-2j * np.pi * (fv[q] - f_full[q]) * tk)       # per-bin re-mixing
        Xb = BOX[:, q, :] * ph
        for wname, (b0, b1) in (('400ns', (0, 4)), ('1us', (0, 10)), ('2us', (0, 20))):
            X = Xb[:, b0:b1].mean(1)
            hows = [('mode', 2.0)] if not (fname == 'full' and wname == '1us') else \
                [('mode', 2.0), ('mode', 1.5), ('mode', 3.0), ('modeonly', 0), ('median', 0), ('mean', 0)]
            for how, rad in hows:
                z, a = zframe(X, q, rad, how)
                if fname == 'full' and wname == '1us' and how == 'mode' and rad == 2.0:
                    zstore[q] = (z, a)
                if fname == 'late' and wname == '1us':
                    zstore[('late', q)] = (z, a)
                for b in (0, 1):
                    for spn, spm in (('test', split_id == 1), ('pool', np.ones(N, bool))):
                        A[f'{fname}|{wname}|{how}{rad}|Q{q+1}|p{b}|{spn}'] = E3(z, (bits[:, q] == b) & spm)
    print('A', fname, flush=True)
R['A'] = A

# ---------------- (B) CNN cross-tab ----------------
pb = np.load(str(LP.REPORTS / 'preds_MCMitCNN_pb.npy')); tb = np.load(str(LP.REPORTS / 'preds_MCMitCNN_tb.npy'))
te = split_id == 1
idx_te = S['test']['idx']
P = pb[idx_te].astype(int); T_ = tb[idx_te].astype(int)
assert np.array_equal(T_, bits[te])
Bd = {}
for lab, src in (('box_full', lambda q: zstore[q]), ('box_late', lambda q: zstore[('late', q)])):
    tot = [0, 0, 0]
    for q in range(5):
        z, a = src(q)
        zq = z[te]; aq = a[te]; bq = bits[te, q]; pq = P[:, q]; err = pq != bq
        for b in (0, 1):
            s = bq == b
            ne = int(np.sum(err & s))
            ep = int(np.sum(err & s & (zq > 3))); em = int(np.sum(err & s & (zq < -3)))
            Bd[f'{lab}|Q{q+1}|p{b}|errors'] = ne
            Bd[f'{lab}|Q{q+1}|p{b}|leaked_share_of_errors'] = est(ep, em, max(ne, 1))
            # where are the flagged errors along the axis
            for lo, hi in ((-9, 0.5), (0.5, 1.2), (1.2, 99)):
                m = err & s & (aq >= lo) & (aq < hi)
                Bd[f'{lab}|Q{q+1}|p{b}|flagged errors a in [{lo},{hi})'] = [int(np.sum(m & (zq > 3))), int(np.sum(m & (zq < -3)))]
        ne = int(err.sum()); ep = int(np.sum(err & (zq > 3))); em = int(np.sum(err & (zq < -3)))
        Bd[f'{lab}|Q{q+1}|all|leaked_share_of_errors'] = est(ep, em, ne)
        if q != 3:
            tot[0] += ep; tot[1] += em; tot[2] += ne
    Bd[f'{lab}|pooled Q1Q2Q3Q5'] = est(*tot)
R['B'] = Bd
print('B done', flush=True)

# ---------------- (C) Q3 L-lifetime, fixed clean per-bin reference ----------------
C = {}
for q in (2, 4):
    z1, a1 = zstore[q]
    Xw2 = BOX[:, q, :].mean(1)
    z2, a2 = zframe(Xw2, q)
    clean0 = (bits[:, q] == 0) & (np.abs(z2) < 2) & (np.abs(a2) < 0.3)
    clean1_early = (bits[:, q] == 1) & (np.abs(z1) < 2) & (np.abs(a1 - 1) < 0.3)
    clean1_late = (bits[:, q] == 1) & (np.abs(z2) < 2) & (np.abs(a2 - 1) < 0.3)
    Ak = np.zeros((N, 20)); Vk = np.zeros((N, 20)); info = []
    for k in range(20):
        Xk = BOX[:, q, k]
        c0 = Xk[clean0].mean()
        c1 = Xk[clean1_late if k >= 10 else clean1_early].mean()
        u = (c1 - c0) / abs(c1 - c0)
        w = (Xk - c0) * np.conj(u)
        _, s0 = trunc(w.imag[clean0])
        Ak[:, k] = w.real / abs(c1 - c0); Vk[:, k] = w.imag / s0
        info.append([float(np.degrees(np.angle(c1 - c0))), float(abs(c1 - c0) / s0)])
    C[f'Q{q+1}|bin_angle_sep'] = info
    v3 = Vk[:, 2:5].mean(1) * np.sqrt(3); a3 = Ak[:, 2:5].mean(1)
    for b in (1, 0):
        rows = bits[:, q] == b
        for thr in (4, 5):
            grp = {'L': rows & (v3 > thr) & (a3 > 1.2), 'mirror': rows & (v3 < -thr) & (a3 > 1.2)}
            res = {}
            for lab, m in grp.items():
                res[lab] = {'n': int(m.sum()), 'a': Ak[m].mean(0).tolist() if m.sum() else None,
                            'vz': Vk[m].mean(0).tolist() if m.sum() else None}
            if res['L']['n'] > 50:
                vz = np.array(res['L']['vz']); aa = np.array(res['L']['a'])
                t = (np.arange(5, 20) - 5) * 0.1
                try:
                    popt, pcov = curve_fit(lambda t, A_, tau, B_: A_ * np.exp(-t / tau) + B_, t, vz[5:20],
                                           p0=[vz[5], 1.0, 0.5], maxfev=20000)
                    res['fit_vz_exp_plus_const'] = {'A': popt[0], 'tau_us': popt[1], 'B': popt[2],
                                                    'tau_se': float(np.sqrt(pcov[1, 1]))}
                except Exception as e:
                    res['fit_vz_exp_plus_const'] = repr(e)
                try:
                    popt, pcov = curve_fit(lambda t, A_, tau: A_ * np.exp(-t / tau), t, vz[5:20], p0=[vz[5], 1.0])
                    res['fit_vz_exp_to_zero'] = {'A': popt[0], 'tau_us': popt[1], 'tau_se': float(np.sqrt(pcov[1, 1]))}
                except Exception as e:
                    res['fit_vz_exp_to_zero'] = repr(e)
                # destination: late along position vs perp fraction; reference L location from bins 3-4
                vL = vz[3:5].mean(); aL = aa[3:5].mean()
                PL = vz / vL
                P1 = aa - PL * aL                    # a = PL*aL + P1*1 + P0*0
                res['decomp'] = {'PL': PL.tolist(), 'P1': P1.tolist(), 'P0': (1 - PL - P1).tolist(),
                                 'aL': aL, 'vL': vL}
            C[f'Q{q+1}|p{b}|thr{thr}'] = res
    # |1> reference: in-|1> at bins 2-4 -> mean along decay
    m1 = (bits[:, q] == 1) & (np.abs(v3) < 1) & (np.abs(a3 - 1) < 0.25)
    C[f'Q{q+1}|in1'] = {'n': int(m1.sum()), 'a': Ak[m1].mean(0).tolist(), 'vz': Vk[m1].mean(0).tolist(),
                         'frac_a_lt_0.5': (Ak[m1] < 0.5).mean(0).tolist()}
R['C'] = C
json.dump(R, open(f'{INP}/vreproduction_followup.json', 'w'), indent=0, default=float)
print('done', flush=True)
