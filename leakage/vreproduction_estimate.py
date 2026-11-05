#!/usr/bin/env python3
"""vreproduction: independent leakage estimators on the independently demodulated subsample.

Frame (own): for qubit q, window w, demod frame fr, split s and each of the 16 neighbour
configurations c: c0 / c1 = 2-D density mode refined by an iterated 2-sigma-circle trimmed
mean of the prepared-0 / prepared-1 window means; |0>-|1> line through c0 and c1;
v = signed distance from that line (+ = 90 deg CCW of c0->c1 in the I+iQ plane),
a = along position in separations (0 at c0, 1 at c1); sigma0 = truncated-Gaussian width of v
for the prepared-0 shots of the same configuration.

Statistics:
  E_k  = [N(z>k) - N(z<-k)] / N,  z = v / sigma0   (the headline statistic, re-implemented)
  variants: 'pc'  per-state perp centre (truncated mean) subtracted
            'own' own-state width (prepared-1 width for prepared-1 shots)
            'glob' one rotation per qubit (median config angle) + per-state perp centre
            'r15' trimmed-mean radius 1.5 sigma instead of 2 for the centres
  D_k  = segment-distance mirror excess: d = distance (in sigma0) to the c0-c1 segment, signed
         by sign(v):  [N(d>k, v>0) - N(d>k, v<0)] / N   (counts along-axis displacement too)
  B_k  = E_k restricted to a > 0.5 (beyond the mid-point)
CI: +-1.96 sqrt(N+ + N-) / N (as in the headline). Also per-split (train/test) values.
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json, argparse
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

p = argparse.ArgumentParser()
p.add_argument('--inp', default=str(LP.stage('vreproduction') / 'out'))
p.add_argument('--reports', default=str(LP.REPORTS))
a_ = p.parse_args()

S = {}
for split in ('train', 'test'):
    d = np.load(f'{a_.inp}/vreproduction_sub_{split}.npz')
    S[split] = {k: d[k] for k in d.files}
y = np.concatenate([S['train']['y'], S['test']['y']])
split_id = np.concatenate([np.zeros(len(S['train']['y']), int), np.ones(len(S['test']['y']), int)])
FR = {fr: np.concatenate([S['train'][fr], S['test'][fr]]) for fr in ('box', 'bw')}
N = len(y)
bits = (y[:, None] >> np.arange(5)) & 1
WIN = {'400ns': (0, 4), '1us': (0, 10), '2us': (0, 20)}
SLIDE = [(k, k + 4) for k in range(0, 17, 2)]
KS = [1, 2, 3, 4, 5, 6, 8]
TRUNC_C = 2.5
FTR = 1 - 2 * TRUNC_C * norm.pdf(TRUNC_C) / (2 * norm.cdf(TRUNC_C) - 1)


def mode2d(P, s):
    re, im = P.real, P.imag
    lr, hr = np.percentile(re, [0.5, 99.5]); li, hi = np.percentile(im, [0.5, 99.5])
    h = s / 3
    nr = int(np.clip((hr - lr) / h, 10, 400)); ni = int(np.clip((hi - li) / h, 10, 400))
    H, er, ei = np.histogram2d(re, im, bins=[nr, ni], range=[[lr, hr], [li, hi]])
    H = gaussian_filter(H, 1.5)
    i, j = np.unravel_index(H.argmax(), H.shape)
    return complex((er[i] + er[i + 1]) / 2, (ei[j] + ei[j + 1]) / 2)


def centre(P, s, rad=2.0):
    m = mode2d(P, s)
    for _ in range(50):
        sel = np.abs(P - m) < rad * s
        mn = P[sel].mean()
        if abs(mn - m) < 1e-5 * s:
            m = mn
            break
        m = mn
    return m


def trunc(v):
    m = np.median(v); s = 1.4826 * np.median(np.abs(v - m))
    for _ in range(50):
        sel = np.abs(v - m) < TRUNC_C * s
        m2 = v[sel].mean(); s2 = np.sqrt(((v[sel] - m2) ** 2).mean() / FTR)
        if abs(m2 - m) < 1e-6 * s and abs(s2 - s) < 1e-6 * s:
            m, s = m2, s2
            break
        m, s = m2, s2
    return m, s


def frame(X, q, sp, rad=2.0):
    """X complex window means (N,), returns dict of per-shot arrays restricted to split sp."""
    out = {k: np.full(N, np.nan) for k in ('v', 'a', 's0', 's1', 'm0', 'm1', 'L', 'vg', 'mg')}
    ang = {}
    nbr = (y & ~(1 << q))
    cfgs = np.unique(nbr)
    cache = {}
    for c in cfgs:
        g0 = np.flatnonzero((nbr == c) & (bits[:, q] == 0) & (split_id == sp))
        g1 = np.flatnonzero((nbr == c) & (bits[:, q] == 1) & (split_id == sp))
        X0, X1 = X[g0], X[g1]
        s_ini = 1.4826 * np.median(np.abs(X0.real - np.median(X0.real)))
        s_ini = 0.5 * (s_ini + 1.4826 * np.median(np.abs(X0.imag - np.median(X0.imag))))
        c0 = centre(X0, s_ini, rad)
        # refine sigma from perp of prepared-0 around c0 with a provisional axis
        c1 = centre(X1, s_ini, rad)
        u = (c1 - c0) / abs(c1 - c0)
        g = np.concatenate([g0, g1])
        w = (X[g] - c0) * np.conj(u)
        v = w.imag; aa = w.real / abs(c1 - c0)
        m0, s0 = trunc(v[:len(g0)])
        m1, s1 = trunc(v[len(g0):])
        out['v'][g] = v; out['a'][g] = aa; out['s0'][g] = s0
        out['s1'][g0] = s0; out['s1'][g1] = s1
        out['m0'][g0] = m0; out['m0'][g1] = m1
        out['L'][g] = abs(c1 - c0)
        ang[int(c)] = np.angle(c1 - c0)
        cache[int(c)] = (g0, g1, c0, c1)
    th = float(np.median(list(ang.values())))
    for c, (g0, g1, c0, c1) in cache.items():
        g = np.concatenate([g0, g1])
        vg = ((X[g] - c0) * np.exp(-1j * th)).imag
        mg0, _ = trunc(vg[:len(g0)]); mg1, _ = trunc(vg[len(g0):])
        out['vg'][g] = vg; out['mg'][g0] = mg0; out['mg'][g1] = mg1
    out['angles_deg'] = {k: float(np.degrees(v)) for k, v in ang.items()}
    out['theta_glob'] = th
    return out


def zvariants(F):
    z = F['v'] / F['s0']
    zv = {'base': z,
          'pc': (F['v'] - F['m0']) / F['s0'],
          'own': F['v'] / F['s1'],
          'glob': (F['vg'] - F['mg']) / F['s0']}
    # segment distance (sigma0 units)
    As = F['a'] * F['L'] / F['s0']; Ls = F['L'] / F['s0']
    dx = np.where(As < 0, -As, np.where(As > Ls, As - Ls, 0.0))
    d = np.hypot(dx, z)
    return zv, d


def counts(z, sel, k):
    zp = int(np.sum(z[sel] > k)); zm = int(np.sum(z[sel] < -k)); n = int(sel.sum())
    return zp, zm, n


def est(zp, zm, n):
    e = (zp - zm) / n
    h = 1.96 * np.sqrt(zp + zm) / n
    return [e, e - h, e + h, zp, zm, n]


R = {'meta': {'N': int(N), 'n_train': int((split_id == 0).sum()), 'n_test': int((split_id == 1).sum())}}
store = {}
# CNN predictions for the test rows
try:
    pb = np.load(f'{a_.reports}/preds_MCMitCNN_pb.npy'); tb = np.load(f'{a_.reports}/preds_MCMitCNN_tb.npy')
    idx_te = S['test']['idx']
    pb_te = pb[idx_te].astype(int); tb_te = tb[idx_te].astype(int)
    R['meta']['cnn_tb_matches_bits'] = bool(np.array_equal(tb_te, bits[split_id == 1]))
    R['meta']['cnn_err_rate_sub'] = (pb_te != tb_te).mean(0).tolist()
    R['meta']['cnn_err_rate_full'] = (pb != tb).mean(0).tolist()
    HAVE_CNN = True
except Exception as e:
    R['meta']['cnn_error'] = repr(e); HAVE_CNN = False

for fr in ('box', 'bw'):
    for wname, (b0, b1) in list(WIN.items()) + [(f'sl{b0}-{b1}', (b0, b1)) for b0, b1 in SLIDE]:
        if fr == 'bw' and wname.startswith('sl'):
            continue
        Xall = FR[fr][:, :, b0:b1].mean(2)
        for q in range(5):
            Fs = {}
            for sp in (0, 1):
                Fsp = frame(Xall[:, q], q, sp)
                Fs[sp] = Fsp
            F = {k: np.where(split_id == 0, Fs[0][k], Fs[1][k]) for k in ('v', 'a', 's0', 's1', 'm0', 'm1', 'L', 'vg', 'mg')}
            zv, d = zvariants(F)
            key = f'{fr}|{wname}|q{q}'
            r = {'angles_test': Fs[1]['angles_deg'], 'theta_glob_test': Fs[1]['theta_glob'],
                 'sep_test_median': float(np.median(F['L'][split_id == 1])),
                 's0_test_median': float(np.median(F['s0'][split_id == 1]))}
            variants = ['base'] if wname.startswith('sl') else ['base', 'pc', 'own', 'glob']
            for b in (0, 1):
                for spn, spm in (('test', split_id == 1), ('train', split_id == 0), ('pool', np.ones(N, bool))):
                    sel = (bits[:, q] == b) & spm
                    for var in variants:
                        for k in (KS if var == 'base' else [3]):
                            r[f'E{k}|{var}|p{b}|{spn}'] = est(*counts(zv[var], sel, k))
                    if not wname.startswith('sl'):
                        for k in (2, 3, 4, 5):
                            dp = int(np.sum((d[sel] > k) & (zv['base'][sel] > 0)))
                            dm = int(np.sum((d[sel] > k) & (zv['base'][sel] < 0)))
                            r[f'D{k}|p{b}|{spn}'] = est(dp, dm, int(sel.sum()))
                        for thr in (0.5, 1.0):
                            s2 = sel & (F['a'] > thr)
                            zp = int(np.sum(zv['base'][s2] > 3)); zm = int(np.sum(zv['base'][s2] < -3))
                            r[f'B3a{thr}|p{b}|{spn}'] = est(zp, zm, int(sel.sum()))
            # per-config E3 (test, base) for prepared 1
            if fr == 'box' and wname == '1us':
                nbr = (y & ~(1 << q))
                pc = {}
                for c in np.unique(nbr):
                    for b in (0, 1):
                        sel = (nbr == c) & (bits[:, q] == b) & (split_id == 1)
                        pc[f'{int(c)}|p{b}'] = est(*counts(zv['base'], sel, 3))
                r['percfg_E3_test'] = pc
            if fr == 'box' and wname in WIN:
                store[f'z_{wname}_q{q}'] = zv['base'].astype(np.float32)
                store[f'a_{wname}_q{q}'] = F['a'].astype(np.float32)
                store[f'd_{wname}_q{q}'] = d.astype(np.float32)
            R[key] = r
        print(fr, wname, 'done', flush=True)

# ---------------- conditional analyses (box, 1us, pooled train+test) ----------------
C = {}
z = {q: store[f'z_1us_q{q}'] for q in range(5)}
a = {q: store[f'a_1us_q{q}'] for q in range(5)}
for spn, spm in (('pool', np.ones(N, bool)), ('test', split_id == 1)):
    # Q5 (q4) prepared 1 by Q3 (q2) and Q4 (q3) prepared bits; Q1 (q0) by Q3 bit
    for q, conds in ((4, [(2, 3)]), (0, [(2,)]), (1, [(2,), (3,)])):
        for cond in conds:
            for b in (0, 1):
                for vals in np.ndindex(*(2,) * len(cond)):
                    sel = spm & (bits[:, q] == b)
                    for cq, cv in zip(cond, vals):
                        sel &= bits[:, cq] == cv
                    lab = ','.join(f'Q{cq+1}={cv}' for cq, cv in zip(cond, vals))
                    C[f'{spn}|Q{q+1} p{b}|{lab}'] = est(*counts(z[q], sel, 3))
    # pairing with Q3 found state: Q3 prepared 1 and target prepared 1
    for q in (4, 0, 1):
        base = spm & (bits[:, 2] == 1) & (bits[:, q] == 1)
        for lab, m in (('Q3 found 0-like (a<0.5)', a[2] < 0.5),
                       ('Q3 found 1-like, on-axis (a>=0.5, |z|<3)', (a[2] >= 0.5) & (np.abs(z[2]) < 3)),
                       ('Q3 off-axis (z>3)', z[2] > 3)):
            C[f'{spn}|Q{q+1} p1, Q3 p1|{lab}'] = est(*counts(z[q], base & m, 3))
        # Q3 found-state rates conditional on target off-axis vs not
    # Q3's own off-axis by Q1/Q5 prepared bits
    for b1 in (0, 1):
        for b5 in (0, 1):
            sel = spm & (bits[:, 2] == 1) & (bits[:, 0] == b1) & (bits[:, 4] == b5)
            C[f'{spn}|Q3 p1|Q1={b1},Q5={b5}'] = est(*counts(z[2], sel, 3))
            C[f'{spn}|Q3 p1 found0 frac|Q1={b1},Q5={b5}'] = float(np.mean(a[2][sel] < 0.5))
R['conditional'] = C

# ---------------- CNN cross-tab (test rows, box 1us) ----------------
if HAVE_CNN:
    te = split_id == 1
    CN = {}
    tot_err = 0; tot_leak_p = 0; tot_leak_m = 0
    for q in range(5):
        zq = z[q][te]; bq = bits[te, q]; pq = pb_te[:, q]
        err = pq != bq
        CN[f'Q{q+1} cnn_err_rate'] = float(err.mean())
        for b in (0, 1):
            s = bq == b
            npl = int(np.sum(s & (zq > 3))); nmi = int(np.sum(s & (zq < -3)))
            rp = int(np.sum(s & (zq > 3) & (pq == 1))); rm = int(np.sum(s & (zq < -3) & (pq == 1)))
            CN[f'Q{q+1} p{b} leaked_read_as_1 (mirror)'] = [(rp - rm) / max(npl - nmi, 1), rp, rm, npl, nmi]
        ep = int(np.sum(err & (zq > 3))); em = int(np.sum(err & (zq < -3))); ne = int(err.sum())
        CN[f'Q{q+1} leaked_share_of_errors (mirror)'] = est(ep, em, ne)
        if q != 3:
            tot_err += ne; tot_leak_p += ep; tot_leak_m += em
    CN['pooled Q1,Q2,Q3,Q5 leaked share of errors'] = est(tot_leak_p, tot_leak_m, tot_err)
    R['cnn'] = CN

# ---------------- time-resolved: per-bin global frames (box), Q3 (q2), Q5, Q1, Q2 ----------------
TR = {}
B = FR['box']
for q in (2, 4, 0, 1):
    for b in (1, 0):
        rows = (bits[:, q] == b)
        # per-bin centres from all shots pooled over configs (q-specific bit)
        perbin = []
        for k in range(20):
            Xk = B[:, q, k]
            s_ini = 1.4826 * np.median(np.abs(Xk[bits[:, q] == 0].real - np.median(Xk[bits[:, q] == 0].real)))
            c0 = centre(Xk[bits[:, q] == 0], s_ini); c1 = centre(Xk[bits[:, q] == 1], s_ini)
            u = (c1 - c0) / abs(c1 - c0)
            w = (Xk - c0) * np.conj(u)
            _, s0 = trunc(w.imag[bits[:, q] == 0])
            perbin.append((w.real / abs(c1 - c0), w.imag / s0, float(np.degrees(np.angle(c1 - c0))), s0, abs(c1 - c0)))
        Ak = np.stack([p_[0] for p_ in perbin], 1); Vk = np.stack([p_[1] for p_ in perbin], 1)
        # selection on bins 2-4: mean perp z (3-bin mean has width ~ 1/sqrt(3) if bins independent)
        v3 = Vk[:, 2:5].mean(1) * np.sqrt(3); a3 = Ak[:, 2:5].mean(1)
        out = {'bin_angle_deg': [p_[2] for p_ in perbin], 'bin_sep_over_s0': [p_[4] / p_[3] for p_ in perbin]}
        for lab, m in (('L(+)', rows & (v3 > 4) & (a3 > 1.2)), ('mirror(-)', rows & (v3 < -4) & (a3 > 1.2)),
                       ('in1', rows & (np.abs(v3) < 1) & (np.abs(a3 - 1) < 0.25))):
            out[lab] = {'n': int(m.sum()), 'mean_a': Ak[m].mean(0).tolist(), 'mean_vz': Vk[m].mean(0).tolist()}
        # lag-1 correlation of bin noise (prepared 0, bins 5-15) for the record
        V0 = Vk[(bits[:, q] == 0)][:, 5:16]
        V0 = V0 - V0.mean(0)
        out['lag1_corr_perp'] = float(np.mean([np.corrcoef(V0[:, i], V0[:, i + 1])[0, 1] for i in range(10)]))
        TR[f'Q{q+1}|p{b}'] = out
R['time'] = TR

json.dump(R, open(f'{a_.inp}/vreproduction_results.json', 'w'), indent=0, default=float)
np.savez_compressed(f'{a_.inp}/vreproduction_pershot.npz', y=y, split=split_id, **store)
print('all done', flush=True)
