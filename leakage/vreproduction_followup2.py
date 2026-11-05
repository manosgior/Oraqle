#!/usr/bin/env python3
"""vreproduction follow-up 2:
 (D) diagnose the CNN 'leaked share of errors' frame dependence: an 'R-like' frame (one rotation per
     qubit from the MEAN centroid difference, per-state perp centre = truncated median of perp near the
     along mode, sigma0 = MAD of prepared-0 perp per configuration) versus my mode-line frame;
     mean z of prepared-1 CNN errors that sit near |0> (a < 0.5) must be ~0 in an unbiased frame.
 (E) Q3 L destination by nearest-reference classification of segment means (S2 = bins 5-9,
     S3 = bins 12-19) for shots selected as L in bins 2-4.
Own code only."""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

INP = str(LP.stage('vreproduction') / 'out')
S = {sp: dict(np.load(f'{INP}/vreproduction_sub_{sp}.npz')) for sp in ('train', 'test')}
PS = np.load(f'{INP}/vreproduction_pershot.npz')
y = np.concatenate([S['train']['y'], S['test']['y']])
split_id = np.concatenate([np.zeros(len(S['train']['y']), int), np.ones(len(S['test']['y']), int)])
N = len(y)
bits = (y[:, None] >> np.arange(5)) & 1
BOX = np.concatenate([S['train']['box'], S['test']['box']])
te = split_id == 1
pb = np.load(str(LP.REPORTS / 'preds_MCMitCNN_pb.npy')); tb = np.load(str(LP.REPORTS / 'preds_MCMitCNN_tb.npy'))
P = pb[S['test']['idx']].astype(int)
assert np.array_equal(tb[S['test']['idx']].astype(int), bits[te])


def est(zp, zm, n):
    e = (zp - zm) / n; h = 1.96 * np.sqrt(zp + zm) / n
    return [e, e - h, e + h, int(zp), int(zm), int(n)]


def amode(a, s):
    lo, hi = np.percentile(a, [0.5, 99.5]); nb = int(np.clip((hi - lo) / (s / 4), 20, 2000))
    H, e = np.histogram(a, nb, (lo, hi)); H = gaussian_filter1d(H.astype(float), 2)
    i = H.argmax(); return 0.5 * (e[i] + e[i + 1])


R = {}
D = {}
for q in (0, 1, 2, 4):
    z_m = PS[f'z_1us_q{q}']; a_m = PS[f'a_1us_q{q}']
    X = BOX[:, q, 0:10].mean(1)
    zR = np.full(N, np.nan); aR = np.full(N, np.nan)
    nbr = y & ~(1 << q)
    for sp in (0, 1):
        msp = split_id == sp
        d = X[msp & (bits[:, q] == 1)].mean() - X[msp & (bits[:, q] == 0)].mean()
        rot = np.conj(d) / abs(d)
        W = X * rot
        for c in np.unique(nbr):
            m0 = msp & (nbr == c) & (bits[:, q] == 0)
            p0 = W[m0].imag
            s0 = 1.4826 * np.median(np.abs(p0 - np.median(p0)))
            c0r = np.median(W[m0].real)
            for b in (0, 1):
                m = msp & (nbr == c) & (bits[:, q] == b)
                aa, pp = W[m].real, W[m].imag
                am = amode(aa, s0); near = np.abs(aa - am) < s0
                mu = np.median(pp[near])
                for _ in range(5):
                    mu = np.median(pp[near & (np.abs(pp - mu) < 2.5 * s0)])
                zR[m] = (pp - mu) / s0
                aR[m] = (aa - c0r) / abs(d)
    dd = {}
    for lab, z, a in (('mine', z_m, a_m), ('Rlike', zR, aR)):
        zt, at, bq, pq = z[te], a[te], bits[te, q], P[:, q]
        err = pq != bq
        r = {}
        for b in (0, 1):
            s = bq == b
            r[f'p{b} E3'] = est(np.sum(s & (zt > 3)), np.sum(s & (zt < -3)), s.sum())
            ne = int(np.sum(err & s))
            r[f'p{b} leaked share of errors'] = est(np.sum(err & s & (zt > 3)), np.sum(err & s & (zt < -3)), max(ne, 1))
            e_near0 = err & s & (at < 0.5)
            r[f'p{b} CNN errors with a<0.5: n, mean z, median z'] = [int(e_near0.sum()), float(np.mean(zt[e_near0])), float(np.median(zt[e_near0]))]
            near0 = s & (at < 0.5)
            r[f'p{b} all shots with a<0.5: n, mean z'] = [int(near0.sum()), float(np.mean(zt[near0]))]
            for lo, hi in ((-9, 0.5), (0.5, 1.2), (1.2, 99)):
                mm = err & s & (at >= lo) & (at < hi)
                r[f'p{b} flagged errors a in [{lo},{hi}) +/-'] = [int(np.sum(mm & (zt > 3))), int(np.sum(mm & (zt < -3)))]
        r['all leaked share of errors'] = est(np.sum(err & (zt > 3)), np.sum(err & (zt < -3)), err.sum())
        r['p1 leaked read as 1 (mirror)'] = float((np.sum((bq == 1) & (zt > 3) & (pq == 1)) - np.sum((bq == 1) & (zt < -3) & (pq == 1))) /
                                                  max(np.sum((bq == 1) & (zt > 3)) - np.sum((bq == 1) & (zt < -3)), 1))
        dd[lab] = r
    D[f'Q{q+1}'] = dd
    print('D', q, flush=True)
R['D'] = D

# ---------------- (E) Q3 L destination ----------------
E = {}
q = 2
z1 = PS['z_1us_q2']; a1 = PS['a_1us_q2']; z2 = PS['z_2us_q2']; a2 = PS['a_2us_q2']
seg = {'S1': (2, 5), 'S2': (5, 10), 'S3': (12, 20)}
SM = {k: BOX[:, q, b0:b1].mean(1) for k, (b0, b1) in seg.items()}
clean0 = (bits[:, q] == 0) & (np.abs(z2) < 2) & (np.abs(a2) < 0.3)
clean1 = (bits[:, q] == 1) & (np.abs(z2) < 2) & (np.abs(a2 - 1) < 0.3)
deepL = (z2 > 8)
refs = {}
for k, Xs in SM.items():
    c0 = Xs[clean0].mean(); c1 = Xs[clean1].mean()
    cL = np.median(Xs[deepL].real) + 1j * np.median(Xs[deepL].imag)
    u = (c1 - c0) / abs(c1 - c0)
    sig = 1.4826 * np.median(np.abs(((Xs[clean0] - c0) * np.conj(u)).imag))
    refs[k] = (c0, c1, cL, sig)
    E[f'ref {k}: |c1-c0|/sig, |cL-c1|/sig, |cL-c0|/sig, n_deepL'] = [abs(c1 - c0) / sig, abs(cL - c1) / sig, abs(cL - c0) / sig, int(deepL.sum())]


def classify(Xs, k):
    c0, c1, cL, sig = refs[k]
    dist = np.stack([np.abs(Xs - c0), np.abs(Xs - c1), np.abs(Xs - cL)], 1) / sig
    lab = dist.argmin(1)
    srt = np.sort(dist, 1)
    conf = (srt[:, 1] - srt[:, 0]) > 1.0
    return lab, conf


labs = {k: classify(SM[k], k) for k in SM}
for b in (1, 0):
    rows = bits[:, q] == b
    for selname, sel in (('L in S1', rows & (labs['S1'][0] == 2) & labs['S1'][1]),
                         ('1 in S1', rows & (labs['S1'][0] == 1) & labs['S1'][1]),
                         ('0 in S1', rows & (labs['S1'][0] == 0) & labs['S1'][1]),
                         ('L in S2', rows & (labs['S2'][0] == 2) & labs['S2'][1])):
        out = {'n': int(sel.sum())}
        for k in ('S2', 'S3'):
            lab, conf = labs[k]
            m = sel & conf
            if m.sum():
                out[k] = {'n_conf': int(m.sum()), 'frac0': float(np.mean(lab[m] == 0)), 'frac1': float(np.mean(lab[m] == 1)),
                          'fracL': float(np.mean(lab[m] == 2))}
        E[f'p{b} | {selname}'] = out
# L-group mean position per segment relative to refs (in sigma units)
for k in SM:
    c0, c1, cL, sig = refs[k]
    for lab_name, sel in (('L in S1', (labs['S1'][0] == 2) & labs['S1'][1] & (bits[:, q] == 1)),):
        m = SM[k][sel].mean()
        E[f'{lab_name} mean in {k}: dist to c0, c1, cL (sigma)'] = [abs(m - c0) / sig, abs(m - c1) / sig, abs(m - cL) / sig]
R['E'] = E
json.dump(R, open(f'{INP}/vreproduction_followup2.json', 'w'), indent=0, default=float)
print('done', flush=True)
