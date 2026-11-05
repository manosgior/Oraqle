"""vphysics_raw.py -- adversarial physics checks on raw traces (tag vphysics).

A) Class-mean raw traces (complex, 1000 samples) for selected 5-bit prepared states, classes picked from the
   shared features: core0 / core1 (stay put all 2 us), Lstart (third state already in bins 2-4 and still in 8-9),
   Llate (|1> in bins 2-4, third state in 10-19), L0 (prepared-0 shots in the third state in bins 8-19).
   Ring-up (bins 0-1) is never used for selection, so the class-mean ring-up is unbiased.
B) Frequency check: same class means let us measure the phase drift of the pure |0> and |1> responses.
C) E3 (one-sided 3-sigma perpendicular mirror excess, 1 us = samples 0-499, FIR on 0-499 like feat1us) recomputed
   from raw traces with the shared-feature IFs and with the HERQULES hard-coded IFs, per neighbour configuration.
Run in oraqle-torch:latest with /feat, /data (ro) and /out.
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json, os, sys, time
import numpy as np, h5py
from scipy.signal import firwin, fftconvolve

FS = 500e6
TAPS = firwin(101, 5e6, fs=FS)
t = np.arange(1000) / FS
calib = json.load(open(str(LP.FEAT / 'calib.json')))['used_for_features']
F_OURS = np.array([u['f_Hz'] for u in calib]); TH_OURS = np.array([u['theta_rad'] for u in calib])
F_HERQ = -np.array([-64.729e6, -25.366e6, 24.79e6, 70.269e6, 127.282e6])
y = np.load(str(LP.FEAT / 'y_test.npy'))
F = np.load(str(LP.FEAT / 'feat_test.npy'), mmap_mode='r')
F1 = np.load(str(LP.FEAT / 'feat1us_test.npy'), mmap_mode='r')
H = h5py.File(str(LP.RAW_TEST), 'r')['X_test']
OUT = LP.outdir(LP.stage('vphysics') / 'out')
T0 = time.time()
def log(*a):
    print(f'[{time.time()-T0:6.0f}s]', *a, flush=True)

def rows(Y):
    r = np.flatnonzero(y == Y)
    assert len(r) == 35000 and r[-1] - r[0] == 34999
    return r

_cache = {}
def raw_block(Y):
    if Y not in _cache:
        r = rows(Y)
        X = H[r[0]:r[-1] + 1]
        _cache.clear()
        _cache[Y] = (X[..., 0] + 1j * X[..., 1])
        log('read block', Y)
    return _cache[Y]

def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))

WIN = {'A': slice(2, 5), 'B': slice(5, 8), 'C': slice(8, 10), 'D': slice(10, 20)}

def classes(q, y0, y1, T):
    """per-shot classes from shared features for qubit q; returns dict name -> (Y, local index array)"""
    r0, r1 = rows(y0), rows(y1)
    f0 = np.asarray(F[r0[0]:r0[-1] + 1, q]).astype(np.float64)  # (35000,20,2)
    f1 = np.asarray(F[r1[0]:r1[-1] + 1, q]).astype(np.float64)
    z0 = f0[..., 0] + 1j * f0[..., 1]; z1 = f1[..., 0] + 1j * f1[..., 1]
    out = {}
    W0 = {k: z0[:, s].mean(1) for k, s in WIN.items()}
    W1 = {k: z1[:, s].mean(1) for k, s in WIN.items()}
    geo = {}
    for k in WIN:
        c0 = np.median(W0[k].real) + 1j * np.median(W0[k].imag)
        a1 = W1[k]
        c1 = np.median(a1.real) + 1j * np.median(a1.imag)
        # refine c1 with shots near it (avoid decay/L pull)
        for _ in range(3):
            u = (c1 - c0) / abs(c1 - c0)
            al = ((a1 - c0) / u).real / abs(c1 - c0); pe = ((a1 - c0) / u).imag
            sel = (al > 0.6) & (al < 1.4) & (np.abs(pe - np.median(pe)) < 3 * mad(pe))
            c1 = np.median(a1[sel].real) + 1j * np.median(a1[sel].imag)
        u = (c1 - c0) / abs(c1 - c0); sep = abs(c1 - c0)
        p0 = ((W0[k] - c0) / u).imag; s0 = mad(p0); m0p = np.median(p0)
        geo[k] = dict(c0=c0, c1=c1, u=u, sep=sep, s0=s0, m0p=m0p)
    def coords(W, k):
        g = geo[k]; v = (W[k] - g['c0']) / g['u']
        return v.real / g['sep'], (v.imag - g['m0p']) / g['s0']
    A0 = {k: coords(W0, k) for k in WIN}; A1 = {k: coords(W1, k) for k in WIN}
    core = lambda A, c: np.all([(np.abs(A[k][0] - c) < 0.25) & (np.abs(A[k][1]) < 2) for k in WIN], axis=0)
    out['core0'] = (y0, np.flatnonzero(core(A0, 0.0)))
    out['core1'] = (y1, np.flatnonzero(core(A1, 1.0)))
    if q == 3:  # Q4: sub-cloud vs far cloud, nearest of three centres in (along sep, perp sep) using 1-us HMM geometry
        cen = {'c0': (0.0, 0.0), 'sub': (0.95, -0.30), 'far': (2.0, 1.04)}
        lab = {}
        for k in WIN:
            g = geo[k]; v = (W1[k] - g['c0']) / g['u']; a = v.real / g['sep']; p = (v.imag - g['m0p']) / g['sep']
            d = np.stack([np.hypot(a - c[0], p - c[1]) for c in cen.values()], 1)
            lab[k] = d.argmin(1)
        for i, nm in enumerate(cen):
            if nm == 'c0':
                continue
            out['Q4' + nm] = (y1, np.flatnonzero(np.all([lab[k] == i for k in WIN], axis=0)))
    else:
        out['Lstart'] = (y1, np.flatnonzero((A1['A'][1] > T) & (A1['C'][1] > T) & (A1['A'][0] > 0.6)))
        out['Llate'] = (y1, np.flatnonzero((np.abs(A1['A'][1]) < 1.5) & (np.abs(A1['A'][0] - 1) < 0.25) & (A1['D'][1] > T)))
        out['L0'] = (y0, np.flatnonzero((A0['C'][1] > T) & (A0['D'][1] > T)))
    g = {k: {kk: (vv if not isinstance(vv, complex) else [vv.real, vv.imag]) for kk, vv in v.items()} for k, v in geo.items()}
    for k in g:
        for kk in ('c0', 'c1', 'u'):
            v = geo[k][kk]; g[k][kk] = [float(v.real), float(v.imag)]
        for kk in ('sep', 's0', 'm0p'):
            g[k][kk] = float(geo[k][kk])
    return out, g

PLAN = [  # (q, y0, y1, T)
    (2, 0, 4, 4.0),
    (1, 0, 2, 4.0),
    (4, 4, 20, 2.5),
    (4, 0, 16, 2.5),
    (0, 4, 5, 2.5),
    (3, 0, 8, 4.0),
]
res = {'F_OURS': F_OURS.tolist(), 'TH_OURS': TH_OURS.tolist(), 'F_HERQ': F_HERQ.tolist(), 'classes': {}}
means = {}
for q, y0, y1, T in PLAN:
    cl, geo = classes(q, y0, y1, T)
    key = f'q{q}_y{y0}_{y1}'
    res['classes'][key] = {'geo_feat': geo, 'n': {k: int(len(v[1])) for k, v in cl.items()}}
    for nm, (Y, idx) in cl.items():
        if len(idx) == 0:
            continue
        Z = raw_block(Y)[idx]
        means[f'{key}__{nm}'] = Z.mean(0)
        means[f'{key}__{nm}__featmean'] = np.asarray(F[rows(Y)[0] + idx, q]).mean(0)
    log(key, res['classes'][key]['n'])
np.savez(os.path.join(OUT, 'vphysics_classmeans.npz'), **means)
json.dump(res, open(os.path.join(OUT, 'vphysics_raw_classes.json'), 'w'), indent=1)

# ---------------- C) E3 with both IF sets, per configuration ----------------
def fir_same(x):
    return fftconvolve(x, TAPS[None, :], mode='same', axes=1)

def winmeans(Z, f, nsamp):
    mixed = Z[:, :nsamp] * np.exp(-1j * 2 * np.pi * f * t[:nsamp])[None, :]
    return fir_same(mixed).mean(1)

def e3(w0, w1):
    c0 = np.median(w0.real) + 1j * np.median(w0.imag)
    c1 = np.median(w1.real) + 1j * np.median(w1.imag)
    for _ in range(3):
        u = (c1 - c0) / abs(c1 - c0)
        v = (w1 - c0) / u; al = v.real / abs(c1 - c0); pe = v.imag
        sel = (al > 0.6) & (al < 1.4) & (np.abs(pe - np.median(pe)) < 3 * mad(pe))
        c1 = np.median(w1[sel].real) + 1j * np.median(w1[sel].imag)
    u = (c1 - c0) / abs(c1 - c0)
    p0 = ((w0 - c0) / u).imag; p1 = ((w1 - c0) / u).imag
    s0 = mad(p0)
    out = {}
    for nm, p in (('p0', p0), ('p1', p1)):
        # truncated-median centre of the core (|dev| < 2.5 s0), as in r_spotcheck
        cen = np.median(p)
        for _ in range(5):
            cen = np.median(p[np.abs(p - cen) < 2.5 * s0])
        z = (p - cen) / s0
        npl, nmi = int((z > 3).sum()), int((z < -3).sum())
        N = len(z)
        out[nm] = dict(E3=(npl - nmi) / N, ci=1.96 * np.sqrt(npl + nmi) / N, n_plus=npl, n_minus=nmi, N=N)
    out['axis_deg'] = float(np.degrees(np.angle(u)))
    out['sep'] = float(abs(c1 - c0)); out['s0'] = float(s0)
    return out

E3PLAN = {2: [(0, 4), (2, 6), (8, 12), (16, 20)], 1: [(0, 2), (4, 6), (8, 10), (16, 18)]}
res['E3'] = {}
for q, pairs in E3PLAN.items():
    for y0, y1 in pairs:
        key = f'q{q}_y{y0}_{y1}'
        res['E3'][key] = {}
        W = {}
        for Y in (y0, y1):
            Z = raw_block(Y)
            for lab, f in (('ours', F_OURS[q]), ('herq', F_HERQ[q])):
                for nm, ns in (('1us', 500), ('2us', 1000)):
                    W[(Y, lab, nm)] = np.concatenate([winmeans(Z[i:i + 5000], f, ns) for i in range(0, 35000, 5000)])
            ff = np.asarray(F1[rows(Y)[0]:rows(Y)[-1] + 1, q]).mean(1).astype(np.float64)
            W[(Y, 'featfile', '1us')] = ff[:, 0] + 1j * ff[:, 1]
        for lab in ('ours', 'herq', 'featfile'):
            for nm in ('1us', '2us'):
                if (y0, lab, nm) in W:
                    res['E3'][key][f'{lab}_{nm}'] = e3(W[(y0, lab, nm)], W[(y1, lab, nm)])
        log('E3', key, {k: (round(v['p1']['E3'] * 100, 3), round(v['p0']['E3'] * 100, 3)) for k, v in res['E3'][key].items()})
        json.dump(res, open(os.path.join(OUT, 'vphysics_raw_classes.json'), 'w'), indent=1)
log('done')
