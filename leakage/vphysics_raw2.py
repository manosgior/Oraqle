"""vphysics_raw2.py (noise-scaled core classes, Q2/Q4/Q1 rerun) -- adversarial physics checks on raw traces (tag vphysics).

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
    # noise-scaled core: along within 2 sigma_along(prep0, that window) of the target, |z_perp| < 2
    sal = {k: mad(A0[k][0]) for k in WIN}
    core = lambda A, c: np.all([(np.abs(A[k][0] - c) < 2 * sal[k]) & (np.abs(A[k][1]) < 2) for k in WIN], axis=0)
    out['core0'] = (y0, np.flatnonzero(core(A0, 0.0)))
    out['core1'] = (y1, np.flatnonzero(core(A1, 1.0)))
    if q == 3:  # Q4: 3-means on prepared-1 bins 2-9 window means (ADC); class = same cluster in all 4 windows
        Wall = z1[:, 2:10].mean(1)
        cen = np.array([np.median(W0['A'].real) + 1j * np.median(W0['A'].imag), np.percentile(Wall.real, 30) + 1j * np.median(Wall.imag),
                        np.percentile(Wall.real, 85) + 1j * np.median(Wall.imag)])
        for _ in range(30):
            lab = np.abs(Wall[:, None] - cen[None, :]).argmin(1)
            cen = np.array([Wall[lab == i].mean() if (lab == i).any() else cen[i] for i in range(3)])
        labs = {k: np.abs(W1[k][:, None] - cen[None, :]).argmin(1) for k in WIN}
        for i in range(3):
            out[f'Q4k{i}'] = (y1, np.flatnonzero(np.all([labs[k] == i for k in WIN], axis=0)))
        out['Q4centres'] = (y1, np.array([], int))
        geo['kmeans_centres'] = {'c0': cen[0], 'c1': cen[1], 'u': cen[2], 'sep': 0.0, 's0': 0.0, 'm0p': 0.0}
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
    (1, 0, 2, 4.0),
    (1, 4, 6, 4.0),
    (3, 0, 8, 4.0),
    (0, 4, 5, 2.0),
    (2, 16, 20, 4.0),
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
np.savez(os.path.join(OUT, 'vphysics_classmeans2.npz'), **means)
json.dump(res, open(os.path.join(OUT, 'vphysics_raw_classes2.json'), 'w'), indent=1)
log('done')
