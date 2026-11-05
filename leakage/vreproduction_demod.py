#!/usr/bin/env python3
"""vreproduction: independent demodulation of a stratified random subsample of the raw
5-qubit readout traces (Lienhard et al. 2022 dataset).

Independent of the shared feature files and of the other agents' code:
  * own stratified random subsample (TRAIN 4000/state, TEST 8000/state -> 384k shots)
  * own IF estimate: peak of the coherent periodogram of the TRAIN-subsample mean
    difference trace D_q(t) = <z | bit q = 1>(t) - <z | bit q = 0>(t), z = I + iQ,
    refined on a 100 Hz grid (no phase-slope fit, no FFT-bin quantisation)
  * frame 'box': mix with exp(-i 2 pi f_q t), t = n / 500e6, then plain boxcar mean over
    20 contiguous 50-sample bins (no FIR)
  * frame 'bw': HERQULES-style causal Butterworth order 3, 10 MHz, after mixing (no
    per-trace DC / IQ-imbalance normalisation), then the same 20-bin boxcar means
Outputs (to --out): vreproduction_sub_{split}.npz with idx, y, box (N,5,20) c64, bw (N,5,20) c64;
vreproduction_calib.json.
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import argparse, json, time
import numpy as np
import h5py
from scipy.signal import butter, sosfilt

FS = 500e6
T = 1000
p = argparse.ArgumentParser()
p.add_argument('--data', default=str(LP.DATA))
p.add_argument('--out', default=str(LP.stage('vreproduction') / 'out'))
p.add_argument('--ntr', type=int, default=4000)
p.add_argument('--nte', type=int, default=8000)
p.add_argument('--seed', type=int, default=20261104)
a = p.parse_args()

FILES = {'train': ('DRaw_C_Tr_v0-001', 'X_train', 'y_train', a.ntr),
         'test': ('DRaw_C_Te_v0-002', 'X_test', 'y_test', a.nte)}
rng = np.random.default_rng(a.seed)
t = np.arange(T) / FS
meta = {'seed': a.seed, 'fs': FS}


def pick(y, nper):
    idx = []
    for s in range(32):
        rows = np.flatnonzero(y == s)
        idx.append(rng.choice(rows, nper, replace=False))
    return np.sort(np.concatenate(idx))


def read_rows(ds, idx, chunk=16000):
    for i0 in range(0, len(idx), chunk):
        ii = idx[i0:i0 + chunk]
        x = ds[ii]                      # (n,1000,2) float64, ii increasing
        yield i0, x[..., 0] + 1j * x[..., 1]


# ---------------- subsample + calibration on TRAIN subsample ----------------
subs = {}
for split, (fn, xk, yk, nper) in FILES.items():
    with h5py.File(f'{a.data}/{fn}', 'r') as f:
        y = f[yk][:]
    idx = pick(y, nper)
    subs[split] = (idx, y[idx].astype(np.int32))
    meta[f'n_{split}'] = int(len(idx))
    # verify block structure claim
    meta[f'{split}_counts_per_state'] = np.bincount(y, minlength=32).tolist()

fn, xk, yk, _ = FILES['train']
idx, ys = subs['train']
bits = (ys[:, None] >> np.arange(5)) & 1
S1 = np.zeros((5, T), complex); S0 = np.zeros((5, T), complex)
n1 = bits.sum(0); n0 = len(ys) - n1
mean_all = np.zeros(T, complex)
t0 = time.time()
with h5py.File(f'{a.data}/{fn}', 'r') as f:
    ds = f[xk]
    for i0, z in read_rows(ds, idx):
        b = bits[i0:i0 + len(z)]
        mean_all += z.sum(0)
        for q in range(5):
            S1[q] += z[b[:, q] == 1].sum(0)
            S0[q] += z[b[:, q] == 0].sum(0)
print('calib pass', time.time() - t0, flush=True)
D = S1 / n1[:, None] - S0 / n0[:, None]
dc = (mean_all / len(ys)).mean()            # global DC (complex), removed below
meta['global_dc'] = [float(dc.real), float(dc.imag)]


def dft(x, f, tt):
    return np.exp(-2j * np.pi * np.outer(f, tt)) @ x


calib = []
fq = np.zeros(5)
for q in range(5):
    # coarse: zero-padded FFT of full-trace D_q
    nf = 1 << 16
    F = np.fft.fftshift(np.fft.fft(D[q], nf))
    ff = np.fft.fftshift(np.fft.fftfreq(nf, 1 / FS))
    pk = np.argsort(np.abs(F))[::-1]
    f_coarse = ff[pk[0]]
    # other local maxima (for the record): top 5 separated by > 2 MHz
    tops = []
    for k in pk:
        if all(abs(ff[k] - u) > 2e6 for u, _ in tops):
            tops.append((float(ff[k]), float(np.abs(F[k]))))
        if len(tops) == 5:
            break
    res = {}
    for name, sl in (('full', slice(0, 1000)), ('1us', slice(0, 500)), ('late', slice(100, 1000))):
        grid = f_coarse + np.arange(-200e3, 200e3 + 1, 100.0)
        P = np.abs(dft(D[q][sl], grid, t[sl]))
        res[name] = float(grid[np.argmax(P)])
    fq[q] = res['full']
    calib.append({'q': q, 'paper': f'Q{q+1}', 'f_coarse': float(f_coarse), 'f_peak': res,
                  'top_peaks': tops})
    print(q, res, tops[:3], flush=True)
meta['calib'] = calib
meta['f_used'] = fq.tolist()
meta['herqules_freq'] = [64.729e6, 25.366e6, -24.79e6, -70.269e6, -127.282e6]

sos = butter(3, 10e6, btype='low', fs=FS, output='sos')
mix = np.exp(-2j * np.pi * np.outer(fq, t))       # (5,T)

for split, (fn, xk, yk, _) in FILES.items():
    idx, ys = subs[split]
    N = len(idx)
    box = np.zeros((N, 5, 20), np.complex64)
    bw = np.zeros((N, 5, 20), np.complex64)
    t0 = time.time()
    with h5py.File(f'{a.data}/{fn}', 'r') as f:
        ds = f[xk]
        assert np.array_equal(f[yk][idx], ys)
        for i0, z in read_rows(ds, idx):
            z = z - dc
            n = len(z)
            for q in range(5):
                zm = z * mix[q]
                box[i0:i0 + n, q] = zm.reshape(n, 20, 50).mean(-1)
                zf = sosfilt(sos, zm.real, axis=1) + 1j * sosfilt(sos, zm.imag, axis=1)
                bw[i0:i0 + n, q] = zf.reshape(n, 20, 50).mean(-1)
    print(split, 'demod', time.time() - t0, flush=True)
    np.savez(f'{a.out}/vreproduction_sub_{split}.npz', idx=idx, y=ys, box=box, bw=bw)

# store D traces for the record (small)
np.savez(f'{a.out}/vreproduction_Dtraces.npz', D=D.astype(np.complex64), f=fq)
json.dump(meta, open(f'{a.out}/vreproduction_calib.json', 'w'), indent=1)
print('done', flush=True)
