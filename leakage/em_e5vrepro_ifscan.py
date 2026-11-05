# e5vrepro: independent IF estimation from raw TRAIN traces (no shared features/code).
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import h5py, numpy as np, json, sys
FS = 500e6
h = h5py.File(str(LP.RAW_TRAIN), 'r')
X = h['X_train']; y = h['y_train'][:]
n = 120000
x = X[:n].astype(np.float32)          # (n,1000,2)
yy = y[:n]
s = x[..., 0] + 1j * x[..., 1]        # complex trace
M = s.mean(0)
t = np.arange(1000) / FS
NF = 1 << 17
F = np.fft.fftfreq(NF, 1 / FS)
P = np.abs(np.fft.fft(M - M[700:].mean() * 0, NF))
# difference traces per qubit: mean(prep1) - mean(prep0)
out = {}
for q in range(5):
    b = (yy >> q) & 1
    D = s[b == 1].mean(0) - s[b == 0].mean(0)
    PD = np.abs(np.fft.fft(D, NF))
    k = np.argmax(PD)
    out[q] = dict(f_peak_diff=float(F[k]))
    print('q', q, 'diff-peak IF (MHz)', F[k] / 1e6, flush=True)
# top peaks of mean trace
idx = np.argsort(P)[::-1]
peaks = []
for k in idx:
    if all(abs(F[k] - p) > 3e6 for p in peaks):
        peaks.append(F[k])
    if len(peaks) >= 8: break
print('mean-trace peaks (MHz):', [round(p / 1e6, 3) for p in peaks])
# per-sample stats
print('raw dtype stats', float(x.mean()), float(x.std()), float(np.abs(x).max()))
json.dump(dict(diff=out, peaks=peaks), open(_os.path.join(LP.outdir(LP.stage('em/e5vrepro') / 'out'), 'em_e5vrepro_ifscan.json'), 'w'), indent=1)
