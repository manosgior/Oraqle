"""m4conf_rawstats.py -- per-shot raw-trace statistics for the confounder screens (agent tag m4conf).

One pass over the raw HDF5 (train + test). For every shot, from the raw complex trace
x[n] = I[n] + iQ[n] (n = 0..999, 2 ns/sample):
  dcI1, dcQ1   mean I, mean Q over samples 0..499 (paper "1 us")            -> DC / baseline glitch
  p1, p2       mean |x|^2 over 0..499 and 0..999                            -> total trace power
  oob1, oob2   mean power OUTSIDE the readout bands (FFT of 0..499 / 0..999; excluded: |f - f_q| < 8 MHz
               and |f + f_q| < 8 MHz (mixer image) for all 5 tones, and |f| < 3 MHz (DC))
               -> broadband glitch / ADC / feedline noise burst
  maxabs       max(|I|, |Q|) over 0..999                                     -> saturation
  dmax         max |x[n] - x[n-1]| over 0..999                               -> spikes
Also accumulates the per-state mean trace (32 x 1000 complex) for each split (used for residuals in
m4conf_rawprobe.py).
Usage (inside oraqle-torch:latest):  python m4conf_rawstats.py /data /feat /out
"""
import json, os, sys, time
import numpy as np
import h5py
from multiprocessing import Pool

DATA, FEAT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FS = 500e6
CAL = json.load(open(os.path.join(FEAT, "calib.json")))["used_for_features"]
FQ = np.array([c["f_Hz"] for c in CAL])
FILES = {"train": ("DRaw_C_Tr_v0-001", "X_train", "y_train"),
         "test": ("DRaw_C_Te_v0-002", "X_test", "y_test")}
CH = 8000


def oob_mask(L):
    f = np.fft.fftfreq(L, 1 / FS)
    keep = np.abs(f) >= 3e6
    for fq in FQ:
        keep &= np.abs(f - fq) >= 8e6
        keep &= np.abs(f + fq) >= 8e6
    return keep


M1, M2 = oob_mask(500), oob_mask(1000)


def work(args):
    split, a, b = args
    fn, kx, ky = FILES[split]
    with h5py.File(os.path.join(DATA, fn), "r") as h:
        X = h[kx][a:b]
        y = h[ky][a:b]
    x = X[:, :, 0] + 1j * X[:, :, 1]
    out = {}
    out["dcI1"] = X[:, :500, 0].mean(1)
    out["dcQ1"] = X[:, :500, 1].mean(1)
    ax2 = np.abs(x) ** 2
    out["p1"] = ax2[:, :500].mean(1)
    out["p2"] = ax2.mean(1)
    F1 = np.fft.fft(x[:, :500], axis=1)
    out["oob1"] = (np.abs(F1[:, M1]) ** 2).sum(1) / 500 ** 2
    F2 = np.fft.fft(x, axis=1)
    out["oob2"] = (np.abs(F2[:, M2]) ** 2).sum(1) / 1000 ** 2
    out["maxabs"] = np.abs(X).max(axis=(1, 2))
    out["dmax"] = np.abs(np.diff(x, axis=1)).max(1)
    sums = {}
    for s in np.unique(y):
        m = y == s
        sums[int(s)] = (x[m].sum(0), int(m.sum()))
    return split, a, b, {k: v.astype(np.float32) for k, v in out.items()}, sums


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    jobs = []
    N = {}
    for split, (fn, kx, ky) in FILES.items():
        with h5py.File(os.path.join(DATA, fn), "r") as h:
            N[split] = h[kx].shape[0]
        jobs += [(split, a, min(a + CH, N[split])) for a in range(0, N[split], CH)]
    res = {s: {k: np.zeros(N[s], np.float32) for k in ["dcI1", "dcQ1", "p1", "p2", "oob1", "oob2", "maxabs", "dmax"]} for s in N}
    msum = {s: np.zeros((32, 1000), np.complex128) for s in N}
    mcnt = {s: np.zeros(32, np.int64) for s in N}
    with Pool(24) as pool:
        for i, (split, a, b, out, sums) in enumerate(pool.imap_unordered(work, jobs)):
            for k, v in out.items():
                res[split][k][a:b] = v
            for s, (v, c) in sums.items():
                msum[split][s] += v
                mcnt[split][s] += c
            if i % 40 == 0:
                print(f"{i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    for s in N:
        np.savez(os.path.join(OUT, f"m4conf_rawstats_{s}.npz"), **res[s],
                 meantrace=(msum[s] / mcnt[s][:, None]).astype(np.complex64), meancount=mcnt[s])
        print(s, {k: float(np.median(v)) for k, v in res[s].items()}, flush=True)
    print("done", time.time() - t0, flush=True)
