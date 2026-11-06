import sys, glob, os, numpy as np, h5py
c = sys.argv[1]
fs = sorted(glob.glob(f"{c}/*"));
print(len(fs), [os.path.basename(x) for x in fs[:3]])
f0 = fs[0]
with h5py.File(f0, "r") as f:
    def show(n, o):
        if isinstance(o, h5py.Dataset): print("  ", n, o.shape, o.dtype, (o[:5] if o.ndim == 1 else ""))
    f.visititems(show)
    print("attrs", dict(f.attrs))
    I = f["I_data"][:2000]; Q = f["Q_data"][:2000]
print("mean trace |z| per 10 samples", np.round(np.abs((I + 1j * Q).mean(0))[::10], 2))
print("noise std per sample", np.round(I.std(0)[::10], 2))
# time correlation of the noise, a hint at the digitizer bandwidth
print("adjacent-sample corr", np.round(np.corrcoef(I[:, 40], I[:, 41])[0, 1], 3), "lag2", np.round(np.corrcoef(I[:, 40], I[:, 42])[0, 1], 3))
