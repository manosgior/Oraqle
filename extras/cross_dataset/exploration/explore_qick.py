"""First look at the QICK traces: column layout, baseband or IF, signal onset, SNR vs time."""
import sys, json, numpy as np
d = sys.argv[1]; out = sys.argv[2]
X = np.load(f"{d}/0528_X_test_0_770.npy", mmap_mode="r", allow_pickle=False)
y = np.load(f"{d}/0528_y_test_0_770.npy", allow_pickle=False)
print("X", X.shape, X.dtype, "y", y.shape, y.dtype, "y head", y[:3].tolist(), "unique", np.unique(y, axis=0)[:4].tolist())
yb = y[:, 1] if y.ndim == 2 else y
yb = yb.astype(int)
print("counts", np.bincount(yb), "first change at", int(np.argmax(yb != yb[0])))
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(len(X), 20000, replace=False))
A = np.asarray(X[idx], dtype=np.float64); ys = yb[idx]
print("value range", A.min(), A.max(), "mean", A.mean(), "std", A.std())
print("integer-valued?", np.allclose(A[:50], np.round(A[:50])))
# interleaved I,Q,I,Q (as the authors' window script implies)
I, Q = A[:, ::2], A[:, 1::2]
z = I + 1j * Q
m0, m1 = z[ys == 0].mean(0), z[ys == 1].mean(0)
s0, s1 = z[ys == 0].std(0), z[ys == 1].std(0)
d10 = np.abs(m1 - m0); noise = np.sqrt((s0**2 + s1**2) / 2)
np.savez(out, m0=m0, m1=m1, s0=s0, s1=s1)
print("sample | |m0| |m1| |m1-m0| noise SNR")
for t in list(range(0, 770, 35)) + [769]:
    print(f"{t:4d} {t*2500/770:7.0f}ns {abs(m0[t]):9.1f} {abs(m1[t]):9.1f} {d10[t]:9.1f} {noise[t]:9.1f} {d10[t]/noise[t]:7.3f}")
# concatenated-layout check: under I...I,Q...Q the two halves would be unrelated in time structure
Ic, Qc = A[:, :770], A[:, 770:]
print("corr(adjacent cols even-odd) interleaved %.3f, same-quad(0,2) %.3f" % (np.corrcoef(A[:,600],A[:,601])[0,1], np.corrcoef(A[:,600],A[:,602])[0,1]))
# residual IF in the mean trace
for name, m in (("m0", m0), ("m1", m1), ("m1-m0", m1 - m0)):
    w = 200; seg = m[w:w+400]; f = np.fft.fftfreq(len(seg), d=2500e-9/770)
    sp = np.abs(np.fft.fft(seg - seg.mean())); k = np.argsort(sp)[-3:][::-1]
    print(name, "top spectral lines MHz", [round(f[i]/1e6, 2) for i in k], "rel", [round(sp[i]/sp.sum(), 3) for i in k],
          "phase slope rad/sample %.4f" % np.polyfit(np.arange(len(seg)), np.unwrap(np.angle(seg)), 1)[0])
