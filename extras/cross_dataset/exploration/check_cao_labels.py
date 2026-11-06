"""Validate Cao et al. labels: is the sequence period really 120? Do prep_g/e/f separate at short delay? What decays with delay?"""
import sys, glob, os, numpy as np, h5py
from sklearn.mixture import GaussianMixture
c = sys.argv[1]; WIN = (20, 80); ROT = np.exp(1j * np.pi / 2)  # notebook rotates by -90 deg
def load(state, k):
    with h5py.File(f"{c}/001_prep_{state}_natural_decay_{k}", "r") as f:
        I, Q = f["I_data"][:], f["Q_data"][:]
    z = (I + 1j * Q) * ROT
    return z
Z = {s: load(s, 0) for s in "gef"}
print({s: Z[s].shape for s in Z})
for s in "gef":
    z = Z[s].astype(np.complex64); w = z[:, WIN[0]:WIN[1]].mean(1)
    first, second = w[0::2], w[1::2]
    print(f"prep_{s}: first-msmt mean {first.mean():.2f}, second-msmt mean {second.mean():.2f}, std {first.std():.2f}/{second.std():.2f}")
# periodicity of the second measurement: autocorrelation of the decaying population along the row index
s = "e"; w = Z[s][:, WIN[0]:WIN[1]].mean(1)[1::2]; x = w.real - w.real.mean(); y = w.imag - w.imag.mean()
print("n pairs", len(w))
ac = {lag: float((x[:-lag] * x[lag:]).mean() + (y[:-lag] * y[lag:]).mean()) for lag in [1, 2, 10, 60, 100, 110, 119, 120, 121, 130, 200, 240]}
print("autocorr", {k: round(v, 2) for k, v in ac.items()})
# global clusters from the first measurement of all three preps
A = np.concatenate([Z[s][::2, WIN[0]:WIN[1]].mean(1) for s in "gef"]); P = np.c_[A.real, A.imag]
gm = GaussianMixture(3, covariance_type="tied", random_state=0, n_init=3).fit(P[::3])
cent = gm.means_[:, 0] + 1j * gm.means_[:, 1]; wts = gm.weights_
print("first-msmt clusters", np.round(cent, 2), "weights", np.round(wts, 3), "sigma", np.round(np.sqrt(np.diag(gm.covariances_)), 2))
g = int(np.argmax(wts)); print("g cluster index", g)
def assign(w):
    return np.argmin(np.abs(w[:, None] - cent[None]), axis=1)
S = 120
for s in "gef":
    w = Z[s][:, WIN[0]:WIN[1]].mean(1); first = w[0::2]; second = w[1::2]
    ok = assign(first) == g
    idx = np.arange(len(second)) % S
    print(f"prep_{s}: herald pass {ok.mean():.3f}; assigned populations of the 2nd msmt (cluster order) after herald, by delay index:")
    for j in [0, 1, 2, 4, 8, 16, 32, 64, 119]:
        m = ok & (idx == j)
        a = assign(second[m]); print(f"   j={j:3d} t={250*j:6d}ns n={m.sum():4d} pop", np.round(np.bincount(a, minlength=3) / max(1, m.sum()), 3))
# trailing rows: do they keep the same pattern?
w = Z["e"][:, WIN[0]:WIN[1]].mean(1)[1::2]; ok = assign(Z["e"][:, WIN[0]:WIN[1]].mean(1)[0::2]) == g
for a, b in ((0, 66000), (66000, 110000)):
    idx = np.arange(a, b) % S; sel = ok[a:b]
    print(f"rows {a}-{b}: pop near-e at j<4: {np.mean(assign(w[a:b][sel & (idx<4)]) == g):.3f} (fraction assigned to g); at j>100: {np.mean(assign(w[a:b][sel & (idx>100)]) == g):.3f}")
# time structure
for s in "gef":
    z = Z[s][1::2]; m = ok_ = assign(Z[s][:, WIN[0]:WIN[1]].mean(1)[0::2]) == g; j = np.arange(len(z)) % S
    mt = z[m & (j < 4)].mean(0); print(f"prep_{s} mean trace every 8 samples", np.round(mt[::8], 2))
