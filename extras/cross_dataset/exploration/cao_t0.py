import sys, numpy as np, h5py
c = "/w/external/cao_DPMID3"; ROT = np.exp(1j*np.pi/2); ref = np.array([-3.47+5.9j, -4.91+0.31j, 1.99-1.11j])
acc = {}
for s in "gef":
    tot = 0; n = 0
    for k in range(6):
        with h5py.File(f"{c}/001_prep_{s}_natural_decay_{k}", "r") as f:
            Z = (f["I_data"][:] + 1j*f["Q_data"][:]) * ROT
        w = Z[:, 20:80].mean(1); first = w[0::2]; m = min(len(first), len(Z[1::2]))
        ok = np.argmin(np.abs(first[:m, None] - ref[None]), axis=1) == 0; j = np.arange(m) % 120
        sel = Z[1::2][:m][ok & (j < 4)]; tot = tot + sel.sum(0); n += len(sel)
    acc[s] = tot / n; print(s, "shots", n)
np.savez("/w/cross_dataset/out/cao_means.npz", **acc)
g, e, f_ = acc["g"], acc["e"], acc["f"]
print(" i   |g|    |e|   |e-g|  |f-g|")
for i in range(0, 80, 2):
    print(f"{i:3d} {abs(g[i]):6.2f} {abs(e[i]):6.2f} {abs(e[i]-g[i]):6.2f} {abs(f_[i]-g[i]):6.2f}")
