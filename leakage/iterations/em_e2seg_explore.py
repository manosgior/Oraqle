import numpy as np, json
D = "/feat"
F = np.load(f"{D}/featbox_train.npy", mmap_mode="r")
y = np.load(f"{D}/y_train.npy")
B = ((y[:, None] >> np.arange(5)) & 1).astype(np.int8)
X = np.asarray(F[:, :, :10, :], dtype=np.float64)   # (N,5,10,2)
def cfg_of(B, q):
    o = [k for k in range(5) if k != q]
    return (B[:, o] * (1 << np.arange(4))).sum(1)
def mad(x, axis=0):
    return 1.4826 * np.median(np.abs(x - np.median(x, axis=axis)), axis=axis)
out = {}
for q in range(5):
    cf = cfg_of(B, q)
    res = {}
    for c in (0, 15):
        T = {}
        for b in (0, 1):
            m = (cf == c) & (B[:, q] == b)
            P = X[m, q]                       # (n,10,2)
            T[b] = np.median(P, 0)            # coordinate-wise per-bin median (crude)
        D_ = T[1] - T[0]                      # (10,2)
        sep = np.linalg.norm(D_, axis=1)
        u = D_ / sep[:, None]
        m0 = (cf == c) & (B[:, q] == 0)
        x0 = (((X[m0, q] - T[0]) * u).sum(-1)) / sep     # normalized
        m1 = (cf == c) & (B[:, q] == 1)
        x1 = (((X[m1, q] - T[0]) * u).sum(-1)) / sep
        s0 = mad(x0); s1 = mad(x1 - 1)
        # GK correlation adjacent
        def gk(x):
            s = mad(x)
            z = x / s
            K = x.shape[1]
            R = np.eye(K)
            for i in range(K):
                for j in range(i + 1, K):
                    a = mad(z[:, i] + z[:, j]) ** 2; d = mad(z[:, i] - z[:, j]) ** 2
                    R[i, j] = R[j, i] = (a - d) / (a + d)
            return s, R
        g0s, R0 = gk(x0)
        g1s, R1 = gk(x1)
        ang = np.degrees(np.arctan2(u[:, 1], u[:, 0]))
        res[c] = {"sep_raw": sep.round(3).tolist(), "angle": ang.round(1).tolist(),
                  "sig0_norm": g0s.round(3).tolist(), "sig1_norm": g1s.round(3).tolist(),
                  "R0_adj": np.diag(R0, 1).round(3).tolist(), "R1_adj": np.diag(R1, 1).round(3).tolist(),
                  "R0_lag2": np.diag(R0, 2).round(3).tolist(),
                  "mean_sig0_window": float(mad(x0.mean(1))), "mean_sig1_window": float(mad(x1.mean(1)))}
    out[q] = res
    print(q, json.dumps(res[0]), flush=True)
