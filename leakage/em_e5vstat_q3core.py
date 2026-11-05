"""em_e5vstat_q3core.py (tag e5vstat): why does the Q3 prepared-|1> late-core overlap rate depend 3x on the core centre?
Per neighbour configuration: e4recon late |1> core (from em_e4recon_glrt_raw.json) vs the own e5vstat core, distance,
sizes, CNN error rates inside each, in the symmetric difference, and along the line joining them; plus a coarse 2-D
density map of prepared-|1> late-window points with the CNN error rate per cell, in late-frame (u, v) coordinates.
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import numpy as np

D, R, E4, O = str(LP.FEAT), str(LP.REPORTS), str(LP.stage("em/e4recon")), LP.outdir(LP.stage("em/e5vstat") / "out")
q = 2


def bits(y):
    return ((y[:, None] >> np.arange(5)) & 1).astype(np.int8)


def cfg_of(B, q):
    o = [k for k in range(5) if k != q]
    return (B[:, o].astype(np.int64) * (1 << np.arange(4))).sum(1)


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def mode2d(P, s, seed, bw=0.5):
    c = np.asarray(seed, float)
    for _ in range(300):
        d2 = ((P - c) ** 2).sum(1)
        m = d2 < (2 * s) ** 2
        w = np.exp(-d2[m] / (2 * (bw * s) ** 2))
        cn = (w[:, None] * P[m]).sum(0) / w.sum()
        if np.linalg.norm(cn - c) < 1e-6 * s:
            return cn
        c = cn
    return c


ytr, yte = np.load(D + "/y_train.npy"), np.load(D + "/y_test.npy")
Btr, Bte = bits(ytr), bits(yte)
cft, cfe = cfg_of(Btr, q), cfg_of(Bte, q)
ptr, pte = Btr[:, q], Bte[:, q]
Xtr = np.asarray(np.load(f"{D}/featbox_train.npy", mmap_mode="r")[:, q], dtype=np.float64)
Xte = np.asarray(np.load(f"{D}/featbox_test.npy", mmap_mode="r")[:, q], dtype=np.float64)
Ltr, Lte = Xtr[:, 10:20].mean(1), Xte[:, 10:20].mean(1)
pb = np.load(R + "/preds_MCMitCNN_pb.npy")[:, q]
err = pb != pte
raw = json.load(open(f"{E4}/out/em_e4recon_glrt_raw.json"))
info = {(d["c"], d["b"]): d for d in raw["latecore_info"][f"q{q}"]}
lc4 = np.load(f"{E4}/out/em_e4recon_latecore.npz")[f"latecore_q{q}"]
out = []
maps = {}
for c in range(16):
    m0t, m1t = (cft == c) & (ptr == 0), (cft == c) & (ptr == 1)
    s = mad(Ltr[m0t, 1])
    c0 = mode2d(Ltr[m0t], s, np.median(Ltr[m0t], 0))
    a0 = mode2d(Xtr[m0t, 5:10].mean(1), s, np.median(Xtr[m0t, 5:10].mean(1), 0))
    a1 = mode2d(Xtr[m1t, 5:10].mean(1), s, np.median(Xtr[m1t, 5:10].mean(1), 0))
    mine = mode2d(Ltr[m1t], s, c0 + (a1 - a0))
    e4c = np.array(info[(c, 1)]["core"])
    s4 = info[(c, 1)]["s_late"]
    sep = np.linalg.norm(mine - c0)
    u = (mine - c0) / sep
    n = np.array([-u[1], u[0]])
    me1 = (cfe == c) & (pte == 1)
    X = Lte[me1]
    inm = np.linalg.norm(X - mine, axis=1) < 1.5 * s
    in4 = lc4[me1, 1]
    e = err[me1]
    rec = {"c": c, "s_mine": s, "s_e4": s4, "dist_mine_e4_over_s": float(np.linalg.norm(mine - e4c) / s),
           "e4core_uv": [float(((e4c - c0) @ u) / sep), float(((e4c - c0) @ n) / s)],
           "mine_uv": [1.0, 0.0], "seed_uv": [float(((c0 + a1 - a0 - c0) @ u) / sep), float(((c0 + a1 - a0 - c0) @ n) / s)],
           "sep_over_s": float(sep / s), "n_prep1": int(me1.sum()),
           "n_mine": int(inm.sum()), "err_mine_pct": float(100 * e[inm].mean()),
           "n_e4": int(in4.sum()), "err_e4_pct": float(100 * e[in4].mean()) if in4.any() else None,
           "n_mine_not_e4": int((inm & ~in4).sum()), "err_mine_not_e4_pct": float(100 * e[inm & ~in4].mean()) if (inm & ~in4).any() else None,
           "n_e4_not_mine": int((~inm & in4).sum()), "err_e4_not_mine_pct": float(100 * e[~inm & in4].mean()) if (~inm & in4).any() else None}
    # density/error map in (u in sep units, v in sigma units)
    uu = ((X - c0) @ u) / sep
    vv = ((X - c0) @ n) / s
    ue = np.arange(-0.3, 2.31, 0.1)
    ve = np.arange(-4, 16.01, 1.0)
    H = np.histogram2d(uu, vv, bins=[ue, ve])[0]
    He = np.histogram2d(uu[e], vv[e], bins=[ue, ve])[0]
    maps[c] = {"u_edges": ue.round(2).tolist(), "v_edges": ve.tolist(), "n": H.astype(int).tolist(), "err": He.astype(int).tolist()}
    out.append(rec)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
json.dump({"note": __doc__, "per_cfg": out, "maps": maps}, open(f"{O}/em_e5vstat_q3core.json", "w"), indent=1)
