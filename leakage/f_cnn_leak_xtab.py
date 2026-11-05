"""f_cnn_leak_xtab.py (tag f): what a leakage (L) flag would add to the MCMit-CNN's soft-readout erasures.

Line-referenced frame (the verifiers' correction): for every qubit, prepared bit and each of the 16
neighbour configurations, c0 / c1 = density modes of the prepared-|0> / prepared-|1> window means, the
|0>-|1> line goes through c0 and c1, z = signed perpendicular distance from that line in units of the
prepared-|0> perpendicular noise width (1.4826 MAD) of the same configuration. + side = the side of the
off-axis populations (positive imag in the shared frame). Frames are fitted on TRAIN and applied to TEST.
For Q4 (q3) c1 = coordinate-wise median of prepared |1>, because its densest |1> peak is a sub-cloud.

L flag = z > 3; mirror = z < -3. Mirror-subtracted counts estimate the off-axis (L) population.

Cross-tabs on TEST against the per-shot CNN probabilities of sims/soft_readout_cnn.py
(data/sims/soft/cnn_soft_{1000,600,400}ns.npz, same model/recipe as the paper) and against the
canonical hard predictions preds_MCMitCNN_pb.npy:
  - E3 per qubit and prepared bit;
  - leaked share of CNN errors, misread rate of L shots;
  - share of the L population that the paper's confidence flag (least confident f of pooled Q1/Q3/Q4/Q5
    measurements, exactly as sweep_soft.py) already catches;
  - the L flag used as an erasure: error of flagged shots, and e_u versus a confidence flag of equal size;
  - where 1 us L-flagged shots are in the 1.2-2.0 us window (data the CNN never sees).

Windows: 1 us = feat1us bins 0-9 (raw samples 0-499, causal FIR, the CNN input);
600 ns / 400 ns = featbox bins 0-5 / 0-3 (samples 0-299 / 0-199, causal boxcar);
late = feat bins 12-19 (samples 600-999).
Docker (graham, oraqle-torch): -v features:/feat:ro -v f/in:/soft:ro -v ~/oraqle_reports:/reports:ro -v f:/work
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter

D, S, R, O = str(LP.FEAT), str(LP.SOFT), str(LP.REPORTS), str(LP.stage("f") / "out")
os.makedirs(O, exist_ok=True)
QUBITS = [0, 2, 3, 4]          # pooled as in sweep_soft.py (Q2 excluded)
FS = [0.01, 0.02, 0.05, 0.10]


def bits(y):
    return ((y[:, None] >> np.arange(5)) & 1).astype(np.int8)


def wmean(split, src, b0, b1):
    F = np.load(f"{D}/{src}_{split}.npy", mmap_mode="r")
    out = np.empty((F.shape[0], 5, 2))
    for s in range(0, F.shape[0], 100000):
        out[s:s + 100000] = np.asarray(F[s:s + 100000, :, b0:b1, :], dtype=np.float64).mean(2)
    return out


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def mode2d(P, s):
    lo = np.percentile(P, 0.5, axis=0) - s
    hi = np.percentile(P, 99.5, axis=0) + s
    nb = np.minimum(np.ceil((hi - lo) / (s / 3)).astype(int), 600)
    H, ex, ey = np.histogram2d(P[:, 0], P[:, 1], bins=nb, range=[[lo[0], hi[0]], [lo[1], hi[1]]])
    H = gaussian_filter(H, 1.5)
    i, j = np.unravel_index(np.argmax(H), H.shape)
    c = np.array([(ex[i] + ex[i + 1]) / 2, (ey[j] + ey[j + 1]) / 2])
    for _ in range(8):
        m = np.sum((P - c) ** 2, 1) < s ** 2
        c = P[m].mean(0)
    return c


def cfg_of(B, q):
    o = [k for k in range(5) if k != q]
    return (B[:, o] * (1 << np.arange(4))).sum(1)


def fit_frames(A, B):
    fr = {}
    for q in range(5):
        cf = cfg_of(B, q)
        for c in range(16):
            m0 = (cf == c) & (B[:, q] == 0)
            m1 = (cf == c) & (B[:, q] == 1)
            P0, P1 = A[m0, q], A[m1, q]
            s = mad(P0[:, 1])
            c0 = mode2d(P0, s)
            c1 = np.median(P1, 0) if q == 3 else mode2d(P1, s)
            u = (c1 - c0) / np.linalg.norm(c1 - c0)
            n = np.array([-u[1], u[0]])
            if n[1] < 0:
                n = -n
            sig = mad((P0 - c0) @ n)
            fr[(q, c)] = (c0, u, n, sig, np.linalg.norm(c1 - c0))
    return fr


def apply(A, B, fr):
    z = np.empty((A.shape[0], 5))
    t = np.empty((A.shape[0], 5))
    for q in range(5):
        cf = cfg_of(B, q)
        for c in range(16):
            m = cf == c
            c0, u, n, sig, sep = fr[(q, c)]
            z[m, q] = ((A[m, q] - c0) @ n) / sig
            t[m, q] = ((A[m, q] - c0) @ u) / sep
    return z, t


def ci_mirror(npl, nmi, n):
    return [float((npl - nmi) / n - 1.96 * np.sqrt(npl + nmi) / n),
            float((npl - nmi) / n + 1.96 * np.sqrt(npl + nmi) / n)]


ytr, yte = np.load(D + "/y_train.npy"), np.load(D + "/y_test.npy")
Btr, Bte = bits(ytr), bits(yte)
WIN = {"1us": ("feat1us", 0, 10, 1000), "600ns": ("featbox", 0, 6, 600), "400ns": ("featbox", 0, 4, 400)}
res = {"note": __doc__, "windows": {}}

pb_canon = np.load(R + "/preds_MCMitCNN_pb.npy").astype(np.int8)
tb_canon = np.load(R + "/preds_MCMitCNN_tb.npy").astype(np.int8)
res["canon_tb_equals_prepared"] = bool(np.array_equal(tb_canon, Bte))

# late window frame for destinations
Atr_l, Ate_l = wmean("train", "feat", 12, 20), wmean("test", "feat", 12, 20)
fr_l = fit_frames(Atr_l, Btr)
zl, tl = apply(Ate_l, Bte, fr_l)
del Atr_l, Ate_l
late_state = np.where(zl > 3, 2, np.where(tl < 0.5, 0, 1))   # 0, 1, 2 = L

for wname, (src, b0, b1, T) in WIN.items():
    print("window", wname, flush=True)
    Atr, Ate = wmean("train", src, b0, b1), wmean("test", src, b0, b1)
    fr = fit_frames(Atr, Btr)
    z, t = apply(Ate, Bte, fr)
    ztr, _ = apply(Atr, Btr, fr)
    soft = np.load(f"{S}/cnn_soft_{T}ns.npz")
    prob = soft["prob"].astype(np.float32)
    lab = soft["label"].astype(np.int8)
    W = {"soft_label_equals_prepared": bool(np.array_equal(lab, Bte)),
         "frames": {f"q{q}": {"sep_raw_median": float(np.median([fr[(q, c)][4] for c in range(16)])),
                              "sigma0_median": float(np.median([fr[(q, c)][3] for c in range(16)])),
                              "angle_deg_range": [float(np.degrees(np.arctan2(fr[(q, c)][1][1], fr[(q, c)][1][0])))
                                                  for c in (0, 15)]} for q in range(5)},
         "qubits": {}}
    err = (prob > 0.5).astype(np.int8) != lab
    errc = pb_canon != Bte
    conf = np.abs(prob - 0.5)
    # sweep_soft-style confidence flag, pooled over QUBITS
    pooled = conf[:, QUBITS].ravel()
    order = np.argsort(pooled)
    cflag = {}
    for f in FS:
        k = int(round(f * len(pooled)))
        fl = np.zeros(len(pooled), bool)
        fl[order[:k]] = True
        full = np.zeros_like(conf, dtype=bool)
        full[:, QUBITS] = fl.reshape(-1, len(QUBITS))
        cflag[f] = full
    for q in range(5):
        Q = {}
        pl, mi = z[:, q] > 3, z[:, q] < -3
        for b in (0, 1):
            m = Bte[:, q] == b
            n = int(m.sum())
            npl, nmi = int((pl & m).sum()), int((mi & m).sum())
            mt = Btr[:, q] == b
            Ek = {f"E{k}": float(((z[m, q] > k).sum() - (z[m, q] < -k).sum()) / n) for k in (2, 3, 4, 5)}
            ntr = int(mt.sum())
            e_pl, e_mi = int((err[:, q] & pl & m).sum()), int((err[:, q] & mi & m).sum())
            ne = int((err[:, q] & m).sum())
            ec_pl, ec_mi = int((errc[:, q] & pl & m).sum()), int((errc[:, q] & mi & m).sum())
            nec = int((errc[:, q] & m).sum())
            d = {"n": n, "n_plus": npl, "n_minus": nmi, "E3": (npl - nmi) / n, "E3_ci95": ci_mirror(npl, nmi, n),
                 **Ek,
                 "E3_train": float(((ztr[mt, q] > 3).sum() - (ztr[mt, q] < -3).sum()) / ntr),
                 "soft_errors": ne, "soft_err_plus": e_pl, "soft_err_minus": e_mi,
                 "leaked_share_of_soft_errors": (e_pl - e_mi) / max(ne, 1),
                 "leaked_share_of_soft_errors_ci95": ci_mirror(e_pl, e_mi, max(ne, 1)),
                 "misread_rate_of_L": (e_pl - e_mi) / max(npl - nmi, 1),
                 "canon_errors": nec, "leaked_share_of_canon_errors": (ec_pl - ec_mi) / max(nec, 1),
                 "leaked_share_of_canon_errors_ci95": ci_mirror(ec_pl, ec_mi, max(nec, 1)),
                 "p1_median_plus": float(np.median(prob[pl & m, q])) if npl else None,
                 "p1_median_minus": float(np.median(prob[mi & m, q])) if nmi else None,
                 "conf_median_plus": float(np.median(conf[pl & m, q])) if npl else None,
                 "conf_median_all": float(np.median(conf[m, q]))}
            if q in QUBITS:
                d["L_in_conf_flag"] = {str(f): ((cflag[f][:, q] & pl & m).sum() - (cflag[f][:, q] & mi & m).sum())
                                       / max(npl - nmi, 1) for f in FS}
                d["conf_flag_rate"] = {str(f): float(cflag[f][m, q].mean()) for f in FS}
            if wname == "1us":
                dest = {}
                for nm, sel in (("plus", pl & m), ("minus", mi & m), ("core", (np.abs(z[:, q]) < 1) & m)):
                    cnt = np.bincount(late_state[sel, q], minlength=3)
                    dest[nm] = cnt.tolist()
                ex = np.array(dest["plus"]) - np.array(dest["minus"])
                dest["L_excess_late_fraction_0_1_L"] = (ex / max(ex.sum(), 1)).tolist()
                dest["core_late_fraction_0_1_L"] = (np.array(dest["core"]) / max(sum(dest["core"]), 1)).tolist()
                d["late_destination"] = dest
            Q[f"prep{b}"] = d
        n = Bte.shape[0]
        npl, nmi = int(pl.sum()), int(mi.sum())
        ne = int(err[:, q].sum())
        e_pl, e_mi = int((err[:, q] & pl).sum()), int((err[:, q] & mi).sum())
        nec = int(errc[:, q].sum())
        ec_pl, ec_mi = int((errc[:, q] & pl).sum()), int((errc[:, q] & mi).sum())
        Q["overall"] = {"E3": (npl - nmi) / n, "E3_ci95": ci_mirror(npl, nmi, n),
                        "soft_err_rate": ne / n, "canon_err_rate": nec / n,
                        "leaked_share_of_soft_errors": (e_pl - e_mi) / ne,
                        "leaked_share_of_soft_errors_ci95": ci_mirror(e_pl, e_mi, ne),
                        "leaked_share_of_canon_errors": (ec_pl - ec_mi) / nec,
                        "leaked_share_of_canon_errors_ci95": ci_mirror(ec_pl, ec_mi, nec),
                        "Lflag_rate": npl / n, "Lflag_err": e_pl / max(npl, 1),
                        "n_err_soft": ne, "n_err_soft_on_L_mirror": e_pl - e_mi}
        W["qubits"][f"q{q}"] = Q
    # pooled erasure comparison over QUBITS (and without Q4)
    pool = {}
    for name, qs in (("Q1Q3Q4Q5", QUBITS), ("Q1Q3Q5", [0, 2, 4])):
        E = err[:, qs]
        L = z[:, qs] > 3
        C = conf[:, qs]
        tot = E.size
        o = np.argsort(C.ravel())
        Ef = E.ravel()
        def conf_only(frac):
            k = int(round(frac * tot))
            keep = np.ones(tot, bool)
            keep[o[:k]] = False
            return float(Ef[keep].mean()), float(Ef[~keep].mean()) if k else 0.0
        fL = float(L.mean())
        P = {"hard_error": float(E.mean()), "L_flag_fraction": fL, "L_flagged_error": float(E[L].mean()),
             "e_u_L_only": float(E[~L].mean())}
        P["e_u_conf_same_fraction"], P["conf_flagged_error_same_fraction"] = conf_only(fL)
        un = {}
        for f in FS:
            k = int(round(f * tot))
            cf = np.zeros(tot, bool)
            cf[o[:k]] = True
            U = cf | L.ravel()
            fu = float(U.mean())
            eu_conf, ef_conf = conf_only(fu)
            un[str(f)] = {"conf_only_e_u": float(Ef[~cf].mean()), "conf_only_flagged_err": float(Ef[cf].mean()),
                          "union_fraction": fu, "union_e_u": float(Ef[~U].mean()),
                          "union_flagged_err": float(Ef[U].mean()),
                          "conf_only_at_union_fraction_e_u": eu_conf,
                          "conf_only_at_union_fraction_flagged_err": ef_conf,
                          "L_not_in_conf": float((L.ravel() & ~cf).mean())}
        P["union_vs_conf"] = un
        pool[name] = P
    W["pooled_erasure"] = pool
    res["windows"][wname] = W
    json.dump(res, open(O + "/f_cnn_leak_xtab.json", "w"), indent=1, default=float)
print("done", flush=True)
