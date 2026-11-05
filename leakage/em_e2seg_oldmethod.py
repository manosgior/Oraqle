"""em_e2seg_oldmethod.py (tag e2seg): reproduce EXACTLY the old error-mode procedure behind tab:error_modes.

Old pipeline (Oraqle/Discriminators/runners/_train_and_analyze_hybridcnn.py + herqules_error_report.py):
  raw (N, L, 2) -> windowed-sinc sparse-FIR demod (helpers/sparse_fir_demod.py: HERQULES IFs, 101-tap Hamming,
  5 MHz, lfilter + group-delay shift, keep every 40th sample -> T_ds = ceil(L/40)) -> (N, 10, T_ds) "physical" demod.
  HybridCNN (m=16) on the z-normed demod -> P(1) per qubit.
  Error categories (exclusive priority t1 > amp > near > other):
    near : P(correct) < 0.60                                   (true for EVERY error since P(correct) < 0.5)
    amp  : |z| > 3, z = (time-mean |IQ| of the physical demod - TRAIN mean) / TRAIN std, pooled over all states
    t1   : flatten [I0,Q0,I1,Q1,...] (2*T_ds values), Wiener envelope env = mean(x0-x1)/var(x0-x1) from randomly paired
           prepared-|0>/|1> TRAIN traces (RandomState(0)); mf1/mf2 = projections of the first/second half of the
           flattened vector; flag if |mf1| > 0.5 median|mf1|, |mf2| > 0.5 median|mf2| and sign(mf1) != sign(mf2).
Here: (1) validate the reproduction by re-running HybridCNN_len{500,200}.pth and matching the CSVs
(herqules_error_modes_len{500,200}_hybridcnn_*.csv), (2) apply the identical old masks to the canonical MCMit-CNN
predictions (1 us) and to the soft MCMit-CNN models (1000/600/400 ns), (3) store per-shot old masks.
Docker (steve): -v five_qubit_data:/data:ro -v ~/oraqle_models:/models:ro -v ~/oraqle_reports:/reports:ro
                -v f/in:/soft:ro -v em/e2seg:/work
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import csv
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np
import torch
from scipy.signal import firwin, lfilter

sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "vendor"))  # networks/HybridCNN.py (byte copy, was /work/scripts)
from networks.HybridCNN import HybridCNN   # verbatim copy of Oraqle/Discriminators/networks/HybridCNN.py

RAW_TR, RAW_TE = str(LP.RAW_TRAIN), str(LP.RAW_TEST)
O = LP.outdir(LP.stage("em/e2seg") / "out")
FREQ = -np.array([-64.729e6, -25.366e6, 24.79e6, 70.269e6, 127.282e6])
FS, DS, TAPS, CUT = 500e6, 40, 101, 5e6
NEAR_THR, AMP_Z = 0.60, 3.0
NPROC = int(os.environ.get("E2SEG_NPROC", "40"))
GX = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def demod_block(args):
    """Verbatim windowed_sinc_demod on GX['X'][s0:s1] (float32 input -> float64 math -> float32 output)."""
    s0, s1 = args
    X = GX["X"][s0:s1]
    fir = firwin(TAPS, CUT, window="hamming", fs=FS).astype(np.float64)
    shift = (TAPS - 1) // 2
    T = X.shape[1]
    t_axis = np.arange(T) / FS
    T_ds = (T + DS - 1) // DS
    Xc = X[:, :, 0].astype(np.float64) + 1j * X[:, :, 1].astype(np.float64)
    chans = []
    for q in range(5):
        lo = np.exp(-1j * 2 * np.pi * FREQ[q] * t_axis)
        filt = lfilter(fir, [1.0], Xc * lo, axis=-1)
        yc = np.zeros_like(filt)
        if T - shift > 0:
            yc[:, :T - shift] = filt[:, shift:]
        chans.append(yc[:, ::DS][:, :T_ds])
    st = np.stack(chans, axis=2)
    out = np.stack([st.real, st.imag], axis=-1).astype(np.float32)          # (n, T_ds, 5, 2)
    n = out.shape[0]
    return out.reshape(n, T_ds, 10).transpose(0, 2, 1).astype(np.float32)   # (n, 10, T_ds)


def build(X):
    GX["X"] = X
    blocks = [(s, min(s + 20000, len(X))) for s in range(0, len(X), 20000)]
    with Pool(NPROC) as p:
        parts = p.map(demod_block, blocks)
    GX.pop("X")
    return np.concatenate(parts, 0)


def bitsf(y):
    return np.stack([(y >> q) & 1 for q in range(5)], axis=1).astype(np.float32)


def masks(Xtr_phys, Xte_phys, Ytr):
    T_ds = Xtr_phys.shape[2]
    F = 2 * T_ds
    out = {}

    def flat(Xp, q):
        I, Q = Xp[:, 2 * q, :], Xp[:, 2 * q + 1, :]
        return np.stack([I, Q], axis=2).reshape(Xp.shape[0], F)

    for q in range(5):
        ampte = np.linalg.norm(np.stack([Xte_phys[:, 2 * q, :], Xte_phys[:, 2 * q + 1, :]], -1), axis=-1).mean(1)
        amptr = np.linalg.norm(np.stack([Xtr_phys[:, 2 * q, :], Xtr_phys[:, 2 * q + 1, :]], -1), axis=-1).mean(1)
        z = (ampte - amptr.mean()) / (amptr.std() + 1e-12)
        ftr, fte = flat(Xtr_phys, q), flat(Xte_phys, q)
        yq = Ytr[:, q].astype(np.int64)
        zz, oo = ftr[yq == 0], ftr[yq == 1]
        n = min(len(zz), len(oo))
        rng = np.random.RandomState(0)
        zz = zz[rng.choice(len(zz), n, replace=False)]
        oo = oo[rng.choice(len(oo), n, replace=False)]
        env = np.mean(zz - oo, axis=0) / (np.var(zz - oo, axis=0) + 1e-12)
        h = F // 2
        mf1, mf2 = fte[:, :h] @ env[:h], fte[:, h:] @ env[h:]
        sig = (np.abs(mf1) > 0.5 * np.median(np.abs(mf1))) & (np.abs(mf2) > 0.5 * np.median(np.abs(mf2)))
        out[q] = {"amp": np.abs(z) > AMP_Z, "z": z.astype(np.float32), "t1": sig & (np.sign(mf1) != np.sign(mf2)),
                  "mf1": mf1.astype(np.float32), "mf2": mf2.astype(np.float32), "sig": sig}
    return out


def categorise(M, prob_or_pred, Yte, hard):
    rows = []
    for q in range(5):
        tq = Yte[:, q].astype(np.int64)
        if hard:
            pq = prob_or_pred[:, q].astype(np.int64)
            conf = (pq == tq).astype(np.float64)      # P(correct) of a hard decision: 1 or 0
        else:
            P = prob_or_pred[:, q].astype(np.float64)
            pq = (P > 0.5).astype(np.int64)
            conf = np.where(tq == 1, P, 1.0 - P)
        err = pq != tq
        near = conf < NEAR_THR
        t1, amp = M[q]["t1"], M[q]["amp"]
        e = np.where(err)[0]
        c_t1 = t1[e]
        c_amp = amp[e] & ~c_t1
        c_near = near[e] & ~c_t1 & ~c_amp
        n = len(tq)
        r = {"qubit": q, "n_test": n, "n_error": int(err.sum()), "err_rate": float(err.mean()),
             "excl_n_t1_pattern": int(c_t1.sum()), "excl_n_amp_outlier": int(c_amp.sum()),
             "excl_n_near_boundary": int(c_near.sum()), "excl_n_other": int((~c_t1 & ~c_amp & ~c_near).sum()),
             "any_n_t1_pattern": int(t1[e].sum()), "any_n_amp_outlier": int(amp[e].sum()),
             "any_n_near_boundary": int(near[e].sum())}
        for b in (0, 1):
            eb = e[tq[e] == b]
            r[f"prep{b}"] = {"n_error": int(len(eb)), "t1": int(t1[eb].sum()), "amp": int(amp[eb].sum()),
                             "amp_z_pos": int((M[q]["z"][eb] > AMP_Z).sum()), "amp_z_neg": int((M[q]["z"][eb] < -AMP_Z).sum())}
        rows.append(r)
    return rows


def main():
    t0 = time.time()
    res = {"note": __doc__, "L": {}}
    pb = np.load(str(LP.REPORTS / "preds_MCMitCNN_pb.npy")).astype(np.int8)
    soft = {T: np.load(f"{LP.SOFT}/cnn_soft_{T}ns.npz")["prob"].astype(np.float32) for T in (1000, 600, 400)}
    with h5py.File(RAW_TR, "r") as hf:
        Xtr_all = hf["X_train"][:, :500, :].astype(np.float32)
        ytr = hf["y_train"][:].astype(np.int64)
    with h5py.File(RAW_TE, "r") as hf:
        Xte_all = hf["X_test"][:, :500, :].astype(np.float32)
        yte = hf["y_test"][:].astype(np.int64)
    log("raw loaded", Xtr_all.shape, Xte_all.shape)
    Ytr, Yte = bitsf(ytr), bitsf(yte)
    store = {}
    for L, ns, csvglob, ck, preds in (
            (500, 1000, "herqules_error_modes_len500_hybridcnn_*.csv", "HybridCNN_len500.pth",
             {"canonical_MCMitCNN": (pb, True), "soft1000_MCMitCNN": (soft[1000], False)}),
            (300, 600, None, None, {"soft600_MCMitCNN": (soft[600], False)}),
            (200, 400, "herqules_error_modes_len200_hybridcnn_*.csv", "HybridCNN_len200.pth",
             {"soft400_MCMitCNN": (soft[400], False)})):
        Xtr = build(np.ascontiguousarray(Xtr_all[:, :L, :]))
        Xte = build(np.ascontiguousarray(Xte_all[:, :L, :]))
        log("demod", L, Xtr.shape, Xte.shape)
        M = masks(Xtr, Xte, Ytr)
        R_ = {"T_ds": int(Xtr.shape[2]), "window_ns": ns}
        # diagnostics: why the T1 test is blind on some qubits (no offset is subtracted before the sign test)
        R_["diag"] = {}
        for q in range(5):
            dq = {}
            for b in (0, 1):
                mb = Yte[:, q] == b
                dq[f"prep{b}"] = {"frac_mf1_pos": float((M[q]["mf1"][mb] > 0).mean()), "frac_mf2_pos": float((M[q]["mf2"][mb] > 0).mean()),
                                  "median_mf1": float(np.median(M[q]["mf1"][mb])), "median_mf2": float(np.median(M[q]["mf2"][mb])),
                                  "t1_rate_all_shots": float(M[q]["t1"][mb].mean()), "amp_rate_all_shots": float(M[q]["amp"][mb].mean()),
                                  "amp_zpos_rate": float((M[q]["z"][mb] > AMP_Z).mean()), "amp_zneg_rate": float((M[q]["z"][mb] < -AMP_Z).mean())}
            R_["diag"][f"q{q}"] = dq
        if ck is not None:
            mu = Xtr.mean(axis=(0, 2), keepdims=True)
            sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-10
            Xn = (Xte - mu) / sd
            st = torch.load(f"{LP.MODELS}/{ck}", map_location="cpu")
            m_param = int(st["conv1.weight"].shape[0])
            model = HybridCNN(n_channels=10, m_param=m_param, num_qubits=5, n_levels=2)
            model.load_state_dict(st)
            model.eval()
            torch.set_num_threads(16)
            Ps = []
            with torch.no_grad():
                Xt = torch.tensor(Xn, dtype=torch.float32)
                for s in range(0, len(Xt), 4096):
                    Ps.append(torch.sigmoid(model(Xt[s:s + 4096])).numpy())
            P = np.concatenate(Ps, 0)
            rows = categorise(M, P, Yte, hard=False)
            f = sorted(glob.glob(f"{LP.REPORTS}/{csvglob}"))[-1]
            old = list(csv.DictReader(open(f)))
            cmp_ = []
            for r, o in zip(rows, old):
                cmp_.append({k: [r[k], int(o[k])] for k in ("n_error", "excl_n_t1_pattern", "excl_n_amp_outlier",
                                                           "excl_n_near_boundary", "excl_n_other", "any_n_t1_pattern",
                                                           "any_n_amp_outlier", "any_n_near_boundary")})
            R_["hybridcnn_reproduced"] = rows
            R_["hybridcnn_vs_csv"] = {"csv": os.path.basename(f), "m_param": m_param, "per_qubit_[ours,csv]": cmp_,
                                      "exact_match": all(v[0] == v[1] for c in cmp_ for v in c.values())}
            log("hybrid", L, "exact" if R_["hybridcnn_vs_csv"]["exact_match"] else "MISMATCH")
        R_["mcmit"] = {}
        for pname, (pr, hard) in preds.items():
            R_["mcmit"][pname] = categorise(M, pr, Yte, hard)
        res["L"][str(L)] = R_
        store[L] = np.stack([np.stack([M[q]["t1"], M[q]["amp"]], 1) for q in range(5)], 1)   # (N, 5, 2)
        json.dump(res, open(f"{O}/em_e2seg_oldmethod.json", "w"), indent=1, default=float)
    np.savez_compressed(f"{O}/em_e2seg_oldmethod_masks.npz", **{f"L{L}_t1_amp": np.packbits(v, axis=0) for L, v in store.items()},
                        n=len(yte))
    log("all done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
