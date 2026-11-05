"""s2cluster_recompute_perp.py -- re-derive the statistic behind three_level_labels.hdf5.

Re-implements, byte-for-byte in logic, build_three_level_labels.py (May 2026):
  * first MAX_LENGTH=500 samples (= 1 us at 500 MS/s) of every train+test trace
  * trainers/train_HERQULES.demodulate_all_qubits: per-trace DC removal, per-trace
    I/Q amplitude-imbalance correction (Q *= std(I)/std(Q)), mix with cos/sin at
    FREQ_READOUT[q], Butterworth order 3 LPF @ 10 MHz (sosfilt, causal), float32
  * MTV = time-mean of the demodulated (N,500,2) trace (float32 mean -> float64)
  * pooled over train+test: c_s = median(MTV | prepared bit s), u = unit(c1-c0),
    perp = distance from the c0-c1 LINE, sigma_perp = median(perp)/0.6745
  * label 2 iff perp > k * sigma_perp
and then (a) brackets the k that reproduces the stored labels, (b) tabulates
the perp/sigma tail by prepared bit for several k, (c) compares with the
Gaussian floor using per-prepared-state robust perpendicular sigmas, (d) saves
per-shot z_perp and t/sep for other analyses.

Run (graham, oraqle-torch:latest):
  docker run -d --name leak-s2cluster-perp --user $(id -u):$(id -g) \
    -e HDF5_USE_FILE_LOCKING=FALSE \
    -v ~/qubit_readout_klinq/data/five_qubit_data:/data:ro \
    -v ~/oraqle-ics-sims/leakage/s2cluster:/out \
    oraqle-torch:latest python /out/s2cluster_recompute_perp.py
"""
from __future__ import annotations

import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import math
import os
import time
from multiprocessing import Pool

import h5py
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.stats import norm

NQ = 5
MAX_LENGTH = 500
SAMPLING_RATE = 500e6
FREQ_READOUT = -np.array([-64.729e6, -25.366e6, 24.79e6, 70.269e6, 127.282e6])
FILTER_CUTOFF = 10e6
TR = (str(LP.RAW_TRAIN), "X_train", "y_train")
TE = (str(LP.RAW_TEST), "X_test", "y_test")
LAB = str(LP.DATA / "three_level_labels.hdf5")
OUT = LP.outdir(LP.stage("s2cluster"))
CHUNK = 20000
WORKERS = int(os.environ.get("WORKERS", "32"))
PAPER = {q: f"Q{q+1}" for q in range(NQ)}


def demod_mtv(X: np.ndarray) -> np.ndarray:
    """Exact copy of train_HERQULES.demodulate_all_qubits followed by
    build_three_level_labels._mtv. Returns (N, 5, 2) float64."""
    T = X.shape[1]
    dt = 1.0 / SAMPLING_RATE
    DataI = X[:, :, 0].astype(np.float64)
    DataQ = X[:, :, 1].astype(np.float64)
    DataI -= np.mean(DataI, axis=1, keepdims=True)
    DataQ -= np.mean(DataQ, axis=1, keepdims=True)
    corr = np.std(DataI, axis=1, keepdims=True) / (np.std(DataQ, axis=1, keepdims=True) + 1e-30)
    DataQ *= corr
    vTime = np.arange(T) * dt
    sos = butter(3, FILTER_CUTOFF, btype="low", fs=SAMPLING_RATE, output="sos")
    out = np.empty((X.shape[0], NQ, 2), np.float64)
    for i, freq in enumerate(FREQ_READOUT):
        vCos = np.cos(2 * np.pi * vTime * freq)
        vSin = np.sin(2 * np.pi * vTime * freq)
        i_filt = sosfilt(sos, DataI * vCos + DataQ * vSin, axis=1)
        q_filt = sosfilt(sos, DataQ * vCos - DataI * vSin, axis=1)
        d = np.stack((i_filt, q_filt), axis=-1).astype(np.float32)
        out[:, i, :] = d.mean(axis=1).astype(np.float64)
    return out


def work(args):
    path, key, s, e = args
    with h5py.File(path, "r") as hf:
        X = hf[key][s:e, :MAX_LENGTH, :]
    return s, demod_mtv(X)


def run_file(path, key, n):
    out = np.empty((n, NQ, 2), np.float64)
    jobs = [(path, key, s, min(s + CHUNK, n)) for s in range(0, n, CHUNK)]
    with Pool(WORKERS) as p:
        for s, m in p.imap_unordered(work, jobs):
            out[s:s + m.shape[0]] = m
    return out


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [float("nan")] * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [p, max(0.0, c - h), min(1.0, c + h)]


def main():
    t0 = time.time()
    with h5py.File(TR[0], "r") as f:
        ntr = f[TR[1]].shape[0]
        ytr = f[TR[2]][:]
    with h5py.File(TE[0], "r") as f:
        nte = f[TE[1]].shape[0]
        yte = f[TE[2]][:]
    with h5py.File(LAB, "r") as f:
        y3 = np.concatenate([f["y3_train"][:], f["y3_test"][:]])
    mtv = np.concatenate([run_file(TR[0], TR[1], ntr), run_file(TE[0], TE[1], nte)])
    print(f"demod done {time.time()-t0:.0f}s  mtv={mtv.shape}", flush=True)
    y = np.concatenate([ytr, yte]).astype(np.int64)
    split = np.r_[np.zeros(ntr, np.int8), np.ones(nte, np.int8)]  # 0 train, 1 test
    res = {"window": "samples [0,500) = 0-1000 ns at 500 MS/s (paper '1 us')",
           "n_pool": int(len(y)), "per_qubit": {}}
    zs = np.empty((len(y), NQ), np.float32)
    ts = np.empty((len(y), NQ), np.float32)
    for q in range(NQ):
        pb = ((y >> q) & 1).astype(np.float64)
        m0, m1 = pb == 0, pb == 1
        c0 = np.median(mtv[m0, q], axis=0)
        c1 = np.median(mtv[m1, q], axis=0)
        axis = c1 - c0
        sep = float(np.linalg.norm(axis))
        u = axis / (sep or 1e-12)
        rel = mtv[:, q] - c0
        t = rel @ u
        perp = np.linalg.norm(rel - np.outer(t, u), axis=1)
        sigma = float(np.median(perp)) / 0.6745 or 1e-12
        z = perp / sigma
        zs[:, q] = z
        ts[:, q] = t / sep
        lab2 = y3[:, q] == 2
        k_lo = float(z[~lab2].max()) if (~lab2).any() else float("nan")
        k_hi = float(z[lab2].min()) if lab2.any() else float("nan")
        d = {"paper": PAPER[q], "c0": c0.tolist(), "c1": c1.tolist(), "separation": sep,
             "sigma_perp_pooled": sigma, "sep_over_sigma_perp": sep / sigma,
             "k_bracket": {"max_z_unlabelled": k_lo, "min_z_labelled": k_hi,
                           "k_3.0_reproduces": bool(k_lo <= 3.0 < k_hi)}}
        for kk in (3.0,):
            rec = z > kk
            d["agreement_at_k3"] = {"n_label2_stored": int(lab2.sum()),
                                    "n_recomputed": int(rec.sum()),
                                    "n_mismatch": int((rec != lab2).sum())}
        # robust per-prepared-state perpendicular sigma (signed perp)
        perp_dir = np.array([-u[1], u[0]])
        sperp = rel @ perp_dir
        sig_state = {}
        for b, mb in ((0, m0), (1, m1)):
            med = float(np.median(sperp[mb]))
            mad = float(np.median(np.abs(sperp[mb] - med))) / 0.6745
            sig_state[b] = (med, mad)
        d["per_state_signed_perp"] = {f"prep{b}": {"median": v[0], "robust_sigma": v[1],
                                                    "robust_sigma_over_pooled": v[1] / sigma}
                                      for b, v in sig_state.items()}
        tails = {}
        for kk in (2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0):
            row = {}
            for b, mb in ((0, m0), (1, m1)):
                for sname, sv in (("all", None), ("train", 0), ("test", 1)):
                    sel = mb if sv is None else (mb & (split == sv))
                    k_ = int((z[sel] > kk).sum()); n_ = int(sel.sum())
                    row[f"prep{b}_{sname}"] = {"k": k_, "n": n_, "rate_ci95": wilson(k_, n_)}
                # Gaussian null for this prepared cluster using its own robust
                # centre/sigma along the perpendicular: P(|s| > k*sigma_pool)
                med, sg = sig_state[b]
                thr = kk * sigma
                pnull = float(norm.sf((thr - med) / sg) + norm.cdf((-thr - med) / sg))
                row[f"prep{b}_gauss_null_own_sigma"] = pnull
                row[f"prep{b}_excess_over_null"] = row[f"prep{b}_all"]["rate_ci95"][0] - pnull
            row["gauss_null_pooled_sigma_two_sided"] = float(2 * norm.sf(kk))
            tails[str(kk)] = row
        d["tails"] = tails
        # where the labelled shots sit
        if lab2.any():
            for b, mb in ((0, m0), (1, m1)):
                sel = lab2 & mb
                if sel.any():
                    d[f"label2_prep{b}_t_over_sep_pct"] = np.percentile(t[sel] / sep, [5, 25, 50, 75, 95]).tolist()
                    d[f"label2_prep{b}_z_pct"] = np.percentile(z[sel], [5, 25, 50, 75, 95]).tolist()
                    d[f"label2_prep{b}_median_mtv"] = np.median(mtv[sel, q], axis=0).tolist()
        d["zperp_pct_pool"] = {str(p): float(np.percentile(z, p)) for p in (50, 90, 99, 99.9, 99.99)}
        res["per_qubit"][f"q{q}"] = d
        print(f"{PAPER[q]}(q{q}) sep/sig={sep/sigma:.2f} k in ({k_lo:.4f},{k_hi:.4f}] "
              f"mismatch@3={d['agreement_at_k3']['n_mismatch']} "
              f"sig0/pool={sig_state[0][1]/sigma:.3f} sig1/pool={sig_state[1][1]/sigma:.3f}", flush=True)
    with open(os.path.join(OUT, "s2cluster_recompute_perp.json"), "w") as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(os.path.join(OUT, "s2cluster_perp_z.npz"), z_perp=zs, t_over_sep=ts,
                        mtv=mtv.astype(np.float32), y=y.astype(np.int8), split=split)
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
