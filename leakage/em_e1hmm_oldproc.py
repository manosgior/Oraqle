#!/usr/bin/env python3
"""em_e1hmm_oldproc.py (tag e1hmm): exact replica of the OLD error-mode procedure behind the paper's
Table tab:error_modes, validated on the HybridCNN it was run on, then applied to the canonical MCMit-CNN.

OLD procedure (Oraqle/Discriminators/runners/_train_and_analyze_hybridcnn.py, aggregated by
herqules_error_report.py into herqules_error_breakdown_hybridcnn.csv):
  front end   : sparse-FIR windowed-sinc demod (helpers/sparse_fir_demod.windowed_sinc_demod; HERQULES IFs,
                101-tap Hamming 5 MHz LPF, zero-phase shift, keep every 40th sample) of raw samples [0, L)
                -> (N, 10, T_ds) "physical" (pre-z-norm) I/Q per qubit, T_ds = ceil(L/40)
  near        : P(correct) < 0.60 (NEAR_THR). For a 0.5-threshold binary classifier EVERY error has
                P(correct) < 0.5, so this is true for every error by construction.
  amp outlier : |z| > 3 where z = (time-mean |IQ| - mean_train) / std_train, train stats pooled over ALL states.
                Reported as "Excitation" by herqules_error_report.py and as "Leakage" in the paper table.
  T1 pattern  : Wiener matched-filter envelope env = mean(z - o) / var(z - o) over the flattened interleaved
                [I0, Q0, I1, Q1, ...] train vectors (z = prepared 0, o = prepared 1); split the flattened vector
                at F//2 (F = 2 T_ds; the split falls inside time point T_ds//2 for odd T_ds);
                mf1 = first half . env, mf2 = second half . env (NOT centred: no threshold/offset is subtracted);
                "significant" if |mf1| > 0.5 median|mf1| and |mf2| > 0.5 median|mf2| (medians over all test shots);
                T1 = significant & sign(mf1) != sign(mf2). Applied to every error regardless of prepared bit.
  attribution : exclusive priority T1 > amp > near > other ("other" is always empty).

This script: (1) recomputes the front end from the raw HDF5 for L = 500 / 300 / 200 (1 us / 600 ns / 400 ns);
(2) re-infers the saved HybridCNN checkpoints (len500, len200) and re-runs the old categorisation, to check the
replica against the CSV counts; (3) applies the identical categorisation to the canonical MCMit-CNN errors
(preds_MCMitCNN_pb vs tb, 1 us) and to the soft MCMit-CNN models (cnn_soft_{1000,600,400}ns.npz, threshold 0.5);
(4) saves the per-shot old flags for cross-tabulation with the new (HMM / line-frame) attribution.

Docker (graham, oraqle-torch):
  -v ~/qubit_readout_klinq/data/five_qubit_data:/data:ro -v ~/oraqle_models:/models:ro
  -v ~/oraqle_reports:/reports:ro -v ~/oraqle-ics-sims/leakage/f/in:/soft:ro -v .../em/e1hmm:/work
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import argparse
import csv
import glob
import json
import os
import time
from multiprocessing import Pool

import h5py
import numpy as np
from scipy.signal import firwin, lfilter

NUM_QUBITS = 5
SAMPLING_RATE = 500e6
FREQ_READOUT = -np.array([-64.729e6, -25.366e6, 24.79e6, 70.269e6, 127.282e6])
FIR_TAPS, FIR_CUTOFF, DS = 101, 5e6, 40
NEAR_THR, AMP_Z = 0.60, 3.0
TRAIN_FILE, TEST_FILE = str(LP.RAW_TRAIN), str(LP.RAW_TEST)
NAMES = ["Q1", "Q2", "Q3", "Q4", "Q5"]


# ------------------------------------------------------------------ verbatim front end (sparse_fir_demod.py)
def design_sinc_fir(fir_taps=FIR_TAPS, fir_cutoff=FIR_CUTOFF, fs=SAMPLING_RATE):
    return firwin(fir_taps, fir_cutoff, window="hamming", fs=fs).astype(np.float64)


def windowed_sinc_demod(X, downsample=40, fir_taps=FIR_TAPS, fir_cutoff=FIR_CUTOFF, fs=SAMPLING_RATE,
                        freqs=FREQ_READOUT, chunk=20_000):
    fir = design_sinc_fir(fir_taps, fir_cutoff, fs)
    shift = (fir_taps - 1) // 2
    N, T = X.shape[0], X.shape[1]
    t_axis = np.arange(T) / fs
    T_ds = (T + downsample - 1) // downsample
    parts = []
    for s0 in range(0, N, chunk):
        s1 = min(s0 + chunk, N)
        Xc = X[s0:s1, :, 0].astype(np.float64) + 1j * X[s0:s1, :, 1].astype(np.float64)
        chans = []
        for q in range(NUM_QUBITS):
            lo = np.exp(-1j * 2 * np.pi * freqs[q] * t_axis)
            mixed = Xc * lo
            filt_causal = lfilter(fir, [1.0], mixed, axis=-1)
            y_centered = np.zeros_like(filt_causal)
            if T - shift > 0:
                y_centered[:, :T - shift] = filt_causal[:, shift:]
            chans.append(y_centered[:, ::downsample][:, :T_ds])
        stacked = np.stack(chans, axis=2)
        parts.append(np.stack([stacked.real, stacked.imag], axis=-1).astype(np.float32))
    return np.concatenate(parts, axis=0)


def _chunk_job(args):
    path, key, s, e, L = args
    with h5py.File(path, "r") as hf:
        X = hf[f"X_{key}"][s:e, :L, :].astype(np.float32)        # _load_raw casts to float32
    arr = windowed_sinc_demod(X, downsample=DS, fir_taps=FIR_TAPS, fir_cutoff=FIR_CUTOFF)
    N, T_ds = arr.shape[0], arr.shape[1]
    return arr.reshape(N, T_ds, NUM_QUBITS * 2).transpose(0, 2, 1).astype(np.float32)   # (N, 10, T_ds)


def build(path, key, L, procs, chunk=50_000):
    with h5py.File(path, "r") as hf:
        n = hf[f"X_{key}"].shape[0]
        y = hf[f"y_{key}"][:].astype(np.int64)
    jobs = [(path, key, s, min(s + chunk, n), L) for s in range(0, n, chunk)]
    with Pool(procs) as p:
        parts = p.map(_chunk_job, jobs, chunksize=1)
    return np.concatenate(parts, 0), y


# ------------------------------------------------------------------ HybridCNN (verbatim networks/HybridCNN.py)
def hybrid_model(m_param=16):
    import torch.nn as nn
    import torch.nn.functional as F

    class _ResBlock2D(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 1), padding=(1, 0))
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 1), padding=(1, 0))
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.skip = (nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1)) if in_ch != out_ch else nn.Identity())

        def forward(self, x):
            s = self.skip(x)
            y = F.relu(self.bn1(self.conv1(x)))
            y = self.bn2(self.conv2(y))
            return F.relu(y + s)

    class HybridCNN(nn.Module):
        def __init__(self, n_channels=10, m_param=8, num_qubits=5, n_levels=2, strides=(2, 2)):
            super().__init__()
            self.n_levels = n_levels
            self.conv1 = nn.Conv2d(1, m_param, kernel_size=(3, 1), stride=(strides[0], 1), padding=(1, 0))
            self.bn1 = nn.BatchNorm2d(m_param)
            self.conv2 = nn.Conv2d(m_param, m_param, kernel_size=(3, 1), stride=(strides[1], 1), padding=(1, 0))
            self.bn2 = nn.BatchNorm2d(m_param)
            self.res = _ResBlock2D(m_param, m_param)
            head_in = m_param * n_channels
            self.heads = nn.ModuleList([nn.Linear(head_in, 1) for _ in range(num_qubits)])

        def forward(self, x):
            import torch
            x = x.transpose(1, 2).unsqueeze(1)
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.res(x)
            x = x.mean(dim=2)
            x = x.reshape(x.shape[0], -1)
            return torch.cat([h(x) for h in self.heads], dim=1)

    return HybridCNN(n_channels=10, m_param=m_param, num_qubits=5, n_levels=2)


# ------------------------------------------------------------------ old categorisation (verbatim logic)
def flat_phys(Xphys, q):
    I = Xphys[:, 2 * q, :]
    Q = Xphys[:, 2 * q + 1, :]
    return np.stack([I, Q], axis=2).reshape(Xphys.shape[0], 2 * Xphys.shape[2])


def mf_envelope(flat, yq):
    z = flat[yq == 0]
    o = flat[yq == 1]
    n = min(len(z), len(o))
    rng = np.random.RandomState(0)
    z = z[rng.choice(len(z), n, replace=False)]
    o = o[rng.choice(len(o), n, replace=False)]
    return np.mean(z - o, axis=0) / (np.var(z - o, axis=0) + 1e-12)


def old_flags(Xtr_phys, Xte_phys, Ytr, Yte):
    """per qubit: t1_mask, amp_mask (all test shots) + diagnostics."""
    F_ = 2 * Xtr_phys.shape[2]
    out = {}
    for q in range(NUM_QUBITS):
        ampte = np.linalg.norm(np.stack([Xte_phys[:, 2 * q, :], Xte_phys[:, 2 * q + 1, :]], -1), axis=-1).mean(1)
        amptr = np.linalg.norm(np.stack([Xtr_phys[:, 2 * q, :], Xtr_phys[:, 2 * q + 1, :]], -1), axis=-1).mean(1)
        z = (ampte - amptr.mean()) / (amptr.std() + 1e-12)
        amp_mask = np.abs(z) > AMP_Z
        ftr = flat_phys(Xtr_phys, q)
        fte = flat_phys(Xte_phys, q)
        env = mf_envelope(ftr, Ytr[:, q])
        h = F_ // 2
        mf1 = fte[:, :h] @ env[:h]
        mf2 = fte[:, h:] @ env[h:]
        sig = (np.abs(mf1) > 0.5 * np.median(np.abs(mf1))) & (np.abs(mf2) > 0.5 * np.median(np.abs(mf2)))
        t1_mask = sig & (np.sign(mf1) != np.sign(mf2))
        diag = {}
        for b in (0, 1):
            m = Yte[:, q] == b
            diag[f"prep{b}"] = {"mf1_median": float(np.median(mf1[m])), "mf2_median": float(np.median(mf2[m])),
                                "frac_mf1_pos": float((mf1[m] > 0).mean()), "frac_mf2_pos": float((mf2[m] > 0).mean()),
                                "t1_flag_rate": float(t1_mask[m].mean()), "amp_flag_rate": float(amp_mask[m].mean()),
                                "z_amp_median": float(np.median(z[m]))}
        out[q] = {"t1": t1_mask, "amp": amp_mask, "z": z.astype(np.float32), "mf1": mf1.astype(np.float32),
                  "mf2": mf2.astype(np.float32), "diag": diag, "split_index_h": int(h), "F": int(F_)}
    return out


def categorise(flags, pred, true, p1=None):
    """exclusive t1 > amp > near > other on the errors of (pred vs true). p1 = P(qubit=1) or None (hard preds)."""
    rows = {}
    for q in range(NUM_QUBITS):
        is_err = pred[:, q] != true[:, q]
        err_idx = np.where(is_err)[0]
        if p1 is None:
            near_mask = is_err.copy()          # P(correct) < 0.5 < 0.6 for every error of a 0.5-threshold classifier
        else:
            conf = np.where(true[:, q] == 1, p1[:, q], 1.0 - p1[:, q])
            near_mask = conf < NEAR_THR
        t1, amp = flags[q]["t1"], flags[q]["amp"]
        c_t1 = t1[err_idx]
        c_amp = amp[err_idx] & ~c_t1
        c_near = near_mask[err_idx] & ~c_t1 & ~c_amp
        n_e = len(err_idx)
        n = len(is_err)
        r = {"n_test": n, "n_error": n_e, "err_rate": n_e / n,
             "excl_n_t1_pattern": int(c_t1.sum()), "excl_n_amp_outlier": int(c_amp.sum()),
             "excl_n_near_boundary": int(c_near.sum()), "excl_n_other": int(n_e - c_t1.sum() - c_amp.sum() - c_near.sum()),
             "any_n_t1_pattern": int(t1[err_idx].sum()), "any_n_amp_outlier": int(amp[err_idx].sum()),
             "any_n_near_boundary": int(near_mask[err_idx].sum())}
        for k in ("t1_pattern", "amp_outlier", "near_boundary"):
            r[f"{k}_share_pct"] = 100.0 * r[f"excl_n_{k}"] / max(n_e, 1)
        for b in (0, 1):
            m = true[err_idx, q] == b
            r[f"prep{b}_n_error"] = int(m.sum())
            r[f"prep{b}_excl_t1"] = int((c_t1 & m).sum())
            r[f"prep{b}_excl_amp"] = int((c_amp & m).sum())
        rows[NAMES[q]] = r
    tot = {k: sum(rows[n_][k] for n_ in NAMES) for k in ("n_error", "excl_n_t1_pattern", "excl_n_amp_outlier",
                                                          "excl_n_near_boundary")}
    for k in ("t1_pattern", "amp_outlier", "near_boundary"):
        tot[f"{k}_share_pct"] = 100.0 * tot[f"excl_n_{k}"] / tot["n_error"]
    rows["pooled_all5"] = tot
    return rows


def bits(y):
    return ((y[:, None] >> np.arange(5)) & 1).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--out", default=str(LP.stage("em/e1hmm") / "oldproc"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torch
    torch.set_num_threads(16)
    t0 = time.time()
    res = {"script": "em_e1hmm_oldproc.py", "doc": __doc__, "lengths": {}}
    pb = np.load(str(LP.REPORTS / "preds_MCMitCNN_pb.npy")).astype(np.int64)
    tb = np.load(str(LP.REPORTS / "preds_MCMitCNN_tb.npy")).astype(np.int64)
    csv_ref = {}
    for f in sorted(glob.glob(f"{LP.REPORTS}/herqules_error_modes_len*_hybridcnn_*.csv")):
        L = int(os.path.basename(f).split("_len")[1].split("_")[0])
        with open(f) as fh:
            csv_ref[L] = {NAMES[int(r["qubit"])]: {k: (int(v) if v.isdigit() else v) for k, v in r.items()}
                          for r in csv.DictReader(fh)}
    for L, soft_T in ((500, 1000), (300, 600), (200, 400)):
        print(f"L={L}: building front end", flush=True)
        Xtr, ytr = build(TRAIN_FILE, "train", L, a.procs)
        Xte, yte = build(TEST_FILE, "test", L, a.procs)
        Ytr, Yte = bits(ytr), bits(yte)
        print(f"L={L}: built {Xtr.shape} {Xte.shape} ({time.time()-t0:.0f}s)", flush=True)
        flags = old_flags(Xtr, Xte, Ytr, Yte)
        R = {"T_ds": int(Xtr.shape[2]), "flag_diagnostics": {NAMES[q]: flags[q]["diag"] for q in range(5)},
             "split_index_h": flags[0]["split_index_h"], "F": flags[0]["F"],
             "last_time_point_all_zero_test": bool(np.all(Xte[:, :, -1] == 0))}
        R["labels_equal_canon_tb"] = bool(np.array_equal(Yte, tb))
        # (2) HybridCNN validation
        ck = f"{LP.MODELS}/HybridCNN_len{L}.pth"
        if os.path.exists(ck):
            mu = Xtr.mean(axis=(0, 2), keepdims=True)
            sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-10
            model = hybrid_model(16)
            model.load_state_dict(torch.load(ck, map_location="cpu"))
            model.eval()
            Ps = []
            with torch.no_grad():
                for s in range(0, len(Xte), 8192):
                    xb = torch.tensor((Xte[s:s + 8192] - mu) / sd, dtype=torch.float32)
                    Ps.append(torch.sigmoid(model(xb)).numpy())
            P = np.concatenate(Ps, 0)
            predH = (P > 0.5).astype(np.int64)
            catH = categorise(flags, predH, Yte, P)
            R["hybridcnn_reinferred"] = catH
            if L in csv_ref:
                R["hybridcnn_csv_reference"] = csv_ref[L]
                R["hybridcnn_replica_vs_csv"] = {
                    n_: {k: [catH[n_][k], csv_ref[L][n_][k]] for k in ("n_error", "excl_n_t1_pattern",
                                                                      "excl_n_amp_outlier", "excl_n_near_boundary")}
                    for n_ in NAMES}
            np.save(f"{a.out}/hybridcnn_pred_len{L}.npy", predH.astype(np.int8))
            print(f"L={L}: HybridCNN replica {R.get('hybridcnn_replica_vs_csv')}", flush=True)
        # (3) MCMit-CNN
        if L == 500:
            R["mcmit_canonical_hard"] = categorise(flags, pb, tb, None)
        sp = np.load(f"{LP.SOFT}/cnn_soft_{soft_T}ns.npz")
        p1 = sp["prob"].astype(np.float64)
        R["mcmit_soft_label_equals_prepared"] = bool(np.array_equal(sp["label"].astype(np.int64), Yte))
        R["mcmit_soft"] = categorise(flags, (p1 > 0.5).astype(np.int64), Yte, p1)
        res["lengths"][str(L)] = R
        np.savez_compressed(f"{a.out}/oldflags_len{L}.npz",
                            **{f"t1_q{q}": flags[q]["t1"] for q in range(5)},
                            **{f"amp_q{q}": flags[q]["amp"] for q in range(5)},
                            **{f"zamp_q{q}": flags[q]["z"] for q in range(5)},
                            **{f"mf1_q{q}": flags[q]["mf1"] for q in range(5)},
                            **{f"mf2_q{q}": flags[q]["mf2"] for q in range(5)})
        json.dump(res, open(f"{a.out}/em_e1hmm_oldproc.json", "w"), indent=1, default=float)
        del Xtr, Xte, flags
    res["runtime_s"] = time.time() - t0
    json.dump(res, open(f"{a.out}/em_e1hmm_oldproc.json", "w"), indent=1, default=float)
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
