"""Train HybridCNN end-to-end (sparse-FIR demod -> (N,10,T_ds) -> HybridCNN)
and run the SAME error-mode categorisation used for HERQULES, in one process.

End-to-end ⇒ self-consistent: data-prep, training and evaluation all share the
identical pipeline, so there is no stored-checkpoint / lost-data provenance risk
(the HybridCNN has no pretrained weights anyway).

Outputs (to /app/optimization_reports -> oraqle_reports):
  HybridCNN_len{L}.pth                                  (checkpoint)
  herqules_error_modes_len{L}_hybridcnn_{ts}.csv        (same schema as the
                                                         HERQULES error CSVs)
Env: HERQULES_TRACE_LENGTH (L), HYBRID_M (default 16),
     HYBRID_DS (downsample, default 40), HYBRID_EPOCHS (default 40).
"""
import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from loguru import logger

from networks.HybridCNN import HybridCNN
import runners.train_hybrid_cnn as TH   # reuse _load_raw/_build_input/_per_qubit_bits/_val_split

NUM_QUBITS = 5
NEAR_THR = 0.60          # P(correct) below this -> near-boundary (matches HERQULES)
AMP_Z = 3.0              # |z(amp)| above this   -> amplitude outlier
FIR_TAPS, FIR_CUTOFF = 101, 5e6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPORTS = (str(REPORTS_OUT))

L = int(os.environ.get("HERQULES_TRACE_LENGTH", "500"))
M = int(os.environ.get("HYBRID_M", "16"))
DS = int(os.environ.get("HYBRID_DS", "40"))
EPOCHS = int(os.environ.get("HYBRID_EPOCHS", "40"))
LR, BATCH, PATIENCE = 1e-3, 512, 8
NS = L * 2               # readout window in ns (500 samples = 1000 ns)


def _acc_per_qubit(P, Ybits):
    pred = (P > 0.5).astype(np.int64)
    return [float(np.mean(pred[:, q] == Ybits[:, q])) for q in range(NUM_QUBITS)]


def _mf_envelope(flat, yq):
    """Wiener MF envelope on flattened (N, 2*T_ds) traces for one qubit."""
    z = flat[yq == 0]
    o = flat[yq == 1]
    n = min(len(z), len(o))
    rng = np.random.RandomState(0)
    z = z[rng.choice(len(z), n, replace=False)]
    o = o[rng.choice(len(o), n, replace=False)]
    return np.mean(z - o, axis=0) / (np.var(z - o, axis=0) + 1e-12)


def main():
    logger.info(f"=== HybridCNN train+analyze | L={L} ({NS} ns) m={M} ds={DS} ===")
    Xtr_raw, ytr = TH._load_raw(TH.RAW_TRAIN_FILE, "train", L)
    Xte_raw, yte = TH._load_raw(TH.RAW_TEST_FILE, "test", L)

    logger.info(f"Sparse-FIR demod (ds={DS}) ...")
    Xtr = TH._build_input(Xtr_raw, DS, FIR_TAPS, FIR_CUTOFF)   # (N,10,T_ds)
    Xte = TH._build_input(Xte_raw, DS, FIR_TAPS, FIR_CUTOFF)
    T_ds = Xtr.shape[2]
    logger.info(f"  Xtr={Xtr.shape}  Xte={Xte.shape}  T_ds={T_ds}")

    # Keep pre-znorm demod for the (physical) error diagnostics; znorm feeds CNN.
    Xtr_phys, Xte_phys = Xtr.copy(), Xte.copy()
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-10
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    Ytr = TH._per_qubit_bits(ytr)
    Yte = TH._per_qubit_bits(yte)
    tr_idx, val_idx = TH._val_split(Xtr.shape[0], val_frac=0.1)

    model = HybridCNN(n_channels=10, m_param=M, num_qubits=NUM_QUBITS, n_levels=2).to(DEVICE)
    nparams = sum(p.numel() for p in model.parameters())
    logger.info(f"HybridCNN(m={M}) params = {nparams}")
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    Xtr_t = torch.tensor(Xtr[tr_idx], dtype=torch.float32)
    Ytr_t = torch.tensor(Ytr[tr_idx], dtype=torch.float32)
    Xv = torch.tensor(Xtr[val_idx], dtype=torch.float32).to(DEVICE)
    Yv = torch.tensor(Ytr[val_idx], dtype=torch.float32).to(DEVICE)
    loader = DataLoader(TensorDataset(Xtr_t, Ytr_t), batch_size=BATCH, shuffle=True)

    best_val, best_state, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        for Xb, Yb in loader:
            opt.zero_grad()
            loss = crit(model(Xb.to(DEVICE)), Yb.to(DEVICE))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xv)).cpu().numpy()
        vacc = float(np.mean((pv > 0.5).astype(np.int64) == Yv.cpu().numpy()))
        if vacc > best_val:
            best_val, bad = vacc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        logger.info(f"  epoch {ep:3d}  val_bitacc={vacc:.4f}  best={best_val:.4f}")
        if bad >= PATIENCE:
            logger.info(f"  early stop @ epoch {ep}")
            break

    model.load_state_dict(best_state)
    ckpt = os.path.join(str(MODELS_OUT),
                        f"HybridCNN_len{L}.pth")
    torch.save(model.state_dict(), ckpt)
    logger.info(f"saved {ckpt}")

    # ---- Inference on test -> per-qubit P(qubit=1) ----
    model.eval()
    Ps = []
    with torch.no_grad():
        Xte_t = torch.tensor(Xte, dtype=torch.float32)
        for s in range(0, len(Xte_t), 4096):
            Ps.append(torch.sigmoid(model(Xte_t[s:s + 4096].to(DEVICE))).cpu().numpy())
    P = np.concatenate(Ps, axis=0)                         # (N,5)
    accs = _acc_per_qubit(P, Yte)
    logger.info("per-qubit test acc: " + " ".join(f"Q{q+1}={accs[q]:.4f}" for q in range(NUM_QUBITS)))

    # ---- error diagnostics on the physical (pre-znorm) demod ----
    F = 2 * T_ds
    # per-qubit flattened (N, 2*T_ds) as [I0,Q0,I1,Q1,...] over time
    def flat_phys(Xphys, q):
        I = Xphys[:, 2 * q, :]
        Q = Xphys[:, 2 * q + 1, :]
        return np.stack([I, Q], axis=2).reshape(Xphys.shape[0], F)

    rows = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for q in range(NUM_QUBITS):
        true_q = Yte[:, q].astype(np.int64)
        pred_q = (P[:, q] > 0.5).astype(np.int64)
        is_err = pred_q != true_q
        err_idx = np.where(is_err)[0]
        conf = np.where(true_q == 1, P[:, q], 1.0 - P[:, q])
        near_mask = conf < NEAR_THR

        # amplitude outlier (physical), z vs train
        ampte = np.linalg.norm(np.stack([Xte_phys[:, 2 * q, :], Xte_phys[:, 2 * q + 1, :]], -1), axis=-1).mean(1)
        amptr = np.linalg.norm(np.stack([Xtr_phys[:, 2 * q, :], Xtr_phys[:, 2 * q + 1, :]], -1), axis=-1).mean(1)
        z = (ampte - amptr.mean()) / (amptr.std() + 1e-12)
        amp_mask = np.abs(z) > AMP_Z

        # T1: half-time MF sign disagreement (env fit on train)
        ftr = flat_phys(Xtr_phys, q)
        fte = flat_phys(Xte_phys, q)
        env = _mf_envelope(ftr, Ytr[:, q].astype(np.int64))
        h = F // 2
        mf1 = fte[:, :h] @ env[:h]
        mf2 = fte[:, h:] @ env[h:]
        sig = (np.abs(mf1) > 0.5 * np.median(np.abs(mf1))) & (np.abs(mf2) > 0.5 * np.median(np.abs(mf2)))
        t1_mask = sig & (np.sign(mf1) != np.sign(mf2))

        # exclusive priority t1 > amp > near > other
        cat = np.full(err_idx.shape, "other", dtype=object)
        c_t1 = t1_mask[err_idx]
        c_amp = amp_mask[err_idx] & ~c_t1
        c_near = near_mask[err_idx] & ~c_t1 & ~c_amp
        cat[c_t1] = "t1"; cat[c_amp] = "amp"; cat[c_near] = "near"
        n = len(true_q)
        rows.append({
            "split": "hybridcnn", "qubit": q, "n_test": n,
            "n_correct": int((~is_err).sum()), "n_error": int(is_err.sum()),
            "accuracy_pct": f"{100.0*accs[q]:.4f}",
            "excl_n_t1_pattern": int((cat == "t1").sum()),
            "excl_n_amp_outlier": int((cat == "amp").sum()),
            "excl_n_near_boundary": int((cat == "near").sum()),
            "excl_n_other": int((cat == "other").sum()),
            "any_n_t1_pattern": int(t1_mask[err_idx].sum()),
            "any_n_amp_outlier": int(amp_mask[err_idx].sum()),
            "any_n_near_boundary": int(near_mask[err_idx].sum()),
        })
        logger.info(f"qubit {q} acc={100*accs[q]:.3f}% err={int(is_err.sum())} "
                    f"| t1={rows[-1]['excl_n_t1_pattern']} amp={rows[-1]['excl_n_amp_outlier']} "
                    f"near={rows[-1]['excl_n_near_boundary']} other={rows[-1]['excl_n_other']}")

    out = os.path.join(REPORTS, f"herqules_error_modes_len{L}_hybridcnn_{ts}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
