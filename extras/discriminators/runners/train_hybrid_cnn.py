"""
train_hybrid_cnn.py
===================
Train the HybridCNN on the standard 2-level real dataset.

Pipeline:
    raw (N, T, 2) from /data/five_qubit_data/DRaw_C_Tr_v0-001
    -> sparse-FIR windowed-sinc demod (helpers.sparse_fir_demod.windowed_sinc_demod)
    -> z-norm per channel (train stats)
    -> HybridCNN(n_levels=2): (3,1)-kernel Conv2d body, channel-independent
       through the conv body, 5 per-qubit sigmoid heads on a shared embedding
    -> BCEWithLogitsLoss summed over qubits, Adam, early-stop on val accuracy
    -> per-qubit binary test accuracy + geomean fidelity F_5Q

No ternary labels, no leakage harness - just per-qubit binary readout against
the prepared bit ((y_packed >> q) & 1).

Run inside the container, from /app/Discriminators:
    python -m runners.train_hybrid_cnn \
        --length 500 --m-param 8 --downsample 40 \
        --epochs 40 --lr 1e-3 \
        --out-json /app/optimization_reports/hybrid_cnn_2level.json
"""

from __future__ import annotations

import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import argparse
import copy
import json
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.sparse_fir_demod import windowed_sinc_demod
from networks.HybridCNN import HybridCNN

NUM_QUBITS = 5
SAMPLE_SEED = 42

RAW_TRAIN_FILE = os.environ.get(
    "HERQULES_TRAIN_FILE", str(RAW_TRAIN))
RAW_TEST_FILE = os.environ.get(
    "HERQULES_TEST_FILE", str(RAW_TEST))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = str(MODELS_OUT)


def _load_raw(path, key, length, max_samples=None):
    with h5py.File(path, "r") as hf:
        total = hf[f"X_{key}"].shape[0]
        if max_samples and max_samples < total:
            rng = np.random.RandomState(SAMPLE_SEED)
            idx = np.sort(rng.choice(total, size=max_samples, replace=False))
            X = hf[f"X_{key}"][idx, :length, :]
            y = hf[f"y_{key}"][idx]
        else:
            X = hf[f"X_{key}"][:, :length, :]
            y = hf[f"y_{key}"][:]
    return X.astype(np.float32), y.astype(np.int64)


def _build_input(X, downsample, fir_taps, fir_cutoff, chunk=50_000):
    """Raw (N, T, 2) -> (N, 10, T_ds) via sparse-FIR windowed-sinc demod."""
    parts = []
    for s in range(0, len(X), chunk):
        parts.append(windowed_sinc_demod(
            X[s:s + chunk], downsample=downsample,
            fir_taps=fir_taps, fir_cutoff=fir_cutoff))
    arr = np.concatenate(parts, axis=0)                              # (N, T_ds, 5, 2)
    N, T_ds = arr.shape[0], arr.shape[1]
    arr = arr.reshape(N, T_ds, NUM_QUBITS * 2).transpose(0, 2, 1)    # (N, 10, T_ds)
    return arr.astype(np.float32)


def _per_qubit_bits(y_packed):
    """(N,) packed -> (N, 5) float32 bit matrix for BCEWithLogitsLoss."""
    return np.stack(
        [(y_packed >> q) & 1 for q in range(NUM_QUBITS)], axis=1
    ).astype(np.float32)


def _val_split(n, val_frac=0.1, seed=SAMPLE_SEED):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    nv = int(round(n * val_frac))
    return perm[nv:], perm[:nv]


def run(length, m_param, downsample, epochs, lr, batch_size,
        max_train, max_test, out_ckpt, out_json, fir_taps, fir_cutoff,
        patience):
    logger.info(f"Loading raw train ({RAW_TRAIN_FILE}) ...")
    Xtr, ytr = _load_raw(RAW_TRAIN_FILE, "train", length, max_train)
    Xte, yte = _load_raw(RAW_TEST_FILE, "test", length, max_test)
    logger.info(f"  shapes: Xtr={Xtr.shape}  Xte={Xte.shape}")

    logger.info(f"Sparse-FIR demod (downsample={downsample}, taps={fir_taps}, "
                f"cutoff={fir_cutoff/1e6:.1f} MHz) ...")
    Xtr = _build_input(Xtr, downsample, fir_taps, fir_cutoff)
    Xte = _build_input(Xte, downsample, fir_taps, fir_cutoff)
    logger.info(f"  post-demod: Xtr={Xtr.shape}  Xte={Xte.shape}")

    # z-norm per channel from train stats.
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-10
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    Ytr = _per_qubit_bits(ytr)                                       # (N, 5) float
    Yte = _per_qubit_bits(yte)

    tr_idx, val_idx = _val_split(Xtr.shape[0], val_frac=0.1)
    logger.info(f"train={len(tr_idx)}  val={len(val_idx)}  test={Xte.shape[0]}")

    model = HybridCNN(
        n_channels=10, m_param=m_param, num_qubits=NUM_QUBITS, n_levels=2
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"HybridCNN(m={m_param}) params = {n_params}")

    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xtr_t = torch.tensor(Xtr[tr_idx], dtype=torch.float32)
    Ytr_t = torch.tensor(Ytr[tr_idx], dtype=torch.float32)
    Xv_t = torch.tensor(Xtr[val_idx], dtype=torch.float32).to(DEVICE)
    Yv_t = torch.tensor(Ytr[val_idx], dtype=torch.float32).to(DEVICE)
    loader = DataLoader(TensorDataset(Xtr_t, Ytr_t), batch_size=batch_size,
                        shuffle=True, drop_last=False)

    best_val_acc, best_state, since = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        last_loss = float("nan")
        for Xb, Yb in loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(Xb), Yb)
            loss.backward()
            opt.step()
            last_loss = loss.item()
        model.eval()
        with torch.no_grad():
            outs = []
            for s in range(0, len(Xv_t), 1024):
                outs.append(model(Xv_t[s:s + 1024]))
            outs = torch.cat(outs, dim=0)
            pred = (torch.sigmoid(outs) > 0.5).float()
            val_acc = (pred == Yv_t).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc, best_state, since = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
        logger.info(f"  ep {ep+1}/{epochs} loss={last_loss:.4f} "
                    f"val_acc={val_acc:.4f} best={best_val_acc:.4f}")
        if since >= patience:
            logger.info(f"  early stop at ep {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test set eval.
    model.eval()
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    with torch.no_grad():
        outs = []
        for s in range(0, len(Xte_t), 1024):
            outs.append(model(Xte_t[s:s + 1024].to(DEVICE)).cpu())
        outs = torch.cat(outs, dim=0)
        pred = (torch.sigmoid(outs) > 0.5).numpy()
    per_q = [float((pred[:, q] == Yte[:, q]).mean()) for q in range(NUM_QUBITS)]
    gmean = float(np.exp(np.mean(np.log(np.clip(per_q, 1e-9, 1.0)))))

    logger.info("=" * 64)
    logger.info(f"HybridCNN 2-level | per-qubit acc = {['%.4f' % x for x in per_q]}")
    logger.info(f"                  | gmean (F_5Q)  = {gmean:.4f}")
    logger.info(f"                  | params        = {n_params}")
    logger.info("=" * 64)

    if out_ckpt is None:
        out_ckpt = os.path.join(SAVE_DIR, f"HybridCNN_len{length}.pth")
    os.makedirs(os.path.dirname(os.path.abspath(out_ckpt)) or ".", exist_ok=True)
    torch.save(model.state_dict(), out_ckpt)
    logger.info(f"Saved checkpoint -> {out_ckpt}")

    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
        with open(out_json, "w") as f:
            json.dump({
                "model": "HybridCNN", "level": 2, "length": length,
                "m_param": m_param, "downsample": downsample,
                "fir_taps": fir_taps, "fir_cutoff": fir_cutoff,
                "lr": lr, "batch_size": batch_size, "epochs": epochs,
                "n_params": int(n_params),
                "per_qubit_acc": per_q, "gmean_acc": gmean,
                "best_val_acc": float(best_val_acc),
                "n_train": int(len(tr_idx)), "n_test": int(Xte.shape[0]),
                "ckpt": out_ckpt,
            }, f, indent=2)
        logger.info(f"Saved metrics  -> {out_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    p.add_argument("--length", type=int, default=500)
    p.add_argument("--m-param", type=int, default=8, help="HybridCNN filter width.")
    p.add_argument("--downsample", type=int, default=40,
                   help="Sparse-FIR decimation factor (40 = paper's example @ L=500).")
    p.add_argument("--fir-taps", type=int, default=101)
    p.add_argument("--fir-cutoff", type=float, default=5e6,
                   help="LPF cutoff in Hz (5e6 = 5 MHz, default).")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--out-ckpt", default=None,
                   help=f"Default {SAVE_DIR}/HybridCNN_len{{length}}.pth.")
    p.add_argument("--out-json", default=None,
                   help="Optional JSON output with the metrics + config.")
    args = p.parse_args()
    run(args.length, args.m_param, args.downsample, args.epochs, args.lr,
        args.batch_size, args.max_train, args.max_test, args.out_ckpt,
        args.out_json, args.fir_taps, args.fir_cutoff, args.patience)
