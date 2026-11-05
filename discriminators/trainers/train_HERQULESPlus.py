"""
train_HERQULESPlus.py
======================
Train HERQULESPlus (HERQULES Net_rmf backbone + zero-init residual MLP over
multi-class LDA features) with fixed hyperparameters, mirroring
train_HERQULES.py.

Pipeline per trace_length in TRACE_LENGTHS:
  1. Load + demodulate raw IQ traces (cached across lengths; full MAX_LENGTH).
  2. Compute HERQULES MF/RMF envelopes once on the full-length training set.
  3. Per length, build_features at this length (truncated envelopes + traces)
     -> 10-D MF feature vector for train/test.
  4. Concatenate per-qubit demodulated traces at this length, fit sklearn
     LinearDiscriminantAnalysis(shrinkage='auto') -> up to 31 LDA directions
     optimised for joint 32-class discrimination.
  5. Project train/test through LDA -> 31-D feature vector.
  6. Per-feature z-score on both feature sets (train-stats only).
  7. Build HERQULESPlus. If a HERQULES_*len{L}*.pth checkpoint is available,
     warm-start the backbone (and freeze it by default; flip FREEZE_BACKBONE
     to also fine-tune it). Otherwise the backbone trains from scratch.
  8. Train with the same fixed hyperparameters as HERQULES
     (Adam, lr=0.01 step-decay at 30/60/90, batch=512, 100 epochs).
  9. Save best-val-acc checkpoint and a CSV report per length.

Run:
    python -m trainers.train_HERQULESPlus
"""

from __future__ import annotations

import csv
import glob
import os
import re
import sys
import time
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from disc_paths import RAW_TRAIN, RAW_TEST, MODELS_OUT, REPORTS_OUT  # [unify] replaces the old docker mount paths
from networks.HERQULESPlus import HERQULESPlus
from trainers.train_HERQULES import (
    demodulate_all_qubits,
    compute_all_envelopes,
    build_features as herqules_build_features,
    adjust_lr,
)


# ============================================================================
# Configuration (fixed, matches train_HERQULES.py)
# ============================================================================

RAW_TRAIN_FILE = str(RAW_TRAIN)
RAW_TEST_FILE  = str(RAW_TEST)
NUM_QUBITS     = 5
NUM_CLASSES    = 2 ** NUM_QUBITS
LDA_COMPONENTS = NUM_CLASSES - 1

# Extra-feature mode for the residual MLP path:
#   "subwindow" -> NUM_QUBITS * SUBWINDOW_K * 2  per-window MF + RMF scalars
#                  (default; targets short-readout regime by preserving timing)
#   "lda"       -> up to NUM_CLASSES - 1 joint multi-class LDA directions
#   "both"      -> concatenate sub-window MFs and LDA directions
EXTRA_FEATURE_MODE = "subwindow"
SUBWINDOW_K        = 5    # number of sub-windows per (qubit, filter)

MAX_LENGTH    = 500
TRACE_LENGTHS = [100, 200, 300, 400, 500]

# Match HERQULES exactly
LRN_RATE    = 0.01
BATCH_SIZE  = 512
MAX_EPOCHS  = 100
VAL_SPLIT   = 0.2
SAMPLE_SEED = 42

# Residual MLP defaults (no hyperparam search — tiny model, pick reasonable)
RESIDUAL_HIDDEN_DIM   = 64
RESIDUAL_NUM_LAYERS   = 2
RESIDUAL_DROPOUT      = 0.1
# Strong weight decay on the residual MLP keeps its weights from drifting
# away from zero when the extra features carry no marginal signal. This
# turns HERQULESPlus into a strict ">= HERQULES" floor: if sub-window /
# LDA features don't help, the residual stays ~zero and the model just
# reproduces the frozen HERQULES backbone exactly.
RESIDUAL_WEIGHT_DECAY = 1e-2

# Whether to warm-start the backbone from a HERQULES checkpoint and freeze it.
# If True (default) and a checkpoint is found, only the residual MLP trains —
# this is the cleanest "strict improvement over HERQULES" study.
WARMSTART_BACKBONE = True
FREEZE_BACKBONE    = True

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = str(MODELS_OUT)  # [unify]
CSV_DIR = str(REPORTS_OUT)  # [unify]

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CSV_DIR,  exist_ok=True)


# ============================================================================
# Data
# ============================================================================

def load_hdf5_full(filepath: str, is_test: bool):
    key = "test" if is_test else "train"
    with h5py.File(filepath, "r") as hf:
        X = hf[f"X_{key}"][:, :MAX_LENGTH, :]
        y = hf[f"y_{key}"][:]
    logger.info(f"Loaded {key}: X={X.shape} y={y.shape}")
    return X, y


def _concat_demod_at_length(demod: dict, trace_length: int) -> np.ndarray:
    """Per-qubit demod -> truncate -> flatten -> concat across qubits.

    Output shape: (N, NUM_QUBITS * trace_length * 2)
    """
    N = demod[0].shape[0]
    chunks = [
        demod[q][:, :trace_length, :].reshape(N, -1)
        for q in range(NUM_QUBITS)
    ]
    return np.concatenate(chunks, axis=1).astype(np.float32, copy=False)


def _build_lda_features(demod_train, demod_test, y_train, trace_length):
    """Fit 32-class LDA on concatenated demod traces; return (train, test)."""
    X_train_concat = _concat_demod_at_length(demod_train, trace_length)
    X_test_concat  = _concat_demod_at_length(demod_test,  trace_length)

    n_present = int(len(np.unique(y_train)))
    n_components = min(LDA_COMPONENTS, n_present - 1, X_train_concat.shape[1])
    lda = LinearDiscriminantAnalysis(
        n_components=n_components, solver="eigen", shrinkage="auto",
    )
    lda.fit(X_train_concat, y_train)
    return (
        lda.transform(X_train_concat).astype(np.float32, copy=False),
        lda.transform(X_test_concat).astype(np.float32, copy=False),
    )


def _build_subwindow_features(demod, mf_envs, rmf_envs, trace_length, K):
    """Project per-qubit demodulated traces onto K equal-length sub-window
    pieces of the MF/RMF envelopes, producing K * 2 scalars per qubit.

    Generalises HERQULES's `build_features`: K=1 recovers the standard
    10-D MF+RMF vector. Higher K preserves *when* the matched-filter mass
    accumulates -- useful for distinguishing fast vs. slow relaxation that
    the global MF averages over, especially at short readout.

    Output shape: (N, NUM_QUBITS * K * 2), layout
        [MF_q0_w0..MF_q0_w(K-1), MF_q1_w0..., ..., MF_q4_w(K-1),
         RMF_q0_w0..RMF_q0_w(K-1), ..., RMF_q4_w(K-1)].
    """
    N = demod[0].shape[0]
    F = trace_length * 2
    window = F // K
    if window == 0:
        raise ValueError(
            f"trace_length={trace_length} too short for K={K} sub-windows (F={F})"
        )

    mf_out  = np.zeros((N, NUM_QUBITS * K), dtype=np.float32)
    rmf_out = np.zeros((N, NUM_QUBITS * K), dtype=np.float32)

    for q in range(NUM_QUBITS):
        flat    = demod[q][:, :trace_length, :].reshape(N, F).astype(np.float32, copy=False)
        mf_env  = mf_envs[q][:F].astype(np.float32, copy=False)
        rmf_env = rmf_envs[q][:F].astype(np.float32, copy=False)
        for k in range(K):
            s = k * window
            # last sub-window absorbs any remainder so F doesn't have to be
            # divisible by K (e.g., F=200, K=3 -> windows of 66, 66, 68).
            e = F if k == K - 1 else (k + 1) * window
            mf_out[:,  q * K + k] = flat[:, s:e] @ mf_env[s:e]
            rmf_out[:, q * K + k] = flat[:, s:e] @ rmf_env[s:e]

    return np.concatenate([mf_out, rmf_out], axis=1)


def build_extra_features(demod_train, demod_test, y_train, mf_envs, rmf_envs,
                         trace_length):
    """Dispatch on EXTRA_FEATURE_MODE; return (extra_train, extra_test)."""
    if EXTRA_FEATURE_MODE == "lda":
        return _build_lda_features(demod_train, demod_test, y_train, trace_length)

    if EXTRA_FEATURE_MODE == "subwindow":
        sw_train = _build_subwindow_features(
            demod_train, mf_envs, rmf_envs, trace_length, SUBWINDOW_K,
        )
        sw_test = _build_subwindow_features(
            demod_test, mf_envs, rmf_envs, trace_length, SUBWINDOW_K,
        )
        return sw_train, sw_test

    if EXTRA_FEATURE_MODE == "both":
        lda_tr, lda_te = _build_lda_features(
            demod_train, demod_test, y_train, trace_length,
        )
        sw_tr = _build_subwindow_features(
            demod_train, mf_envs, rmf_envs, trace_length, SUBWINDOW_K,
        )
        sw_te = _build_subwindow_features(
            demod_test, mf_envs, rmf_envs, trace_length, SUBWINDOW_K,
        )
        return (
            np.concatenate([sw_tr, lda_tr], axis=1).astype(np.float32, copy=False),
            np.concatenate([sw_te, lda_te], axis=1).astype(np.float32, copy=False),
        )

    raise ValueError(
        f"Unknown EXTRA_FEATURE_MODE={EXTRA_FEATURE_MODE!r}; "
        f"expected 'subwindow', 'lda', or 'both'."
    )


def _find_pretrained_herqules(trace_length: int) -> str | None:
    candidates = [
        os.path.join(SAVE_DIR, f"HERQULES_len{trace_length}.pth"),
        os.path.join(SAVE_DIR, f"HERQULES_best_len{trace_length}.pth"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # Glob is too permissive on its own ("len50" matches "len500.pth"); require
    # len{N} to be followed by a non-digit so the length is anchored exactly.
    pattern = re.compile(rf"_len{trace_length}(?:\D|$)")
    matches = sorted(
        f for f in glob.glob(os.path.join(SAVE_DIR, "HERQULES*.pth"))
        if pattern.search(os.path.basename(f))
    )
    return matches[0] if matches else None


# ============================================================================
# Training
# ============================================================================

def train_one_length(MF_train, MF_test, LDA_train, LDA_test,
                     y_train, y_test, trace_length: int, pretrained_path):
    """Train HERQULESPlus at this trace_length and return (model, best_val_acc, test_acc, per_q)."""

    # Train/val split
    idx_tr, idx_val = train_test_split(
        np.arange(MF_train.shape[0]), test_size=VAL_SPLIT,
        random_state=SAMPLE_SEED, stratify=y_train,
    )

    # MFs are passed RAW. HERQULES Net_rmf is trained on raw build_features
    # output (no normalization); z-scoring here would feed the frozen backbone
    # inputs ~sigma times smaller than it expects and collapse its logits.
    # The residual MLP's input LayerNorm handles the joint-feature scaling.
    # LDA features are z-scored using train-only stats (the backbone never
    # sees them, so this is safe).
    lda_m = LDA_train[idx_tr].mean(0, keepdims=True)
    lda_s = LDA_train[idx_tr].std(0,  keepdims=True) + 1e-10
    LDA_train_n = (LDA_train - lda_m) / lda_s
    LDA_test_n  = (LDA_test  - lda_m) / lda_s

    train_ds = TensorDataset(
        torch.tensor(MF_train[idx_tr],    dtype=torch.float32),
        torch.tensor(LDA_train_n[idx_tr], dtype=torch.float32),
        torch.tensor(y_train[idx_tr],     dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(MF_train[idx_val],    dtype=torch.float32),
        torch.tensor(LDA_train_n[idx_val], dtype=torch.float32),
        torch.tensor(y_train[idx_val],     dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model = HERQULESPlus(
        mf_dim=MF_train.shape[1],
        lda_dim=LDA_train.shape[1],
        num_classes=NUM_CLASSES,
        hidden_dim=RESIDUAL_HIDDEN_DIM,
        num_hidden_layers=RESIDUAL_NUM_LAYERS,
        dropout=RESIDUAL_DROPOUT,
    ).to(DEVICE)

    if WARMSTART_BACKBONE and pretrained_path is not None:
        try:
            model.load_backbone(torch.load(pretrained_path, map_location=DEVICE))
            logger.info(f"  Warm-started backbone from {pretrained_path}")
            if FREEZE_BACKBONE:
                model.freeze_backbone()
                logger.info("  Backbone frozen (only residual MLP trains).")
        except Exception as e:
            logger.warning(f"  Backbone warm-start failed ({e}); training from scratch.")

    criterion = nn.CrossEntropyLoss()
    # AdamW with weight decay only on Linear weights (not LayerNorm / biases).
    # Decay is the regularizer that keeps the residual MLP near zero unless
    # the extra features carry real signal.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = optim.AdamW(
        [
            {"params": decay,    "weight_decay": RESIDUAL_WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=LRN_RATE,
    )

    best_val_acc = -1.0
    best_epoch   = 0
    model_path   = os.path.join(SAVE_DIR, f"HERQULESPlus_best_len{trace_length}.pth")

    for epoch in range(MAX_EPOCHS):
        lr = adjust_lr(LRN_RATE, optimizer, epoch)

        model.train()
        epoch_loss = 0.0
        for mf_b, lda_b, y_b in train_loader:
            mf_b, lda_b, y_b = mf_b.to(DEVICE), lda_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(mf_b, lda_b), y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for mf_b, lda_b, y_b in val_loader:
                mf_b, lda_b, y_b = mf_b.to(DEVICE), lda_b.to(DEVICE), y_b.to(DEVICE)
                preds = model(mf_b, lda_b).argmax(dim=1)
                correct += (preds == y_b).sum().item()
                total   += y_b.size(0)
        val_acc = correct / total

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), model_path)

        if epoch % 10 == 0:
            logger.info(
                f"    epoch {epoch:3d}/{MAX_EPOCHS}  loss={epoch_loss:.4f}  "
                f"val_acc={val_acc*100:.2f}%  lr={lr:.5f}"
            )

    logger.info(f"  Best val_acc={best_val_acc*100:.2f}% at epoch {best_epoch}")

    # Reload best, evaluate on test
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    test_ds = TensorDataset(
        torch.tensor(MF_test,    dtype=torch.float32),
        torch.tensor(LDA_test_n, dtype=torch.float32),
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    with torch.no_grad():
        for mf_b, lda_b in test_loader:
            preds.append(model(mf_b.to(DEVICE), lda_b.to(DEVICE)).argmax(dim=1).cpu().numpy())
    pred_labels = np.concatenate(preds)
    overall_acc = 100.0 * np.mean(pred_labels == y_test)
    per_q = [
        100.0 * np.mean(((pred_labels >> q) & 1) == ((y_test >> q) & 1))
        for q in range(NUM_QUBITS)
    ]
    return model, best_val_acc, overall_acc, per_q, best_epoch, model_path


def save_csv(model, trace_length, best_val_acc, best_epoch,
             overall_acc, per_q, pretrained_path, model_path):
    total_params = sum(p.numel() for p in model.parameters())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(CSV_DIR, f"HERQULESPlus_len{trace_length}_{ts}.csv")
    row = {
        "timestamp":           ts,
        "model_name":          "HERQULESPlus",
        "trace_length":        trace_length,
        "learning_rate":       LRN_RATE,
        "batch_size":          BATCH_SIZE,
        "epochs":              MAX_EPOCHS,
        "best_epoch":          best_epoch,
        "best_val_acc":        f"{best_val_acc*100:.4f}",
        "residual_hidden_dim": RESIDUAL_HIDDEN_DIM,
        "residual_num_layers": RESIDUAL_NUM_LAYERS,
        "residual_dropout":    RESIDUAL_DROPOUT,
        "residual_weight_decay": RESIDUAL_WEIGHT_DECAY,
        "warmstart_backbone":  WARMSTART_BACKBONE,
        "freeze_backbone":     FREEZE_BACKBONE,
        "pretrained_backbone": pretrained_path or "none",
        "extra_feature_mode":  EXTRA_FEATURE_MODE,
        "subwindow_K":         SUBWINDOW_K if EXTRA_FEATURE_MODE in ("subwindow", "both") else "",
        "total_parameters":    total_params,
        "overall_accuracy":    f"{overall_acc:.4f}",
        **{f"qubit_{q}_accuracy": f"{per_q[q]:.4f}" for q in range(NUM_QUBITS)},
        "device":              str(DEVICE),
        "model_path":          model_path,
    }
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        w.writeheader()
        w.writerow(row)
    logger.info(f"  CSV: {filepath}")


# ============================================================================
# Main
# ============================================================================

def run():
    logger.info(
        f"=== HERQULESPlus Training "
        f"(extra={EXTRA_FEATURE_MODE}"
        f"{f', K={SUBWINDOW_K}' if EXTRA_FEATURE_MODE in ('subwindow', 'both') else ''}"
        f", zero-init residual MLP) ==="
    )
    logger.info(
        f"Device: {DEVICE}  |  lr={LRN_RATE}  |  batch={BATCH_SIZE}  "
        f"|  epochs={MAX_EPOCHS}"
    )
    logger.info(
        f"Residual: hidden={RESIDUAL_HIDDEN_DIM} layers={RESIDUAL_NUM_LAYERS} "
        f"dropout={RESIDUAL_DROPOUT}  wd={RESIDUAL_WEIGHT_DECAY}  "
        f"warmstart={WARMSTART_BACKBONE}  freeze={FREEZE_BACKBONE}"
    )

    X_train_raw, y_train = load_hdf5_full(RAW_TRAIN_FILE, is_test=False)
    X_test_raw,  y_test  = load_hdf5_full(RAW_TEST_FILE,  is_test=True)

    logger.info("Demodulating training traces...")
    demod_train = demodulate_all_qubits(X_train_raw)
    logger.info("Demodulating test traces...")
    demod_test  = demodulate_all_qubits(X_test_raw)
    del X_train_raw, X_test_raw

    logger.info("Computing MF/RMF envelopes from full-length training data...")
    np.random.seed(SAMPLE_SEED)
    mf_envs, rmf_envs = compute_all_envelopes(demod_train, y_train)

    summary = []
    for length in TRACE_LENGTHS:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Trace length: {length}  ({length * 2} ns)")
        logger.info("=" * 60)
        t0 = time.perf_counter()

        MF_train = herqules_build_features(demod_train, mf_envs, rmf_envs, length)
        MF_test  = herqules_build_features(demod_test,  mf_envs, rmf_envs, length)

        logger.info(f"  Building extra features (mode={EXTRA_FEATURE_MODE})...")
        LDA_train, LDA_test = build_extra_features(
            demod_train, demod_test, y_train, mf_envs, rmf_envs, length,
        )
        logger.info(f"  Extra features shape: {LDA_train.shape}")

        pretrained_path = _find_pretrained_herqules(length)
        if pretrained_path:
            logger.info(f"  Pretrained HERQULES backbone: {pretrained_path}")

        model, best_val, test_acc, per_q, best_epoch, model_path = train_one_length(
            MF_train, MF_test, LDA_train, LDA_test,
            y_train, y_test, length, pretrained_path,
        )
        logger.info(
            f"  TEST overall={test_acc:.2f}%  "
            f"per_qubit={['%.2f' % v for v in per_q]}"
        )

        save_csv(model, length, best_val, best_epoch, test_acc, per_q,
                 pretrained_path, model_path)

        summary.append({
            "length":      length,
            "best_val":    best_val * 100,
            "test_acc":    test_acc,
            "per_q":       per_q,
            "elapsed_s":   time.perf_counter() - t0,
        })

    logger.info("")
    logger.info("=" * 60)
    logger.info("  SUMMARY")
    logger.info("=" * 60)
    for s in summary:
        logger.info(
            f"  len={s['length']:3d}  val={s['best_val']:.2f}%  "
            f"test={s['test_acc']:.2f}%  "
            f"per_q={['%.2f' % v for v in s['per_q']]}  "
            f"({s['elapsed_s']:.0f}s)"
        )


if __name__ == "__main__":
    run()
