"""
three_level_common.py
=====================
Shared utilities for the 3-level (qutrit |0>/|1>/|2>) readout benchmark. Every
survey model reuses these so the 2-level vs 3-level comparison is consistent:

  * load_raw_and_labels  - raw multiplexed IQ traces + the row-aligned ternary
                           labels produced by build_three_level_labels.py.
  * demodulate_all_qubits- per-qubit demodulation (re-exported from
                           train_HERQULES, the same routine the labeler used).
  * per_qubit_fidelity   - per-qubit ternary accuracy (matches the codebase's
                           2-level "fidelity" = classification accuracy).
  * f5q                  - geometric mean over the 5 per-qubit fidelities, the
                           same F_5Q the 2-level evaluate_truncated reports.

Metric definitions are deliberately IDENTICAL to the 2-level harness (per-qubit
accuracy, gmean over qubits) - only the number of classes per qubit changes
(3 instead of 2) - so the headline table is apples-to-apples.

The ternary labels are aligned ROW-FOR-ROW with the FULL raw X_train/X_test.
When we subsample we apply the SAME index set to both X and y3.
"""

from __future__ import annotations

import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import os
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainers.train_HERQULES import demodulate_all_qubits  # noqa: F401 (re-export)

NUM_QUBITS = 5
MAX_LENGTH = 500
NUM_LEVELS = 3
SAMPLE_SEED = 42

RAW_TRAIN_FILE = os.environ.get(
    "HERQULES_TRAIN_FILE", str(RAW_TRAIN))
RAW_TEST_FILE = os.environ.get(
    "HERQULES_TEST_FILE", str(RAW_TEST))
# Default location of the ternary labels HDF5 (produced by
# build_three_level_labels.py). Override per-call with --labels or globally
# via the THREE_LEVEL_LABELS env var.
LABELS_FILE = os.environ.get(
    "THREE_LEVEL_LABELS", str(THREE_LEVEL_LABELS))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_raw(path, key, trace_length, max_samples):
    """Load X_{key}[:, :trace_length, :] plus the row indices kept (for aligning
    the ternary labels). Subsampling uses SAMPLE_SEED so X and y3 stay in step."""
    with h5py.File(path, "r") as hf:
        total = hf[f"X_{key}"].shape[0]
        if max_samples is not None and max_samples < total:
            rng = np.random.RandomState(SAMPLE_SEED)
            idx = np.sort(rng.choice(total, size=max_samples, replace=False))
            X = hf[f"X_{key}"][idx, :trace_length, :]
        else:
            idx = np.arange(total)
            X = hf[f"X_{key}"][:, :trace_length, :]
    return X.astype(np.float32), idx


def load_raw_and_labels(labels_path, trace_length=MAX_LENGTH,
                        max_train=None, max_test=None, with_2level=False):
    """Raw multiplexed IQ traces + aligned ternary labels.

    Returns (Xtr, y3tr, Xte, y3te) where X* are (N, trace_length, 2) float32 and
    y3* are (N, 5) int in {0,1,2}, taken from the build_three_level_labels.py
    output (which is row-aligned with the full raw arrays).

    If with_2level=True, also returns the ORIGINAL packed 2-level labels
    (y2tr, y2te), inserted after each y3: (Xtr, y3tr, y2tr, Xte, y3te, y2te).
    HERQULES needs these to build its binary (|0> vs |1>) MF/RMF envelopes.
    """
    with h5py.File(labels_path, "r") as f:
        y3tr_full = f["y3_train"][:]
        y3te_full = f["y3_test"][:]
        y2tr_full = f["y_train"][:] if with_2level else None
        y2te_full = f["y_test"][:] if with_2level else None
    Xtr, idx_tr = _load_raw(RAW_TRAIN_FILE, "train", trace_length, max_train)
    Xte, idx_te = _load_raw(RAW_TEST_FILE, "test", trace_length, max_test)
    if y3tr_full.shape[0] < idx_tr.max() + 1 or y3te_full.shape[0] < idx_te.max() + 1:
        raise ValueError(
            "Label/raw row-count mismatch: the labels file was not built from "
            f"these raw files. y3_train rows={y3tr_full.shape[0]} vs raw needs "
            f"{idx_tr.max() + 1}. Rebuild with build_three_level_labels.py.")
    y3tr, y3te = y3tr_full[idx_tr], y3te_full[idx_te]
    leak_tr = (y3tr == 2).mean(axis=0)
    logger.info(f"Loaded 3-level data: Xtr={Xtr.shape} Xte={Xte.shape} "
                f"per-qubit |2> frac (train)={np.round(leak_tr * 100, 3)}%")
    if with_2level:
        return Xtr, y3tr, y2tr_full[idx_tr], Xte, y3te, y2te_full[idx_te]
    return Xtr, y3tr, Xte, y3te


def per_qubit_fidelity(y_true, y_pred):
    """Per-qubit ternary accuracy. y_true/y_pred: (N, 5) int in {0,1,2}.
    Returns a list of 5 accuracies in [0, 1] (same definition as the 2-level
    harness, just over 3 classes)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return [float(np.mean(y_pred[:, q] == y_true[:, q])) for q in range(NUM_QUBITS)]


def per_qubit_balanced_fidelity(y_true, y_pred):
    """Per-qubit balanced (macro-recall) ternary accuracy: mean over the 3
    prepared states of P(correct | state). Reported alongside plain accuracy
    because |2> is rare (<5%), so plain accuracy can look high while the model
    never recovers |2>. Returns a list of 5 values in [0, 1]."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = []
    for q in range(NUM_QUBITS):
        recalls = []
        for s in range(NUM_LEVELS):
            m = y_true[:, q] == s
            if m.any():
                recalls.append(float(np.mean(y_pred[m, q] == s)))
        out.append(float(np.mean(recalls)) if recalls else float("nan"))
    return out


def per_qubit_leakage_recall(y_true, y_pred):
    """Per-qubit |2> recall = P(predict |2> | truly |2>). THE discriminating
    quantity for the survey: hand-built binary matched filters (HERQULES) are
    structurally blind to the off-axis |2>, so this is ~0 for them, while
    trace-based models recover it. NaN for a qubit with no true |2>."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = []
    for q in range(NUM_QUBITS):
        m = y_true[:, q] == 2
        out.append(float(np.mean(y_pred[m, q] == 2)) if m.any() else float("nan"))
    return out


def f5q(per_qubit):
    """Geometric mean over the 5 per-qubit fidelities (the F_5Q used at 2-level)."""
    a = np.clip(np.asarray(per_qubit, dtype=np.float64), 1e-9, 1.0)
    return float(np.exp(np.mean(np.log(a))))


def confusion_per_qubit(y_true, y_pred):
    """3x3 confusion matrix per qubit (rows=true, cols=pred). Returns
    (5, 3, 3) int array. Useful to see HOW a model fails on |2> (e.g. calls
    every |2> a |1>)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cms = np.zeros((NUM_QUBITS, NUM_LEVELS, NUM_LEVELS), dtype=np.int64)
    for q in range(NUM_QUBITS):
        for t in range(NUM_LEVELS):
            for p in range(NUM_LEVELS):
                cms[q, t, p] = int(np.sum((y_true[:, q] == t) & (y_pred[:, q] == p)))
    return cms


# --------------------------------------------------------------------------- #
# Base-3 joint encoding (for the joint 3^5=243-class models: FNN/Transformer/  #
# HERQULES). The 2-level harness used base-2 ((pred >> q) & 1); this is the    #
# direct ternary analog.                                                       #
# --------------------------------------------------------------------------- #
_POW3 = (NUM_LEVELS ** np.arange(NUM_QUBITS)).astype(np.int64)  # [1,3,9,27,81]


def encode_base3(y3):
    """(N, 5) ternary -> (N,) joint class in [0, 243)."""
    return (np.asarray(y3, dtype=np.int64) * _POW3).sum(axis=1)


def decode_base3(joint):
    """(N,) joint class in [0, 243) -> (N, 5) ternary."""
    joint = np.asarray(joint, dtype=np.int64)
    return np.stack([(joint // _POW3[q]) % NUM_LEVELS for q in range(NUM_QUBITS)], axis=1)


# --------------------------------------------------------------------------- #
# Generic torch training / inference (reused by every model that fits a plain  #
# multiclass classifier on a feature/representation tensor).                   #
# --------------------------------------------------------------------------- #
def fit_classifier(model, X_train, y_train, X_val, y_val, *, epochs=40,
                   batch_size=512, lr=1e-3, weight_decay=0.0, patience=8,
                   class_weight=None, log_tag=""):
    """CrossEntropy + Adam loop, keeping the best-val-accuracy state.

    X_* may be any shape the model accepts (N, ...); y_* are int (N,) joint
    class labels. class_weight (len = n_classes) up-weights rare classes (|2>).
    Returns the best state_dict (already reloaded into `model`).
    """
    model = model.to(DEVICE)
    Xt = torch.as_tensor(X_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    Xv = torch.as_tensor(X_val, dtype=torch.float32).to(DEVICE)
    yv = torch.as_tensor(y_val, dtype=torch.long).to(DEVICE)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size,
                        shuffle=True, drop_last=False)
    cw = (torch.as_tensor(class_weight, dtype=torch.float32).to(DEVICE)
          if class_weight is not None else None)
    crit = torch.nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc, best_state, since = -1.0, None, 0
    import copy
    for ep in range(epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_acc = (model(Xv).argmax(1) == yv).float().mean().item()
        if val_acc > best_acc:
            best_acc, best_state, since = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
        logger.info(f"  [{log_tag}] epoch {ep+1}/{epochs} loss={loss.item():.4f} "
                    f"val_joint_acc={val_acc:.4f} best={best_acc:.4f}")
        if since >= patience:
            logger.info(f"  [{log_tag}] early stop at epoch {ep+1}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_state


def fit_multihead(model, X_train, Y_train, X_val, Y_val, *, epochs=40,
                  batch_size=512, lr=1e-3, weight_decay=0.0, patience=10,
                  log_tag=""):
    """Per-qubit multi-head training (e.g. the CNN's 5 ternary heads).

    Y_* are (N, n_heads) int label matrices; the model returns (N, n_heads,
    n_classes). Loss = sum over heads of CrossEntropy. Model is selected on the
    mean per-head validation accuracy. Returns the best state_dict (reloaded).
    """
    import copy
    model = model.to(DEVICE)
    Xt = torch.as_tensor(X_train, dtype=torch.float32)
    Yt = torch.as_tensor(Y_train, dtype=torch.long)
    Yv = np.asarray(Y_val)
    loader = DataLoader(TensorDataset(Xt, Yt), batch_size=batch_size, shuffle=True)
    crit = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc, best_state, since = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for Xb, Yb in loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            opt.zero_grad()
            out = model(Xb)  # (B, H, C)
            loss = sum(crit(out[:, q, :], Yb[:, q]) for q in range(out.shape[1]))
            loss.backward()
            opt.step()
        val_pred = predict_logits(model, X_val).argmax(axis=2)  # (Nv, H)
        val_acc = float((val_pred == Yv).mean())
        if val_acc > best_acc:
            best_acc, best_state, since = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
        logger.info(f"  [{log_tag}] epoch {ep+1}/{epochs} loss={loss.item():.4f} "
                    f"val_mean_acc={val_acc:.4f} best={best_acc:.4f}")
        if since >= patience:
            logger.info(f"  [{log_tag}] early stop at epoch {ep+1}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_state


def predict_logits(model, X, batch_size=1024):
    """Run `model` over X (N, ...) and return its outputs as a (N, ...) numpy
    array (concatenated over the batch dim). Caller interprets the shape."""
    model = model.to(DEVICE).eval()
    Xt = torch.as_tensor(X, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(Xt), batch_size):
            xb = Xt[i:i + batch_size].to(DEVICE)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


def make_val_split(n, val_frac=0.1, seed=SAMPLE_SEED):
    """Return (train_idx, val_idx) for an n-row dataset."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_frac))
    return perm[n_val:], perm[:n_val]
