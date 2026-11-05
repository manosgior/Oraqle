"""
error_analysis_sweep.py
=======================
Per-trace error-mode categorization for HERQULES / CNN / Transformer across a
sweep of trace lengths, on the SAME test traces. Focused by default on the
high-fidelity qubits (q0, q4) where the categorization is most reliable.

Categories (exclusive, priority order):

    leakage       y3[:, q] == 2 (ternary-label ground truth from
                  build_three_level_labels.py; HERQULES is blind to |2>).
    t1_pattern    first- and second-half MF scores disagree in sign AND both are
                  significantly above the half-trace MF noise floor: a structural
                  decay signature mid-readout (the RMF is built for this).
    near_boundary the model's confidence P(true class | trace) < 0.60: the
                  Q-function noise tail. Length-dependent (shorter readout =>
                  smaller deflection => fatter tail => more of these).
    state_prep    confident wrong (>= 0.60), not T1, not leakage: prep failed
                  and the qubit was actually in the OTHER state, OR cross-talk.
    other         residual (rare; usually small).

Length-specific checkpoints required: HERQULES_len{L}.pth, CNN_len{L}*.pth,
Transformer_len{L}*.pth under ./saved_models (or /app/saved_models in-container).
(model, L) pairs with no matching checkpoint are skipped with a warning.

The 2-level labels (y_train, y_test packed) come from the labels file written by
build_three_level_labels.py, which preserves them aligned with y3. MF envelopes
are computed from PURE prepared train traces (no |2> on any qubit) per
HERQULES's design intent.

Run inside the container, from /app/Discriminators:
    python -m runners.error_analysis_sweep \
        --labels /data/five_qubit_data/three_level_labels.hdf5 \
        --lengths 500 400 300 200 100 \
        --qubits 0 4 \
        --models herqules cnn \
        --out /app/optimization_reports/error_analysis_sweep.csv
"""

from __future__ import annotations

import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import argparse
import csv
import glob
import os
import re
import sys
from datetime import datetime

import h5py
import numpy as np
import torch
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.CNN import CNN
from networks.HERQULES import Net_rmf
from networks.Transformer import QubitClassifierTransformer
from runners.synth_to_cnn import _demod_downsample
from runners.three_level_common import LABELS_FILE
from trainers.train_HERQULES import (
    build_features, compute_all_envelopes, demodulate_all_qubits,
)

NUM_QUBITS = 5
NUM_CLASSES_JOINT = 2 ** NUM_QUBITS  # 32 (2-level binary task)
MAX_LENGTH = 500
SAMPLE_SEED = 42
NEAR_BOUNDARY_CONF = 0.60

SAVE_DIR = str(MODELS_OUT)
CSV_DIR = str(REPORTS_OUT)

RAW_TRAIN_FILE = os.environ.get(
    "HERQULES_TRAIN_FILE", str(RAW_TRAIN))
RAW_TEST_FILE = os.environ.get(
    "HERQULES_TEST_FILE", str(RAW_TEST))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Data + envelope setup (once)                                                #
# --------------------------------------------------------------------------- #

def _load_raw(path, key, max_samples=None):
    with h5py.File(path, "r") as hf:
        total = hf[f"X_{key}"].shape[0]
        if max_samples and max_samples < total:
            rng = np.random.RandomState(SAMPLE_SEED)
            idx = np.sort(rng.choice(total, size=max_samples, replace=False))
            X = hf[f"X_{key}"][idx, :MAX_LENGTH, :]
            y = hf[f"y_{key}"][idx]
        else:
            idx = np.arange(total)
            X = hf[f"X_{key}"][:, :MAX_LENGTH, :]
            y = hf[f"y_{key}"][:]
    return X.astype(np.float32), y.astype(np.int64), idx


def _load_y3(labels_path, idx_tr, idx_te):
    with h5py.File(labels_path, "r") as f:
        y3_tr = f["y3_train"][:][idx_tr]
        y3_te = f["y3_test"][:][idx_te]
    return y3_tr, y3_te


# --------------------------------------------------------------------------- #
# Checkpoint discovery                                                        #
# --------------------------------------------------------------------------- #

def _find_ckpt(model_name, length):
    """Return the most recent .pth matching `<model>_len{L}*.pth`, or None."""
    patterns = {
        "herqules":    [f"HERQULES_len{length}.pth", f"HERQULES_best_len{length}.pth"],
        "cnn":         [f"CNN_len{length}_*.pth", f"CNN_len{length}.pth"],
        "transformer": [f"Transformer_len{length}_*.pth", f"Transformer_len{length}.pth"],
    }
    cands = []
    for pat in patterns[model_name]:
        cands.extend(glob.glob(os.path.join(SAVE_DIR, pat)))
    # Defend against len50 matching len500: filter by exact len token boundary.
    rx = re.compile(rf"_len{length}(?:\D|$)")
    cands = [c for c in cands if rx.search(os.path.basename(c))]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


# --------------------------------------------------------------------------- #
# Trace property flags (model-agnostic)                                       #
# --------------------------------------------------------------------------- #

def _half_mf_signs(demod_q_at_L, mf_env_q):
    """First- vs second-half MF scores. demod_q_at_L: (N, L, 2). Returns
    (mf_first (N,), mf_second (N,))."""
    N, L, _ = demod_q_at_L.shape
    F = 2 * L
    flat = demod_q_at_L.reshape(N, F).astype(np.float32, copy=False)
    env = mf_env_q[:F].astype(np.float32, copy=False)
    h = F // 2
    return flat[:, :h] @ env[:h], flat[:, h:] @ env[h:]


def _is_t1_pattern(demod_q_at_L, mf_env_q):
    """T1-decay flag per trace: first/second-half MF sign disagreement when
    both halves are significantly above their median magnitudes."""
    mf1, mf2 = _half_mf_signs(demod_q_at_L, mf_env_q)
    med1 = np.median(np.abs(mf1)) + 1e-12
    med2 = np.median(np.abs(mf2)) + 1e-12
    sig = (np.abs(mf1) > 0.5 * med1) & (np.abs(mf2) > 0.5 * med2)
    return sig & (np.sign(mf1) != np.sign(mf2))


# --------------------------------------------------------------------------- #
# Per-qubit marginalization                                                   #
# --------------------------------------------------------------------------- #

def _per_qubit_marginal_from_joint(logits_joint):
    """logits_joint: (N, 32) -> P(qubit_q = 1): (N, 5). Sums softmax mass over
    joint states whose bit q is set."""
    probs = torch.softmax(logits_joint, dim=-1)
    states = torch.arange(NUM_CLASSES_JOINT)
    out = torch.zeros(probs.shape[0], NUM_QUBITS, dtype=probs.dtype)
    for q in range(NUM_QUBITS):
        mask = ((states >> q) & 1).bool()
        out[:, q] = probs[:, mask].sum(dim=-1)
    return out.cpu().numpy()


def _per_qubit_sigmoid(logits_multitask):
    """logits_multitask: (N, 5) -> P(qubit_q = 1): (N, 5)."""
    return torch.sigmoid(logits_multitask).cpu().numpy()


# --------------------------------------------------------------------------- #
# Per-model predict                                                           #
# --------------------------------------------------------------------------- #

def _predict_herqules(ckpt_path, demod_test_full, mf_envs, rmf_envs, length):
    """HERQULES at length L: build the 10-D MF/RMF feature vector by dotting
    the demod trace TRUNCATED TO L with the full-length envelopes sliced to L
    (matches build_features). Returns (N, 32) joint logits and pq1 (N, 5)."""
    feats = build_features(demod_test_full, mf_envs, rmf_envs, length)
    model = Net_rmf().to(DEVICE).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    with torch.no_grad():
        logits = []
        Xt = torch.tensor(feats, dtype=torch.float32)
        for s in range(0, Xt.shape[0], 4096):
            logits.append(model(Xt[s:s + 4096].to(DEVICE)).cpu())
        logits = torch.cat(logits, dim=0)
    return _per_qubit_marginal_from_joint(logits)


def _predict_cnn(ckpt_path, X_test_at_L, train_stats, downsample=20):
    """CNN at length L: demod+downsample raw truncated to L, z-norm per channel
    with TRAIN stats, run model. Returns pq1 (N, 5)."""
    # _demod_downsample expects (N, T, 2); produces (N, T_ds, 5, 2).
    parts = []
    for s in range(0, len(X_test_at_L), 50_000):
        parts.append(_demod_downsample(X_test_at_L[s:s + 50_000], downsample))
    arr = np.concatenate(parts, axis=0)
    N, T_ds = arr.shape[0], arr.shape[1]
    arr = arr.reshape(N, T_ds, NUM_QUBITS * 2).transpose(0, 2, 1).astype(np.float32)
    mu, sd = train_stats
    arr = (arr - mu) / sd
    # Infer m_param from checkpoint.
    sd_state = torch.load(ckpt_path, map_location=DEVICE)
    m_param = sd_state["conv1.weight"].shape[0]
    model = CNN(in_channels=10, m_param=int(m_param), num_qubits=NUM_QUBITS).to(DEVICE).eval()
    model.load_state_dict(sd_state)
    with torch.no_grad():
        logits = []
        Xt = torch.tensor(arr, dtype=torch.float32)
        for s in range(0, Xt.shape[0], 4096):
            logits.append(model(Xt[s:s + 4096].to(DEVICE)).cpu())
        logits = torch.cat(logits, dim=0)
    return _per_qubit_sigmoid(logits)


def _predict_transformer(ckpt_path, X_test_at_L, length):
    """Transformer at length L: take raw IQ, infer hparams from state_dict,
    load, predict joint 32. Returns pq1 (N, 5)."""
    sd_state = torch.load(ckpt_path, map_location=DEVICE)
    # Infer hparams (mirrors evaluate_truncated).
    embedding_dim = sd_state["cls_token"].shape[2] if "cls_token" in sd_state else 128
    if "patch_embed.projection.weight" in sd_state:
        w = sd_state["patch_embed.projection.weight"]
        patch_size = w.shape[1] // 2 if w.dim() == 2 else w.shape[2]
    else:
        patch_size = 10
    layer_idx = set()
    for k in sd_state:
        m = re.match(r"transformer_encoder\.layers\.(\d+)\.", k)
        if m:
            layer_idx.add(int(m.group(1)))
    num_layers = max(layer_idx) + 1 if layer_idx else 4
    num_heads = 8 if embedding_dim % 8 == 0 else 4
    model = QubitClassifierTransformer(
        num_classes=NUM_CLASSES_JOINT, patch_size=int(patch_size),
        embedding_dim=int(embedding_dim), num_heads=int(num_heads),
        num_layers=int(num_layers), dropout=0.1).to(DEVICE).eval()
    model.load_state_dict(sd_state)
    with torch.no_grad():
        logits = []
        Xt = torch.tensor(X_test_at_L, dtype=torch.float32)
        for s in range(0, Xt.shape[0], 2048):
            logits.append(model(Xt[s:s + 2048].to(DEVICE)).cpu())
        logits = torch.cat(logits, dim=0)
    return _per_qubit_marginal_from_joint(logits)


# --------------------------------------------------------------------------- #
# Categorization                                                              #
# --------------------------------------------------------------------------- #

def _categorize_per_qubit(true_q, pq1_q, is_leak_q, is_t1_q,
                          conf_threshold=NEAR_BOUNDARY_CONF):
    """Return counts dict for one (model, length, qubit) cell."""
    pred_q = (pq1_q > 0.5).astype(np.int64)
    correct = pred_q == true_q
    err_idx = np.flatnonzero(~correct)
    conf_correct = np.where(true_q == 1, pq1_q, 1.0 - pq1_q)

    # Boolean masks restricted to error indices.
    e_leak = is_leak_q[err_idx]
    e_t1 = is_t1_q[err_idx]
    e_near = conf_correct[err_idx] < conf_threshold

    # Priority ordering (exclusive): leakage > t1 > near_boundary > state_prep > other.
    cat = np.full(err_idx.shape, "other", dtype=object)
    cat[~e_leak & e_near & ~e_t1] = "near_boundary"
    cat[~e_leak & e_t1] = "t1_pattern"
    cat[e_leak] = "leakage"
    # state_prep = confident wrong NOT explained by leakage/t1/boundary.
    spam_mask = (cat == "other") & ~e_near
    cat[spam_mask] = "state_prep"

    return {
        "n_test": int(true_q.size),
        "n_correct": int(correct.sum()),
        "n_error": int((~correct).sum()),
        "n_leakage": int((cat == "leakage").sum()),
        "n_t1": int((cat == "t1_pattern").sum()),
        "n_boundary": int((cat == "near_boundary").sum()),
        "n_spam": int((cat == "state_prep").sum()),
        "n_other": int((cat == "other").sum()),
        "accuracy_pct": float(100.0 * correct.mean()),
        # Diagnostic counts: non-exclusive memberships (sum can exceed n_error).
        "any_leakage": int(is_leak_q[err_idx].sum()),
        "any_t1": int(is_t1_q[err_idx].sum()),
        "any_boundary": int(e_near.sum()),
    }


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def run(labels_path, lengths, qubits, models, out_path, max_train, max_test):
    logger.info(f"Loading raw + y3 labels (max_train={max_train}, max_test={max_test}) ...")
    Xtr, y2tr, idx_tr = _load_raw(RAW_TRAIN_FILE, "train", max_train)
    Xte, y2te, idx_te = _load_raw(RAW_TEST_FILE, "test", max_test)
    y3tr, y3te = _load_y3(labels_path, idx_tr, idx_te)
    logger.info(f"  Xtr={Xtr.shape} Xte={Xte.shape} y3tr={y3tr.shape} y3te={y3te.shape}")

    logger.info("Demodulating train + test (full length) ...")
    demod_tr_full = demodulate_all_qubits(Xtr)
    demod_te_full = demodulate_all_qubits(Xte)

    logger.info("Computing binary MF/RMF envelopes from PURE train traces ...")
    np.random.seed(SAMPLE_SEED)
    pure = ~(y3tr == 2).any(axis=1)
    demod_pure = {q: demod_tr_full[q][pure] for q in range(NUM_QUBITS)}
    mf_envs, rmf_envs = compute_all_envelopes(demod_pure, y2tr[pure])

    # CNN train norm stats (full-length, all train traces, post demod+downsample).
    logger.info("Computing CNN train normalization stats (per channel) ...")
    parts = []
    for s in range(0, len(Xtr), 50_000):
        parts.append(_demod_downsample(Xtr[s:s + 50_000], 20))
    cnn_tr = np.concatenate(parts, axis=0)
    cnn_tr = cnn_tr.reshape(cnn_tr.shape[0], cnn_tr.shape[1], NUM_QUBITS * 2)\
                   .transpose(0, 2, 1).astype(np.float32)
    cnn_mu = cnn_tr.mean(axis=(0, 2), keepdims=True)
    cnn_sd = cnn_tr.std(axis=(0, 2), keepdims=True) + 1e-10
    del cnn_tr
    cnn_train_stats = (cnn_mu, cnn_sd)

    del Xtr  # free memory; raw test kept (Transformer needs it)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    rows = []

    for L in lengths:
        logger.info(f"=== Length L={L} ===")
        # Per-qubit T1 flag at this length (depends on L because demod is sliced).
        is_t1 = {}
        for q in qubits:
            demod_q_L = demod_te_full[q][:, :L, :]
            is_t1[q] = _is_t1_pattern(demod_q_L, mf_envs[q])
            logger.info(f"  q{q}: is_t1 frac = {is_t1[q].mean()*100:.3f}%")

        # Predict per model, then categorize per qubit.
        for model_name in models:
            ckpt = _find_ckpt(model_name, L)
            if ckpt is None:
                logger.warning(f"  no {model_name} checkpoint for L={L}; skipping")
                continue
            logger.info(f"  {model_name}: {os.path.basename(ckpt)}")
            if model_name == "herqules":
                pq1 = _predict_herqules(ckpt, demod_te_full, mf_envs, rmf_envs, L)
            elif model_name == "cnn":
                pq1 = _predict_cnn(ckpt, Xte[:, :L, :], cnn_train_stats)
            elif model_name == "transformer":
                pq1 = _predict_transformer(ckpt, Xte[:, :L, :], L)
            else:
                logger.error(f"unknown model {model_name!r}")
                continue

            for q in qubits:
                true_q = ((y2te >> q) & 1).astype(np.int64)
                is_leak_q = (y3te[:, q] == 2)
                cell = _categorize_per_qubit(true_q, pq1[:, q], is_leak_q, is_t1[q])
                row = {"length": L, "model": model_name, "qubit": q,
                       "checkpoint": os.path.basename(ckpt), **cell}
                rows.append(row)
                logger.info(
                    f"    q{q}: acc={cell['accuracy_pct']:.3f}% err={cell['n_error']} | "
                    f"leak={cell['n_leakage']} t1={cell['n_t1']} "
                    f"bnd={cell['n_boundary']} spam={cell['n_spam']} other={cell['n_other']}"
                )

    if not rows:
        logger.error("No rows produced — no (model, length) combos had checkpoints.")
        return
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Saved {out_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    p.add_argument("--labels", default=LABELS_FILE,
                   help=f"Ternary labels HDF5. Default: {LABELS_FILE} "
                        "(override globally via the THREE_LEVEL_LABELS env var).")
    p.add_argument("--lengths", type=int, nargs="+", default=[500, 400, 300, 200, 100])
    p.add_argument("--qubits", type=int, nargs="+", default=[0, 4])
    p.add_argument("--models", nargs="+", default=["herqules", "cnn"],
                   choices=["herqules", "cnn", "transformer"])
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--out", default=os.path.join(CSV_DIR, "error_analysis_sweep.csv"))
    args = p.parse_args()
    run(args.labels, args.lengths, args.qubits, args.models, args.out,
        args.max_train, args.max_test)
