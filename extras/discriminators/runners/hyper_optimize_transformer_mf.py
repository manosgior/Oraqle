"""
hyper_optimize_transformer_mf.py
================================
Optuna hyper-optimization for the HERQULES-aided transformer
(`QubitClassifierTransformerMF`).

Forked from `hyper_optimize_transformer.py` with three additions:

  1. One-time HERQULES feature precomputation. At startup we load the full
     (MAX_LENGTH=500) raw IQ traces, run `demodulate_all_qubits` and
     `compute_all_envelopes` from `trainers.train_HERQULES`, and cache the
     demodulated arrays and MF/RMF envelopes. Each per-length data prep then
     calls `build_features` at the matching trained length to materialise the
     10-D MF+RMF feature vector for that length's train/val/test splits
     (envelope and trace are truncated consistently — paper-faithful).

  2. New Optuna axis `mf_hidden_dim`. Controls whether the 10-D MF feature
     vector is passed through a small MLP before concatenation with the [CLS]
     embedding (0 = none, else width from a small candidate set).

  3. Per-feature z-score on the MF vector. MF scalars differ by orders of
     magnitude across qubits; without per-qubit normalization the early
     CLS+MF concat would be dominated by whichever qubit's envelope happens
     to produce the largest output.

Model forward signature is `model(X_b, MF_b)`; the inner sampling loop indexes
a parallel MF tensor alongside the trace tensor.

Run inside the project Docker container:
    python -m runners.hyper_optimize_transformer_mf
"""

from __future__ import annotations

import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import csv
import math
import os
import sys
import threading
import time
from datetime import datetime, timedelta

import h5py
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks import QubitClassifierTransformerMF
from trainers.train_HERQULES import (
    demodulate_all_qubits,
    compute_all_envelopes,
    build_features as herqules_build_features,
)

# Input shape per length is constant -> let cuDNN cache the fastest kernel.
torch.backends.cudnn.benchmark = True


# ============================================================================
# User Configuration
# ============================================================================

RAW_TRAIN_FILE = str(RAW_TRAIN)
RAW_TEST_FILE  = str(RAW_TEST)
NUM_QUBITS  = 5
NUM_CLASSES = 2 ** NUM_QUBITS

# Envelopes are computed once at MAX_LENGTH; each evaluated length truncates
# both trace and envelope (matches train_HERQULES.build_features semantics).
MAX_LENGTH    = 500
TRACE_LENGTHS = [100, 200, 300, 400, 500]

N_OPTUNA_TRIALS   = 50
EPOCHS_PER_TRIAL  = 30
N_PARALLEL_TRIALS = 8

WARMUP_EPOCHS = 3
MIN_LR_RATIO  = 0.01
MAX_GRAD_NORM = 1.0
VAL_SPLIT     = 0.2

MAX_TRAIN_SAMPLES: int | None = None
MAX_TEST_SAMPLES:  int | None = None
SAMPLE_SEED = 42

PATCH_SIZE_CANDIDATES    = [5, 10, 20, 25, 50]
HARD_FRACTION_CANDIDATES = [0.0]
# 0 = no MLP, just LayerNorm on raw MF features. >0 = MLP hidden width.
MF_HIDDEN_DIM_CANDIDATES = [0, 32, 64]

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = str(MODELS_OUT)
CSV_DIR  = str(REPORTS_OUT)

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CSV_DIR,  exist_ok=True)

_RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
logger.add(
    os.path.join(CSV_DIR, f"transformer_mf_optimize_{_RUN_TAG}.log"),
    level="INFO", enqueue=True, backtrace=True, diagnose=False,
)

_FILE_LOCK = threading.Lock()
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _fmt_dur(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    if td.total_seconds() >= 3600:
        return str(td)
    m, s = divmod(int(td.total_seconds()), 60)
    return f"{m:02d}:{s:02d}"


# ============================================================================
# Data Loading + Preprocessing
# ============================================================================

def _load_h5_split(filepath: str, trace_length: int, is_test: bool,
                   max_samples: int | None):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file missing: {filepath}")

    file_mb = os.path.getsize(filepath) / (1024 * 1024)
    key_suffix = "test" if is_test else "train"
    t0 = time.perf_counter()
    logger.info(f"Loading {filepath}  ({file_mb:.1f} MB)")

    with h5py.File(filepath, "r") as hf:
        total   = hf[f"X_{key_suffix}"].shape[0]
        T_avail = hf[f"X_{key_suffix}"].shape[1]
        if trace_length > T_avail:
            raise ValueError(
                f"trace_length={trace_length} exceeds available raw length "
                f"{T_avail} in {filepath}"
            )
        if max_samples is not None and max_samples < total:
            rng = np.random.RandomState(SAMPLE_SEED)
            idx = np.sort(rng.choice(total, size=max_samples, replace=False))
            X = hf[f"X_{key_suffix}"][idx, :trace_length, :]
            y = hf[f"y_{key_suffix}"][idx]
            logger.info(f"  subsampled {key_suffix}: {total:,} -> {max_samples:,} traces")
        else:
            X = hf[f"X_{key_suffix}"][:, :trace_length, :]
            y = hf[f"y_{key_suffix}"][:]

    logger.info(
        f"  loaded {key_suffix}: X={X.shape} y={y.shape} "
        f"in {_fmt_dur(time.perf_counter() - t0)}"
    )
    return X, y


def prepare_global_data() -> dict:
    """Load FULL-length raw traces once, demodulate, and fit MF/RMF envelopes.

    Per the HERQULES paper / `train_HERQULES.py`, envelopes are computed on
    the full-length demodulated training set; `build_features` then truncates
    both trace and envelope for any shorter trained length. Computing this
    once at startup makes the per-length data prep cheap (one dot product
    per qubit + a train/val split).
    """
    logger.info(f"--- Global data prep (one-time, MAX_LENGTH={MAX_LENGTH}) ---")
    X_train_raw, y_train = _load_h5_split(
        RAW_TRAIN_FILE, MAX_LENGTH, is_test=False, max_samples=MAX_TRAIN_SAMPLES,
    )
    X_test_raw, y_test = _load_h5_split(
        RAW_TEST_FILE, MAX_LENGTH, is_test=True, max_samples=MAX_TEST_SAMPLES,
    )

    logger.info("Demodulating training traces (per-qubit, 5x)...")
    t0 = time.perf_counter()
    demod_train = demodulate_all_qubits(X_train_raw)
    logger.info(f"  done in {_fmt_dur(time.perf_counter() - t0)}")

    logger.info("Demodulating test traces (per-qubit, 5x)...")
    t0 = time.perf_counter()
    demod_test = demodulate_all_qubits(X_test_raw)
    logger.info(f"  done in {_fmt_dur(time.perf_counter() - t0)}")

    logger.info("Computing MF/RMF envelopes from purified train traces...")
    t0 = time.perf_counter()
    np.random.seed(SAMPLE_SEED)  # deterministic class-balancing subsampling
    mf_envs, rmf_envs = compute_all_envelopes(demod_train, y_train)
    logger.info(f"  done in {_fmt_dur(time.perf_counter() - t0)}")

    return {
        "X_train_raw": X_train_raw,
        "y_train":     y_train,
        "X_test_raw":  X_test_raw,
        "y_test":      y_test,
        "demod_train": demod_train,
        "demod_test":  demod_test,
        "mf_envs":     mf_envs,
        "rmf_envs":    rmf_envs,
    }


def _hardness_order_desc(X_train: np.ndarray,
                         y_train_packed: np.ndarray) -> np.ndarray:
    """Rank training traces by per-qubit MF hardness, hardest first.

    Identical to the base trainer: per qubit, build an MF envelope from
    (gnd-ext), classify each trace by midpoint threshold, count wrong qubits
    per trace, break ties by summed wrong-side margin.
    """
    N = X_train.shape[0]
    X_flat = X_train.reshape(N, -1).astype(np.float32, copy=False)
    wrong  = np.zeros((N, NUM_QUBITS), dtype=bool)
    margin = np.zeros(N, dtype=np.float32)

    for q in range(NUM_QUBITS):
        y_q = ((y_train_packed >> q) & 1).astype(np.int64)
        gnd_mask = y_q == 0
        ext_mask = y_q == 1
        n = int(min(gnd_mask.sum(), ext_mask.sum()))
        gnd = X_flat[gnd_mask][:n]
        ext = X_flat[ext_mask][:n]
        diff = gnd - ext
        envelope = (diff.mean(axis=0) / (diff.var(axis=0) + 1e-10)).astype(
            np.float32, copy=False
        )
        mf  = X_flat @ envelope
        thr = 0.5 * (mf[gnd_mask].mean() + mf[ext_mask].mean())
        pred_gnd     = mf > thr
        wrong[:, q]  = pred_gnd != gnd_mask
        margin      += np.where(wrong[:, q], np.abs(mf - thr), 0.0).astype(np.float32)

    n_wrong    = wrong.sum(axis=1)
    order_desc = np.lexsort((-margin, -n_wrong)).astype(np.int64)

    n_hard_summary = np.bincount(n_wrong, minlength=NUM_QUBITS + 1)
    logger.info(
        f"  MF wrong-count histogram (0..{NUM_QUBITS}): "
        f"{n_hard_summary.tolist()}  (total={n_hard_summary.sum():,})"
    )
    return order_desc


def prepare_split_for_length(global_data: dict, trace_length: int) -> dict:
    """Slice cached full-length data to `trace_length`, build MF features, split."""
    # --- Slice raw trace to evaluated length ---
    X = global_data["X_train_raw"][:, :trace_length, :].astype(np.float32, copy=False)
    y = global_data["y_train"].astype(np.int64, copy=False)

    # --- Build MF features at this length (build_features truncates envelopes) ---
    MF = herqules_build_features(
        global_data["demod_train"],
        global_data["mf_envs"], global_data["rmf_envs"],
        trace_length,
    )

    # --- Train/val split (stratified, parallel index for X and MF) ---
    idx_tr, idx_val = train_test_split(
        np.arange(X.shape[0]), test_size=VAL_SPLIT,
        random_state=SAMPLE_SEED, stratify=y,
    )
    X_train,  X_val  = X[idx_tr],  X[idx_val]
    MF_train, MF_val = MF[idx_tr], MF[idx_val]
    y_train,  y_val  = y[idx_tr],  y[idx_val]

    # --- Per-channel z-score for trace (train stats reused for val + test) ---
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std(axis=(0, 1),  keepdims=True) + 1e-10
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std

    # --- Per-feature z-score for MF (per-qubit scales differ by ~OOM) ---
    mf_mean = MF_train.mean(axis=0, keepdims=True)
    mf_std  = MF_train.std(axis=0,  keepdims=True) + 1e-10
    MF_train = (MF_train - mf_mean) / mf_std
    MF_val   = (MF_val   - mf_mean) / mf_std

    # --- MF-based hardness ranking on the normalized training pool ---
    t0 = time.perf_counter()
    order_desc = _hardness_order_desc(X_train, y_train)
    logger.info(
        f"  MF hardness ranking done in {_fmt_dur(time.perf_counter() - t0)}  "
        f"(n_train={X_train.shape[0]:,})"
    )

    return {
        "X_train":  torch.from_numpy(X_train).to(DEVICE, non_blocking=True),
        "y_train":  torch.from_numpy(y_train).to(DEVICE, non_blocking=True),
        "X_val":    torch.from_numpy(X_val).to(DEVICE,   non_blocking=True),
        "y_val":    torch.from_numpy(y_val).to(DEVICE,   non_blocking=True),
        "MF_train": torch.from_numpy(MF_train.astype(np.float32)).to(DEVICE, non_blocking=True),
        "MF_val":   torch.from_numpy(MF_val.astype(np.float32)).to(DEVICE,   non_blocking=True),
        "hardness_order_desc": torch.from_numpy(order_desc).to(DEVICE, non_blocking=True),
        "mean":    mean,    "std":    std,
        "mf_mean": mf_mean, "mf_std": mf_std,
    }


def prepare_test_for_length(global_data: dict, trace_length: int,
                            mean: np.ndarray, std: np.ndarray,
                            mf_mean: np.ndarray, mf_std: np.ndarray):
    """Test split for one length; normalized with the matching TRAIN statistics."""
    X = global_data["X_test_raw"][:, :trace_length, :].astype(np.float32, copy=False)
    X = (X - mean) / std
    y = global_data["y_test"].astype(np.int64, copy=False)

    MF = herqules_build_features(
        global_data["demod_test"],
        global_data["mf_envs"], global_data["rmf_envs"],
        trace_length,
    )
    MF = (MF - mf_mean) / mf_std

    return (
        torch.from_numpy(X).to(DEVICE, non_blocking=True),
        torch.from_numpy(MF.astype(np.float32)).to(DEVICE, non_blocking=True),
        torch.from_numpy(y).to(DEVICE, non_blocking=True),
    )


# ============================================================================
# Training utilities
# ============================================================================

def _make_warmup_cosine(optimizer, warmup_epochs: int, total_epochs: int,
                        min_lr_ratio: float = MIN_LR_RATIO):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _param_groups_no_decay_for_norm_bias(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _train_one_trial(trial: optuna.Trial, data: dict, trace_length: int) -> float:
    valid_patch_sizes = [p for p in PATCH_SIZE_CANDIDATES if trace_length % p == 0]
    if not valid_patch_sizes:
        raise optuna.exceptions.TrialPruned()

    lr           = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    patch_size   = trial.suggest_categorical("patch_size", valid_patch_sizes)
    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
    num_heads    = trial.suggest_categorical("num_heads", [4, 8])
    num_layers   = trial.suggest_int("num_layers", 2, 6)
    dropout      = trial.suggest_float("dropout", 0.05, 0.3)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    hard_fraction = trial.suggest_categorical("hard_fraction", HARD_FRACTION_CANDIDATES)
    mf_hidden_sentinel = trial.suggest_categorical("mf_hidden_dim", MF_HIDDEN_DIM_CANDIDATES)
    mf_hidden_dim = None if mf_hidden_sentinel == 0 else int(mf_hidden_sentinel)

    if embedding_dim % num_heads != 0:
        raise optuna.exceptions.TrialPruned()

    logger.info(
        f"[L={trace_length}] Trial #{trial.number} START  "
        f"lr={lr:.2e}  bs={batch_size}  patch={patch_size}  "
        f"d={embedding_dim}  h={num_heads}  layers={num_layers}  "
        f"drop={dropout:.3f}  wd={weight_decay:.2e}  "
        f"hard_frac={hard_fraction:.2f}  mf_hidden={mf_hidden_sentinel}"
    )
    trial_t0 = time.perf_counter()

    X_train: torch.Tensor  = data["X_train"]
    y_train: torch.Tensor  = data["y_train"]
    MF_train: torch.Tensor = data["MF_train"]
    X_val: torch.Tensor    = data["X_val"]
    y_val: torch.Tensor    = data["y_val"]
    MF_val: torch.Tensor   = data["MF_val"]
    order_desc: torch.Tensor = data["hardness_order_desc"]
    n_train_pool = X_train.shape[0]
    n_val        = X_val.shape[0]

    n_drop   = int(round(n_train_pool * hard_fraction))
    keep_idx = order_desc[n_drop:]
    n_keep   = int(keep_idx.shape[0])

    model = QubitClassifierTransformerMF(
        num_classes=NUM_CLASSES,
        patch_size=patch_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        mf_feature_dim=int(MF_train.shape[1]),
        mf_hidden_dim=mf_hidden_dim,
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        _param_groups_no_decay_for_norm_bias(model, weight_decay), lr=lr,
    )
    scheduler = _make_warmup_cosine(optimizer, WARMUP_EPOCHS, EPOCHS_PER_TRIAL)

    log_every  = max(1, EPOCHS_PER_TRIAL // 3)
    val_loss   = float("inf")
    train_loss = float("inf")

    for epoch in range(EPOCHS_PER_TRIAL):
        # ---- train ----
        model.train()
        perm = keep_idx[torch.randperm(n_keep, device=DEVICE)]
        train_loss_t = torch.zeros((), device=DEVICE)
        for start in range(0, n_keep, batch_size):
            idx  = perm[start:start + batch_size]
            X_b  = X_train[idx]
            mf_b = MF_train[idx]
            y_b  = y_train[idx]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X_b, mf_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            train_loss_t = train_loss_t + loss.detach() * X_b.size(0)
        scheduler.step()
        train_loss = (train_loss_t / n_keep).item()

        # ---- val ----
        model.eval()
        val_loss_t = torch.zeros((), device=DEVICE)
        with torch.no_grad():
            for start in range(0, n_val, batch_size):
                X_b  = X_val[start:start + batch_size]
                mf_b = MF_val[start:start + batch_size]
                y_b  = y_val[start:start + batch_size]
                val_loss_t = val_loss_t + criterion(model(X_b, mf_b), y_b) * X_b.size(0)
        val_loss = (val_loss_t / n_val).item()

        if (epoch + 1) % log_every == 0 or epoch == EPOCHS_PER_TRIAL - 1:
            logger.info(
                f"[L={trace_length}] Trial #{trial.number} "
                f"epoch {epoch + 1:>3d}/{EPOCHS_PER_TRIAL}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        trial.report(val_loss, epoch)
        if trial.should_prune():
            logger.info(
                f"[L={trace_length}] Trial #{trial.number} PRUNED at epoch "
                f"{epoch + 1}  val={val_loss:.4f}"
            )
            raise optuna.exceptions.TrialPruned()

    model_path = os.path.join(
        SAVE_DIR,
        f"TransformerMF_len{trace_length}_trial{trial.number}.pth",
    )
    with _FILE_LOCK:
        torch.save(model.state_dict(), model_path)
    trial.set_user_attr("model_path",      model_path)
    trial.set_user_attr("wall_time_s",     time.perf_counter() - trial_t0)
    trial.set_user_attr("n_kept_train",    n_keep)
    trial.set_user_attr("n_dropped_train", int(n_drop))
    return val_loss


# ============================================================================
# Evaluation + CSV
# ============================================================================

def _evaluate_test(model: nn.Module, X_test: torch.Tensor, MF_test: torch.Tensor,
                   y_test: torch.Tensor, batch_size: int):
    model.eval()
    n = X_test.shape[0]
    pred_chunks = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            X_b  = X_test[start:start + batch_size]
            mf_b = MF_test[start:start + batch_size]
            pred_chunks.append(model(X_b, mf_b).argmax(dim=1))
    preds = torch.cat(pred_chunks, dim=0).cpu().numpy().astype(np.int64)
    truth = y_test.cpu().numpy().astype(np.int64)

    overall = 100.0 * np.mean(preds == truth)
    per_q = []
    for q in range(NUM_QUBITS):
        pred_q = (preds >> q) & 1
        true_q = (truth >> q) & 1
        per_q.append(100.0 * np.mean(pred_q == true_q))
    return overall, per_q


def _save_csv_report(study: optuna.Study, model: nn.Module, trace_length: int,
                     overall_acc: float, per_qubit_accs: list[float]):
    best = study.best_trial
    total_params = sum(p.numel() for p in model.parameters())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"TransformerMF_len{trace_length}_{timestamp}.csv"
    filepath = os.path.join(CSV_DIR, filename)

    row = {
        "timestamp":         timestamp,
        "model_name":        "TransformerMF",
        "trace_length":      trace_length,
        "optimizer":         "AdamW",
        "lr_schedule":       f"warmup({WARMUP_EPOCHS})+cosine(min_ratio={MIN_LR_RATIO})",
        "learning_rate":     best.params.get("lr",            "N/A"),
        "weight_decay":      best.params.get("weight_decay",  "N/A"),
        "batch_size":        best.params.get("batch_size",    "N/A"),
        "patch_size":        best.params.get("patch_size",    "N/A"),
        "embedding_dim":     best.params.get("embedding_dim", "N/A"),
        "num_heads":         best.params.get("num_heads",     "N/A"),
        "num_layers":        best.params.get("num_layers",    "N/A"),
        "dropout":           best.params.get("dropout",       "N/A"),
        "hard_fraction":     best.params.get("hard_fraction", "N/A"),
        "mf_hidden_dim":     best.params.get("mf_hidden_dim", "N/A"),
        "n_kept_train":      best.user_attrs.get("n_kept_train",    "N/A"),
        "n_dropped_train":   best.user_attrs.get("n_dropped_train", "N/A"),
        "epochs":            EPOCHS_PER_TRIAL,
        "n_optuna_trials":   len(study.trials),
        "best_trial_number": best.number,
        "best_val_loss":     f"{best.value:.6f}",
        "total_parameters":  total_params,
        "overall_accuracy":  f"{overall_acc:.4f}",
        **{f"qubit_{q}_accuracy": f"{per_qubit_accs[q]:.4f}" for q in range(NUM_QUBITS)},
        "device":            str(DEVICE),
        "model_path":        best.user_attrs.get("model_path", "N/A"),
    }

    with _FILE_LOCK:
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)
    logger.info(f"CSV report saved: {filepath}")


# ============================================================================
# Main Loop
# ============================================================================

def _make_trial_callback(trace_length: int):
    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        elapsed = trial.user_attrs.get("wall_time_s", None)
        elapsed_s = f"  ({_fmt_dur(elapsed)})" if elapsed is not None else ""

        completed = sum(
            1 for t in study.trials
            if t.state in (
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED,
                optuna.trial.TrialState.FAIL,
            )
        )

        if trial.state == optuna.trial.TrialState.COMPLETE:
            best_so_far = min(
                (t.value for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value is not None),
                default=float("inf"),
            )
            logger.success(
                f"[L={trace_length}] Trial #{trial.number} DONE "
                f"({completed}/{N_OPTUNA_TRIALS})  val={trial.value:.6f}  "
                f"best_so_far={best_so_far:.6f}{elapsed_s}"
            )
        elif trial.state == optuna.trial.TrialState.PRUNED:
            logger.info(
                f"[L={trace_length}] Trial #{trial.number} pruned "
                f"({completed}/{N_OPTUNA_TRIALS}){elapsed_s}"
            )
        else:
            logger.error(
                f"[L={trace_length}] Trial #{trial.number} FAILED "
                f"({completed}/{N_OPTUNA_TRIALS}){elapsed_s}"
            )
    return _cb


def optimize_transformer_mf():
    overall_t0 = time.perf_counter()
    summary: list[dict] = []

    logger.info("=" * 78)
    logger.info("TransformerMF Hyper-Optimization (HERQULES MF/RMF concat at CLS)")
    logger.info("=" * 78)
    logger.info(f"  device              : {DEVICE}")
    if DEVICE.type == "cuda":
        logger.info(f"  gpu                 : {torch.cuda.get_device_name(0)}")
    logger.info(f"  trace lengths       : {TRACE_LENGTHS}")
    logger.info(f"  envelopes @         : MAX_LENGTH={MAX_LENGTH}")
    logger.info(f"  trials per length   : {N_OPTUNA_TRIALS}")
    logger.info(f"  parallel trials     : {N_PARALLEL_TRIALS}")
    logger.info(f"  epochs per trial    : {EPOCHS_PER_TRIAL}")
    logger.info(f"  warmup epochs       : {WARMUP_EPOCHS}")
    logger.info(f"  max grad norm       : {MAX_GRAD_NORM}")
    logger.info(f"  patch_size cands    : {PATCH_SIZE_CANDIDATES}")
    logger.info(f"  hard_fraction cands : {HARD_FRACTION_CANDIDATES}")
    logger.info(f"  mf_hidden_dim cands : {MF_HIDDEN_DIM_CANDIDATES}  (0 = no MLP)")
    logger.info(f"  save dir            : {SAVE_DIR}")
    logger.info(f"  csv dir             : {CSV_DIR}")
    logger.info(f"  run tag             : {_RUN_TAG}")
    logger.info("=" * 78)

    # One-time HERQULES feature prep (full-length demod + envelopes).
    try:
        global_data = prepare_global_data()
    except FileNotFoundError as e:
        logger.error(f"Missing raw data: {e}")
        return

    for li, length in enumerate(TRACE_LENGTHS, start=1):
        physical_ns = length * 2
        length_t0 = time.perf_counter()

        logger.info("")
        logger.info("-" * 78)
        logger.info(
            f"[{li}/{len(TRACE_LENGTHS)}] Trace length {length} ({physical_ns} ns)"
        )
        logger.info("-" * 78)

        data = prepare_split_for_length(global_data, length)

        logger.info(
            f"Train/val shapes: X_train={tuple(data['X_train'].shape)}  "
            f"X_val={tuple(data['X_val'].shape)}  "
            f"MF_train={tuple(data['MF_train'].shape)}  "
            f"y_train={tuple(data['y_train'].shape)}"
        )
        logger.info(
            f"  per-channel trace mean: {data['mean'].reshape(-1).tolist()}  "
            f"std: {data['std'].reshape(-1).tolist()}"
        )
        logger.info(
            f"  per-feature MF mean range: "
            f"[{data['mf_mean'].min():.4g}, {data['mf_mean'].max():.4g}]  "
            f"std range: [{data['mf_std'].min():.4g}, {data['mf_std'].max():.4g}]"
        )

        sampler = optuna.samplers.TPESampler(seed=SAMPLE_SEED)
        pruner  = optuna.pruners.MedianPruner(n_warmup_steps=5)
        study   = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

        logger.info(
            f"Starting Optuna study: {N_OPTUNA_TRIALS} trials, "
            f"{N_PARALLEL_TRIALS} in parallel"
        )
        study_t0 = time.perf_counter()
        study.optimize(
            lambda trial: _train_one_trial(trial, data, length),
            n_trials=N_OPTUNA_TRIALS,
            n_jobs=N_PARALLEL_TRIALS,
            gc_after_trial=True,
            callbacks=[_make_trial_callback(length)],
        )
        study_dur = time.perf_counter() - study_t0

        n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        n_pruned   = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        n_failed   = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
        logger.info(
            f"Study finished in {_fmt_dur(study_dur)}  |  "
            f"complete={n_complete}  pruned={n_pruned}  failed={n_failed}"
        )

        best = study.best_trial
        logger.success(
            f"BEST trial #{best.number}  val_loss={best.value:.6f}  "
            f"lr={best.params['lr']:.2e}  bs={best.params['batch_size']}  "
            f"patch={best.params['patch_size']}  "
            f"d={best.params['embedding_dim']}  "
            f"h={best.params['num_heads']}  layers={best.params['num_layers']}  "
            f"drop={best.params['dropout']:.3f}  "
            f"wd={best.params['weight_decay']:.2e}  "
            f"hard_frac={best.params['hard_fraction']:.2f}  "
            f"mf_hidden={best.params['mf_hidden_dim']}"
        )

        # Reload best weights and evaluate on the (unfiltered) test set.
        logger.info("Reloading best weights and evaluating on test set...")
        best_mf_hidden = (None if best.params["mf_hidden_dim"] == 0
                          else int(best.params["mf_hidden_dim"]))
        best_model = QubitClassifierTransformerMF(
            num_classes=NUM_CLASSES,
            patch_size=best.params["patch_size"],
            embedding_dim=best.params["embedding_dim"],
            num_heads=best.params["num_heads"],
            num_layers=best.params["num_layers"],
            dropout=best.params["dropout"],
            mf_feature_dim=int(data["MF_train"].shape[1]),
            mf_hidden_dim=best_mf_hidden,
        ).to(DEVICE)
        best_model.load_state_dict(
            torch.load(best.user_attrs["model_path"], map_location=DEVICE)
        )
        best_model.eval()

        try:
            X_test, MF_test, y_test = prepare_test_for_length(
                global_data, length,
                data["mean"],    data["std"],
                data["mf_mean"], data["mf_std"],
            )
        except FileNotFoundError as e:
            logger.warning(f"Test set not available for length={length}: {e}.")
            summary.append({"length": length, "overall": None, "per_q": None,
                            "elapsed": time.perf_counter() - length_t0})
            del data, best_model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            continue

        eval_t0 = time.perf_counter()
        overall_acc, per_q = _evaluate_test(
            best_model, X_test, MF_test, y_test,
            batch_size=best.params["batch_size"],
        )
        logger.info(f"Test eval done in {_fmt_dur(time.perf_counter() - eval_t0)}")
        logger.success(
            f"TEST  (len {length}, {physical_ns} ns)  "
            f"overall={overall_acc:.2f}%  "
            + "  ".join(f"q{i}={per_q[i]:.2f}%" for i in range(NUM_QUBITS))
        )
        _save_csv_report(study, best_model, length, overall_acc, per_q)
        summary.append({"length": length, "overall": overall_acc, "per_q": per_q,
                        "elapsed": time.perf_counter() - length_t0})

        del data, best_model, X_test, MF_test, y_test
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
            logger.info(
                f"GPU mem after cleanup: "
                f"allocated={torch.cuda.memory_allocated()/1e9:.2f} GB  "
                f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB"
            )

        logger.info(
            f"[{li}/{len(TRACE_LENGTHS)}] length={length} finished in "
            f"{_fmt_dur(time.perf_counter() - length_t0)}"
        )

    # -------------- Final summary --------------
    total_elapsed = time.perf_counter() - overall_t0
    logger.info("")
    logger.info("=" * 78)
    logger.info(f"All lengths done in {_fmt_dur(total_elapsed)}")
    logger.info("=" * 78)
    header = (
        f"{'length':>8} {'ns':>6} {'overall':>9} "
        + " ".join(f"q{i}".rjust(7) for i in range(NUM_QUBITS))
        + f"  {'elapsed':>8}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for row in summary:
        if row["overall"] is None:
            line = (
                f"{row['length']:>8} {row['length']*2:>6} "
                f"{'n/a':>9} "
                + " ".join(f"{'n/a':>7}" for _ in range(NUM_QUBITS))
                + f"  {_fmt_dur(row['elapsed']):>8}"
            )
        else:
            line = (
                f"{row['length']:>8} {row['length']*2:>6} "
                f"{row['overall']:>8.2f}% "
                + " ".join(f"{row['per_q'][i]:>6.2f}%" for i in range(NUM_QUBITS))
                + f"  {_fmt_dur(row['elapsed']):>8}"
            )
        logger.info(line)
    logger.info("=" * 78)


if __name__ == "__main__":
    optimize_transformer_mf()
