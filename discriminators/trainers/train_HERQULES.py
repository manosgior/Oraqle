"""
hyper_optimize_herqules.py
==========================
HERQULES training script — faithful replication of the original paper.

Follows HERQULES_original.py as closely as possible:

  1. Load full ~1.5 M-row dataset (no subsampling).
  2. Digitally demodulate the multiplexed IQ stream into per-qubit traces
     (Butterworth LP filter, order 3, 10 MHz cutoff).
  3. Run the geometric pre-classifier to identify clean |0>/|1> reference
     traces and relaxation events per qubit.
  4. Compute per-qubit MF and RMF envelopes (with paper boxcar windows) from
     the purified traces.
  5. For each trace length, build 10-dim MF+RMF feature vectors and train
     Net_rmf with the paper's exact fixed hyperparameters:
         lr = 0.01, batch_size = 512, 100 epochs,
         step-decay LR schedule at epochs 30 / 60 / 90.
  6. Save the best checkpoint (by val accuracy) as a .pth file and write a
     CSV report — one row per trace length.

Configuration
-------------
Edit RAW_TRAIN_FILE / RAW_TEST_FILE to point to your HDF5 datasets.
"""

import os
import sys
import csv
from datetime import datetime

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy.signal import butter, sosfilt
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from disc_paths import RAW_TRAIN, RAW_TEST, MODELS_OUT, REPORTS_OUT  # [unify] replaces the old docker mount paths
from networks import Net_rmf

# ============================================================================
# Configuration — paper-faithful fixed hyperparameters
# ============================================================================

RAW_TRAIN_FILE = str(RAW_TRAIN)
RAW_TEST_FILE  = str(RAW_TEST)

NUM_QUBITS  = 5
TRACE_LENGTHS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

# ---- Paper hyperparameters (HERQULES_original.py, train()) ----
LRN_RATE   = 0.01    # Adam initial lr
BATCH_SIZE = 512
MAX_EPOCHS = 100
LR_SCHEDULE = [30, 60, 90]   # step-decay milestones (÷10 at each)

# ---- Hardware parameters ----
SAMPLING_RATE = 500e6         # 500 MHz ADC
FREQ_READOUT  = -np.array([-64.729e6, -25.366e6, 24.79e6, 70.269e6, 127.282e6])
FILTER_CUTOFF = 10e6          # Butterworth LP cutoff
BOXCARS       = [1, 1, 9, 2, 9]  # per-qubit boxcar widths (units: 50 ADC samples)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = str(MODELS_OUT)  # [unify]
CSV_DIR = str(REPORTS_OUT)  # [unify]
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CSV_DIR,  exist_ok=True)


# ============================================================================
# 1. Data loading
# ============================================================================

def load_hdf5_full(filepath: str, is_test: bool = False):
    """Load the complete HDF5 dataset without subsampling."""
    suffix = "test" if is_test else "train"
    logger.info(f"Loading {'test' if is_test else 'train'} data from {filepath} ...")
    with h5py.File(filepath, "r") as hf:
        X = hf[f"X_{suffix}"][:]
        y = hf[f"y_{suffix}"][:]
    logger.info(f"  X.shape={X.shape}, y.shape={y.shape}")
    return X, y


# ============================================================================
# 2. Demodulation  (follows demodulate_multiplexed_traces in HERQULES_original)
# ============================================================================

def demodulate_all_qubits(iq_traces: np.ndarray) -> dict:
    """
    Digitally demodulate a multiplexed IQ stream into per-qubit traces.

    Replicates the procedure in HERQULES_original.demodulate_multiplexed_traces:
      1. Per-trace DC-offset removal and IQ amplitude-imbalance correction.
      2. Digital down-conversion (mix to DC) for each qubit's IF frequency.
      3. Butterworth order-3 LP filter at FILTER_CUTOFF.

    Returns
    -------
    dict  {qubit_index (0-based): ndarray (N, T, 2)}
    """
    num_traces, T, _ = iq_traces.shape
    dt = 1.0 / SAMPLING_RATE

    DataI = iq_traces[:, :, 0].astype(np.float64)
    DataQ = iq_traces[:, :, 1].astype(np.float64)

    # DC-offset and IQ amplitude-imbalance correction
    DataI -= np.mean(DataI, axis=1, keepdims=True)
    DataQ -= np.mean(DataQ, axis=1, keepdims=True)
    corr   = np.std(DataI, axis=1, keepdims=True) / (np.std(DataQ, axis=1, keepdims=True) + 1e-30)
    DataQ *= corr

    vTime = np.arange(T) * dt
    sos   = butter(3, FILTER_CUTOFF, btype="low", fs=SAMPLING_RATE, output="sos")

    demod = {}
    for i, freq in enumerate(FREQ_READOUT):
        logger.info(f"  Demodulating qubit {i+1} ({freq/1e6:.3f} MHz) ...")
        vCos = np.cos(2 * np.pi * vTime * freq)
        vSin = np.sin(2 * np.pi * vTime * freq)
        i_filt = sosfilt(sos, DataI * vCos + DataQ * vSin, axis=1)
        q_filt = sosfilt(sos, DataQ * vCos - DataI * vSin, axis=1)
        demod[i] = np.stack((i_filt, q_filt), axis=-1).astype(np.float32)

    return demod


# ============================================================================
# 3. Geometric pre-classifier  (follows get_traces / preclassifier in original)
# ============================================================================

def _dist(x0, y0, x1, y1):
    return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)


def preclassify_qubit(demod_traces: np.ndarray, y: np.ndarray,
                       qubit: int, rscale: float = 1.0) -> dict:
    """
    Geometric purification for a single qubit.

    Replicates the core loop of get_traces() in HERQULES_original.py:
      1. Compute |0> and |1> IQ centroids (time-averaged per trace).
      2. Define acceptance radius = rscale * half-inter-centroid distance.
      3. Keep only traces within radius of their own centroid (purified sets).
      4. Identify relaxation traces: |1>-labelled but time-average near |0> centroid.

    Parameters
    ----------
    demod_traces : (N, T, 2)  per-qubit demodulated traces
    y            : (N,)       packed 5-qubit integer labels
    qubit        : 1-based qubit index (1..5)

    Returns
    -------
    dict  {'traces_0', 'traces_1', 'traces_relax'}  or None on failure.
    """
    zero_state = 0
    one_state  = 2 ** (qubit - 1)

    mask0 = (y == zero_state)
    mask1 = (y == one_state)
    if mask0.sum() < 10 or mask1.sum() < 10:
        logger.warning(f"Qubit {qubit}: insufficient samples for states 0/{one_state}")
        return None

    tr0 = demod_traces[mask0]
    tr1 = demod_traces[mask1]

    i0 = np.mean(tr0[:, :, 0], axis=1); q0 = np.mean(tr0[:, :, 1], axis=1)
    i1 = np.mean(tr1[:, :, 0], axis=1); q1 = np.mean(tr1[:, :, 1], axis=1)
    x0c, y0c = np.mean(i0), np.mean(q0)
    x1c, y1c = np.mean(i1), np.mean(q1)
    radius    = rscale * _dist(x0c, y0c, x1c, y1c) / 2.0

    pure0 = tr0[_dist(i0, q0, x0c, y0c) < radius]
    pure1 = tr1[_dist(i1, q1, x1c, y1c) < radius]

    impure1 = tr1[_dist(i1, q1, x1c, y1c) >= radius]
    if len(impure1) > 0:
        ii = np.mean(impure1[:, :, 0], axis=1)
        qi = np.mean(impure1[:, :, 1], axis=1)
        relax = impure1[_dist(ii, qi, x0c, y0c) <= rscale * radius]
    else:
        relax = np.zeros((0, demod_traces.shape[1], 2), dtype=demod_traces.dtype)

    logger.info(f"  Qubit {qubit}: pure|0>={len(pure0)}, pure|1>={len(pure1)}, relax={len(relax)}")
    return {"traces_0": pure0, "traces_1": pure1, "traces_relax": relax}


# ============================================================================
# 4. MF / RMF envelope computation  (follows get_mf in HERQULES_original)
# ============================================================================

def _get_mf_core(t0: np.ndarray, t1: np.ndarray):
    """
    Wiener-optimal matched filter: E[x0-x1] / Var[x0-x1].
    Balances classes by subsampling the majority.
    t0, t1 must already be flattened to (N, F).
    Returns (envelope, threshold_99.5).
    """
    if t1.shape[0] > t0.shape[0]:
        idx  = np.random.choice(t1.shape[0], t0.shape[0], replace=False)
        diff = t0 - t1[idx]
    else:
        idx  = np.random.choice(t0.shape[0], t1.shape[0], replace=False)
        diff = t0[idx] - t1

    mf        = np.mean(diff, axis=0) / (np.var(diff, axis=0) + 1e-30)
    filtered  = np.sort(np.sum(t0 * mf, axis=1))
    threshold = filtered[int(0.995 * len(filtered))]
    return mf, threshold


def _apply_boxcar(envelope: np.ndarray, trace_length: int, boxcar_width: int) -> np.ndarray:
    """
    Zero-out the envelope beyond the boxcar window.
    boxcar_width is in units of 50 ADC samples (follows relaxation_mf_classifier.fit).
    """
    bc_axis = np.arange(len(envelope))
    window  = np.heaviside((len(envelope) - boxcar_width / 50) - bc_axis, 1)
    return envelope * window


def compute_all_envelopes(demod_train: dict, y_train: np.ndarray):
    """
    Compute full-length MF and RMF envelopes for all 5 qubits from training data.

    Returns
    -------
    mf_envelopes  : list[ndarray]  shape (T_full*2,) per qubit
    rmf_envelopes : list[ndarray]  shape (T_full*2,) per qubit
    """
    mf_envelopes  = []
    rmf_envelopes = []

    for qubit in range(1, NUM_QUBITS + 1):
        logger.info(f"Computing envelopes for qubit {qubit} ...")
        q_traces = demod_train[qubit - 1]
        F_full   = q_traces.shape[1] * 2

        tc = preclassify_qubit(q_traces, y_train, qubit)
        if tc is None:
            mf_envelopes.append(np.zeros(F_full))
            rmf_envelopes.append(np.zeros(F_full))
            continue

        t0_flat    = tc["traces_0"].reshape(-1, F_full)
        t1_flat    = tc["traces_1"].reshape(-1, F_full)
        relax_flat = tc["traces_relax"].reshape(-1, F_full) if len(tc["traces_relax"]) > 10 else None

        mf, _ = _get_mf_core(t0_flat, t1_flat)
        mf    = _apply_boxcar(mf, q_traces.shape[1], BOXCARS[qubit - 1])
        mf_envelopes.append(mf)

        if relax_flat is not None and len(relax_flat) > 10:
            rmf, _ = _get_mf_core(relax_flat, t0_flat)
            rmf    = _apply_boxcar(rmf, q_traces.shape[1], BOXCARS[qubit - 1])
        else:
            logger.warning(f"  Qubit {qubit}: insufficient relaxation traces — RMF set to zero.")
            rmf = np.zeros(F_full)
        rmf_envelopes.append(rmf)

    return mf_envelopes, rmf_envelopes


# ============================================================================
# 5. Feature extraction: apply truncated envelopes → 10-dim vectors
# ============================================================================

def build_features(demod_data: dict, mf_envelopes: list,
                   rmf_envelopes: list, trace_length: int) -> np.ndarray:
    """
    Project demodulated traces onto truncated MF/RMF envelopes.

    Follows relaxation_mf_classifier.predict():
        envelope_to_use = full_envelope[:trace_length * 2]

    Returns
    -------
    features : (N, 10)  — [MF_q0..q4, RMF_q0..q4]
    """
    N   = demod_data[0].shape[0]
    F   = trace_length * 2
    mf_out  = np.zeros((N, NUM_QUBITS), dtype=np.float32)
    rmf_out = np.zeros((N, NUM_QUBITS), dtype=np.float32)

    for q in range(NUM_QUBITS):
        flat    = demod_data[q][:, :trace_length, :].reshape(N, F)
        mf_out[:, q]  = flat @ mf_envelopes[q][:F].astype(np.float32)
        rmf_out[:, q] = flat @ rmf_envelopes[q][:F].astype(np.float32)

    return np.concatenate([mf_out, rmf_out], axis=1)   # (N, 10)


# ============================================================================
# 6. LR schedule — paper's adjust_learning_rate (step-decay at 30/60/90)
# ============================================================================

def adjust_lr(initial_lr: float, optimizer, epoch: int) -> float:
    """Apply the paper's step-decay schedule in-place."""
    lr = initial_lr
    for milestone in LR_SCHEDULE:
        if epoch >= milestone:
            lr *= 0.1
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ============================================================================
# 7. Training  (follows train() in HERQULES_original.py, fixed hyperparams)
# ============================================================================

def train_herqules(X_train_feat: np.ndarray, y_train: np.ndarray,
                   X_val_feat:   np.ndarray, y_val:   np.ndarray,
                   trace_length: int) -> tuple:
    """
    Train Net_rmf with the paper's exact fixed hyperparameters.

    Parameters match train() in HERQULES_original.py:
        lr=0.01, batch=512, 100 epochs, Adam, CrossEntropyLoss,
        step-decay schedule at epochs 30/60/90.

    Returns
    -------
    (model, best_val_acc, best_epoch, model_path)
    """
    train_ds = TensorDataset(
        torch.tensor(X_train_feat, dtype=torch.float32),
        torch.tensor(y_train,      dtype=torch.long))
    val_ds = TensorDataset(
        torch.tensor(X_val_feat, dtype=torch.float32),
        torch.tensor(y_val,      dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model     = Net_rmf().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LRN_RATE)

    best_val_acc  = -1.0
    best_epoch    = 0
    model_path    = os.path.join(SAVE_DIR, f"HERQULES_best_len{trace_length}.pth")

    for epoch in range(MAX_EPOCHS):
        lr = adjust_lr(LRN_RATE, optimizer, epoch)

        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # --- Validate (accuracy, matching paper's best-checkpoint criterion) ---
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                preds     = model(X_b).argmax(dim=1)
                correct  += (preds == y_b).sum().item()
                total    += y_b.size(0)
        val_acc = correct / total

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), model_path)

        if epoch % 10 == 0:
            logger.info(f"    epoch {epoch:3d}/{MAX_EPOCHS}  "
                        f"loss={epoch_loss:.4f}  val_acc={val_acc*100:.2f}%  lr={lr:.5f}")

    logger.info(f"  Best val_acc={best_val_acc*100:.2f}% at epoch {best_epoch}")

    # Reload best checkpoint
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model, best_val_acc, best_epoch, model_path


# ============================================================================
# 8. Evaluation
# ============================================================================

def evaluate_32class(model, X_feat: np.ndarray, y: np.ndarray,
                     batch_size: int = 512):
    """Return (overall_acc %, per_qubit_accs list[%])."""
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_feat, dtype=torch.float32)),
        batch_size=batch_size)
    preds = []
    with torch.no_grad():
        for (X_b,) in loader:
            preds.append(model(X_b.to(DEVICE)).argmax(dim=1).cpu().numpy())
    pred_labels = np.concatenate(preds)
    overall     = 100.0 * np.mean(pred_labels == y)
    per_q, pl, yl = [], pred_labels.copy(), y.copy()
    for _ in range(NUM_QUBITS):
        per_q.append(100.0 * np.mean((pl & 1) == (yl & 1)))
        pl >>= 1; yl >>= 1
    return overall, per_q


def get_model_info(model):
    total  = sum(p.numel() for p in model.parameters())
    layers = [f"{n}: {m.__class__.__name__}"
              for n, m in model.named_modules()
              if n and not isinstance(m, (nn.Sequential, nn.ModuleList))]
    return total, len(layers), layers


# ============================================================================
# 9. CSV report
# ============================================================================

def save_csv(model_name: str, model, trace_length: int, best_val_acc: float,
             best_epoch: int, overall_acc: float, per_qubit_accs: list,
             model_path: str):
    total_p, n_layers, layer_desc = get_model_info(model)
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(CSV_DIR, f"{model_name}_len{trace_length}_{ts}.csv")

    q_cols = {f"qubit_{q}_accuracy": f"{per_qubit_accs[q]:.4f}"
              for q in range(NUM_QUBITS)}
    row = {
        "timestamp":        ts,
        "model_name":       model_name,
        "trace_length":     trace_length,
        "optimizer":        "Adam",
        "learning_rate":    LRN_RATE,
        "batch_size":       BATCH_SIZE,
        "epochs":           MAX_EPOCHS,
        "lr_schedule":      str(LR_SCHEDULE),
        "best_epoch":       best_epoch,
        "best_val_acc":     f"{best_val_acc*100:.4f}",
        "total_parameters": total_p,
        "num_layers":       n_layers,
        "layer_descriptions": " | ".join(layer_desc),
        "overall_accuracy": f"{overall_acc:.4f}",
        **q_cols,
        "device":           str(DEVICE),
        "model_path":       model_path,
    }
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        w.writeheader(); w.writerow(row)
    logger.info(f"CSV report saved: {fname}")
    return fname


# ============================================================================
# 10. Main loop
# ============================================================================

def run():
    logger.info("=== HERQULES Training Pipeline (paper-faithful) ===")
    logger.info(f"Device: {DEVICE}  |  lr={LRN_RATE}  |  batch={BATCH_SIZE}  |  epochs={MAX_EPOCHS}")

    # --- Load full datasets ---
    X_train_raw, y_train = load_hdf5_full(RAW_TRAIN_FILE, is_test=False)
    X_test_raw,  y_test  = load_hdf5_full(RAW_TEST_FILE,  is_test=True)

    # --- Demodulate once (full 500-sample traces) ---
    logger.info("Demodulating training traces ...")
    demod_train = demodulate_all_qubits(X_train_raw)
    logger.info("Demodulating test traces ...")
    demod_test  = demodulate_all_qubits(X_test_raw)
    del X_train_raw, X_test_raw

    # --- Compute MF/RMF envelopes from purified training traces ---
    logger.info("Computing MF / RMF envelopes ...")
    mf_envelopes, rmf_envelopes = compute_all_envelopes(demod_train, y_train)

    # --- Loop over trace lengths ---
    for length in TRACE_LENGTHS:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Trace length: {length} samples")
        logger.info(f"{'='*60}")

        # Build 10-dim feature arrays for this trace length
        X_train_feat = build_features(demod_train, mf_envelopes, rmf_envelopes, length)
        X_test_feat  = build_features(demod_test,  mf_envelopes, rmf_envelopes, length)
        logger.info(f"  Feature shapes: train={X_train_feat.shape}, test={X_test_feat.shape}")

        # Train / validation split (80/20, stratified — paper uses val for best-ckpt selection)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_feat, y_train, test_size=0.2, random_state=42, stratify=y_train)

        # Train with paper's fixed hyperparameters
        model, best_val_acc, best_epoch, model_path = train_herqules(
            X_tr, y_tr, X_val, y_val, length)

        # Evaluate on held-out test set
        overall_acc, per_qubit_accs = evaluate_32class(model, X_test_feat, y_test)
        logger.info(f"  Test overall acc : {overall_acc:.2f}%")
        for q, acc in enumerate(per_qubit_accs):
            logger.info(f"    Qubit {q}: {acc:.2f}%")

        save_csv("HERQULES_Net_rmf", model, length, best_val_acc, best_epoch,
                 overall_acc, per_qubit_accs, model_path)

    logger.info("\n=== Done ===")


if __name__ == "__main__":
    run()
