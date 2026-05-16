"""
hyper_optimize.py
=================
Hyper-optimization script for Qubit Discriminators using Optuna.

This script loops over a specified list of trace lengths, and for each length, 
it runs an Optuna study to find the best hyperparameters (learning rate, batch size, etc.)
for a given model architecture. The best model across all trials for each trace length
is automatically saved.

Configuration
-------------
Edit the `RAW_TRAIN_FILE` and `RAW_TEST_FILE` variables to point to your datasets.
Edit the `MODELS_TO_OPTIMIZE` list to select which models to train.
"""

import os
import csv
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import optuna
from loguru import logger
import argparse
import sys

# Add parent directory to sys.path so we can import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models
from networks import (
    SingleQubitFNN,
    CNN,
    Arxiv240618807FNN,
    Net_rmf,
    Net,
    KLiNQTeacherModel,
    KLiNQStudentModel,
    QubitClassifierTransformer,
)

# Import helpers
from helpers.cnn_helpers import prepare_cnn_data

# ============================================================================
# User Configuration
# ============================================================================

# Paths inside the Docker container (mounted via docker run -v)
RAW_TRAIN_FILE = "/data/five_qubit_data/DRaw_C_Tr_v0-001"
RAW_TEST_FILE = "/data/five_qubit_data/DRaw_C_Te_v0-002"

# CNN model uses a differently preprocessed (downsampled) file.
CNN_TRAIN_FILE = "/data/cnn/Qubit_5Channel_ds20_train.h5"
CNN_TEST_FILE = "/data/cnn/Qubit_5Channel_ds20_test.h5"

NUM_QUBITS = 5
TRACE_LENGTHS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500] 
N_OPTUNA_TRIALS = 30
EPOCHS_PER_TRIAL = 25

# Subsample the dataset to avoid using the full ~1.5M traces.
# Set to None to use all data.
MAX_TOTAL_SAMPLES = 400_000
MAX_TRAIN_SAMPLES = int(MAX_TOTAL_SAMPLES * 0.8) if MAX_TOTAL_SAMPLES else None  # 400k
MAX_TEST_SAMPLES = int(MAX_TOTAL_SAMPLES * 0.2) if MAX_TOTAL_SAMPLES else None   # 100k
SAMPLE_SEED = 42  # For reproducible subsampling
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./saved_models"
CSV_DIR = "./optimization_reports"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# IF Frequencies for demodulation (Hz)
FREQ_READOUT = -np.array([-64.729e6, -25.366e6, 24.79e6, 70.269e6, 127.282e6])

# ============================================================================
# Data Handlers
# ============================================================================

def load_hdf5_data(filepath, trace_length, is_test=False, max_samples=None):
    """Loads and truncates data from HDF5, optionally subsampling.
    
    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    trace_length : int
        Number of time steps to keep per trace.
    is_test : bool
        Whether to load test or train keys.
    max_samples : int or None
        If set, randomly subsample to at most this many traces.
        Uses SAMPLE_SEED for reproducibility.
    """
    key_suffix = "test" if is_test else "train"
    with h5py.File(filepath, "r") as hf:
        total = hf[f"X_{key_suffix}"].shape[0]
        if max_samples is not None and max_samples < total:
            rng = np.random.RandomState(SAMPLE_SEED)
            indices = np.sort(rng.choice(total, size=max_samples, replace=False))
            X = hf[f"X_{key_suffix}"][indices, :trace_length, :]
            y = hf[f"y_{key_suffix}"][indices]
            logger.info(f"Subsampled {key_suffix} data: {total} -> {max_samples} traces")
        else:
            X = hf[f"X_{key_suffix}"][:, :trace_length, :]
            y = hf[f"y_{key_suffix}"][:]
    return X, y

def demodulate_and_average(traces, freq_readout, dt=2e-9):
    """Demodulates and integrates the trace for Arxiv model."""
    N, T, _ = traces.shape
    num_qubits = len(freq_readout)
    t = np.arange(T) * dt
    
    demodulated_acc = np.zeros((N, num_qubits, 2))
    
    for i, freq in enumerate(freq_readout):
        lo_i = np.cos(2 * np.pi * freq * t)
        lo_q = -np.sin(2 * np.pi * freq * t)
        
        I_trace = traces[:, :, 0]
        Q_trace = traces[:, :, 1]
        
        I_demod = I_trace * lo_i - Q_trace * lo_q
        Q_demod = I_trace * lo_q + Q_trace * lo_i
        
        demodulated_acc[:, i, 0] = np.mean(I_demod, axis=1)
        demodulated_acc[:, i, 1] = np.mean(Q_demod, axis=1)
        
    return demodulated_acc

def extract_qubit_labels(y_packed, target_qubit):
    """Extracts binary label for a specific qubit."""
    return (y_packed >> target_qubit) & 1

# ============================================================================
# Model Introspection & CSV Export
# ============================================================================

def get_model_layer_info(model):
    """Returns total parameter count, number of layers, and a description string of each layer."""
    total_params = sum(p.numel() for p in model.parameters())
    layers = []
    for name, module in model.named_modules():
        # Skip the top-level module itself and container modules
        if name == '' or isinstance(module, (nn.Sequential, nn.ModuleList, nn.TransformerEncoder)):
            continue
        layers.append(f"{name}: {module.__class__.__name__}")
    return total_params, len(layers), layers


def evaluate_test_accuracy(model, X_test, y_test, task_type, batch_size=512):
    """Evaluate overall and per-qubit accuracy on the test set.

    Parameters
    ----------
    model : nn.Module
    X_test : np.ndarray
    y_test : np.ndarray
        For 32-class models: integer labels 0-31.
        For per-qubit binary models: binary labels 0/1.
        For CNN multi-task: (N, 5) binary labels.
    task_type : str
        One of '32class', 'binary', 'multitask'.
    batch_size : int

    Returns
    -------
    overall_acc : float
        Overall accuracy as a percentage.
    per_qubit_accs : list[float]
        Per-qubit accuracy percentages (length 5). For binary models that
        target a single qubit, only that qubit's slot is filled; others are NaN.
    """
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size)

    all_preds = []
    with torch.no_grad():
        for (X_b,) in loader:
            X_b = X_b.to(DEVICE)
            out = model(X_b)
            all_preds.append(out.cpu())
    all_preds = torch.cat(all_preds, dim=0)

    if task_type == '32class':
        pred_labels = all_preds.argmax(dim=1).numpy()
        overall_acc = 100.0 * np.mean(pred_labels == y_test)
        # Per-qubit: extract bit for each qubit
        per_qubit_accs = []
        for q in range(NUM_QUBITS):
            pred_q = (pred_labels >> q) & 1
            true_q = (y_test >> q) & 1
            per_qubit_accs.append(100.0 * np.mean(pred_q == true_q))
        return overall_acc, per_qubit_accs

    elif task_type == 'multitask':
        # CNN: output shape (N, 5), apply sigmoid threshold
        pred_binary = (torch.sigmoid(all_preds) >= 0.5).int().numpy()
        y_int = y_test.astype(int)
        # Overall = all 5 qubits correct
        overall_acc = 100.0 * np.mean(np.all(pred_binary == y_int, axis=1))
        per_qubit_accs = [
            100.0 * np.mean(pred_binary[:, q] == y_int[:, q])
            for q in range(NUM_QUBITS)
        ]
        return overall_acc, per_qubit_accs

    elif task_type == 'binary':
        # Per-qubit binary model: output shape (N, 1)
        # Arxiv model already has a Sigmoid at the end and outputs probabilities.
        # Other binary models (like KLiNQ) output raw logits.
        if isinstance(model, Arxiv240618807FNN):
            preds = all_preds.squeeze()
        else:
            preds = torch.sigmoid(all_preds.squeeze())
        
        pred_binary = (preds >= 0.5).int().numpy()
        acc = 100.0 * np.mean(pred_binary == y_test)
        return acc, None  # caller will place into the right qubit slot

    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def save_model_report_csv(
    model_name, model, study, trace_length, epochs,
    overall_acc, per_qubit_accs, target_qubit=None,
    extra_hparams=None,
):
    """Save a CSV report for a completed hyper-optimization run.

    One CSV file is created per (model_name, trace_length, [qubit]) combination.
    """
    best_trial = study.best_trial
    best_lr = best_trial.params.get('lr', 'N/A')
    best_batch_size = best_trial.params.get('batch_size', 'N/A')

    total_params, num_layers, layer_descs = get_model_layer_info(model)

    # Build filename
    qubit_tag = f"_q{target_qubit}" if target_qubit is not None else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}{qubit_tag}_len{trace_length}_{timestamp}.csv"
    filepath = os.path.join(CSV_DIR, filename)

    # Collect per-qubit accuracy values
    q_accs = {}
    for q in range(NUM_QUBITS):
        if per_qubit_accs is not None:
            q_accs[f"qubit_{q}_accuracy"] = f"{per_qubit_accs[q]:.4f}"
        else:
            q_accs[f"qubit_{q}_accuracy"] = "N/A"

    # Collect extra hyperparameters from the trial (e.g. Transformer-specific)
    extra = {}
    if extra_hparams:
        for key in extra_hparams:
            extra[key] = best_trial.params.get(key, 'N/A')

    row = {
        "timestamp": timestamp,
        "model_name": model_name,
        "target_qubit": target_qubit if target_qubit is not None else "all",
        "trace_length": trace_length,
        "optimizer": "Adam",
        "learning_rate": best_lr,
        "batch_size": best_batch_size,
        "epochs": epochs,
        "n_optuna_trials": len(study.trials),
        "best_trial_number": best_trial.number,
        "best_val_loss": f"{best_trial.value:.6f}",
        "total_parameters": total_params,
        "num_layers": num_layers,
        "layer_descriptions": " | ".join(layer_descs),
        "overall_accuracy": f"{overall_acc:.4f}",
        **q_accs,
        **extra,
        "device": str(DEVICE),
        "model_path": best_trial.user_attrs.get("model_path", "N/A"),
    }

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    logger.info(f"CSV report saved: {filepath}")
    return filepath


# ============================================================================
# Optuna Objectives
# ============================================================================

def objective_arxiv(trial, X_q_train, y_q_train, trace_length, target_qubit):
    """Optuna objective for the Arxiv240618807FNN."""
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    # Normalize to [0, 1] as in the paper
    X_min = np.min(X_q_train, axis=0)
    X_range = np.max(X_q_train, axis=0) - X_min
    X_range[X_range == 0] = 1e-10
    X_norm = (X_q_train - X_min) / X_range
    
    X_train, X_val, y_train, y_val = train_test_split(X_norm, y_q_train, test_size=0.2, random_state=42, stratify=y_q_train)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=batch_size)
    
    model = Arxiv240618807FNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE).unsqueeze(1)
                val_loss += criterion(model(X_b), y_b).item() * X_b.size(0)
        val_loss /= len(val_loader.dataset)
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    # Save model if it's the best so far
    model_path = os.path.join(SAVE_DIR, f"Arxiv_q{target_qubit}_len{trace_length}_trial{trial.number}.pth")
    torch.save(model.state_dict(), model_path)
    trial.set_user_attr("model_path", model_path)
    
    return val_loss

def objective_fnn(trial, X_raw, y_raw, trace_length):
    """Optuna objective for SingleQubitFNN (Raw trace, 32-class)."""
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    
    # Flatten IQ dimensions
    X_flat = X_raw.reshape(X_raw.shape[0], -1)
    
    # Z-score normalization
    X_mean = np.mean(X_flat, axis=0)
    X_std = np.std(X_flat, axis=0) + 1e-10
    X_norm = (X_flat - X_mean) / X_std
    
    X_train, X_val, y_train, y_val = train_test_split(X_norm, y_raw, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)), batch_size=batch_size)
    
    model = SingleQubitFNN(input_size=trace_length * 2, output_size=2 ** NUM_QUBITS).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                val_loss += criterion(model(X_b), y_b).item() * X_b.size(0)
        val_loss /= len(val_loader.dataset)
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    model_path = os.path.join(SAVE_DIR, f"FNN_len{trace_length}_trial{trial.number}.pth")
    torch.save(model.state_dict(), model_path)
    trial.set_user_attr("model_path", model_path)
    
    return val_loss

def objective_cnn(trial, trace_length):
    """Optuna objective for CNN (Multi-task, 5-qubit)."""
    if not os.path.exists(CNN_TRAIN_FILE):
        raise FileNotFoundError(f"CNN data file missing: {CNN_TRAIN_FILE}")

    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    m_param = trial.suggest_categorical("m_param", [8, 16, 32, 64])

    with h5py.File(CNN_TRAIN_FILE, "r") as hf:
        total = hf["X_train"].shape[0]
        if MAX_TRAIN_SAMPLES is not None and MAX_TRAIN_SAMPLES < total:
            rng = np.random.RandomState(SAMPLE_SEED)
            indices = np.sort(rng.choice(total, size=MAX_TRAIN_SAMPLES, replace=False))
            X = hf["X_train"][indices]
            y = hf["y_train"][indices]
        else:
            X = hf["X_train"][:]
            y = hf["y_train"][:]

    # Handle feature shape: (N, Channels, Time) for PyTorch
    # If shape is (N, T, Q, IQ) as in extract_data.py, flatten Q*IQ to 10 channels and transpose
    if len(X.shape) == 4: 
        X = X.reshape(X.shape[0], X.shape[1], -1) # (N, T, 10)
        X = np.transpose(X, (0, 2, 1)) # (N, 10, T)
    elif len(X.shape) == 3 and X.shape[1] == 10: 
        pass # Already in correct (N, 10, T) format

    # Handle labels shape: (N, Q) for multitask
    if len(y.shape) == 1: # Packed integers -> bit matrix
        y = np.array([[(label >> i) & 1 for i in range(NUM_QUBITS)] for label in y], dtype=np.float32)

    # Scale trace_length for downsampled data (e.g., 500 samples @ ds=20 -> 25 steps)
    steps = max(1, trace_length // 20)
    X = X[:, :, :steps]
    
    X_train, X_val, y_train, y_val = train_test_split(X.astype(np.float32), y.astype(np.float32), test_size=0.2, random_state=42)

    # Normalization: Calculate stats on training split and apply to validation
    mean = np.mean(X_train, axis=(0, 2), keepdims=True)
    std = np.std(X_train, axis=(0, 2), keepdims=True) + 1e-10
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # Save best trial stats for evaluation later
    trial.set_user_attr("mean", mean.tolist())
    trial.set_user_attr("std", std.tolist())
    trial.set_user_attr("m_param", m_param)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=batch_size)
    
    model = CNN(in_channels=10, m_param=m_param, num_qubits=NUM_QUBITS).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                val_loss += criterion(model(X_b), y_b).item() * X_b.size(0)
        val_loss /= len(val_loader.dataset)
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    model_path = os.path.join(SAVE_DIR, f"CNN_len{trace_length}_trial{trial.number}.pth")
    torch.save(model.state_dict(), model_path)
    trial.set_user_attr("model_path", model_path)
    
    return val_loss

def objective_herqules(trial, X_raw, y_raw, trace_length):
    """Optuna objective for HERQULES Net_rmf (MF + RMF, 32-class).

    Computes both standard matched-filter (MF) and relaxation matched-filter
    (RMF) features per qubit, yielding a 10-dimensional input vector for
    the Net_rmf classifier.

    MF:  E[x_0 - x_1] / Var[x_0 - x_1]   — distinguishes |0⟩ vs |1⟩
    RMF: E[x_relax - x_0] / Var[x_relax - x_0]  — detects |1⟩→|0⟩ relaxation
    """
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    X_flat = X_raw.reshape(X_raw.shape[0], -1)

    # --- Compute MF + RMF features per qubit ---
    mf_features = np.zeros((X_raw.shape[0], NUM_QUBITS))
    rmf_features = np.zeros((X_raw.shape[0], NUM_QUBITS))

    for q in range(NUM_QUBITS):
        y_q = extract_qubit_labels(y_raw, q)
        gnd = X_flat[y_q == 0]  # |0⟩ traces
        ext = X_flat[y_q == 1]  # |1⟩ traces

        # Standard MF: E[x_0 - x_1] / Var[x_0 - x_1]
        n = min(len(gnd), len(ext))
        diff_mf = gnd[:n] - ext[:n]
        mf_envelope = np.mean(diff_mf, axis=0) / (np.var(diff_mf, axis=0) + 1e-10)
        mf_out = X_flat @ mf_envelope
        mf_features[:, q] = mf_out

        # RMF: identify relaxation traces (|1⟩-labelled but MF output near |0⟩)
        # Use the MF threshold to find |1⟩ traces that look like |0⟩
        mf_gnd = mf_out[y_q == 0]
        mf_ext = mf_out[y_q == 1]
        threshold = np.mean(mf_gnd)  # centre of |0⟩ distribution
        sigma_gnd = np.std(mf_gnd) + 1e-10

        # Relaxation: |1⟩-labelled traces within 2σ of the |0⟩ mean
        relax_mask = (y_q == 1) & (np.abs(mf_out - threshold) < 2 * sigma_gnd)
        relax_traces = X_flat[relax_mask]

        if len(relax_traces) > 10:  # need enough traces for a stable estimate
            n_rmf = min(len(relax_traces), len(gnd))
            diff_rmf = relax_traces[:n_rmf] - gnd[:n_rmf]
            rmf_envelope = np.mean(diff_rmf, axis=0) / (np.var(diff_rmf, axis=0) + 1e-10)
            rmf_features[:, q] = X_flat @ rmf_envelope
        else:
            # Fall back to zero RMF if not enough relaxation traces
            rmf_features[:, q] = 0.0

    # Concatenate MF + RMF → 10-dim feature vector
    combined_features = np.concatenate([mf_features, rmf_features], axis=1)

    X_train, X_val, y_train, y_val = train_test_split(
        combined_features, y_raw, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size)

    model = Net_rmf().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                val_loss += criterion(model(X_b), y_b).item() * X_b.size(0)
        val_loss /= len(val_loader.dataset)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    model_path = os.path.join(SAVE_DIR, f"HERQULES_rmf_len{trace_length}_trial{trial.number}.pth")
    torch.save(model.state_dict(), model_path)
    trial.set_user_attr("model_path", model_path)

    return val_loss

def objective_klinq_teacher(trial, X_raw, y_raw, trace_length, target_qubit):
    """Optuna objective for KLiNQ TeacherModel (per-qubit binary).

    Stage 1 of the KLiNQ knowledge-distillation pipeline.
    The teacher receives the full flattened IQ trace and learns binary
    classification for a single qubit.
    """
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    y_q = extract_qubit_labels(y_raw, target_qubit)

    # Flatten IQ and z-score normalise
    X_flat = X_raw.reshape(X_raw.shape[0], -1)
    X_mean = np.mean(X_flat, axis=0)
    X_std = np.std(X_flat, axis=0) + 1e-10
    X_norm = (X_flat - X_mean) / X_std

    X_train, X_val, y_train, y_val = train_test_split(
        X_norm, y_q, test_size=0.2, random_state=42, stratify=y_q
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.float32)),
        batch_size=batch_size)

    model = KLiNQTeacherModel(input_size=trace_length * 2, output_size=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE).unsqueeze(1)
                val_loss += criterion(model(X_b), y_b).item() * X_b.size(0)
        val_loss /= len(val_loader.dataset)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    model_path = os.path.join(SAVE_DIR, f"KLiNQ_teacher_q{target_qubit}_len{trace_length}_trial{trial.number}.pth")
    torch.save(model.state_dict(), model_path)
    trial.set_user_attr("model_path", model_path)

    return val_loss


def objective_klinq_student(trial, X_raw, y_raw, trace_length, target_qubit, teacher_model):
    """Optuna objective for KLiNQ StudentModel with knowledge distillation.

    Stage 2 of the KLiNQ knowledge-distillation pipeline.
    The student receives a compact feature vector (flat trace + averaged
    trace + matched-filter scalar) and is trained using a composite loss:
        L = alpha * MSE(student/T, teacher/T) * T^2 + (1-alpha) * BCE(student, hard_label)

    The teacher is frozen and provides soft targets each batch.
    """
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    temperature = trial.suggest_categorical("temperature", [1, 2, 3, 4, 5])
    alpha = trial.suggest_categorical("alpha", [0.2, 0.3, 0.5, 0.7, 0.8, 0.9])

    target_length = 5  # averaging target bins per channel
    y_q = extract_qubit_labels(y_raw, target_qubit)

    # --- Build student feature vector ---
    # 1) Flat trace (same input the teacher sees, for the MF computation)
    X_flat = X_raw.reshape(X_raw.shape[0], -1).copy()

    # 2) Time-averaged trace (downsample each IQ channel to target_length bins)
    bin_size = max(1, trace_length // target_length)
    n_bins = trace_length // bin_size
    X_avg_I = X_raw[:, :n_bins * bin_size, 0].reshape(X_raw.shape[0], n_bins, bin_size).mean(axis=2)
    X_avg_Q = X_raw[:, :n_bins * bin_size, 1].reshape(X_raw.shape[0], n_bins, bin_size).mean(axis=2)
    X_avg = np.concatenate([X_avg_I, X_avg_Q], axis=1)

    # 3) Matched-filter scalar for this qubit
    gnd, ext = X_flat[y_q == 0], X_flat[y_q == 1]
    n = min(len(gnd), len(ext))
    diff = gnd[:n] - ext[:n]
    envelope = np.mean(diff, axis=0) / (np.var(diff, axis=0) + 1e-10)
    mf_scalar = (X_flat @ envelope).reshape(-1, 1)

    # Z-score normalize each component, then concatenate
    def _znorm(a):
        return (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-10)

    X_student = np.concatenate([_znorm(X_flat), _znorm(X_avg), _znorm(mf_scalar)], axis=1)
    student_input_size = X_student.shape[1]

    # --- Build teacher input (z-score normalised flat trace) ---
    X_flat_norm = _znorm(X_raw.reshape(X_raw.shape[0], -1).copy())

    X_stu_train, X_stu_val, X_tea_train, X_tea_val, y_train, y_val = train_test_split(
        X_student, X_flat_norm, y_q, test_size=0.2, random_state=42, stratify=y_q
    )

    train_ds = TensorDataset(
        torch.tensor(X_stu_train, dtype=torch.float32),
        torch.tensor(X_tea_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_stu_val, dtype=torch.float32),
        torch.tensor(X_tea_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    student = KLiNQStudentModel(input_size=student_input_size).to(DEVICE)
    bce_criterion = nn.BCEWithLogitsLoss()
    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=lr)

    # Freeze teacher
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    for epoch in range(EPOCHS_PER_TRIAL):
        student.train()
        for X_stu_b, X_tea_b, y_b in train_loader:
            X_stu_b = X_stu_b.to(DEVICE)
            X_tea_b = X_tea_b.to(DEVICE)
            y_b = y_b.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()

            # Teacher soft targets (temperature-scaled)
            with torch.no_grad():
                teacher_logits = teacher_model(X_tea_b) / temperature

            # Student predictions (temperature-scaled for distillation)
            student_logits = student(X_stu_b)
            student_logits_scaled = student_logits / temperature

            # Composite loss (following KnowledgeDistillationTrainer_KLiNQ)
            distillation_loss = mse_criterion(student_logits_scaled, teacher_logits) * (temperature ** 2)
            classification_loss = bce_criterion(student_logits, y_b)
            loss = alpha * distillation_loss + (1 - alpha) * classification_loss

            loss.backward()
            optimizer.step()

        # Validation (student only, using hard-label BCE)
        student.eval()
        val_loss = 0
        with torch.no_grad():
            for X_stu_b, X_tea_b, y_b in val_loader:
                X_stu_b = X_stu_b.to(DEVICE)
                y_b = y_b.to(DEVICE).unsqueeze(1)
                val_loss += bce_criterion(student(X_stu_b), y_b).item() * X_stu_b.size(0)
        val_loss /= len(val_loader.dataset)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    model_path = os.path.join(SAVE_DIR, f"KLiNQ_student_q{target_qubit}_len{trace_length}_trial{trial.number}.pth")
    torch.save(student.state_dict(), model_path)
    trial.set_user_attr("model_path", model_path)
    trial.set_user_attr("student_input_size", student_input_size)

    return val_loss

def objective_transformer(trial, X_raw, y_raw, trace_length):
    """Optuna objective for QubitClassifierTransformer (raw IQ, 32-class)."""
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    patch_size = trial.suggest_categorical(
        "patch_size", [p for p in [5, 10, 20, 25, 50] if trace_length % p == 0]
    )
    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
    num_heads = trial.suggest_categorical("num_heads", [4, 8])
    num_layers = trial.suggest_int("num_layers", 2, 6)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)

    if embedding_dim % num_heads != 0:
        raise optuna.exceptions.TrialPruned()

    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size)

    model = QubitClassifierTransformer(
        num_classes=2 ** NUM_QUBITS, patch_size=patch_size,
        embedding_dim=embedding_dim, num_heads=num_heads,
        num_layers=num_layers, dropout=dropout
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                val_loss += criterion(model(X_b), y_b).item() * X_b.size(0)
        val_loss /= len(val_loader.dataset)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    model_path = os.path.join(SAVE_DIR, f"Transformer_len{trace_length}_trial{trial.number}.pth")
    torch.save(model.state_dict(), model_path)
    trial.set_user_attr("model_path", model_path)

    return val_loss


# ============================================================================
# Main Optimizer Loop
# ============================================================================

def _load_best_model(model_class, model_path, **kwargs):
    """Instantiate a model and load the best trial's weights."""
    model = model_class(**kwargs).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def optimize_models(models_to_run=["FNN", "Arxiv240618807FNN", "CNN", "HERQULES_rmf", "Transformer", "KLiNQ_teacher", "KLiNQ_student"]):
    """Main function to run hyper-parameter optimization loops."""
    logger.info("Starting Hyper-Optimization Pipeline")
    
    for length in TRACE_LENGTHS:
        logger.info(f"=== Optimizing for Trace Length: {length} ===")

        # Pre-load all data once per trace length to avoid redundant HDF5 I/O
        X_test_raw, y_test_raw = load_hdf5_data(RAW_TEST_FILE, length, is_test=True, max_samples=MAX_TEST_SAMPLES)
        X_train_raw, y_train_raw = load_hdf5_data(RAW_TRAIN_FILE, length, is_test=False, max_samples=MAX_TRAIN_SAMPLES)
        logger.info(f"Loaded {X_train_raw.shape[0]} train + {X_test_raw.shape[0]} test traces")

        X_test_flat = X_test_raw.reshape(X_test_raw.shape[0], -1)
        X_train_flat = X_train_raw.reshape(X_train_raw.shape[0], -1)

        # Z-score normalization stats for FNN
        fnn_mean = np.mean(X_train_flat, axis=0)
        fnn_std = np.std(X_train_flat, axis=0) + 1e-10
        X_test_fnn = (X_test_flat - fnn_mean) / fnn_std

        # ----------------------------------------------------------------
        # 1. Optimize FNN (raw trace, 32-class)
        # ----------------------------------------------------------------
        if "FNN" in models_to_run:
            logger.info(f"Optimizing FNN")
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective_fnn(trial, X_train_raw, y_train_raw, length), n_trials=N_OPTUNA_TRIALS)
            best_model_path = study.best_trial.user_attrs["model_path"]
            logger.info(f"Best FNN (Length {length}) saved at: {best_model_path}")

            model = _load_best_model(SingleQubitFNN, best_model_path,
                                    input_size=length * 2, output_size=2 ** NUM_QUBITS)
            overall_acc, per_qubit_accs = evaluate_test_accuracy(
                model, X_test_fnn, y_test_raw, task_type='32class')
            save_model_report_csv("FNN", model, study, length, EPOCHS_PER_TRIAL,
                                overall_acc, per_qubit_accs)

        # ----------------------------------------------------------------
        # 2. Optimize Arxiv240618807FNN per qubit
        # ----------------------------------------------------------------
        # Demodulate test data once for all qubits
        if "Arxiv240618807FNN" in models_to_run:
            X_test_demod = demodulate_and_average(X_test_raw, FREQ_READOUT)
            X_train_demod = demodulate_and_average(X_train_raw, FREQ_READOUT)

            arxiv_qubit_accs = [None] * NUM_QUBITS
            for qubit in range(NUM_QUBITS):
                logger.info(f"Optimizing Arxiv240618807FNN (Qubit {qubit})")
                
                # Extract demodulated features and binary labels for the current qubit
                X_q_train = X_train_demod[:, qubit, :]
                y_q_train = extract_qubit_labels(y_train_raw, qubit)

                study = optuna.create_study(direction="minimize")
                study.optimize(lambda trial, q=qubit, x_q=X_q_train, y_q=y_q_train: objective_arxiv(trial, x_q, y_q, length, q), n_trials=N_OPTUNA_TRIALS)
                
                best_model_path = study.best_trial.user_attrs["model_path"]
                logger.info(f"Best Arxiv240618807FNN (Qubit {qubit}, Length {length}) saved at: {best_model_path}")

                model = _load_best_model(Arxiv240618807FNN, best_model_path)
                # Prepare per-qubit test data with same normalization
                X_q_test = X_test_demod[:, qubit, :]
                X_min = np.min(X_q_train, axis=0)
                X_range = np.max(X_q_train, axis=0) - X_min
                X_range[X_range == 0] = 1e-10
                X_q_test_norm = (X_q_test - X_min) / X_range
                y_q_test = extract_qubit_labels(y_test_raw, qubit)

                acc, _ = evaluate_test_accuracy(model, X_q_test_norm, y_q_test, task_type='binary')
                arxiv_qubit_accs[qubit] = acc

                # Save individual per-qubit CSV
                per_q_row = [float('nan')] * NUM_QUBITS
                per_q_row[qubit] = acc
                save_model_report_csv("Arxiv240618807FNN", model, study, length,
                                    EPOCHS_PER_TRIAL, acc, per_q_row, target_qubit=qubit)

        # ----------------------------------------------------------------
        # 3. Optimize Multi-task CNN
        # ----------------------------------------------------------------
        if "CNN" in models_to_run:
            logger.info(f"Optimizing Multi-task CNN")
            try:
                study = optuna.create_study(direction="minimize")
                study.optimize(lambda trial: objective_cnn(trial, length), n_trials=N_OPTUNA_TRIALS)
                
                best_model_path = study.best_trial.user_attrs["model_path"]
                best_m_param = study.best_trial.user_attrs["m_param"]
                logger.info(f"Best CNN (Length {length}) saved at: {best_model_path}")

                model = _load_best_model(CNN, best_model_path,
                                        in_channels=10, m_param=best_m_param, num_qubits=NUM_QUBITS)
                
                # Load CNN test data directly
                with h5py.File(CNN_TEST_FILE, "r") as hf:
                    X_test = hf["X_test"][:]
                    y_test = hf["y_test"][:]
                
                if len(X_test.shape) == 4:
                    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
                    X_test = np.transpose(X_test, (0, 2, 1))
                elif len(X_test.shape) == 3 and X_test.shape[1] == 10:
                    pass

                if len(y_test.shape) == 1:
                    y_test = np.array([[(label >> i) & 1 for i in range(NUM_QUBITS)] for label in y_test], dtype=np.float32)

                steps = max(1, length // 20)
                X_test = X_test[:, :, :steps]
                
                # Normalize test data using best trial stats
                best_mean = np.array(study.best_trial.user_attrs["mean"])
                best_std = np.array(study.best_trial.user_attrs["std"])
                X_test = (X_test.astype(np.float32) - best_mean) / best_std
                
                y_test = y_test.astype(np.float32)

                overall_acc, per_qubit_accs = evaluate_test_accuracy(
                    model, X_test, y_test, task_type='multitask', batch_size=study.best_trial.params['batch_size'])
                save_model_report_csv("CNN", model, study, length, EPOCHS_PER_TRIAL,
                                    overall_acc, per_qubit_accs, extra_hparams=['m_param'])
            except FileNotFoundError:
                logger.warning(f"CNN data file missing. Skipping CNN optimization.")

        # ----------------------------------------------------------------
        # 4. Optimize HERQULES Net_rmf (MF + RMF, 32-class)
        # ----------------------------------------------------------------
        if "HERQULES_rmf" in models_to_run:
            logger.info(f"Optimizing HERQULES Net_rmf")
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective_herqules(trial, X_train_raw, y_train_raw, length), n_trials=N_OPTUNA_TRIALS)
            best_model_path = study.best_trial.user_attrs["model_path"]
            logger.info(f"Best HERQULES Net_rmf (Length {length}) saved at: {best_model_path}")

            model = _load_best_model(Net_rmf, best_model_path)
            # Compute MF + RMF features for test set using training statistics
            mf_test = np.zeros((X_test_raw.shape[0], NUM_QUBITS))
            rmf_test = np.zeros((X_test_raw.shape[0], NUM_QUBITS))
            for q in range(NUM_QUBITS):
                y_q = extract_qubit_labels(y_train_raw, q)
                gnd = X_train_flat[y_q == 0]
                ext = X_train_flat[y_q == 1]
                n = min(len(gnd), len(ext))

                # Standard MF
                diff_mf = gnd[:n] - ext[:n]
                mf_envelope = np.mean(diff_mf, axis=0) / (np.var(diff_mf, axis=0) + 1e-10)
                mf_out_train = X_train_flat @ mf_envelope
                mf_test[:, q] = X_test_flat @ mf_envelope

                # RMF: identify relaxation traces from training data
                mf_gnd = mf_out_train[y_q == 0]
                threshold = np.mean(mf_gnd)
                sigma_gnd = np.std(mf_gnd) + 1e-10

                relax_mask = (y_q == 1) & (np.abs(mf_out_train - threshold) < 2 * sigma_gnd)
                relax_traces = X_train_flat[relax_mask]

                if len(relax_traces) > 10:
                    n_rmf = min(len(relax_traces), len(gnd))
                    diff_rmf = relax_traces[:n_rmf] - gnd[:n_rmf]
                    rmf_envelope = np.mean(diff_rmf, axis=0) / (np.var(diff_rmf, axis=0) + 1e-10)
                    rmf_test[:, q] = X_test_flat @ rmf_envelope
                else:
                    rmf_test[:, q] = 0.0

            mf_rmf_test = np.concatenate([mf_test, rmf_test], axis=1)
            overall_acc, per_qubit_accs = evaluate_test_accuracy(
                model, mf_rmf_test, y_test_raw, task_type='32class')
            save_model_report_csv("HERQULES_Net_rmf", model, study, length, EPOCHS_PER_TRIAL,
                                overall_acc, per_qubit_accs)

        # ----------------------------------------------------------------
        # 5. Optimize KLiNQ (2-stage knowledge distillation) per qubit
        # ----------------------------------------------------------------
        if "KLiNQ_teacher" in models_to_run or "KLiNQ_student" in models_to_run:
            target_length_klinq = 5
            klinq_qubit_accs = [None] * NUM_QUBITS
            for qubit in range(NUM_QUBITS):
                # --- Stage 1: Train teacher ---
                logger.info(f"Optimizing KLiNQ Teacher (Qubit {qubit})")
                teacher_study = optuna.create_study(direction="minimize")
                teacher_study.optimize(
                    lambda trial, q=qubit: objective_klinq_teacher(trial, X_train_raw, y_train_raw, length, q),
                    n_trials=N_OPTUNA_TRIALS,
                )
                best_teacher_path = teacher_study.best_trial.user_attrs["model_path"]
                logger.info(f"Best KLiNQ Teacher (Qubit {qubit}, Length {length}) saved at: {best_teacher_path}")

                # Evaluate teacher on test set
                y_q_test = extract_qubit_labels(y_test_raw, qubit)
                y_q_train = extract_qubit_labels(y_train_raw, qubit)

                # Teacher test data: z-score normalised flat trace (using train stats)
                X_tea_mean = np.mean(X_train_flat, axis=0)
                X_tea_std = np.std(X_train_flat, axis=0) + 1e-10
                X_test_teacher = (X_test_flat - X_tea_mean) / X_tea_std

                teacher_model = _load_best_model(
                    KLiNQTeacherModel, best_teacher_path,
                    input_size=length * 2, output_size=1,
                )
                teacher_acc, _ = evaluate_test_accuracy(
                    teacher_model, X_test_teacher, y_q_test, task_type='binary')
                logger.info(f"KLiNQ Teacher (Qubit {qubit}) test accuracy: {teacher_acc:.2f}%")

                # Save teacher CSV report
                per_q_row_teacher = [float('nan')] * NUM_QUBITS
                per_q_row_teacher[qubit] = teacher_acc
                save_model_report_csv("KLiNQ_Teacher", teacher_model, teacher_study, length,
                                    EPOCHS_PER_TRIAL, teacher_acc, per_q_row_teacher,
                                    target_qubit=qubit)

                # --- Stage 2: Distill to student ---
                logger.info(f"Optimizing KLiNQ Student via distillation (Qubit {qubit})")
                # Keep teacher on device for distillation
                teacher_model.eval()
                for p in teacher_model.parameters():
                    p.requires_grad = False

                student_study = optuna.create_study(direction="minimize")
                student_study.optimize(
                    lambda trial, q=qubit: objective_klinq_student(
                        trial, X_train_raw, y_train_raw, length, q, teacher_model),
                    n_trials=N_OPTUNA_TRIALS,
                )
                best_student_path = student_study.best_trial.user_attrs["model_path"]
                student_input_size = student_study.best_trial.user_attrs["student_input_size"]
                logger.info(f"Best KLiNQ Student (Qubit {qubit}, Length {length}) saved at: {best_student_path}")

                # Prepare student test features
                bin_size = max(1, length // target_length_klinq)
                n_bins = length // bin_size

                X_avg_I = X_test_raw[:, :n_bins * bin_size, 0].reshape(X_test_raw.shape[0], n_bins, bin_size).mean(axis=2)
                X_avg_Q = X_test_raw[:, :n_bins * bin_size, 1].reshape(X_test_raw.shape[0], n_bins, bin_size).mean(axis=2)
                X_avg_test = np.concatenate([X_avg_I, X_avg_Q], axis=1)

                gnd = X_train_flat[y_q_train == 0]
                ext = X_train_flat[y_q_train == 1]
                n = min(len(gnd), len(ext))
                diff = gnd[:n] - ext[:n]
                envelope = np.mean(diff, axis=0) / (np.var(diff, axis=0) + 1e-10)
                mf_scalar_test = (X_test_flat @ envelope).reshape(-1, 1)

                def _znorm(a):
                    return (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-10)

                X_combined_test = np.concatenate(
                    [_znorm(X_test_flat.copy()), _znorm(X_avg_test), _znorm(mf_scalar_test)], axis=1
                )

                student_model = _load_best_model(
                    KLiNQStudentModel, best_student_path, input_size=student_input_size)
                student_acc, _ = evaluate_test_accuracy(
                    student_model, X_combined_test, y_q_test, task_type='binary')
                klinq_qubit_accs[qubit] = student_acc

                per_q_row = [float('nan')] * NUM_QUBITS
                per_q_row[qubit] = student_acc
                save_model_report_csv(
                    "KLiNQ_Student", student_model, student_study, length, EPOCHS_PER_TRIAL,
                    student_acc, per_q_row, target_qubit=qubit,
                    extra_hparams=['temperature', 'alpha'],
                )

        # ----------------------------------------------------------------
        # 6. Optimize Transformer
        # ----------------------------------------------------------------
        if "Transformer" in models_to_run:
            logger.info(f"Optimizing Transformer")
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective_transformer(trial, X_train_raw, y_train_raw, length), n_trials=N_OPTUNA_TRIALS)
            best_model_path = study.best_trial.user_attrs["model_path"]
            logger.info(f"Best Transformer (Length {length}) saved at: {best_model_path}")

            best_params = study.best_trial.params
            model = _load_best_model(
                QubitClassifierTransformer, best_model_path,
                num_classes=2 ** NUM_QUBITS,
                patch_size=best_params['patch_size'],
                embedding_dim=best_params['embedding_dim'],
                num_heads=best_params['num_heads'],
                num_layers=best_params['num_layers'],
                dropout=best_params['dropout'],
            )
            overall_acc, per_qubit_accs = evaluate_test_accuracy(
                model, X_test_raw, y_test_raw, task_type='32class')
            save_model_report_csv(
                "Transformer", model, study, length, EPOCHS_PER_TRIAL,
                overall_acc, per_qubit_accs,
                extra_hparams=['patch_size', 'embedding_dim', 'num_heads', 'num_layers', 'dropout'],
            )


if __name__ == "__main__":
    optimize_models(models_to_run=["CNN"])
