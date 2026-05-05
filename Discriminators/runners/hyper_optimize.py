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
N_OPTUNA_TRIALS = 20
EPOCHS_PER_TRIAL = 50
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

def load_hdf5_data(filepath, trace_length, is_test=False):
    """Loads and truncates data from HDF5."""
    key_suffix = "test" if is_test else "train"
    with h5py.File(filepath, "r") as hf:
        # Load and truncate
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
        pred_binary = (torch.sigmoid(all_preds.squeeze()) >= 0.5).int().numpy()
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

def objective_arxiv(trial, trace_length, target_qubit):
    """Optuna objective for the Arxiv240618807FNN."""
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    X_raw, y_raw = load_hdf5_data(RAW_TRAIN_FILE, trace_length, is_test=False)
    X_demod = demodulate_and_average(X_raw, FREQ_READOUT)
    
    X_q = X_demod[:, target_qubit, :]
    y_q = extract_qubit_labels(y_raw, target_qubit)
    
    # Normalize to [0, 1] as in the paper
    X_min = np.min(X_q, axis=0)
    X_range = np.max(X_q, axis=0) - X_min
    X_range[X_range == 0] = 1e-10
    X_norm = (X_q - X_min) / X_range
    
    X_train, X_val, y_train, y_val = train_test_split(X_norm, y_q, test_size=0.2, random_state=42, stratify=y_q)
    
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

def objective_fnn(trial, trace_length):
    """Optuna objective for SingleQubitFNN (Raw trace, 32-class)."""
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])

    X_raw, y_raw = load_hdf5_data(RAW_TRAIN_FILE, trace_length, is_test=False)
    
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

    # CNN loader resamples, we pass time_slice
    X_tensor, y_tensor = prepare_cnn_data(
        CNN_TRAIN_FILE, 
        downsample_factor=20, 
        original_length=500, # Assuming max length is 500 for the file
        num_qubits=NUM_QUBITS, 
        time_slice=(0, trace_length),
        is_test=False
    )
    
    X_train, X_val, y_train, y_val = train_test_split(X_tensor.numpy(), y_tensor.numpy(), test_size=0.2, random_state=42)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=batch_size)
    
    model = CNN(in_channels=10, m_param=8, num_qubits=NUM_QUBITS).to(DEVICE)
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

def objective_herqules(trial, trace_length):
    """Optuna objective for HERQULES Net (MF-based, 32-class)."""
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    X_raw, y_raw = load_hdf5_data(RAW_TRAIN_FILE, trace_length, is_test=False)
    X_flat = X_raw.reshape(X_raw.shape[0], -1)

    # Compute matched-filter scalar per qubit
    mf_features = np.zeros((X_raw.shape[0], NUM_QUBITS))
    for q in range(NUM_QUBITS):
        y_q = extract_qubit_labels(y_raw, q)
        gnd, ext = X_flat[y_q == 0], X_flat[y_q == 1]
        n = min(len(gnd), len(ext))
        diff = gnd[:n] - ext[:n]
        envelope = np.mean(diff, axis=0) / (np.var(diff, axis=0) + 1e-10)
        mf_features[:, q] = X_flat @ envelope

    X_train, X_val, y_train, y_val = train_test_split(
        mf_features, y_raw, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size)

    model = Net().to(DEVICE)
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

    model_path = os.path.join(SAVE_DIR, f"HERQULES_len{trace_length}_trial{trial.number}.pth")
    torch.save(model.state_dict(), model_path)
    trial.set_user_attr("model_path", model_path)

    return val_loss

def objective_klinq(trial, trace_length, target_qubit):
    """Optuna objective for KLiNQ StudentModel (per-qubit binary)."""
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    target_length = 5  # averaging target bins per channel

    X_raw, y_raw = load_hdf5_data(RAW_TRAIN_FILE, trace_length, is_test=False)
    y_q = extract_qubit_labels(y_raw, target_qubit)

    # 1) Flat trace
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

    X_combined = np.concatenate([_znorm(X_flat), _znorm(X_avg), _znorm(mf_scalar)], axis=1)
    input_size = X_combined.shape[1]

    X_train, X_val, y_train, y_val = train_test_split(
        X_combined, y_q, test_size=0.2, random_state=42, stratify=y_q
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.float32)),
        batch_size=batch_size)

    model = KLiNQStudentModel(input_size=input_size).to(DEVICE)
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

    model_path = os.path.join(SAVE_DIR, f"KLiNQ_q{target_qubit}_len{trace_length}_trial{trial.number}.pth")
    torch.save(model.state_dict(), model_path)
    trial.set_user_attr("model_path", model_path)

    return val_loss

def objective_transformer(trial, trace_length):
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

    X_raw, y_raw = load_hdf5_data(RAW_TRAIN_FILE, trace_length, is_test=False)

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


def optimize_models():
    """Main function to run hyper-parameter optimization loops."""
    logger.info("Starting Hyper-Optimization Pipeline")
    
    for length in TRACE_LENGTHS:
        logger.info(f"=== Optimizing for Trace Length: {length} ===")

        # Pre-load test data once per trace length (raw traces)
        X_test_raw, y_test_raw = load_hdf5_data(RAW_TEST_FILE, length, is_test=True)
        X_test_flat = X_test_raw.reshape(X_test_raw.shape[0], -1)

        # Z-score normalization stats for FNN (recomputed from train for consistency)
        X_train_raw_tmp, _ = load_hdf5_data(RAW_TRAIN_FILE, length, is_test=False)
        X_train_flat_tmp = X_train_raw_tmp.reshape(X_train_raw_tmp.shape[0], -1)
        fnn_mean = np.mean(X_train_flat_tmp, axis=0)
        fnn_std = np.std(X_train_flat_tmp, axis=0) + 1e-10
        X_test_fnn = (X_test_flat - fnn_mean) / fnn_std

        # ----------------------------------------------------------------
        # 1. Optimize FNN (raw trace, 32-class)
        # ----------------------------------------------------------------
        logger.info(f"Optimizing FNN")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective_fnn(trial, length), n_trials=N_OPTUNA_TRIALS)
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
        X_test_demod = demodulate_and_average(X_test_raw, FREQ_READOUT)
        X_train_demod = demodulate_and_average(X_train_raw_tmp, FREQ_READOUT)

        arxiv_qubit_accs = [None] * NUM_QUBITS
        for qubit in range(NUM_QUBITS):
            logger.info(f"Optimizing Arxiv240618807FNN (Qubit {qubit})")
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective_arxiv(trial, length, qubit), n_trials=N_OPTUNA_TRIALS)
            
            best_model_path = study.best_trial.user_attrs["model_path"]
            logger.info(f"Best Arxiv240618807FNN (Qubit {qubit}, Length {length}) saved at: {best_model_path}")

            model = _load_best_model(Arxiv240618807FNN, best_model_path)
            # Prepare per-qubit test data with same normalization
            X_q_test = X_test_demod[:, qubit, :]
            X_q_train = X_train_demod[:, qubit, :]
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
        logger.info(f"Optimizing Multi-task CNN")
        try:
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective_cnn(trial, length), n_trials=N_OPTUNA_TRIALS)
            
            best_model_path = study.best_trial.user_attrs["model_path"]
            logger.info(f"Best CNN (Length {length}) saved at: {best_model_path}")

            model = _load_best_model(CNN, best_model_path,
                                     in_channels=10, m_param=8, num_qubits=NUM_QUBITS)
            # Load CNN test data
            X_cnn_test, y_cnn_test = prepare_cnn_data(
                CNN_TEST_FILE, downsample_factor=20, original_length=500,
                num_qubits=NUM_QUBITS, time_slice=(0, length), is_test=True)
            overall_acc, per_qubit_accs = evaluate_test_accuracy(
                model, X_cnn_test.numpy(), y_cnn_test.numpy(), task_type='multitask')
            save_model_report_csv("CNN", model, study, length, EPOCHS_PER_TRIAL,
                                  overall_acc, per_qubit_accs)
        except FileNotFoundError:
            logger.warning(f"CNN data file missing. Skipping CNN optimization.")

        # ----------------------------------------------------------------
        # 4. Optimize HERQULES Net (MF-based, 32-class)
        # ----------------------------------------------------------------
        logger.info(f"Optimizing HERQULES Net")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective_herqules(trial, length), n_trials=N_OPTUNA_TRIALS)
        best_model_path = study.best_trial.user_attrs["model_path"]
        logger.info(f"Best HERQULES Net (Length {length}) saved at: {best_model_path}")

        model = _load_best_model(Net, best_model_path)
        # Compute MF features for test set using training statistics
        _, y_train_raw = load_hdf5_data(RAW_TRAIN_FILE, length, is_test=False)
        mf_test = np.zeros((X_test_raw.shape[0], NUM_QUBITS))
        for q in range(NUM_QUBITS):
            y_q = extract_qubit_labels(y_train_raw, q)
            gnd = X_train_flat_tmp[y_q == 0]
            ext = X_train_flat_tmp[y_q == 1]
            n = min(len(gnd), len(ext))
            diff = gnd[:n] - ext[:n]
            envelope = np.mean(diff, axis=0) / (np.var(diff, axis=0) + 1e-10)
            mf_test[:, q] = X_test_flat @ envelope
        overall_acc, per_qubit_accs = evaluate_test_accuracy(
            model, mf_test, y_test_raw, task_type='32class')
        save_model_report_csv("HERQULES_Net", model, study, length, EPOCHS_PER_TRIAL,
                              overall_acc, per_qubit_accs)

        # ----------------------------------------------------------------
        # 5. Optimize KLiNQ Student per qubit
        # ----------------------------------------------------------------
        target_length_klinq = 5
        klinq_qubit_accs = [None] * NUM_QUBITS
        for qubit in range(NUM_QUBITS):
            logger.info(f"Optimizing KLiNQ Student (Qubit {qubit})")
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective_klinq(trial, length, qubit), n_trials=N_OPTUNA_TRIALS)
            best_model_path = study.best_trial.user_attrs["model_path"]
            logger.info(f"Best KLiNQ Student (Qubit {qubit}, Length {length}) saved at: {best_model_path}")

            # Prepare KLiNQ test features (same pipeline as objective_klinq)
            y_q_test = extract_qubit_labels(y_test_raw, qubit)
            y_q_train = extract_qubit_labels(y_train_raw, qubit)

            bin_size = max(1, length // target_length_klinq)
            n_bins = length // bin_size

            # Test features
            X_avg_I = X_test_raw[:, :n_bins * bin_size, 0].reshape(X_test_raw.shape[0], n_bins, bin_size).mean(axis=2)
            X_avg_Q = X_test_raw[:, :n_bins * bin_size, 1].reshape(X_test_raw.shape[0], n_bins, bin_size).mean(axis=2)
            X_avg_test = np.concatenate([X_avg_I, X_avg_Q], axis=1)

            gnd = X_train_flat_tmp[y_q_train == 0]
            ext = X_train_flat_tmp[y_q_train == 1]
            n = min(len(gnd), len(ext))
            diff = gnd[:n] - ext[:n]
            envelope = np.mean(diff, axis=0) / (np.var(diff, axis=0) + 1e-10)
            mf_scalar_test = (X_test_flat @ envelope).reshape(-1, 1)

            def _znorm(a):
                return (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-10)

            X_combined_test = np.concatenate(
                [_znorm(X_test_flat.copy()), _znorm(X_avg_test), _znorm(mf_scalar_test)], axis=1
            )
            input_size = X_combined_test.shape[1]

            model = _load_best_model(KLiNQStudentModel, best_model_path, input_size=input_size)
            acc, _ = evaluate_test_accuracy(model, X_combined_test, y_q_test, task_type='binary')
            klinq_qubit_accs[qubit] = acc

            per_q_row = [float('nan')] * NUM_QUBITS
            per_q_row[qubit] = acc
            save_model_report_csv("KLiNQ_Student", model, study, length, EPOCHS_PER_TRIAL,
                                  acc, per_q_row, target_qubit=qubit)

        # ----------------------------------------------------------------
        # 6. Optimize Transformer
        # ----------------------------------------------------------------
        logger.info(f"Optimizing Transformer")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective_transformer(trial, length), n_trials=N_OPTUNA_TRIALS)
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
    optimize_models()
