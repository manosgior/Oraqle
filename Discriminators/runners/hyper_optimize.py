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
    Net_rmf
)

# Import helpers
from helpers.cnn_helpers import prepare_cnn_data

# ============================================================================
# User Configuration
# ============================================================================

RAW_TRAIN_FILE = "/path/to/DRaw_C_Tr_v0-001.h5"
RAW_TEST_FILE = "/path/to/DRaw_C_Te_v0-002.h5"

# CNN model uses a differently preprocessed (downsampled) file.
CNN_TRAIN_FILE = "/path/to/Qubit_5Channel_ds20_train.h5"
CNN_TEST_FILE = "/path/to/Qubit_5Channel_ds20_test.h5"

NUM_QUBITS = 5
TRACE_LENGTHS = [100, 200, 300, 400, 500] 
N_OPTUNA_TRIALS = 15
EPOCHS_PER_TRIAL = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./saved_models"

os.makedirs(SAVE_DIR, exist_ok=True)

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

def objective_singlequbitfnn(trial, trace_length, target_qubit):
    """Optuna objective for SingleQubitFNN (Raw trace)."""
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])

    X_raw, y_raw = load_hdf5_data(RAW_TRAIN_FILE, trace_length, is_test=False)
    y_q = extract_qubit_labels(y_raw, target_qubit)
    
    # Flatten IQ dimensions
    X_flat = X_raw.reshape(X_raw.shape[0], -1)
    
    # Z-score normalization
    X_mean = np.mean(X_flat, axis=0)
    X_std = np.std(X_flat, axis=0) + 1e-10
    X_norm = (X_flat - X_mean) / X_std
    
    X_train, X_val, y_train, y_val = train_test_split(X_norm, y_q, test_size=0.2, random_state=42, stratify=y_q)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=batch_size)
    
    model = SingleQubitFNN(input_size=trace_length * 2, output_size=1).to(DEVICE)
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
            
    model_path = os.path.join(SAVE_DIR, f"SingleQubitFNN_q{target_qubit}_len{trace_length}_trial{trial.number}.pth")
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


# ============================================================================
# Main Optimizer Loop
# ============================================================================

def optimize_models():
    """Main function to run hyper-parameter optimization loops."""
    logger.info("Starting Hyper-Optimization Pipeline")
    
    for length in TRACE_LENGTHS:
        logger.info(f"=== Optimizing for Trace Length: {length} ===")
        
        # 1. Optimize SingleQubitFNN per qubit
        for qubit in range(NUM_QUBITS):
            logger.info(f"Optimizing SingleQubitFNN (Qubit {qubit})")
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective_singlequbitfnn(trial, length, qubit), n_trials=N_OPTUNA_TRIALS)
            
            best_model_path = study.best_trial.user_attrs["model_path"]
            logger.info(f"Best SingleQubitFNN (Qubit {qubit}, Length {length}) saved at: {best_model_path}")
            
        # 2. Optimize Arxiv240618807FNN per qubit
        for qubit in range(NUM_QUBITS):
            logger.info(f"Optimizing Arxiv240618807FNN (Qubit {qubit})")
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective_arxiv(trial, length, qubit), n_trials=N_OPTUNA_TRIALS)
            
            best_model_path = study.best_trial.user_attrs["model_path"]
            logger.info(f"Best Arxiv240618807FNN (Qubit {qubit}, Length {length}) saved at: {best_model_path}")
            
        # 3. Optimize Multi-task CNN
        logger.info(f"Optimizing Multi-task CNN")
        try:
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective_cnn(trial, length), n_trials=N_OPTUNA_TRIALS)
            
            best_model_path = study.best_trial.user_attrs["model_path"]
            logger.info(f"Best CNN (Length {length}) saved at: {best_model_path}")
        except FileNotFoundError:
            logger.warning(f"CNN data file missing. Skipping CNN optimization.")


if __name__ == "__main__":
    optimize_models()
