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

RAW_TRAIN_FILE = "/home/manosgior/qubit_readout_klinq/data/five_qubit_data/DRaw_C_Tr_v0-001"
RAW_TEST_FILE = "/home/manosgior/qubit_readout_klinq/data/five_qubit_data/DRaw_C_Te_v0-002"

# CNN model uses a differently preprocessed (downsampled) file.
CNN_TRAIN_FILE = "/home/sandra/Qubit_5Channel_ds20_train.h5"
CNN_TEST_FILE = "/home/sandra/Qubit_5Channel_ds20_test.h5"

NUM_QUBITS = 5
TRACE_LENGTHS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500] 
N_OPTUNA_TRIALS = 20
EPOCHS_PER_TRIAL = 50
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

def optimize_models():
    """Main function to run hyper-parameter optimization loops."""
    logger.info("Starting Hyper-Optimization Pipeline")
    
    for length in TRACE_LENGTHS:
        logger.info(f"=== Optimizing for Trace Length: {length} ===")
        
        # 1. Optimize FNN (raw trace, 32-class)
        logger.info(f"Optimizing FNN")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective_fnn(trial, length), n_trials=N_OPTUNA_TRIALS)
        best_model_path = study.best_trial.user_attrs["model_path"]
        logger.info(f"Best FNN (Length {length}) saved at: {best_model_path}")
            
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

        # 4. Optimize HERQULES Net (MF-based, 32-class)
        logger.info(f"Optimizing HERQULES Net")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective_herqules(trial, length), n_trials=N_OPTUNA_TRIALS)
        best_model_path = study.best_trial.user_attrs["model_path"]
        logger.info(f"Best HERQULES Net (Length {length}) saved at: {best_model_path}")

        # 5. Optimize KLiNQ Student per qubit
        for qubit in range(NUM_QUBITS):
            logger.info(f"Optimizing KLiNQ Student (Qubit {qubit})")
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective_klinq(trial, length, qubit), n_trials=N_OPTUNA_TRIALS)
            best_model_path = study.best_trial.user_attrs["model_path"]
            logger.info(f"Best KLiNQ Student (Qubit {qubit}, Length {length}) saved at: {best_model_path}")

        # 6. Optimize Transformer
        logger.info(f"Optimizing Transformer")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective_transformer(trial, length), n_trials=N_OPTUNA_TRIALS)
        best_model_path = study.best_trial.user_attrs["model_path"]
        logger.info(f"Best Transformer (Length {length}) saved at: {best_model_path}")


if __name__ == "__main__":
    optimize_models()
