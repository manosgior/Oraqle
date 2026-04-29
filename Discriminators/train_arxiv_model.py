"""
train_arxiv_model.py
====================
Training script for the arXiv:2406.18807 FNN on 5-qubit multiplexed readout data.

Workflow
--------
1. Load raw IQ traces from two HDF5 files (fast HDF5 hyperslab slicing).
2. Pool training and test sets into a single array.
3. Demodulate the multiplexed signal: for each qubit's IF frequency, multiply the
   trace by a complex LO and average over the readout window to obtain a single
   (I, Q) point per qubit per shot.
4. For each of the 5 qubits independently:
     a. Extract per-qubit (I, Q) column.
     b. Extract per-qubit binary label from the packed integer label.
     c. Min-max normalise (I, Q) to [0, 1] as recommended by the paper.
     d. Split into 60% train / 40% test (stratified).
     e. Build PyTorch DataLoaders.
     f. Train a fresh Arxiv240618807FNN instance (2-hidden-layer FNN).
     g. Evaluate and save the per-qubit model.

Paper details (arXiv:2406.18807)
----------------------------------
- The paper proposes a minimal FNN (2→8→4→1, Sigmoid) for single-qubit IQ readout.
- Input: time-averaged (I, Q) point after digital demodulation.
- Normalisation: min-max scaling to [0, 1] per column.
- Training: Binary Cross-Entropy loss, Adam optimiser, 40 epochs, batch size 64.
- Split: 60,000 train / 40,000 test (out of 100,000 total per qubit).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# Project-level imports
from helpers.data_loader import *      # QubitData, QubitTraceDataset
from helpers.data_utils import *       # hdf5_data_load, custom_hdf5_data_loader, etc.
from networks.Arxiv240618807FNN import Arxiv240618807FNN


# ---------------------------------------------------------------------------
# Signal processing utility
# ---------------------------------------------------------------------------

def demodulate_and_average(traces, freq_readout, dt=2e-9):
    """
    Digitally demodulate a frequency-multiplexed IQ trace and average over the
    readout window to obtain a single (I, Q) point per qubit per shot.

    In the multiplexed readout scheme all 5 qubits are probed simultaneously
    using resonators at different IF frequencies.  The ADC captures the
    composite waveform.  To isolate each qubit's contribution, we:
      1. Multiply the trace by a complex LO at the target IF frequency:
             I_demod = I·cos(2πf·t) - Q·(-sin(2πf·t))
             Q_demod = I·(-sin(2πf·t)) + Q·cos(2πf·t)
      2. Average over the readout window (low-pass filter effect).
    This is the standard digital IQ demodulation used in quantum computing
    hardware (e.g. Qiskit Experiments, QubiC, etc.).

    Parameters
    ----------
    traces : ndarray, shape (N, time_steps, 2)
        Raw IQ traces for all shots.  Axis 2: index 0 = I, index 1 = Q.
    freq_readout : array-like, shape (num_qubits,)
        IF frequencies in Hz for each qubit's resonator.
    dt : float
        ADC sampling interval in seconds.  Default: 2e-9 (500 MHz ADC).

    Returns
    -------
    ndarray, shape (N, num_qubits, 2)
        Demodulated and time-averaged (I, Q) point for each shot and qubit.
    """
    N, time_steps, _ = traces.shape
    num_qubits = len(freq_readout)
    # Time axis in seconds: [0, dt, 2·dt, ..., (T-1)·dt]
    t = np.arange(time_steps) * dt  # shape: (time_steps,)

    # Pre-allocate output array: (N, num_qubits, 2)
    demodulated_acc = np.zeros((N, num_qubits, 2))

    for i, freq in enumerate(freq_readout):
        # LO waveforms at this qubit's IF frequency
        lo_i =  np.cos(2 * np.pi * freq * t)   # in-phase LO   shape: (T,)
        lo_q = -np.sin(2 * np.pi * freq * t)   # quadrature LO shape: (T,)

        # Separate channels
        I_trace = traces[:, :, 0]  # shape: (N, T)
        Q_trace = traces[:, :, 1]  # shape: (N, T)

        # Complex demodulation:
        #   (I + jQ) × (lo_i + j·lo_q)
        #   Real part: I·lo_i − Q·lo_q
        #   Imag part: I·lo_q + Q·lo_i
        I_demod = I_trace * lo_i - Q_trace * lo_q  # shape: (N, T), broadcasts correctly
        Q_demod = I_trace * lo_q + Q_trace * lo_i  # shape: (N, T)

        # Average over the readout window (time axis=1) -> shape: (N,)
        demodulated_acc[:, i, 0] = np.mean(I_demod, axis=1)
        demodulated_acc[:, i, 1] = np.mean(Q_demod, axis=1)

    return demodulated_acc  # shape: (N, num_qubits, 2)


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Run one training epoch.

    Iterates over all batches in ``dataloader``, performing the standard
    forward → loss → backward → step loop.

    Note: labels are unsqueezed to shape (batch_size, 1) to match BCELoss's
    expected format when the model outputs a single-node Sigmoid.

    Parameters
    ----------
    model     : Arxiv240618807FNN
    dataloader: DataLoader
    criterion : nn.BCELoss
    optimizer : torch.optim.Adam
    device    : torch.device

    Returns
    -------
    float
        Average loss per *sample* (total loss / dataset size).
    """
    model.train()
    total_loss = 0.0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        # BCELoss expects target shape (N, 1) to match the model's (N, 1) output
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()          # Clear gradients from previous batch
        outputs = model(inputs)        # Forward pass: (N, 2) -> (N, 1)
        loss = criterion(outputs, labels)

        loss.backward()                # Compute gradients
        optimizer.step()               # Update model parameters

        # Accumulate the unscaled (sum, not mean) loss for the whole dataset
        total_loss += loss.item() * inputs.size(0)

    # Divide by total number of samples to get the per-sample average loss
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model accuracy and loss on a held-out dataset.

    Uses ``torch.no_grad()`` to disable gradient tracking (faster and less memory).
    Threshold for binary decision: output >= 0.5 → predict |1⟩, else |0⟩.

    Parameters
    ----------
    model     : Arxiv240618807FNN
    dataloader: DataLoader
    criterion : nn.BCELoss
    device    : torch.device

    Returns
    -------
    avg_loss : float  -- per-sample average BCELoss
    accuracy : float  -- percentage of correctly classified shots (0–100)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            # Hard decision: Sigmoid output >= 0.5 -> predicted state |1⟩
            predictions = (outputs >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Data loading: fast HDF5 hyperslab slicing
    # The paper uses 100k samples total (60k train / 40k test) per qubit.
    # We load more (~300k) and let train_test_split impose the 60/40 ratio.
    # ------------------------------------------------------------------
    data_path       = "/home/manosgior/Documents/GitHub/KLiNQ/qubit_readout_klinq/data/five_qubit_data"
    train_file_name = "DRaw_C_Tr_v0-001"
    test_file_name  = "DRaw_C_Te_v0-002"

    print("Loading data...")
    dataset_keys = {"Train": ("X_train", "y_train"), "Test": ("X_test", "y_test")}

    import h5py

    # Load 160,000 traces from the training file via uniform sub-sampling.
    # HDF5 hyperslab slicing [::step] only reads the selected rows from disk
    # (much faster than building a random index list for large files).
    with h5py.File(f"{data_path}/{train_file_name}", 'r') as hf:
        N_train_total = hf["X_train"].shape[0]
        # Compute step to cover the file with at most 160,000 rows
        step_train = max(1, N_train_total // 160000)

        # Truncate traces to 375 samples (first 750 ns @ 2 ns/sample).
        # The paper uses a 750 ns readout window.
        X_train_raw = np.array(hf["X_train"][::step_train, 0:375, :][:160000])
        y_train_raw = np.array(hf["y_train"][::step_train][:160000])

    # Load 140,000 traces from the test file
    with h5py.File(f"{data_path}/{test_file_name}", 'r') as hf:
        N_test_total = hf["X_test"].shape[0]
        step_test = max(1, N_test_total // 140000)

        X_test_raw = np.array(hf["X_test"][::step_test, 0:375, :][:140000])
        y_test_raw = np.array(hf["y_test"][::step_test][:140000])

    print("Data loaded into memory.")

    # Pool all ~300k shots; the final split is done per-qubit below.
    X_full_raw = np.concatenate((X_train_raw, X_test_raw), axis=0)  # (300k, 375, 2)
    y_full     = np.concatenate((y_train_raw, y_test_raw), axis=0)  # (300k,)
    print("Raw Full Data Shape:", X_full_raw.shape)

    # ------------------------------------------------------------------
    # Demodulation: extract per-qubit (I, Q) from the multiplexed trace
    # IF frequencies (Hz) for the 5 qubits' resonators, negated to match
    # the sign convention used during data acquisition.
    # ------------------------------------------------------------------
    print("Demodulating data...")
    freq_readout = -np.array([-64.729e6, -25.366e6, 24.79e6, 70.269e6, 127.282e6])
    # Result: (300k, 5, 2) — one averaged (I, Q) point per shot per qubit
    X_full_demod = demodulate_and_average(X_full_raw, freq_readout)

    # ------------------------------------------------------------------
    # Per-qubit training loop
    # ------------------------------------------------------------------
    for target_qubit in range(5):
        print(f"\n{'='*40}")
        print(f"Training Model for Qubit {target_qubit}")
        print(f"{'='*40}")

        # Select the (I, Q) column for this qubit only: shape (300k, 2)
        X_full_q = X_full_demod[:, target_qubit, :]

        # Extract single-qubit binary label from the packed integer.
        # The state of qubit q is stored as bit q of the integer label:
        #   state_q = (y_combined >> q) & 1
        y_full_q = (y_full >> target_qubit) & 1

        # ------ Min-max normalisation per column (paper §6.2) ------
        # Scale each column (I and Q separately) to [0, 1].
        # Statistics computed on the full pooled dataset (before the split)
        # because we replicate the paper's strategy of computing scale on
        # the combined dataset before partitioning.
        X_min   = np.min(X_full_q, axis=0)
        X_max   = np.max(X_full_q, axis=0)
        X_range = X_max - X_min
        # Guard against zero range (e.g. flat signal)
        X_range[X_range == 0] = 1e-10
        X_full_norm = (X_full_q - X_min) / X_range  # shape: (300k, 2), values in [0, 1]

        # ------ Train / test split (60% / 40%, stratified by label) ------
        # Stratification ensures the same class ratio in both sets,
        # which is important when qubit states are not equally probable.
        X_train, X_test, y_train, y_test = train_test_split(
            X_full_norm,
            y_full_q,
            test_size=0.4,
            random_state=42,
            stratify=y_full_q
        )

        # ------ DataLoaders ------
        BATCH_SIZE = 64  # Paper uses batch size 64

        train_dataset = QubitTraceDataset(X_train, y_train)
        test_dataset  = QubitTraceDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

        # ------ Model, loss, optimiser ------
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model     = Arxiv240618807FNN().to(device)
        criterion = nn.BCELoss()                          # Binary Cross-Entropy
        optimizer = torch.optim.Adam(model.parameters())  # Default lr = 1e-3

        # ------ Training loop ------
        print("start training")
        NUM_EPOCHS = 40  # Paper trains for 40 epochs

        for epoch in range(NUM_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

            # Evaluate on test set every 20 epochs and at epoch 0 for progress monitoring
            if (epoch + 1) % 20 == 0 or epoch == 0:
                test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
                print(
                    f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Test Loss: {test_loss:.4f} | "
                    f"Test Accuracy: {test_accuracy:.2f}%"
                )

        # ------ Final evaluation and model saving ------
        print("\n--- Final Test Evaluation ---")
        final_test_loss, final_test_accuracy = evaluate(model, test_loader, criterion, device)
        print(
            f"Qubit {target_qubit} Test Loss: {final_test_loss:.4f} | "
            f"Per-Qubit Test Accuracy: {final_test_accuracy:.2f}%"
        )

        # Save per-qubit model state dict
        torch.save(model.state_dict(), f"qubit_arxiv_fnn_model_q{target_qubit}.pth")
        print(f"Saved model to qubit_arxiv_fnn_model_q{target_qubit}.pth")
