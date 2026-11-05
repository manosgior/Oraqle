"""
cnn_helpers.py
==============
Helper functions for CNN-based qubit classification, including data preprocessing,
label formatting, and evaluation utilities.

This module handles:
  - Loading downsampled HDF5 data files
  - Converting integer labels to per-qubit bit matrices
  - Time windowing and reshaping for PyTorch CNN input
  - Per-qubit and global accuracy evaluation
"""

import numpy as np
import h5py
import os
import torch
from typing import Tuple, Dict


# ============================================================================
# Data Loading & Preprocessing
# ============================================================================

def prepare_cnn_data(
    path: str,
    downsample_factor: int = 20,
    original_length: int = 1000,
    num_qubits: int = 5,
    time_slice: Tuple[int, int] = (0, 1000),
    is_test: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load, preprocess, and reshape CNN training/test data from HDF5.

    Performs the following steps:
    1. Loads X and y from HDF5 file (expects keys: X_train/y_train or X_test/y_test)
    2. Converts integer labels to per-qubit bit matrix
    3. Reshapes and transposes X to (Samples, Channels, Downsampled_Steps) for PyTorch
    4. Slices the time dimension according to time_slice parameter
    5. Converts to PyTorch Tensors

    Args:
        path (str): Path to HDF5 file containing downsampled data.
        downsample_factor (int): Downsampling factor used when creating the data.
            Default: 20 (original 1000 → 50 downsampled steps).
        original_length (int): Original trace length before downsampling.
            Default: 1000 samples.
        num_qubits (int): Number of qubits. Default: 5.
        time_slice (Tuple[int, int]): Time window in original indices [start, end).
            Indices are scaled to downsampled coordinates. Default: (0, 1000) = full.
        is_test (bool): If True, load test split; else load training split. Default: False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(X_tensor, y_tensor)`` where

        - **X_tensor** (torch.Tensor): Shaped as
          ``(N_samples, channels, time_steps_sliced)`` (float32).
        - **y_tensor** (torch.Tensor): Bit matrix of shape
          ``(N_samples, num_qubits)`` (float32, suitable for BCEWithLogitsLoss).

    Raises:
        FileNotFoundError: If the HDF5 file does not exist.
        KeyError: If expected keys are missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    # Determine HDF5 key suffix
    key_suffix = "test" if is_test else "train"

    # Load data from HDF5
    with h5py.File(path, "r") as hf:
        X = hf[f"X_{key_suffix}"][:]
        y = hf[f"y_{key_suffix}"][:]

    # Convert integer labels to per-qubit bit matrix
    y_bits = np.array([
        [(label >> i) & 1 for i in range(num_qubits)]
        for label in y
    ], dtype=np.float32)

    # Reshape to (Samples, Downsampled_Steps, Channels)
    # The original TF model reshaped to (-1, total_ds_steps, 1, 10)
    total_ds_steps = original_length // downsample_factor
    X_reshaped = X.reshape(-1, total_ds_steps, 10)

    # Scale time_slice indices
    orig_start, orig_end = time_slice
    start_ds = orig_start // downsample_factor
    end_ds = (orig_end // downsample_factor) if orig_end is not None else None

    # Slice the time dimension: (Samples, Sliced_Steps, Channels)
    X_sliced = X_reshaped[:, start_ds:end_ds, :]
    
    # Transpose for PyTorch Conv1d: (Samples, Channels, Sliced_Steps)
    X_transposed = np.transpose(X_sliced, (0, 2, 1)).astype(np.float32)

    return torch.tensor(X_transposed), torch.tensor(y_bits)


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_per_qubit_accuracy(
    predictions: np.ndarray,
    ground_truth: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """Compute per-qubit and global accuracy metrics.

    Evaluates model predictions against ground truth labels and computes:
    - Per-qubit accuracy (independent accuracy for each qubit)
    - Global accuracy (all qubits correct on single shot)
    - Geometric mean (fidelity-like metric)

    Args:
        predictions (np.ndarray): Predicted binary labels of shape
            (N_samples, num_qubits). Values should be in {0, 1}.
        ground_truth (np.ndarray): Ground truth bit matrix of same shape.

    Returns:
        Tuple[np.ndarray, float, float]: ``(qubit_accs, global_acc, fidelity)`` where

        - **qubit_accs** (np.ndarray): Per-qubit accuracy array of shape
          (num_qubits,).
        - **global_acc** (float): Fraction of shots where all qubits matched.
        - **fidelity** (float): Geometric mean of per-qubit accuracies.
    """
    num_qubits = predictions.shape[1]

    # Per-qubit accuracy
    qubit_accs = np.array([
        np.mean(predictions[:, i] == ground_truth[:, i])
        for i in range(num_qubits)
    ])

    # Global accuracy (all qubits correct on single shot)
    global_acc = np.mean(np.all(predictions == ground_truth, axis=1))

    # Fidelity: geometric mean of per-qubit accuracies
    # Add small epsilon to avoid log(0)
    fidelity = np.exp(np.mean(np.log(qubit_accs + 1e-12)))

    return qubit_accs, global_acc, fidelity


def evaluate_cnn_predictions(
    model_predictions: np.ndarray,
    ground_truth_bits: np.ndarray,
    dataset_name: str = "Dataset",
    threshold: float = 0.5
) -> None:
    """Print formatted evaluation report for CNN predictions.

    Takes sigmoid outputs from the model (values in [0, 1]) and ground truth
    bit matrix, applies threshold to convert to binary predictions, then
    computes and displays metrics.

    Args:
        model_predictions (np.ndarray): Sigmoid probabilities.
            Shape: (N_samples, num_qubits), values in [0, 1].
        ground_truth_bits (np.ndarray): Ground truth bit matrix of shape
            (N_samples, num_qubits).
        dataset_name (str): Name of dataset for display. Default: "Dataset".
        threshold (float): Threshold for converting sigmoid outputs to binary.
            Default: 0.5.

    Returns:
        None. Prints evaluation report to stdout.
    """
    # Threshold sigmoid outputs
    binary_preds = (model_predictions > threshold).astype(int)

    # Compute metrics
    qubit_accs, global_acc, fidelity = compute_per_qubit_accuracy(
        binary_preds, ground_truth_bits
    )

    # Print report
    print(f"\n[Evaluation: {dataset_name}]")
    for i in range(len(qubit_accs)):
        print(f"  Qubit {i} Accuracy: {qubit_accs[i]*100:.2f}%")
    print(f"  Global Accuracy:  {global_acc*100:.2f}%")
    print(f"  Fidelity (G-Mean): {fidelity*100:.2f}%")
