"""
cnn_helpers.py
==============
Helper functions for CNN-based qubit classification, including data preprocessing,
label formatting, and evaluation utilities.

This module handles:
  - Loading downsampled HDF5 data files
  - Converting integer labels to per-qubit bit matrices
  - Time windowing and reshaping for CNN input
  - Multi-output formatting for Keras multi-task learning
  - Per-qubit and global accuracy evaluation
"""

import numpy as np
import h5py
import os
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Load, preprocess, and reshape CNN training/test data from HDF5.

    Performs the following steps:
    1. Loads X and y from HDF5 file (expects keys: X_train/y_train or X_test/y_test)
    2. Converts integer labels to per-qubit bit matrix
    3. Reshapes to (Samples, Downsampled_Steps, 1, Channels)
    4. Slices the time dimension according to time_slice parameter

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
        Tuple[np.ndarray, np.ndarray]: ``(X_preprocessed, y_bits)`` where

        - **X_preprocessed** (np.ndarray): Shaped as
          ``(N_samples, time_steps_sliced, 1, channels)`` with time_steps_sliced
          determined by the time_slice parameter.
        - **y_bits** (np.ndarray): Bit matrix of shape
          ``(N_samples, num_qubits)`` where each row is the qubit-by-qubit
          binary decomposition of the integer label.

    Raises:
        FileNotFoundError: If the HDF5 file does not exist.
        KeyError: If expected keys (X_train/y_train or X_test/y_test) are missing.

    Examples
    --------
    >>> X_train, y_train = prepare_cnn_data(
    ...     'data_train.h5',
    ...     downsample_factor=20,
    ...     time_slice=(0, 1000)
    ... )
    >>> print(X_train.shape, y_train.shape)
    (10000, 50, 1, 10) (10000, 5)
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
    # For each label, extract bits 0 through num_qubits-1
    y_bits = np.array([
        [(label >> i) & 1 for i in range(num_qubits)]
        for label in y
    ])

    # Reshape to initial form: (Samples, Downsampled_Steps, 1, Channels)
    total_ds_steps = original_length // downsample_factor
    X_reshaped = X.reshape(-1, total_ds_steps, 1, 10)

    # Scale time_slice indices from original scale [0, 1000] to downsampled scale
    orig_start, orig_end = time_slice
    start_ds = orig_start // downsample_factor
    end_ds = (orig_end // downsample_factor) if orig_end is not None else None

    # Slice the time dimension
    X_sliced = X_reshaped[:, start_ds:end_ds, :, :]

    return X_sliced, y_bits


# ============================================================================
# Label Formatting
# ============================================================================

def format_labels_for_multitask(
    y_bits: np.ndarray,
    num_qubits: int = 5
) -> Dict[str, np.ndarray]:
    """Convert bit matrix to multi-task Keras format (dict of arrays).

    Takes a bit matrix and converts it to a dictionary where each key is a
    qubit identifier (q0, q1, ..., q4) and each value is the corresponding
    column from the bit matrix. This format is required by Keras when using
    multiple outputs with different loss functions.

    Args:
        y_bits (np.ndarray): Bit matrix of shape (N_samples, num_qubits).
            Each row is the qubit-by-qubit binary decomposition of a state.
        num_qubits (int): Number of qubits. Default: 5.

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping qubit names to label arrays.
        Example: {'q0': array([0, 1, 0, ...]), 'q1': array([...]), ...}

    Examples
    --------
    >>> y_bits = np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])
    >>> y_dict = format_labels_for_multitask(y_bits, num_qubits=5)
    >>> print(y_dict.keys())
    dict_keys(['q0', 'q1', 'q2', 'q3', 'q4'])
    >>> print(y_dict['q0'])
    [0 1]
    """
    return {f"q{i}": y_bits[:, i] for i in range(num_qubits)}


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
          (num_qubits,). Entry i is the fraction of shots where qubit i
          prediction matched ground truth.
        - **global_acc** (float): Fraction of shots where all qubits matched.
        - **fidelity** (float): Geometric mean of per-qubit accuracies
          (approximates process fidelity for independent qubits).

    Examples
    --------
    >>> preds = np.array([[0, 1, 0], [1, 0, 1]])
    >>> truth = np.array([[0, 1, 0], [1, 0, 0]])
    >>> qubit_accs, g_acc, fid = compute_per_qubit_accuracy(preds, truth)
    >>> print(f"Qubit accuracies: {qubit_accs}")
    Qubit accuracies: [1.  1.  0.5]
    >>> print(f"Global: {g_acc:.2%}, Fidelity: {fid:.2%}")
    Global: 50.00%, Fidelity: 88.91%
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
    """Print formatted evaluation report for CNN multi-task predictions.

    Takes sigmoid outputs from the model (values in [0, 1]) and ground truth
    bit matrix, applies threshold to convert to binary predictions, then
    computes and displays per-qubit, global, and fidelity metrics.

    Args:
        model_predictions (np.ndarray): Raw model outputs (sigmoid activation).
            Shape: (N_samples, num_qubits), values in [0, 1].
        ground_truth_bits (np.ndarray): Ground truth bit matrix of shape
            (N_samples, num_qubits).
        dataset_name (str): Name of dataset for display (e.g., "Validation",
            "Test Set"). Default: "Dataset".
        threshold (float): Threshold for converting sigmoid outputs to binary.
            Default: 0.5.

    Returns:
        None. Prints evaluation report to stdout.

    Examples
    --------
    >>> model_out = np.array([[0.1, 0.9, 0.2], [0.8, 0.3, 0.7]])
    >>> ground_truth = np.array([[0, 1, 0], [1, 0, 1]])
    >>> evaluate_cnn_predictions(model_out, ground_truth, "Test")
    [Evaluation: Test]
      Qubit 0 Accuracy: 100.00%
      Qubit 1 Accuracy: 100.00%
      Qubit 2 Accuracy: 50.00%
      Global Accuracy:  50.00%
      Fidelity (G-Mean): 81.65%
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
