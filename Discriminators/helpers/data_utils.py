"""
data_utils.py
=============
Low-level utility functions for preprocessing qubit IQ readout data.

This module contains all the building-block operations used by
``helpers/data_loader.py``.  The functions here are intentionally kept pure
(no class state) so they can be imported and tested independently.

Data shapes / conventions
--------------------------
  Raw traces   : ndarray  (N, trace_length, 2)
                 N            – number of measurement shots
                 trace_length – number of ADC time samples (e.g. 500 @ 2 ns/sample)
                 2            – quadrature channels: index 0 = I (in-phase),
                                                     index 1 = Q (quadrature)

  Flattened    : ndarray  (N, 2 × trace_length)
                 After ``flatten_iq_dimensions``, the two channels are
                 interleaved into a single 1-D feature vector per sample.

  Labels       : ndarray  (N,)
                 Integers encoding the qubit state(s):
                 - Single qubit: 0 or 1
                 - 5 multiplexed qubits: 0–31 (bit-packed: bit k = state of qubit k)

Contents
--------
  HDF5 I/O
    hdf5_data_load            Load X and y from an HDF5 file.
    custom_hdf5_data_loader   Load only a fraction of an HDF5 file (memory-efficient).

  PyTorch Dataset
    QubitTraceDataset         torch.utils.data.Dataset wrapper for trace/label arrays.

  Normalisation
    normalize_data            z-score  (mean/std) – creates new arrays.
    normalize_data_inplace    z-score in-place    – avoids an extra copy.
    normalize_data_forb       Frobenius-norm division.
    normalize_data_forb_weighted  Frobenius / 4× (for large-amplitude signals).
    normalize_data_forb_subtraction  Per-sample norm subtraction variant.
    normalize_data_std_p2     z-score with std rounded to nearest power of 2
                              (hardware-friendly for fixed-point arithmetic).
    nearest_power_of_2        Helper: round values to nearest power of 2.

  Preprocessing
    reduce_trace_duration     Truncate traces to a shorter length.
    flatten_iq_dimensions     Reshape (N, T, 2) -> (N, 2T).
    stratified_split          Class-balanced train / validation split.

  Matched Filter
    apply_mf_rmf              Project traces onto MF or RMF pulse envelope.

  Hardware-Friendly Normalisation
    find_nearest_power_of_two Round up to nearest power of 2 (integer).
    compute_normalization_params  Per-component max / mean for fixed-point scaling.
    apply_normalization       Apply fixed-point-friendly normalisation parameters.
"""

import os

import h5py
from typing import Tuple, Optional
from loguru import logger
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


# ---------------------------------------------------------------------------
# Matched Filter application
# ---------------------------------------------------------------------------

def apply_mf_rmf(traces, mf_rmf_envelope_I, mf_rmf_envelope_Q):
    """
    Compute the matched-filter (or reverse matched-filter) output for a batch
    of IQ traces.

    The matched filter is the optimal linear detector for a known signal shape
    in additive white Gaussian noise (AWGN).  For dispersive qubit readout, the
    pulse envelope (the expected IQ waveform shape) is used as the filter kernel.

    Operation
    ---------
    For each trace the filter computes:
        output_I = trace_I · mf_envelope_I   (dot product over time)
        output_Q = trace_Q · mf_envelope_Q
        output   = output_I + output_Q

    This is equivalent to projecting the trace onto the matched-filter axis in
    the IQ plane, yielding a single scalar that maximises the SNR for detecting
    the target pulse shape.

    Parameters
    ----------
    traces : ndarray, shape (num_traces, trace_length, 2)
        Batch of IQ traces.
    mf_rmf_envelope_I : ndarray, shape (trace_length,)
        I-channel component of the MF or RMF pulse template.
    mf_rmf_envelope_Q : ndarray, shape (trace_length,)
        Q-channel component of the MF or RMF pulse template.

    Returns
    -------
    ndarray, shape (num_traces,)
        Scalar MF (or RMF) output for each trace in the batch.
    """
    # Separate I and Q components from the batch of traces
    I_traces = traces[..., 0]  # Shape: (num_traces, trace_length)
    Q_traces = traces[..., 1]  # Shape: (num_traces, trace_length)

    # Dot product along the time axis (equivalent to cross-correlation at lag 0)
    mf_rmf_output_I = np.dot(I_traces, mf_rmf_envelope_I)  # Shape: (num_traces,)
    mf_rmf_output_Q = np.dot(Q_traces, mf_rmf_envelope_Q)  # Shape: (num_traces,)

    # Sum I and Q contributions to get the total MF output scalar
    mf_rmf_outputs = mf_rmf_output_I + mf_rmf_output_Q  # Shape: (num_traces,)

    return mf_rmf_outputs


# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------

def hdf5_data_load(data_path: str, data_file_name: str, data_type: str) -> (
        Tuple)[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load feature matrix and label vector from an HDF5 file.

    The HDF5 file is expected to contain two top-level datasets:
      - For data_type='Train': keys 'X_train' and 'y_train'
      - For data_type='Test':  keys 'X_test'  and 'y_test'

    Parameters
    ----------
    data_path : str
        Directory containing the HDF5 file.
    data_file_name : str
        File name (not full path).
    data_type : str
        'Train' or 'Test'.

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray]]
        (X, y) where X contains the IQ traces and y contains integer labels.
        Returns (None, None) if data_type is unrecognised or the file is missing.
    """
    # Map data_type string to the corresponding HDF5 dataset keys
    data_keys = {
        "Train": ("X_train", "y_train"),
        "Test":  ("X_test",  "y_test")
    }

    dataset_keys = data_keys.get(data_type)

    if not dataset_keys:
        logger.error(f"Data type '{data_type}' is not supported. Choose 'Train' or 'Test'.")
        return None, None

    # Validate file existence before opening
    file_path = os.path.join(data_path, data_file_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Load all rows into memory at once.
    # For very large datasets consider custom_hdf5_data_loader instead.
    with h5py.File(f"{data_path}/{data_file_name}", 'r') as hf:
        X = np.array(hf[dataset_keys[0]])
        y = np.array(hf[dataset_keys[1]])
    return X, y


def custom_hdf5_data_loader(data_path: str, data_file_name: str, data_type: str, percent: float = 0.2):
    """
    Load a fraction of an HDF5 dataset without reading the entire file into memory.

    Useful for quick prototyping or when RAM is limited.  Loads the first
    ``percent`` × 100% of rows from the dataset.

    Parameters
    ----------
    data_path : str
        Directory containing the HDF5 file.
    data_file_name : str
        File name (not full path).
    data_type : str
        'Train' or 'Test'.
    percent : float
        Fraction of rows to load, in (0, 1].  Default: 0.2 (20%).

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray]]
        (X, y) as NumPy arrays loaded from HDF5.
        X is always truncated to the first 500 time samples along axis=1.
    """
    data_keys = {
        "Train": ("X_train", "y_train"),
        "Test":  ("X_test",  "y_test")
    }
    dataset_keys = data_keys.get(data_type)

    if not dataset_keys:
        logger.error(f"Data type '{data_type}' is not supported. Choose 'Train' or 'Test'.")
        return None, None

    file_path = os.path.join(data_path, data_file_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    print(f"{data_path}/{data_file_name}")
    with h5py.File(f"{data_path}/{data_file_name}", 'r') as hf:
        X_dset = hf[dataset_keys[0]]
        y_dset = hf[dataset_keys[1]]

        # Calculate how many rows correspond to the requested fraction
        num_total_rows = X_dset.shape[0]
        num_rows_to_load = int(num_total_rows * percent)

        # HDF5 slice: first num_rows_to_load rows, first 500 time samples, all channels
        # Slicing an HDF5 dataset is lazy — only the requested bytes are read from disk.
        X = X_dset[0:num_rows_to_load, 0:500, :]
        y = y_dset[0:num_rows_to_load]

    return X, y


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------

class QubitTraceDataset(Dataset):
    """
    Minimal ``torch.utils.data.Dataset`` wrapper for NumPy trace / label arrays.

    Converts NumPy arrays to properly-typed PyTorch tensors:
      - traces: float32  (required by all standard PyTorch layers)
      - labels: long     (required by nn.CrossEntropyLoss)

    Parameters
    ----------
    traces : array-like, shape (N, feature_dim)
        Pre-processed and normalised IQ feature vectors.
    labels : array-like, shape (N,)
        Integer class labels.
    """

    def __init__(self, traces, labels):
        self.traces = torch.tensor(traces, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        """Return the (trace, label) pair at index ``idx``."""
        return self.traces[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Normalisation utilities
# ---------------------------------------------------------------------------

def normalize_data(X_train, X_test):
    """
    Z-score normalise X_test using statistics computed from X_train.

    Applies:  X_norm = (X - mean(X_train)) / std(X_train)

    Statistics are computed **per feature** (axis=0), so each feature is
    independently centred and scaled.  Only training statistics are used;
    test data is transformed with the same parameters to prevent data leakage.

    Parameters
    ----------
    X_train : ndarray, shape (N_train, n_features)
    X_test  : ndarray, shape (N_test,  n_features)

    Returns
    -------
    (X_train_norm, X_test_norm) : Tuple of normalised arrays (new copies).
    """
    X_mean = np.mean(X_train, axis=0)
    X_std  = np.std(X_train, axis=0)

    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm  = (X_test  - X_mean) / X_std

    return X_train_norm, X_test_norm


def normalize_data_inplace(X_train, X_test):
    """
    Z-score normalise in-place, modifying the input arrays directly.

    Functionally identical to ``normalize_data`` but avoids creating copies of
    potentially large arrays.

    Parameters
    ----------
    X_train : ndarray, shape (N_train, n_features)  [modified in-place]
    X_test  : ndarray, shape (N_test,  n_features)  [modified in-place]

    Returns
    -------
    (X_train, X_test) : The same array objects, now normalised.

    Note
    ----
    The input arrays must be of a mutable float dtype (e.g. float64).
    Using integer or read-only arrays will raise a TypeError.
    """
    X_mean = np.mean(X_train, axis=0)
    X_std  = np.std(X_train, axis=0)

    X_train -= X_mean
    X_train /= X_std

    X_test -= X_mean
    X_test /= X_std

    return X_train, X_test


def nearest_power_of_2(std_array: np.ndarray) -> np.ndarray:
    """
    Round each element of an array to the nearest power of 2.

    Used when targeting fixed-point hardware (FPGA) that can implement
    division by powers of 2 as a single bit-shift operation.

    Parameters
    ----------
    std_array : ndarray
        Input array of positive floats (typically column-wise standard deviations).

    Returns
    -------
    ndarray
        Array where each value is 2^round(log2(x)).

    Example
    -------
    nearest_power_of_2(np.array([3.0, 7.5, 0.6]))
    # -> array([4., 8., 0.5])   (2^2, 2^3, 2^(−1))
    """
    powers_of_2 = np.power(2, np.round(np.log2(std_array)))
    return powers_of_2


def normalize_data_std_p2(X_train, X_test):
    """
    Z-score normalise with the standard deviation rounded to the nearest power of 2.

    This is a hardware-friendly variant of z-score normalisation designed for
    FPGA fixed-point arithmetic.  By rounding std to a power of 2, the division
    step becomes a cheap bit-shift in hardware.

    Statistics are computed on X_train and applied to both X_train and X_test.

    Parameters
    ----------
    X_train : ndarray, shape (N_train, n_features)
    X_test  : ndarray, shape (N_test,  n_features)

    Returns
    -------
    (X_train_norm, X_test_norm) : Tuple of normalised arrays.
    """
    X_mean = np.mean(X_train, axis=0)
    X_std  = np.std(X_train, axis=0)

    # Replace exact std with nearest power of 2 for hardware compatibility
    X_std = nearest_power_of_2(X_std)

    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm  = (X_test  - X_mean) / X_std

    print('power_2 std:', np.unique(X_std))
    return X_train_norm, X_test_norm


def normalize_data_forb(X_train, X_test):
    """
    Normalise by the Frobenius norm of the training data matrix.

    The Frobenius norm F = sqrt(sum of all elements squared) provides a
    single global scale factor.  Dividing by F rescales the data so that
    the training matrix has unit Frobenius norm.

    Parameters
    ----------
    X_train : ndarray, shape (N_train, n_features)
    X_test  : ndarray, shape (N_test,  n_features)

    Returns
    -------
    (X_train_norm, X_test_norm) : Normalised arrays.
    """
    frob_norm = np.linalg.norm(X_train)  # Scalar: sqrt(sum(X_train^2))
    X_train_norm = X_train / frob_norm
    X_test_norm  = X_test  / frob_norm
    return X_train_norm, X_test_norm


def normalize_data_forb_weighted(X_train, X_test):
    """
    Frobenius-norm normalisation with a 4× dampening factor.

    Identical to ``normalize_data_forb`` but divides by 4 × Frobenius norm.
    Used for datasets with particularly high-amplitude raw ADC values where
    plain Frobenius normalisation still leaves values too large.

    Parameters
    ----------
    X_train : ndarray, shape (N_train, n_features)
    X_test  : ndarray, shape (N_test,  n_features)

    Returns
    -------
    (X_train_norm, X_test_norm) : Normalised arrays.
    """
    frob_norm = np.linalg.norm(X_train)
    X_train_norm = X_train / (4 * frob_norm)
    X_test_norm  = X_test  / (4 * frob_norm)
    return X_train_norm, X_test_norm


def normalize_data_forb_subtraction(X_train, X_test):
    """
    Per-sample Frobenius-norm normalisation with mean subtraction.

    Computes the mean of the per-sample L2 norms (mean Frobenius norm per row)
    and uses the mean per-sample std as an additional scale factor.
    This makes each sample's contribution independent of its global amplitude.

    Parameters
    ----------
    X_train : ndarray, shape (N_train, n_features)
    X_test  : ndarray, shape (N_test,  n_features)

    Returns
    -------
    (X_train_norm, X_test_norm) : Normalised arrays.
    """
    # Mean of per-sample standard deviations (shape: (N,) -> scalar)
    X_std  = np.mean(np.std(X_train, axis=1))
    # Mean of per-sample L2 norms (shape: (N,) -> scalar)
    X_norm = np.mean(np.linalg.norm(X_train, axis=1))

    X_train_norm = (X_train / X_norm) / X_std
    X_test_norm  = (X_test  / X_norm) / X_std

    return X_train_norm, X_test_norm


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def reduce_trace_duration(X_train: np.ndarray, X_test: np.ndarray, reduction_size: int = 500) -> (
        np.ndarray, np.ndarray):
    """
    Truncate traces to a shorter time window by slicing the time axis.

    Keeps only the first ``reduction_size`` time samples.  Samples beyond
    this index are discarded.  The truncation does not alter the data values,
    only the duration of each trace.

    Parameters
    ----------
    X_train : ndarray, shape (N_train, original_trace_length, 2)
    X_test  : ndarray, shape (N_test,  original_trace_length, 2)
    reduction_size : int
        Number of time samples to retain.  Must be <= original_trace_length.

    Returns
    -------
    (X_train_reduced, X_test_reduced): Arrays with shape
        (N_*, reduction_size, 2).
    """
    X_train_reduced = X_train[:, :reduction_size, :]
    X_test_reduced  = X_test[:, :reduction_size, :]
    return X_train_reduced, X_test_reduced


def flatten_iq_dimensions(X: np.ndarray) -> np.ndarray:
    """
    Flatten the (trace_length, 2) IQ dimensions into a single feature vector.

    Reshape (N, trace_length, 2) -> (N, trace_length * 2).
    The I and Q channel values are interleaved as stored in memory (row-major /
    C-order), so the output layout is [I_0, Q_0, I_1, Q_1, ..., I_T, Q_T].

    Parameters
    ----------
    X : ndarray, shape (N, trace_length, 2)

    Returns
    -------
    ndarray, shape (N, trace_length * 2)
    """
    n_samples, dim1, dim2 = X.shape
    X_reshaped = X.reshape(n_samples, -1)  # -1 infers dim1 * dim2
    return X_reshaped


def stratified_split(x, y, train_sample_size, val_sample_size):
    """
    Class-balanced split of (x, y) into training and validation subsets.

    For each unique class, independently samples up to ``train_sample_size``
    examples for training and up to ``val_sample_size`` examples for validation.
    Samples are taken from the *same shuffled pool*, so no example appears in
    both subsets.

    This ensures that class imbalance (e.g. many more shots in |0⟩ than |1⟩)
    does not distort the training or validation statistics.

    Parameters
    ----------
    x : ndarray, shape (N, n_features)
    y : ndarray, shape (N,)   -- integer class labels
    train_sample_size : int
        Maximum number of samples per class in the training split.
    val_sample_size : int
        Maximum number of samples per class in the validation split.

    Returns
    -------
    (x_train, x_val, y_train, y_val) : Four NumPy arrays.
        x_train / x_val : shape  (n_classes × train/val_sample_size, n_features)
        y_train / y_val : shape  (n_classes × train/val_sample_size,)
    """
    unique_classes = np.unique(y)
    x_train, x_val, y_train, y_val = [], [], [], []

    for cls in unique_classes:
        # Get indices belonging to this class and shuffle them
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)

        cls_x = x[cls_indices]
        cls_y = y[cls_indices]

        # First train_sample_size examples go to training
        x_train.append(cls_x[:train_sample_size])
        y_train.append(cls_y[:train_sample_size])

        # Next val_sample_size examples go to validation (no overlap with training)
        x_val.append(cls_x[train_sample_size:train_sample_size + val_sample_size])
        y_val.append(cls_y[train_sample_size:train_sample_size + val_sample_size])

    # Stack all class chunks into single arrays
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_val   = np.concatenate(x_val)
    y_val   = np.concatenate(y_val)

    return x_train, x_val, y_train, y_val


# ---------------------------------------------------------------------------
# Hardware-friendly normalisation utilities
# ---------------------------------------------------------------------------

def find_nearest_power_of_two(x):
    """
    Return the smallest power of 2 that is >= x.

    This is the classic "bit-length" trick for integer inputs.
    Used when preparing normalisation parameters for FPGA fixed-point circuits.

    Parameters
    ----------
    x : float or int

    Returns
    -------
    int
        Smallest n = 2^k such that n >= x.

    Examples
    --------
    find_nearest_power_of_two(5)   -> 8
    find_nearest_power_of_two(8)   -> 8
    find_nearest_power_of_two(0)   -> 1
    """
    if x == 0:
        return 1
    x = int(x)
    # (x - 1).bit_length() gives ceil(log2(x)) for x > 1
    return 1 << (x - 1).bit_length()


def compute_normalization_params(train_data):
    """
    Compute per-component normalisation parameters optimised for fixed-point hardware.

    For each IQ component (I and Q separately), computes:
      - ``n``  : nearest power of 2 >= max |value| in that component
      - ``mu`` : mean of that component across all training samples

    These parameters are intended to be applied by ``apply_normalization``.
    Together they implement the hardware-friendly transform:
        y = (x + n - mu) >> (log2(n) + 1)

    Parameters
    ----------
    train_data : ndarray, shape (N_samples, n_features, 2)
        Raw IQ traces.

    Returns
    -------
    dict
        Keys: 'n_0', 'mu_0', 'n_1', 'mu_1'
        (one pair per IQ component: 0 = I, 1 = Q)
    """
    params = {}
    n_samples, n_features, components = train_data.shape

    for component in range(components):
        component_data = train_data[:, :, component]

        # Maximum absolute value -> rounded to nearest power of 2 for bit-shift scaling
        max_value = np.max(np.abs(component_data))
        params[f'n_{component}']  = find_nearest_power_of_two(max_value)

        # Mean for centering
        params[f'mu_{component}'] = np.mean(component_data)

    return params


def apply_normalization(data, params):
    """
    Apply pre-computed fixed-point normalisation parameters to a dataset.

    Implements the transform (per IQ component):
        tmp = data + n - mu                          # centre around n
        tmp_shifted = tmp × 2^17                     # left-shift for integer precision
        norm = tmp_shifted >> (log2(n) + 1)          # right-shift to normalise
        output = norm / 2^(17 - (log2(n) + 1))      # convert back to float

    This scheme is equivalent to z-score normalisation with the divisor rounded
    to a power of 2, enabling efficient FPGA implementation without a divider.

    Parameters
    ----------
    data : ndarray, shape (N, n_features, 2)
        Raw or pre-processed IQ data.
    params : dict
        Output of ``compute_normalization_params`` (keys: 'n_0', 'mu_0', 'n_1', 'mu_1').

    Returns
    -------
    ndarray, shape (N, n_features, 2), dtype float32
        Normalised data.
    """
    n_samples, n_features, components = data.shape
    normalized_data = np.zeros_like(data, dtype=np.float32)

    for component in range(components):
        component_data = data[:, :, component]

        n  = params[f'n_{component}']
        mu = params[f'mu_{component}']

        # Centre the data: shift by n and subtract mean
        tmp = component_data + n - mu

        # Integer fixed-point arithmetic (emulated in float):
        # Multiply by 2^17 to move the decimal point far right
        tmp_shifted = (tmp * (1 << 17)).astype(np.int64)
        # Right-shift by (bit_length(n) + 1) acts as division by ~2n
        data_norm   = tmp_shifted >> (n.bit_length() + 1)

        # Convert back to float by undoing the residual bit-shift
        normalized_data[:, :, component] = data_norm / (1 << (17 - (n.bit_length() + 1)))

    return normalized_data
