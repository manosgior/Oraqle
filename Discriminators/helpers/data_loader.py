"""
data_loader.py
==============
High-level data loading and transformation tools for the qubit readout pipeline.

Overview
--------
This module defines ``QubitData``, the main entry point for turning raw HDF5
dataset files into ready-to-train NumPy arrays.  It wraps several low-level
utility functions from ``helpers/data_utils.py`` and exposes three public
transform variants that cater to different model families:

  load_transform()
      Standard pipeline for single-qubit FNN models and the Transformer.
      Flattens the IQ trace and applies a configurable normalisation strategy.

  load_transform_KLiNQ_KD(target_length)
      Extended pipeline for KLiNQ knowledge-distillation.
      Produces a richer feature vector by combining three representations:
        1. Full flattened IQ trace
        2. Time-averaged IQ (at a lower temporal resolution)
        3. Matched-Filter (MF) / Reverse Matched-Filter (RMF) scalar features

  transform(...)
      Lightweight wrapper that skips the HDF5 loading step and applies
      the standard transformations to arrays that are already in memory.

Data format
-----------
  Raw HDF5 datasets contain:
    X : shape (N, trace_length, 2)  -- IQ traces
          N             : number of shots
          trace_length  : number of ADC samples (e.g. 500 @ 2 ns/sample = 1 µs)
          2             : channels — index 0 is I (in-phase), index 1 is Q (quadrature)
    y : shape (N,)  -- integer class labels
          For 5 multiplexed qubits: integer ∈ [0, 31] encoding all qubit states as bits.
          For individual qubit files: integer ∈ {0, 1}.

Normalisation strategies (selected via data_config['normalize'])
----------------------------------------------------------------
  'mean/std'       : z-score normalisation (subtract column mean, divide by std)
                     Statistics computed on training set and reused for test.
  'forb'           : divide by the Frobenius norm of the training data matrix.
  'forb_s'         : subtract mean of norms, then divide; see normalize_data_forb_subtraction.
  'forb-weighted'  : like 'forb' but divides by 4 × norm (for amplitude-heavy signals).
  'no-norm'        : no normalisation (raw ADC values).
  'mean/p2std'     : z-score with std rounded to the nearest power of 2
                     (hardware-friendly for fixed-point arithmetic).

Validation sampling modes (selected via data_config['val_sampling_mode'])
--------------------------------------------------------------------------
  'stratified' : uses ``stratified_split`` to ensure each class is equally
                 represented in the validation set (preferred for imbalanced data).
  anything else: uses sklearn's ``train_test_split`` with test_size=0.1 and shuffle.
"""

from typing import Optional, Tuple
import pickle
import loguru
import numpy as np
from numpy import ndarray
from loguru import logger
from sklearn.model_selection import train_test_split

from helpers.data_utils import (hdf5_data_load, normalize_data, stratified_split, flatten_iq_dimensions,
                                normalize_data_forb, normalize_data_forb_subtraction, normalize_data_forb_weighted,
                                normalize_data_std_p2, normalize_data_inplace, reduce_trace_duration, apply_mf_rmf)


class QubitData:
    """
    High-level data manager for qubit readout datasets.

    Loads paired train/test HDF5 files and applies preprocessing steps
    (trace truncation, channel flattening, normalisation, and train/val split)
    according to the supplied configuration dictionary.

    Parameters
    ----------
    data_path : str
        Absolute path to the directory containing the HDF5 data files.
    data_train_file_name : str
        File name (not full path) of the training HDF5 file.
    data_test_file_name : str
        File name (not full path) of the test HDF5 file.
    data_config : dict
        Dictionary with the following required keys:
          'train_sample_size'  (int)  : max samples per class in the training split.
          'val_sample_size'    (int)  : max samples per class in the validation split.
          'trace_length'       (int)  : number of ADC time samples to keep (truncates
                                        the raw trace from the left).
          'normalize'          (str)  : normalisation strategy (see module docstring).
          'val_sampling_mode'  (str)  : validation split strategy ('stratified' or other).
    mf_rmf_env_file_name : str, optional
        File name of the pickle file containing pre-computed MF / RMF envelope
        templates.  Only required when calling ``load_transform_KLiNQ_KD``.
    """

    def __init__(self, data_path: str, data_train_file_name: str, data_test_file_name: str, data_config: dict,
                 mf_rmf_env_file_name=''):
        self.data_path = data_path
        self.data_train_file = data_train_file_name
        self.data_test_file = data_test_file_name

        # Unpack data_config
        self.train_sample_size = data_config['train_sample_size']
        self.val_sample_size = data_config['val_sample_size']
        self.trace_length = data_config['trace_length']
        self.normalize = data_config['normalize']
        self.val_sampling_mode = data_config['val_sampling_mode']

        # Optional: path to the MF/RMF envelope file (only needed for KLiNQ KD pipeline)
        self.mf_rmf_envelope_file_name = mf_rmf_env_file_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_data(self) -> Tuple[Optional[ndarray], Optional[ndarray], Optional[ndarray], Optional[ndarray]]:
        """
        Load the raw training and testing arrays from HDF5 files.

        Returns (X_train, y_train, X_test, y_test) as NumPy arrays with shapes:
          X_* : (N, trace_length_original, 2)  -- raw IQ traces
          y_* : (N,)                            -- integer class labels

        All four values are None if loading fails.
        """
        print(self.data_path, self.data_train_file)
        try:
            X_train, y_train = hdf5_data_load(
                data_path=self.data_path, data_file_name=self.data_train_file, data_type='Train')
            X_test, y_test = hdf5_data_load(
                data_path=self.data_path, data_file_name=self.data_test_file, data_type='Test')
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None, None, None, None

        return X_train, y_train, X_test, y_test

    def transform(self, X_train, y_train, X_test, y_test, trace_length):
        """
        Apply preprocessing to pre-loaded arrays (trace truncation, flattening,
        normalisation, and train/val split).

        This method is useful when the raw arrays are already in memory (e.g.
        loaded externally).  It is a lightweight alternative to ``load_transform``.

        Parameters
        ----------
        X_train : ndarray, shape (N_train, full_trace_length, 2)
        y_train : ndarray, shape (N_train,)
        X_test  : ndarray, shape (N_test, full_trace_length, 2)
        y_test  : ndarray, shape (N_test,)
        trace_length : int
            Number of ADC samples to keep (truncates from the front).

        Returns
        -------
        Tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
          All arrays are normalised and flattened.
          X_* has shape (N, 2 * trace_length).
        """
        # --- Step 1: Truncate traces to the requested length ---
        X_train, X_test = reduce_trace_duration(X_train, X_test, reduction_size=trace_length)

        # --- Step 2: Flatten (N, trace_length, 2) -> (N, 2*trace_length) ---
        X_train = flatten_iq_dimensions(X_train)
        X_test = flatten_iq_dimensions(X_test)

        # --- Step 3: Apply normalisation (based on training-set statistics only) ---
        if self.normalize.lower() == 'forb':
            logger.info("Data has been Normalized by Frobenius Norm (div)")
            X_train, X_test = normalize_data_forb(X_train, X_test)
        elif self.normalize.lower() == 'forb_s':
            logger.info("Data has been Normalized by Frobenius Norm (subtr)")
            X_train, X_test = normalize_data_forb_subtraction(X_train, X_test)
        elif self.normalize.lower() == 'false':
            logger.info("No normalization")
        elif self.normalize.lower() == 'forb-weighted':
            X_train, X_test = normalize_data_forb_weighted(X_train, X_test)
        else:
            logger.info("Data has been Normalized by Mean/Std")
            X_train, X_test = normalize_data(X_train, X_test)

        # --- Step 4: Split training data into train and validation subsets ---
        X_train, X_val, y_train, y_val = stratified_split(
            X_train, y_train, self.train_sample_size, self.val_sample_size)

        return X_train, y_train, X_val, y_val, X_test, y_test,

    def load_transform(self) -> Tuple:
        """
        Full standard pipeline: load from HDF5 then transform.

        Combines ``load_data()`` and a standardised transformation sequence:
          1. Truncate traces to ``self.trace_length``.
          2. Flatten IQ channels into a single feature vector.
          3. Normalise using the selected strategy.
          4. Split into train and validation subsets.

        Used by: ``SingleQubitFNN``, ``KLiNQTeacherModel``, Transformer training scripts.

        Returns
        -------
        Tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
          X_* shape : (N, 2 * self.trace_length)
          y_* shape : (N,)
          Returns (None, None, None, None) if data loading fails.
        """
        X_train, y_train, X_test, y_test = self.load_data()

        if X_train is None or y_train is None or X_test is None or y_test is None:
            logger.error("Failed to load data. Exiting transformation process.")
            return None, None, None, None

        # Step 1: Truncate to the configured trace length
        X_train, X_test = reduce_trace_duration(X_train, X_test, reduction_size=self.trace_length)

        # Step 2: Flatten (N, trace_length, 2) -> (N, 2*trace_length)
        X_train = flatten_iq_dimensions(X_train)
        X_test = flatten_iq_dimensions(X_test)

        # Step 3: Normalisation
        if self.normalize.lower() == 'forb':
            logger.info("Data is Normalized by Frobenius Norm (div)")
            X_train, X_test = normalize_data_forb(X_train, X_test)
        elif self.normalize.lower() == 'no-norm':
            logger.info("No normalization")
        else:
            logger.info("Data is Normalized by Mean/Std")
            # Inplace variant modifies arrays in-memory to avoid a copy (memory efficient)
            X_train, X_test = normalize_data_inplace(X_train, X_test)

        # Step 4: Train / validation split
        if self.val_sampling_mode == 'stratified':
            # Guarantees equal class representation in both sets
            X_train, X_val, y_train, y_val = stratified_split(
                X_train, y_train, self.train_sample_size, self.val_sample_size)
        else:
            # Simple random split: 90% train, 10% val
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.1, shuffle=True)

        print(X_train.shape, X_test.shape)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def load_transform_KLiNQ_KD(self, target_length) -> Tuple:
        """
        Extended pipeline for KLiNQ knowledge-distillation: loads HDF5 data
        and constructs a rich composite feature vector for the student model.

        Feature vector composition
        --------------------------
        The student receives three complementary views of each trace,
        all column-stacked along the feature axis:

          1. Full flattened IQ trace  (2 × trace_length features)
             The same representation used by the teacher.

          2. Time-averaged IQ  (2 × target_length features)
             The trace is divided into ``target_length`` equal bins;
             each bin is averaged (via ``average_trace_data_fixed_length``).
             This mimics hardware integrators and reduces noise.

          3. Matched-Filter scalar  (1 feature)
             The dot product of the trace with a pre-computed pulse-shape
             template (MF envelope) for both I and Q channels, summed together.
             The MF/RMF envelopes are loaded from the pickle file at
             ``self.mf_rmf_envelope_file_name`` (must contain keys 'MF_I',
             'MF_Q', 'RMF_I', 'RMF_Q' as 1-D arrays of length trace_length).

        Final input dimension examples:
          trace_length=500, target_length=5  -> 1000 + 10 + 1 = 1011
          trace_length=15,  target_length=5  -> 30  + 10 + 1 = 41

        Note: The raw data is first capped at 500 samples if it is longer,
        then truncated to ``self.trace_length`` for the reduced representations.
        The full flattened trace uses all available samples (up to 500).

        Parameters
        ----------
        target_length : int
            Number of averaged time bins for the low-resolution IQ view.

        Returns
        -------
        Tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
          X_* shape : (N, 2*500 + 2*target_length + 1)  approximately
          y_* shape : (N,)
          Returns (None, None, None, None) if data loading fails.
        """
        X_train, y_train, X_test, y_test = self.load_data()
        if X_train is None or y_train is None or X_test is None or y_test is None:
            logger.error("Failed to load data. Exiting transformation process.")
            return None, None, None, None

        # --- Cap raw trace at 500 samples to ensure consistent baseline ---
        if X_train.shape[1] > 500:
            X_train = X_train[:, :500, :]
            X_test = X_test[:, :500, :]

        # --- Reduce to self.trace_length for the MF and averaged representations ---
        X_train_reduced, X_test_reduced = reduce_trace_duration(
            X_train, X_test, reduction_size=self.trace_length)

        # --- Load pre-computed MF / RMF envelope templates ---
        # The pickle file must contain a dict with keys: 'MF_I', 'MF_Q', 'RMF_I', 'RMF_Q'
        # Each value is a 1-D array of shape (trace_length,) representing the
        # matched-filter or reverse-matched-filter pulse envelope.
        with open(f"{self.data_path}/{self.mf_rmf_envelope_file_name}", "rb") as f:
            envelope_components = pickle.load(f)

        # --- Compute MF scalar for train and test (dot product of trace with envelope) ---
        # Shape: (N_train,) and (N_test,) — one scalar per shot
        train_mf_envelope = apply_mf_rmf(X_train_reduced,
                                         envelope_components[f'MF_I'][:self.trace_length],
                                         envelope_components[f'MF_Q'][:self.trace_length])

        train_rmf_envelope = apply_mf_rmf(X_train_reduced,
                                          envelope_components[f'RMF_I'][:self.trace_length],
                                          envelope_components[f'RMF_Q'][:self.trace_length])

        test_mf_envelope = apply_mf_rmf(X_test_reduced,
                                        envelope_components[f'MF_I'][:self.trace_length],
                                        envelope_components[f'MF_Q'][:self.trace_length])

        test_rmf_envelope = apply_mf_rmf(X_test_reduced,
                                         envelope_components[f'RMF_I'][:self.trace_length],
                                         envelope_components[f'RMF_Q'][:self.trace_length])

        # --- Compute time-averaged IQ at target_length resolution ---
        # Shape before flattening: (N, target_length, 2)
        # Shape after flattening:  (N, 2 * target_length)
        X_train_avg = flatten_iq_dimensions(
            self.average_trace_data_fixed_length(X_train_reduced, target_length))
        X_test_avg = flatten_iq_dimensions(
            self.average_trace_data_fixed_length(X_test_reduced, target_length))

        # --- Flatten full trace (up to 500 samples) ---
        # Shape: (N, 2 * 500) = (N, 1000)
        X_train = flatten_iq_dimensions(X_train)
        X_test = flatten_iq_dimensions(X_test)

        # --- Normalise each representation independently ---
        if self.normalize.lower() == 'forb':
            logger.info("Data is Normalized by Frobenius Norm (div)")
            X_train, X_test = normalize_data_forb(X_train, X_test)
        elif self.normalize.lower() == 'no-norm':
            logger.info("Data is NOT Normalized")
        elif self.normalize.lower() == 'mean/p2std':
            logger.info("Data is Normalized by mean/p2std")
            # Hardware-friendly normalisation: std rounded to nearest power of 2
            X_train_avg, X_test_avg = normalize_data_std_p2(X_train_avg, X_test_avg)
            X_train, X_test = normalize_data(X_train, X_test)
            train_mf_envelope, test_mf_envelope = normalize_data_std_p2(train_mf_envelope, test_mf_envelope)
        else:
            logger.info("Data is Normalized by Mean/Std")
            X_train_avg, X_test_avg = normalize_data(X_train_avg, X_test_avg)
            X_train, X_test = normalize_data(X_train, X_test)
            train_mf_envelope, test_mf_envelope = normalize_data(train_mf_envelope, test_mf_envelope)

        # --- Column-stack the three feature groups ---
        # Final shape: (N, 1000 + 2*target_length + 1)
        X_train = np.column_stack((X_train, X_train_avg, train_mf_envelope))
        X_test = np.column_stack((X_test, X_test_avg, test_mf_envelope))

        # --- Train / validation split ---
        if self.val_sampling_mode == 'stratified':
            X_train, X_val, y_train, y_val = stratified_split(
                X_train, y_train, self.train_sample_size, self.val_sample_size)
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.1, shuffle=True)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def average_trace_data_fixed_length(self, data, target_length=100):
        """
        Downsample a batch of IQ traces by averaging contiguous bins.

        Divides the time axis of each trace into ``target_length`` equal-width
        bins and replaces each bin by its mean over all samples in the bin.
        This is equivalent to a box-car (moving-average) filter followed by
        decimation.

        Physical interpretation
        -----------------------
        A hardware integrator computes the mean IQ value over the entire readout
        window.  This function generalises that to ``target_length`` sub-windows,
        preserving some temporal structure while dramatically reducing the
        feature dimension.

        Parameters
        ----------
        data : ndarray, shape (num_traces, trace_duration, 2)
            IQ traces.  trace_duration should be evenly divisible by target_length.
        target_length : int
            Number of output time bins (= number of averaged samples per channel).
            Default: 100.

        Returns
        -------
        ndarray, shape (num_traces, target_length, 2)
            Averaged IQ traces at the reduced temporal resolution.
        """
        num_traces, trace_duration, _ = data.shape
        sampling_interval_ns = 2  # Each ADC sample represents 2 ns (500 MHz ADC)

        # Number of raw samples that map to a single averaged bin
        samples_per_interval = trace_duration // target_length

        # Reshape to (num_traces, target_length, samples_per_interval, 2)
        # then average over the samples_per_interval axis.
        new_trace_duration = target_length
        reshaped_data = data[:, :new_trace_duration * samples_per_interval, :].reshape(
            num_traces, new_trace_duration, samples_per_interval, 2
        )
        # Mean over the bin-samples axis -> (num_traces, target_length, 2)
        averaged_data = np.mean(reshaped_data, axis=2)

        print(f"Averaged Data Shape: {averaged_data.shape}")
        return averaged_data