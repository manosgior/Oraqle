"""
HERQULES — Hierarchical Efficient Readout with QUbit Learning via Ensemble Stages
==================================================================================

This module implements the **HERQULES** classification pipeline for superconducting
qubit state discrimination from raw IQ readout traces.

Overview
--------
Qubit readout produces raw IQ (in-phase / quadrature) time-series traces captured by
an ADC after heterodyne detection.  For a 5-qubit multiplexed readout all five resonator
signals are captured simultaneously in a single wideband IQ stream; they must first be
**demodulated** to extract the individual qubit contributions before classification.

HERQULES combines three complementary stages into a single pipeline:

1. **Frequency demodulation** (:func:`demodulate_multiplexed_traces`)
   The raw multiplexed IQ stream is digitally down-converted and low-pass filtered for
   each qubit's IF frequency, separating the 5 resonator signals.

2. **Pre-classification / trace purification** (:class:`preclassifier`)
   A semi-supervised geometric pre-filter that identifies *clean* ground (|0⟩) and
   excited (|1⟩) traces by clustering in IQ space.  It also separates known error
   events:

   - **Relaxation** (|1⟩→|0⟩ during readout)
   - **Leakage** (|2⟩-state trajectories)
   - **Thermal excitation** (|0⟩→|1⟩ during readout)

3. **Matched-filter & relaxation matched-filter features** (:class:`relaxation_mf_classifier`)
   Optimal linear envelope filters are computed from the purified traces via the
   :func:`get_mf` helper and the functions in ``matched_filter.py``.  Two independent
   matched-filter outputs are produced per qubit:

   - **Standard MF** — discriminates |0⟩ vs |1⟩.
   - **Relaxation MF (RMF)** — detects |1⟩→|0⟩ relaxations to allow soft relabelling.

4. **Neural network classifier** (:class:`Net` / :class:`Net_rmf`)
   A compact 3-layer MLP takes the concatenated MF + RMF scalar outputs as its
   input and produces a logit vector over 32 states (5 qubits × 2 levels).

Data Format
-----------
- Raw multiplexed traces  : ``(N_shots, trace_length, 2)`` — last axis is [I, Q].
- Per-qubit demodulated   : ``(N_shots, trace_length, 2)`` — one file per qubit.
- Labels                  : ``int ∈ [0, 31]`` — binary encoding, bit *k* is qubit *k+1*.

Dataset files (HDF5) are expected in ``data/five_qubit_data/``:
- ``DRaw_C_Tr_v0-001``  — training split
- ``DRaw_C_Te_v0-002``  — test split

Key Parameters (hardware-specific)
-----------------------------------
- ``num_qubits``    : 5
- ``sampling_rate`` : 500 MHz
- ``trace_length``  : 500 samples (= 1 µs window)
- ``freq_readout``  : [-64.73, -25.37, 24.79, 70.27, 127.28] MHz (IF frequencies)

Entry Points
------------
- :func:`train` — full HERQULES training pipeline (pre-filter → MF → NN).
- :func:`test`  — load a saved ``Net_rmf`` checkpoint and evaluate on the test set.
- :func:`train_baseline` — train the raw-trace baseline MLP (``Net_baseline``).

Dependencies
------------
- ``matched_filter.py`` — matched-filter computation helpers.
- ``helpers/data_loader.py`` — HDF5 data loading via :func:`custom_hdf5_data_loader`.
- ``helpers/data_utils.py`` — dataset wrappers and normalisation utilities.
- ``herqules_demodulation``  — (external) demodulation helper (if available).
"""
### Setting up libraries for data processing
import numpy as np
import h5py
import random
import os
import pickle
import pandas as pd

import itertools

from helpers.data_loader import *
from helpers.data_utils import *

from scipy.signal import butter, sosfilt

from herqules_demodulation import *

## Required for NN
import numpy as np
import torch as T
import os
from pstats import SortKey
import time


def get_train_val_and_test_set(trace, y, num_qubits=5, NUM_TRAIN_VAL = 3000, NUM_TEST = 7000, NUM_VAL_RATIO = 0.35, trace_length=500):
    """Split the full dataset into balanced train / validation / test subsets.

    Each of the 2**num_qubits basis states contributes exactly *NUM_TRAIN*
    samples to the training set, *NUM_VAL* to the validation set, and *NUM_TEST*
    to the test set.  This ensures class-balanced splits regardless of the
    overall class distribution in *trace*.

    Args:
        trace (np.ndarray): Raw or demodulated IQ traces with shape
            ``(N_shots, trace_length_full, 2)``.
        y (np.ndarray): Integer class labels with shape ``(N_shots,)`` and
            values in ``[0, 2**num_qubits - 1]``.
        num_qubits (int): Number of qubits; determines the number of classes
            ``(2**num_qubits)``. Default: 5.
        NUM_TRAIN_VAL (int): Total number of samples per class used for
            training + validation combined. Default: 3000.
        NUM_TEST (int): Number of test samples per class. Default: 7000.
        NUM_VAL_RATIO (float): Fraction of *NUM_TRAIN_VAL* reserved for
            validation. Default: 0.35.
        trace_length (int): Number of time samples to retain from each trace
            (truncated from the beginning of the time axis). Default: 500.

    Returns:
        tuple: A 2-tuple ``(splits, labels)`` where

        - **splits** is a 3-tuple ``(train_set, val_set, test_set)`` each with
          shape ``(2**num_qubits * N_split, trace_length, 2)``.
        - **labels** is a 3-tuple ``(y_train, y_val, y_test)`` each a 1-D
          integer array of length ``2**num_qubits * N_split``.
    """
    
    #NUM_VAL = int(NUM_TRAIN_VAL * NUM_VAL_RATIO) 
    #NUM_TRAIN_VAL = NUM_TRAIN_VAL * 5  
    NUM_VAL = int(NUM_TRAIN_VAL * NUM_VAL_RATIO)
    #NUM_TEST = NUM_TEST * 5
    NUM_TRAIN = NUM_TRAIN_VAL - NUM_VAL
    ## Data for training, validation and testing is in train_set, val_set and test_set
    ## Accordingly labels in y_train, y_val, y_test
    train_set = []
    val_set = []
    test_set = []
    y_train = []
    y_val = []
    y_test = []
    for i in range(0, 2**num_qubits):
        ind = np.where(y==i)[0]
        train_set.append(trace[ind[:NUM_TRAIN], :trace_length, :])
        test_set.append(trace[ind[NUM_TRAIN_VAL:NUM_TRAIN_VAL+NUM_TEST], :trace_length, :])
        val_set.append(trace[ind[NUM_TRAIN:NUM_TRAIN_VAL], :trace_length, :])
        y_train.append(np.array([i for _ in range(NUM_TRAIN)]))
        y_val.append(np.array([i for _ in range(NUM_VAL)]))
        y_test.append(np.array([i for _ in range(NUM_TEST)]))
    
    train_set = np.reshape(np.array(train_set), (2**num_qubits * NUM_TRAIN, trace_length, trace.shape[2]))
    val_set = np.reshape(np.array(val_set), (2**num_qubits * NUM_VAL, trace_length, trace.shape[2]))
    test_set = np.reshape(np.array(test_set), (2**num_qubits * NUM_TEST, trace_length, trace.shape[2]))
    y_train = np.reshape(np.array(y_train), (2**num_qubits * NUM_TRAIN))
    y_val = np.reshape(np.array(y_val), (2**num_qubits * NUM_VAL))
    y_test = np.reshape(np.array(y_test), (2**num_qubits * NUM_TEST))
    return tuple((train_set, val_set, test_set)), tuple((np.array(y_train), np.array(y_val), np.array(y_test)))

def demodulate_multiplexed_traces(
    iq_traces: np.ndarray,
    qubit_frequencies: np.ndarray,
    sampling_rate: float,
    filter_cutoff: float = 10e6,
    normalize: bool = True,
    filename_prefix: str = ""
) -> dict:
    """Digitally demodulate a multiplexed 5-qubit IQ trace into per-qubit streams.

    For each qubit's intermediate frequency (IF) the function:

    1. Optionally normalises the raw I/Q streams to remove DC offsets and
       correct for amplitude imbalance between the I and Q mixer channels.
    2. Multiplies (rotates) the IQ phasor by ``exp(j 2π f_IF t)`` to shift the
       qubit's resonator contribution to DC (digital down-conversion).
    3. Applies a low-pass Butterworth filter (order 3) to suppress contributions
       from all other resonators and from noise outside the readout bandwidth.
    4. Saves the filtered per-qubit IQ traces to individual HDF5 files named
       ``demodulated_q{k}_{filename_prefix}.h5``.

    Args:
        iq_traces (np.ndarray): Raw multiplexed IQ traces with shape
            ``(N_shots, trace_length, 2)``; axis-2 order is ``[I, Q]``.
        qubit_frequencies (np.ndarray): 1-D array of ``num_qubits`` IF
            frequencies in Hz.  Each entry is the demodulation frequency for
            the corresponding qubit (may be negative for LSB signals).
        sampling_rate (float): ADC sampling rate in samples per second (Hz).
        filter_cutoff (float): Low-pass filter −3 dB cut-off frequency in Hz.
            Default: 10 MHz.
        normalize (bool): If ``True``, remove per-trace DC offset and correct
            the I/Q amplitude imbalance before demodulation. Default: ``True``.
        filename_prefix (str): Optional tag appended to output filenames.
            Default: ``""``.

    Returns:
        dict: Mapping ``{qubit_index: demodulated_traces}`` where
        ``demodulated_traces`` has shape ``(N_shots, trace_length, 2)``.
        The same data is also written to HDF5 files on disk.
    """
    # Get dimensions from the input data
    num_traces, trace_length, _ = iq_traces.shape
    dt = 1.0 / sampling_rate

    # Separate the I and Q components of all traces
    DataI = iq_traces[:, :, 0]
    DataQ = iq_traces[:, :, 1]

    if normalize:
        # Normalize each trace to compensate for DC offset
        DataI = DataI - np.mean(DataI, axis=1, keepdims=True)
        DataQ = DataQ - np.mean(DataQ, axis=1, keepdims=True)
        
        # Normalize for amplitude error between I and Q mixers
        corr_factor = np.std(DataI, axis=1, keepdims=True) / np.std(DataQ, axis=1, keepdims=True)
        DataQ = DataQ * corr_factor

    # Create the time vector for generating the local oscillator signals
    vTime = np.arange(trace_length) * dt

    # Design the low-pass filter (Butterworth is a good choice)
    sos = butter(3, filter_cutoff, btype='low', fs=sampling_rate, output='sos')

    demodulated_qubits = {}
    print("Demodulating traces...")

    for i, dFreq in enumerate(qubit_frequencies):
        # 1. Calculate the cosine/sine vectors for the current qubit's frequency
        vCos = np.cos(2 * np.pi * vTime * dFreq)
        vSin = np.sin(2 * np.pi * vTime * dFreq)

        # 2. Perform complex mixing (rotation) to shift the desired frequency to DC
        # Broadcasting automatically applies the 1D vCos/vSin to every trace
        i_mixed = DataI * vCos + DataQ * vSin
        q_mixed = DataQ * vCos - DataI * vSin

        # 3. Apply the low-pass filter to remove other frequency components
        # The filter is applied along the time axis (axis=1)
        i_filtered = sosfilt(sos, i_mixed, axis=1)
        q_filtered = sosfilt(sos, q_mixed, axis=1)
        
        # 4. Stack the I and Q components back together and store in the dictionary
        demodulated_qubits[i] = np.stack((i_filtered, q_filtered), axis=-1)
        print(f"  - Qubit {i} demodulated.")

    print("\nSaving each qubit's data to a separate file...")
    # Loop through the dictionary items
    for qubit_index, traces in demodulated_qubits.items():
        # Create a unique filename for each qubit
        output_filename = f"demodulated_q{qubit_index+1}_{filename_prefix}.h5"
        
        # Open the file in write mode
        with h5py.File(output_filename, 'w') as hf:
            # Save the data into a dataset named 'traces'
            hf.create_dataset('traces', data=traces)
            print(f"  - Saved traces for Qubit {qubit_index+1} to '{output_filename}'")

    return demodulated_qubits

data_path = "data/five_qubit_data"
train_file_name = "DRaw_C_Tr_v0-001"
test_file_name = "DRaw_C_Te_v0-002"

X_train, y_train = custom_hdf5_data_loader(data_path, train_file_name, "Train", trace_length=500)
X_test, y_test = custom_hdf5_data_loader(data_path, test_file_name, "Test", trace_length=500)

all_data = np.concatenate((X_train, X_test), axis=0)

y_data = np.concatenate((y_train, y_test), axis=0)

#demod = demodulation(all_data)
num_Q 			= 5 		# number of qubits
#states_bin 		= demod.states_config(num_Q)
sampling_rate 	= 500*1e6 #Samples/sec
num_samples 	= int(2e-6*sampling_rate)
freq_readout 	= -np.array([-64.729*1e6,-25.366*1e6,24.79*1e6,70.269*1e6,127.282*1e6]) #Hz
num_records 	= int(5e4) #number of repeated acquisitions	
DT_Bin			= 2e-9 # or higher


def distance(x0, y0, x1, y1):
    """Euclidean distance between two points (or arrays of points) in 2-D IQ space.

    Args:
        x0, y0: Coordinates of the first point (scalars or arrays).
        x1, y1: Coordinates of the second point (scalars or arrays).

    Returns:
        float or np.ndarray: ``sqrt((x0−x1)² + (y0−y1)²)``.
    """
    return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)

def get_data(qubit):
    """Load the demodulated IQ traces and labels for a single qubit.

    Reads the HDF5 file ``demodulated_q{qubit}_.h5`` (produced by
    :func:`demodulate_multiplexed_traces`) and retrieves the array stored
    under the ``'traces'`` key.  The shared global label array ``y_data`` is
    returned unchanged as the label set (all qubits share the same experiment
    shots).

    Args:
        qubit (int): 1-based qubit index (1 … 5).

    Returns:
        tuple: ``(traces, y, t)`` where

        - **traces** (np.ndarray): Shape ``(N_shots, trace_length, 2)``.
        - **y** (np.ndarray): Integer labels of shape ``(N_shots,)`` with
          values in ``[0, 31]``.
        - **t** (list): Empty placeholder (reserved for future timing data).
    """
    #file = './DD_10k_f%i_v1'%(qubit)
    file = 'demodulated_q%i_.h5'%(qubit)
    data = h5py.File(file, 'r')
    traces = np.array(data['traces'])
    y = y_data
    t = []
    return traces, y, t

getbinary = lambda x, n: format(x, 'b').zfill(n)

def get_traces(num_qubits=5, plot=False, rscale=1, data_type=0):
    """Purify and characterise per-qubit IQ traces for pre-classification.

    For each qubit the function:

    1. Loads demodulated traces and splits them into train / val / test via
       :func:`get_train_val_and_test_set`.
    2. Computes the IQ centroids of the |0⟩ and |1⟩ clouds and defines a
       *purity radius* (``rscale × half the inter-centroid distance``).
    3. Filters traces within the purity radius to obtain clean |0⟩ and |1⟩
       reference traces and refines the centroids on the purified set.
    4. Identifies traces *outside* the radius and further classifies them as:

       - **Relaxation traces** — |1⟩-labelled traces whose IQ mean falls
         inside the |0⟩ cluster (qubit decayed during readout).
       - **|2⟩ leakage traces** — |1⟩-labelled traces that lie outside both
         clusters (qubit leaked to the second excited state).
       - **Excitation traces** — |0⟩-labelled traces that lie outside the
         |0⟩ cluster (thermal/state-prep errors).

    The results are stored in a per-qubit dictionary and returned together
    with a set of filtered global indices suitable for :class:`preclassifier`.

    Args:
        num_qubits (int): Number of qubits to process. Default: 5.
        plot (bool): If ``True``, print trace-count diagnostics. Default: ``False``.
        rscale (float): Multiplier on the purity radius (> 1 widens the
            acceptance region). Default: 1.
        data_type (int): Which split to analyse: 0 = train, 1 = val, 2 = test.
            Default: 0.

    Returns:
        tuple: ``(qubit_traces, filtered_indices)`` where

        - **qubit_traces** (dict): Keys are 1-based qubit indices; values are
          dicts with entries:

          =========================================  =========================
          Key                                        Content
          =========================================  =========================
          ``'gnd_0'``                                Mean trace for |0⟩
          ``'gnd_1'``                                Mean trace for |1⟩
          ``'relax'``                                Mean relaxation trace
          ``'ket2'``                                 |2⟩ traces (array)
          ``'excite'``                               Mean excitation trace
          ``'mean_0'``, ``'mean_1'``                 Raw IQ centroids
          ``'mean_0_filtered'``, ``'mean_1_filtered'`` Purified IQ centroids
          ``'traces_relax'``                         Relaxation trace array
          ``'traces_0'``                             Purified |0⟩ trace array
          ``'traces_1'``                             Purified |1⟩ trace array
          ``'traces_excite'``                        Excitation trace array
          =========================================  =========================

        - **filtered_indices** (tuple): ``(indices_0, one_indices, indices_1, indices_y)``
          used by :meth:`preclassifier.predict`.
    """
    qubit_traces = {}
    indices_0 = np.arange(10000)
    one_indices = {2**(qubit - 1):None for qubit in range(1, num_qubits+1)}
    for qubit in range(1, num_qubits+1):
        dem, y, t = get_data(qubit)
        print(dem.shape)
        #indices_0 = np.arange(np.array(dem).shape[0])
        traces, ys = get_train_val_and_test_set(dem, y)
        dem = traces[data_type]
        y = ys[data_type]
        del traces
        traces = dem
        # Compute the centroid of the clouds corresponding to 0 and 1
        zero = 0
        one = 2**(qubit - 1)
        i0 = np.mean(traces[np.where(y==zero)[0], :, 0], axis=1)
        q0 = np.mean(traces[np.where(y==zero)[0], :, 1], axis=1)
        i1 = np.mean(traces[np.where(y==one)[0], :, 0], axis=1)
        q1 = np.mean(traces[np.where(y==one)[0], :, 1], axis=1)
        i = np.mean(traces[:, :, 0], axis=1)
        q = np.mean(traces[:, :, 1], axis=1)
        x0 = np.mean(i0)
        y0 = np.mean(q0)
        x1 = np.mean(i1)
        y1 = np.mean(q1)
        midx = (x0 + x1) / 2
        midy = (y0 + y1) / 2
        radius = rscale * distance(x0, y0, x1, y1) / 2 # Radius from the centroid where a trace is considered correct
        traces_0 = traces[np.where(y==zero)[0], :, :]
        traces_1 = traces[np.where(y==one)[0], :, :]
        traces_0 = traces_0[np.where(distance(i0, q0, x0, y0) < radius)[0], :, :]
        traces_1 = traces_1[np.where(distance(i1, q1, x1, y1) < radius)[0], :, :]
        indices_0 = np.intersect1d(indices_0, np.where((y==zero) & (distance(i, q, x0, y0) < radius))[0])
        indices_1 = np.where((y==one) & (distance(i, q, x1, y1) < radius))[0]
        
        # Find the new centroids with the purified traces
        x0_filtered = np.mean(traces_0[:, :, 0], axis=1)
        y0_filtered = np.mean(traces_0[:, :, 1], axis=1)
        x1_filtered = np.mean(traces_1[:, :, 0], axis=1)
        y1_filtered = np.mean(traces_1[:, :, 1], axis=1)
        one_indices[one] = indices_1
        
        if plot:
            print('New # traces:%i; old: %i'%(traces_0.shape[0] + traces_1.shape[0], i0.shape[0] + i1.shape[0]))
        
        new_i0 = np.mean(traces_0[:, :, 0], axis=1)
        new_q0 = np.mean(traces_0[:, :, 1], axis=1)
        new_i1 = np.mean(traces_1[:, :, 0], axis=1)
        new_q1 = np.mean(traces_1[:, :, 1], axis=1)

        # Found correct traces for ground and excited states
        # Now to find traces corresponding to 1->0 relaxations and 0->1 excitations
        # Also need to distinguish between initialization errors and relaxations/excitations
        # Find ground truth (0/1) traces

        ground_0_trace = np.mean(traces_0, axis=0)
        ground_1_trace = np.mean(traces_1, axis=0)

        # Relaxation traces vs. |2> state traces:
        # |2> traces are generally in a different direction than relaxation traces;
        # Vector of relaxation trace -> ground state trace
        # Use centroid of ground state cluster with the incorrect |1> trace to determine
        # vector direction. Then if direction is similar to the vector joining the two
        # Centroids (0, 1), then it is most likely a relaxation trace. 

        incorrect_traces_0 = traces[np.where(y==zero)[0], :, :]
        incorrect_traces_1 = traces[np.where(y==one)[0], :, :]
        incorrect_traces_0 = incorrect_traces_0[np.where(distance(i0, q0, x0, y0) >= rscale * radius)[0], :, :]
        incorrect_traces_1 = incorrect_traces_1[np.where(distance(i1, q1, x1, y1) >= rscale * radius)[0], :, :]

        # Find traces showing relaxation by evaluation if the mean of the trace lies in
        # the circle corresponding to the ground state
        incorrect_i1 = np.mean(incorrect_traces_1[:, :, 0], axis=1)
        incorrect_q1 = np.mean(incorrect_traces_1[:, :, 1], axis=1)
        relax_traces_1 = incorrect_traces_1[np.where(distance(incorrect_i1, incorrect_q1, x0, y0) <= rscale * radius)[0], :, :]
        relax_trace = np.mean(relax_traces_1, axis=0)
        ket_2_traces = incorrect_traces_1[np.where(distance(incorrect_i1, incorrect_q1, x0, y0) > rscale * radius)[0], :, :]
        ket_2_trace = np.mean(ket_2_traces, axis=0)
        excitation_traces_0 = incorrect_traces_0
        excitation_trace = np.mean(excitation_traces_0, axis=0)

        data = {}
        data['gnd_0'] = ground_0_trace
        data['gnd_1'] = ground_1_trace
        data['relax'] = relax_trace
        data['ket2'] = ket_2_traces
        data['excite'] = excitation_trace
        data['mean_0'] = tuple((x0, y0))
        data['mean_1'] = tuple((x1, y1))
        data['mean_0_filtered'] = tuple((np.mean(x0_filtered), np.mean(y0_filtered)))
        data['mean_1_filtered'] = tuple((np.mean(x1_filtered), np.mean(y1_filtered)))
        data['traces_relax'] = relax_traces_1
        data['traces_0'] = traces_0
        data['traces_1'] = traces_1
        data['traces_excite'] = incorrect_traces_0
        qubit_traces[qubit] = data           

    indices_y = np.zeros(indices_0.shape)
    indices_1 = np.array([])
    for key in one_indices.keys():
        indices_y = np.hstack((indices_y, key * np.ones(one_indices[key].shape)))
        indices_1 = np.hstack((indices_1, one_indices[key]))
    filtered_indices = tuple((indices_0, one_indices, indices_1, indices_y))
    return qubit_traces, filtered_indices

def get_mf(traces_0, traces_1):
    """Compute the optimal matched-filter envelope and discrimination threshold.

    The matched filter (MF) envelope is the Wiener-optimal linear discriminant:

    .. math::

        \\mathbf{h} = \\frac{\\mathbb{E}[\\mathbf{x}_0 - \\mathbf{x}_1]}
                            {\\mathrm{Var}[\\mathbf{x}_0 - \\mathbf{x}_1]}

    where :math:`\\mathbf{x}_0` and :math:`\\mathbf{x}_1` are the flattened IQ
    traces for the |0⟩ and |1⟩ states, respectively.  The two classes are
    balanced by random sub-sampling of the majority class before computing the
    envelope.

    The discrimination threshold is set at the 99.5th percentile of the
    distribution of MF outputs on |0⟩ traces, providing a high-confidence
    acceptance region for |0⟩.

    Args:
        traces_0 (np.ndarray): |0⟩ traces with shape
            ``(N_0, trace_length, 2)``.
        traces_1 (np.ndarray): |1⟩ traces with shape
            ``(N_1, trace_length, 2)``.

    Returns:
        tuple: ``(mf, threshold)`` where

        - **mf** (np.ndarray): 1-D MF envelope of length
          ``trace_length * 2`` (flattened IQ).
        - **threshold** (float): 99.5 %-ile decision threshold; a new trace
          is classified as |0⟩ if ``dot(mf, trace) > threshold``.
    """
    traces_0 = traces_0.reshape(-1, traces_0.shape[1] * traces_0.shape[2])
    traces_1 = traces_1.reshape(-1, traces_1.shape[1] * traces_1.shape[2])
    # traces_0 generally more than traces_1 traces. 
    # Do random sampling from the trace_0 set
    mf = None
    if traces_1.shape[0] > traces_0.shape[0]:
        indices = np.random.choice(traces_1.shape[0], traces_0.shape[0], replace=False)
        mf = np.mean(traces_0 - traces_1[indices], axis=0) / np.var(traces_0 - traces_1[indices], axis=0)
    else:
        indices = np.random.choice(traces_0.shape[0], traces_1.shape[0], replace=False)
        mf = np.mean(traces_0[indices] - traces_1, axis=0) / np.var(traces_0[indices] - traces_1, axis=0)
    
    filtered = np.sum(np.multiply(mf, traces_0), axis=1)
    filtered = np.sort(filtered)
    ind = int(0.995 * filtered.shape[0])
    threshold = filtered[ind]

    return mf, threshold

class preclassifier():
    """Semi-supervised geometric pre-filter for IQ trace purification.

    Before training the matched-filter or neural-network classifiers it is
    important to build *clean* reference traces for each qubit state.
    ``preclassifier`` does this by clustering the time-averaged IQ responses
    into |0⟩ and |1⟩ clouds and retaining only traces that fall within a
    user-specified radius around each centroid.

    The class also identifies and catalogues three types of error events:

    - **Relaxation** (|1⟩→|0⟩ mid-readout)
    - **Leakage**    (|2⟩ trajectories)
    - **Excitation** (|0⟩→|1⟩ mid-readout)

    These catalogued trace classes are subsequently used by
    :class:`relaxation_mf_classifier` to build a dedicated relaxation
    matched filter.

    Attributes:
        filtered_indices (tuple or None): 4-tuple
            ``(indices_0, one_indices, indices_1, indices_y)`` populated by
            :meth:`fit`.  Used by :meth:`predict` to extract the purified
            subset from raw data arrays.
        rscale (float): Radius multiplier passed to :func:`get_traces`.
        trace_classes (dict or None): Per-qubit trace-class dictionary
            returned by :func:`get_traces`.

    Args:
        radius_scale (float): Multiplier applied to the acceptance radius.
            Values > 1 include more traces (less strict filtering).
            Default: 1.
    """
    # Can pass train-test ratio and other parameters through constructor later.
    def __init__(self, radius_scale=1) -> None:
        self.filtered_indices = None
        self.rscale = radius_scale
        self.trace_classes = None

    def distance(self, x0, y0, x1, y1):
        return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    
    def fit(self):
        """Run the geometric trace-purification pipeline.

        Calls :func:`get_traces` (with the instance's ``rscale`` radius
        multiplier) to identify clean |0⟩ / |1⟩ traces and to catalogue error
        events for all 5 qubits.  Populates :attr:`filtered_indices` and
        :attr:`trace_classes`.
        """
        qubit_traces, indices_filtered = get_traces()
        self.filtered_indices = indices_filtered
        self.trace_classes = qubit_traces
        return

    def save_state(self, filename):
        """Persist the pre-classifier state to a pickle file.

        Args:
            filename (str): Destination file path (e.g. ``'preclassifier_state.pkl'``).
        """
        state = {
            'filtered_indices': self.filtered_indices,
            'trace_classes': self.trace_classes
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filename):
        """Restore pre-classifier state from a previously saved pickle file.

        Args:
            filename (str): Path to the pickle file created by :meth:`save_state`.
        """
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        self.filtered_indices = state['filtered_indices']
        self.trace_classes = state['trace_classes']

    def predict(self, data, num_qubits=5):
        """Extract purified traces from a raw data array using the stored indices.

        Uses the ``filtered_indices`` computed during :meth:`fit` to select
        only those shots that were classified as clean |0⟩ or |1⟩.

        Args:
            data (np.ndarray): Raw IQ trace array of shape
                ``(N_shots, trace_length, 2)``.
            num_qubits (int): Unused; reserved for future multi-qubit
                generalisation. Default: 5.

        Returns:
            tuple: ``(filtered_traces, labels)`` where
            ``filtered_traces`` has shape ``(N_filtered, trace_length, 2)``
            and ``labels`` is a 1-D integer array. Returns ``-1`` if *data*
            does not have 3 dimensions.
        """
        data = np.array(data)
        temp = None
        if len(data.shape) == 3:
            indices_0 = self.filtered_indices[0]
            indices_1 = self.filtered_indices[2]
            y = self.filtered_indices[3]
            one_indices = self.filtered_indices[1]
            temp = data[indices_0.astype(int), :, :]
            temp = np.vstack((temp, data[indices_1.astype(int), :, :]))
            return temp, y
        else:
            print('Please pass a dataset of the correct shape.')
            return -1
        return

    def get_traces(self):
        """Return the per-qubit trace-class dictionary.

        Returns:
            dict or None: The ``trace_classes`` dict populated by :meth:`fit`,
            keyed by 1-based qubit index.  ``None`` if :meth:`fit` has not
            been called yet.
        """
        return self.trace_classes

class relaxation_mf_classifier(preclassifier):
    """Matched-filter classifier specialised for detecting qubit relaxation events.

    Extends :class:`preclassifier` by computing a *relaxation matched filter*
    (RMF) for each qubit.  The RMF is trained to distinguish between *clean*
    |0⟩ traces and |1⟩→|0⟩ *relaxation* traces (where the qubit decays from
    the excited state during the readout window).

    This is complementary to the standard matched filter computed in
    ``matched_filter.py``:

    - **Standard MF** — discriminates |0⟩ vs |1⟩ at the end of readout.
    - **RMF** — detects the characteristic ringing / trajectory of a qubit
      that started in |1⟩ but relaxed to |0⟩ before the readout window closed.

    Both MF and RMF outputs are concatenated into the feature vector fed to the
    neural network classifier (:class:`Net_rmf`).

    Attributes:
        envelopes (list): Per-qubit RMF envelope arrays of shape
            ``(trace_length * 2,)`` populated by :meth:`fit`.
        thresholds (list): Per-qubit decision thresholds (99.5%-ile) populated
            by :meth:`fit`.
    """
    def __init__(self) -> None:
        super().__init__()
        self.envelopes = []
        self.thresholds = []
        pass

    def fit(self, trace_classes, num_qubits=5, boxcars=None):
        """Compute per-qubit relaxation matched-filter envelopes and thresholds.

        For each qubit the function collects the purified |0⟩ traces and the
        relaxation traces identified by :func:`get_traces`, balances the two
        sets by random sub-sampling, and computes the Wiener-optimal linear
        discriminant envelope:

        ``RMF = E[x_relax - x_0] / Var[x_relax - x_0]``

        An optional *boxcar* window is applied to zero-out the tail of the
        envelope beyond a per-qubit cut-off, reducing sensitivity to late-time
        noise.

        Args:
            trace_classes (dict): Per-qubit trace-class dictionary returned by
                :meth:`preclassifier.get_traces`.
            num_qubits (int): Number of qubits. Default: 5.
            boxcars (list or None): Per-qubit boxcar window widths (in units of
                50 ADC samples).  If ``None``, no boxcar is applied.
        """
        for qubit in range(1, 1 + num_qubits):
            # Distinguish between |0> and |1> -> |0> traces.
            relaxation_traces = trace_classes[qubit]['traces_relax']
            zero_traces = trace_classes[qubit]['traces_0']
            relaxation_traces = relaxation_traces.reshape(-1, relaxation_traces.shape[1] * relaxation_traces.shape[2])
            zero_traces = zero_traces.reshape(-1, zero_traces.shape[1] * zero_traces.shape[2])
            # zero traces generally more than relaxation traces. 
            # Do random sampling from the zero trace set
            mf = None
            if zero_traces.shape[0] > relaxation_traces.shape[0]:
                indices = np.random.choice(zero_traces.shape[0], relaxation_traces.shape[0], replace=False)
                mf = np.mean(relaxation_traces - zero_traces[indices], axis=0) / np.var(relaxation_traces - zero_traces[indices], axis=0)
            else:
                indices = np.random.choice(relaxation_traces.shape[0], zero_traces.shape[0], replace=False)
                mf = np.mean(relaxation_traces[indices] - zero_traces, axis=0) / np.var(relaxation_traces[indices] - zero_traces, axis=0)
            if boxcars is not None:
                boxcar = np.heaviside((len(mf) - boxcars[qubit - 1] / 50) - np.arange(len(mf)), 1)
                mf = mf * boxcar
            zero_filter = np.sum(np.multiply(mf, zero_traces), axis=1)
            zero_filter = np.sort(zero_filter)
            ind = int(0.995 * zero_filter.shape[0])
            threshold = zero_filter[ind]
            self.envelopes.append(mf)
            self.thresholds.append(threshold)
        return

    def save_state(self, filename):
        state = {
            'filtered_indices': self.filtered_indices,
            'trace_classes': self.trace_classes,
            'envelopes': self.envelopes,
            'thresholds': self.thresholds
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        self.filtered_indices = state['filtered_indices']
        self.trace_classes = state['trace_classes']
        self.envelopes = state['envelopes']
        self.thresholds = state['thresholds']

    def predict(self, num_qubits=5, data_type=0, trace_length=500):
        # Use demodulated data dumps
        result = []
        for qubit in range(1, num_qubits + 1):
            dem, y, t = get_data(qubit)
            traces, ys = get_train_val_and_test_set(np.array(dem), np.array(y), trace_length=trace_length)
            dem = traces[data_type]
            y = ys[data_type]
            del traces
            traces = dem
            states = np.unique(y)
            traces = traces.reshape(traces.shape[0], -1)

            full_envelope = self.envelopes[qubit - 1]
            current_trace_length = traces.shape[1]
            envelope_to_use = full_envelope[:current_trace_length]
            

            mf_results = np.empty((states.shape[0], int(traces.shape[0] / states.shape[0])))
            for state in states:
                traces_for_state = traces[y==state]
                mf_results[state] = np.sum(traces_for_state * envelope_to_use, axis=1)
            
            result.append(mf_results)
        result = np.array(result).transpose([1, 2, 0])
        return result

root_dir = './'

class ADCDataset(T.utils.data.Dataset):

    def __init__(self, data_file):
        all_data = np.load(data_file)
        num_samples_per_state = all_data.shape[1]
        num_basis_state = all_data.shape[0]

        all_labels = []
        for i in range(num_basis_state):
            all_labels.append(np.array([i for _ in range(num_samples_per_state)]))

        all_data = all_data.reshape((num_basis_state * num_samples_per_state, -1))
        all_data = all_data[:, :1000]  # only use the first 1000 samples (500 I's and 500 Q's)
        all_labels = np.array(all_labels).reshape((-1))

        self.x_data = T.tensor(all_data, dtype=T.float)
        self.y_data = T.tensor(all_labels, dtype=T.long)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()
        preds = self.x_data[idx]
        lbl = self.y_data[idx]
        sample = {'predictors': preds, 'target': lbl}

        return

def adjust_learning_rate(initial_lr, optimizer, epoch, lr_schedule=[30, 60, 90]):
    lr = initial_lr
    if epoch >= lr_schedule[0]:
        lr *= 0.1

    if epoch >= lr_schedule[1]:
        lr *= 0.1

    if epoch >= lr_schedule[2]:
        lr *= 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def inference(model, dl):
    model.eval()
    all_scores = []
    all_labels = []
    s = T.nn.Softmax(dim=-1)
    with T.no_grad():
        for (batch_idx, batch) in enumerate(dl):
            X = batch[0]
            Y = batch[1]
            oupt = model(X)
            oupt = s(oupt)

            all_scores.extend(oupt.cpu().numpy())
            all_labels.extend(Y.cpu().numpy())

    model.train()
    return np.array(all_scores), np.array(all_labels)

def accuracy(model, dl):
    all_preds, all_labels = inference(model, dl)
    # logger.debug(all_preds.shape)

    pred_indices = np.argmax(all_preds, axis=-1)
    cumulative_acc = np.sum(pred_indices == all_labels) / len(all_labels)

    acc_per_qubit = []
    for _ in range(5):
        pred_qubit = pred_indices % 2
        label_qubit = all_labels % 2
        acc_per_qubit.append(np.sum(pred_qubit == label_qubit) / len(label_qubit))
        pred_indices = pred_indices >> 1
        all_labels = all_labels >> 1

    return cumulative_acc, acc_per_qubit

def load_data_all(NUM_TRAIN_VAL=3000, NUM_TEST = 7000, NUM_VAL_RATIO = 0.35):
    """Load all traces from ``all_traces_10k.npy`` and split into train / val / test.

    Reads the pre-saved NumPy file ``all_traces_10k.npy`` (produced by
    :func:`data_load`) of shape ``(32, 10000, num_samples, 2)`` and shuffles
    + splits the shots for each basis state into three balanced subsets.

    The split arrays are written to disk under ``split_data/``:
    - ``split_data/train.npy`` — shape ``(32, NUM_TRAIN, num_samples, 2)``
    - ``split_data/val.npy``   — shape ``(32, NUM_VAL,   num_samples, 2)``
    - ``split_data/test.npy``  — shape ``(32, NUM_TEST,  num_samples, 2)``

    Args:
        NUM_TRAIN_VAL (int): Combined train+val samples per class. Default: 3000.
        NUM_TEST (int): Test samples per class. Default: 7000.
        NUM_VAL_RATIO (float): Fraction of *NUM_TRAIN_VAL* reserved for
            validation. Default: 0.35.

    Returns:
        tuple: ``(train_set, val_set, test_set)`` NumPy arrays.
    """
    NUM_VAL = int(NUM_TRAIN_VAL * NUM_VAL_RATIO)
    NUM_TRAIN = NUM_TRAIN_VAL - NUM_VAL
    TOTAL_TRACES = NUM_TRAIN_VAL + NUM_TEST

    print('length of training set: {}'.format(NUM_TRAIN))
    print('length of val set: {}'.format(NUM_VAL))
    print('length of test set: {}'.format(NUM_TEST))
    
    train_set = []
    val_set = []
    test_set = []

    data = np.load('all_traces_10k.npy')
    print(data.shape)
    
    for basis_state_data in data:
        random.shuffle(basis_state_data)
        train_set.append(basis_state_data[:NUM_TRAIN])
        val_set.append(basis_state_data[NUM_TRAIN:NUM_TRAIN_VAL])
        test_set.append(basis_state_data[NUM_TRAIN_VAL:NUM_TRAIN_VAL+NUM_TEST])
        

    train_set = np.array(train_set)
    print('Train set: ', train_set.shape)
    val_set = np.array(val_set)
    print('Val set: ', val_set.shape)
    test_set = np.array(test_set)
    print('Test set: ', test_set.shape)


    os.makedirs('split_data', exist_ok=True)
    np.save('split_data/train.npy', train_set)
    np.save('split_data/val.npy', val_set)
    np.save('split_data/test.npy', test_set)
    
    return train_set, val_set, test_set

def data_load(num_Q=5, rep_seq=False):

    ## structure: [QB5, QB4, QB3, QB2, QB1]
    data = all_data
    print(data.shape)
    num_records 	= int(1e4) #number of repeated acquisitions

    print('loading Ch1')
    Ch1 		= data[:2 ** num_Q * num_records, :, 0]  # shape (1600000, 1473) = (32 * num_records, num_samples)
    print('loading Ch2')
    Ch2 		= data[: 2 ** num_Q * num_records, :, 1]  # shape (1600000, 1473) = (32 * num_records, num_samples)

    ## statistics
    num_samples_raw = Ch1.shape[1] #number of samples per acquisition
    print(Ch1.shape)
    print('num_samples_raw: ', num_samples_raw)

    ## data preparation
    DataRaw = np.zeros((2**num_Q, num_records, num_samples_raw, 2))
    print(DataRaw.shape)
    DataRaw[:, :, :, 0] = np.reshape(Ch1, (2 ** num_Q, num_records, num_samples_raw))
    DataRaw[:, :, :, 1] = np.reshape(Ch2, (2 ** num_Q, num_records, num_samples_raw))
    print('DataRaw shape: ', DataRaw.shape)
    
    np.save('all_traces_10k.npy', DataRaw)
    #data.close()
    del Ch1, Ch2
    return

class Net_baseline(T.nn.Module):
    """Baseline fully-connected neural network for raw-trace 5-qubit classification.

    A simple 3-layer MLP that takes the first 1000 raw IQ samples (500 I values
    concatenated with 500 Q values) as its input and maps them to logits over
    the 32 possible 5-qubit states.

    Architecture::

        Input (1000) → Linear(1000→500) → ReLU
                     → Linear(500→250) → ReLU
                     → Linear(250−32)

    Weights are initialised with Xavier uniform and biases are zeroed.
    This model serves as the upper-bound reference point; it requires no
    feature engineering but is impractical for FPGA deployment due to its
    1 000-dimensional input.
    """
    def __init__(self):
        super(Net_baseline, self).__init__()
        self.hid1 = T.nn.Linear(1000, 500)
        self.hid2 = T.nn.Linear(500, 250)
        self.oupt = T.nn.Linear(250, 32)

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)
        self.relu = T.nn.ReLU()

    def forward(self, x):
        z = self.hid1(x)
        z = self.relu(z)
        z = self.hid2(z)
        z = self.relu(z)
        z = self.oupt(z)
        return z

def train_baseline():
    """Train the raw-trace baseline MLP (:class:`Net_baseline`).

    Serves as an upper-bound accuracy reference for the HERQULES pipeline by
    training directly on the full raw IQ traces (no feature extraction).

    Pipeline:
        1. Load and split data using :class:`helpers.data_utils.QubitTraceDataset`
           and ``train_test_split``.
        2. Train ``Net_baseline`` (1000→500→250−32) for up to 100 epochs with
           Adam (lr = 1e-4) and Cross-Entropy loss.
        3. Apply a step-decay learning-rate schedule at epochs 30, 60, 90
           (each step reduces lr by 10×).
        4. Save the best checkpoint (by validation accuracy) to
           ``checkpoints/1000_points/best_epoch.pth``.
        5. Evaluate and print test accuracy (overall + per-qubit).

    Note:
        This function is useful for benchmarking; it is **not** the primary HERQULES
        entry point (:func:`train` is the full pipeline).
    """
    # 0. get started
    T.manual_seed(1)
    np.random.seed(1)

    # 1. create Dataset and DataLoader objects

    #train_ds = ADCDataset(os.path.join(root_dir, 'split_data/train.npy'))
    #val_ds = ADCDataset(os.path.join(root_dir, 'split_data/val.npy'))
    #test_ds = ADCDataset(os.path.join(root_dir, 'split_data/test.npy'))

    BATCH_SIZE = 512
    #train_ldr = T.utils.data.DataLoader(train_ds, batch_size=bat_size, shuffle=True)
    #val_ldr = T.utils.data.DataLoader(val_ds, batch_size=bat_size, shuffle=False)
    #test_ldr = T.utils.data.DataLoader(test_ds, batch_size=bat_size, shuffle=False)

    X_train, X_test, y_train, y_test = train_test_split(
        all_data, 
        y_data, 
        test_size=0.3, 
        random_state=42, # For reproducibility
        stratify=y_data    # Ensures same class balance in train/test
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, 
        y_train, 
        test_size=0.35, 
        random_state=42, # For reproducibility
        stratify=y_train    # Ensures same class balance in train/test
    )

    train_dataset = QubitTraceDataset(X_train, y_train, True)
    val_dataset = QubitTraceDataset(X_val, y_val)
    test_dataset = QubitTraceDataset(X_test, y_test)

    train_ldr = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_ldr = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_ldr = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. create neural network
    # Creating 1000-500-250-32 binary NN classifier ")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net_baseline() #.to(device)

    # 3. train network
    net = net.train()  # set training mode
    lrn_rate = 0.0001
    loss_obj = T.nn.CrossEntropyLoss()  # cross entropy
    optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate)
    max_epochs = 100
    ep_log_interval = 1


    best_acc = -1

    path = os.path.join(".", "checkpoints/1000_points")
    os.makedirs(path, exist_ok=True)

    for epoch in range(0, max_epochs):
        epoch_loss = 0.0  # for one full epoch

        lr = adjust_learning_rate(lrn_rate, optimizer, epoch)

        for (batch_idx, batch) in enumerate(train_ldr):
            X = batch['predictors']
            Y = batch['target']
            oupt = net(X)

            loss_val = loss_obj(oupt, Y)
            epoch_loss += loss_val.item()

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
        if epoch % ep_log_interval == 0:
            print("epoch = %4d   loss = %0.4f  lr = %0.4f" % \
                  (epoch, epoch_loss, lr))

        acc_train, acc_train_per_qubit = accuracy(net, train_ldr)
        acc_val, acc_val_per_qubit = accuracy(net, val_ldr)
        #print(acc_val_per_qubit)

        if acc_val >= best_acc:
            best_acc = acc_val
            T.save(net.state_dict(), os.path.join(path, 'best_epoch.pth'))

    print("Finished  training")
    print("Best acc on val dataset: {}".format(best_acc))

    net.load_state_dict(T.load(os.path.join(path, 'best_epoch.pth')))
    #net.to(device)
    net.eval()
    acc_test, acc_test_per_qubit = accuracy(net, test_ldr)
    print("Acc on test dataset: {}".format(acc_test))
    print("Acc per qubit: {}".format(acc_test_per_qubit))


from matched_filter import search_matched_filter_for_all_qubits, matched_filter_preprocess, search_matched_filter_for_all_qubits_preclass, matched_filter_preprocess_demux, search_matched_filter_for_all_qubits_demux

train_semi_sup_data = all_data

os.makedirs('accuracy', exist_ok=True)
os.makedirs('stats', exist_ok=True)
os.makedirs('logs', exist_ok=True)

#root_dir = '/nobackup/readout_data/new'
root_dir = '.'
# root_dir ='.'

def mf_demux_data_prep(trace_length=500):
    """Prepare per-qubit demultiplexed data for matched-filter computation.

    Loads the demodulated IQ traces for all 5 qubits (from the HDF5 files
    produced by :func:`demodulate_multiplexed_traces`), splits them into
    balanced train / val / test sets via :func:`get_train_val_and_test_set`,
    and reshapes each split into the per-qubit structure expected by
    :func:`matched_filter.search_matched_filter_for_all_qubits_demux` and
    :func:`matched_filter.matched_filter_preprocess_demux`.

    The output structure is a length-5 list (one entry per qubit) where each
    entry is an array of shape ``(32, num_samples_per_state, trace_length, 2)``
    corresponding to all 32 basis states.

    Args:
        trace_length (int): Number of ADC samples to retain per trace.
            Default: 500.

    Returns:
        tuple: ``(data_train, data_val, data_test)`` where each element is a
        length-5 list; ``data_split[q]`` has shape
        ``(32, num_samples_per_state, trace_length, 2)`` for qubit *q*.
    """
    # Prepare data
    data_train = []
    data_val = []
    data_test = []
    for qubit in range(1, 6):
        traces, y, t = get_data(qubit)
        #print(traces.shape)
        traces, y = get_train_val_and_test_set(np.array(traces), np.array(y), trace_length=trace_length)
        data = traces[0].reshape(32, int(traces[0].shape[0] / 32), traces[0].shape[1], traces[0].shape[2])
        data_train.append(data)
        data = traces[1].reshape(32, int(traces[1].shape[0] / 32), traces[1].shape[1], traces[1].shape[2])
        data_val.append(data)
        data = traces[2].reshape(32, int(traces[2].shape[0] / 32), traces[2].shape[1], traces[2].shape[2])
        data_test.append(data)
    return data_train, data_val, data_test

class MFOutputDataset(T.utils.data.Dataset):
    """PyTorch Dataset wrapping matched-filter (or combined MF + RMF) output arrays.

    Takes a NumPy array of shape ``(num_basis_states, num_samples_per_state, num_features)``
    — the output of :func:`matched_filter.matched_filter_preprocess_demux` or a
    concatenation of that with the RMF outputs — and exposes it as a
    ``torch.utils.data.Dataset`` with integer labels derived from the state index.

    Each sample is returned as a dict:

    .. code-block:: python

        {'predictors': Tensor(num_features,), 'target': Tensor(scalar, dtype=long)}

    Args:
        all_data (np.ndarray): Shape ``(num_basis_states, num_samples_per_state, num_features)``.
            Labels are inferred from the first dimension (state index).
    """

    def __init__(self, all_data):
        num_samples_per_state = all_data.shape[1]
        num_basis_state = all_data.shape[0]

        all_labels = []
        for i in range(num_basis_state):
            all_labels.append(np.array([i for _ in range(num_samples_per_state)]))

        all_data = all_data.reshape((num_basis_state * num_samples_per_state, -1))
        all_labels = np.array(all_labels).reshape((-1))

        self.x_data = T.tensor(all_data, dtype=T.float)
        self.y_data = T.tensor(all_labels, dtype=T.long)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()
        preds = self.x_data[idx]
        lbl = self.y_data[idx]
        sample = {'predictors': preds, 'target': lbl}

        return sample


def adjust_learning_rate(initial_lr, optimizer, epoch, lr_schedule=[30, 60, 90]):
    """Apply a step-decay learning-rate schedule in-place on the optimizer.

    At each milestone epoch in *lr_schedule* the current learning rate is
    multiplied by 0.1 (one decade step-down).  This is applied cumulatively,
    so reaching the third milestone divides the initial learning rate by 1000.

    Args:
        initial_lr (float): Starting learning rate used to recompute the
            current lr from scratch on every call.
        optimizer (torch.optim.Optimizer): PyTorch optimiser whose param-group
            ``'lr'`` values will be updated.
        epoch (int): Current training epoch (0-indexed).
        lr_schedule (list[int]): Epoch milestones at which to apply a 10×
            reduction. Default: ``[30, 60, 90]``.

    Returns:
        float: The learning rate value applied to the optimizer for this epoch.
    """
    lr = initial_lr
    if epoch >= lr_schedule[0]:
        lr *= 0.1

    if epoch >= lr_schedule[1]:
        lr *= 0.1

    if epoch >= lr_schedule[2]:
        lr *= 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def inference(model, dl):
    model.eval()
    all_scores = []
    all_labels = []
    s = T.nn.Softmax(dim=-1)
    with T.no_grad():
        for (batch_idx, batch) in enumerate(dl):
            X = batch['predictors']
            Y = batch['target']
            oupt = model(X)
            oupt = s(oupt)

            all_scores.extend(oupt.cpu().numpy())
            all_labels.extend(Y.cpu().numpy())

    model.train()
    return np.array(all_scores), np.array(all_labels)


def accuracy(model, dl):
    all_preds, all_labels = inference(model, dl)
    pred_indices = np.argmax(all_preds, axis=-1)
    cumulative_acc = np.sum(pred_indices == all_labels) / len(all_labels)
    data = {'preds':pred_indices, 'labels':all_labels}
    with open('mf_rmf_nn_train.pkl', 'wb') as file:
        pickle.dump(data, file)
    acc_per_qubit = []
    for i in range(5):
        pred_qubit = pred_indices % 2
        label_qubit = all_labels % 2
        acc_per_qubit.append(np.sum(pred_qubit == label_qubit) / len(label_qubit))
        pred_indices = pred_indices >> 1
        all_labels = all_labels >> 1

    return cumulative_acc, acc_per_qubit


class Net(T.nn.Module):
    """Compact MLP classifier that operates on standard matched-filter (MF) outputs.

    Takes a 5-dimensional input vector (one MF scalar per qubit) and produces
    logits over the 32 possible 5-qubit basis states.  This architecture is used
    when only the standard MF features are available (i.e. when the RMF is
    disabled via ``run_rmf=False`` in :func:`train`).

    Architecture::

        Input (5) → Linear(5→10) → ReLU
                  → Linear(10→20) → ReLU
                  → Linear(20−32)

    Weights are initialised with Xavier uniform and biases with zeros.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.hid1 = T.nn.Linear(5, 10)
        self.hid2 = T.nn.Linear(10, 20)
        self.oupt = T.nn.Linear(20, 32)

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)
        self.relu = T.nn.ReLU()

    def forward(self, x):
        z = self.hid1(x)
        z = self.relu(z)
        z = self.hid2(z)
        z = self.relu(z)
        z = self.oupt(z)
        return z

class Net_rmf(T.nn.Module):
    """Primary HERQULES neural-network classifier (MF + RMF inputs).

    Accepts a 10-dimensional feature vector (5 standard MF scalars + 5 RMF
    scalars, one per qubit each) and maps them to logits over the 32 possible
    5-qubit states.  This is the main production classifier in the HERQULES
    pipeline when ``run_rmf=True`` in :func:`train`.

    Architecture::

        Input (10) → Linear(10→10) → ReLU
                   → Linear(10→20) → ReLU
                   → Linear(20−32)

    The compact 10→ 10→20→32 structure is highly FPGA-friendly: the entire
    forward pass involves only a few hundred multiply-accumulates.

    Weights are initialised with Xavier uniform and biases with zeros.
    """
    def __init__(self):
        super(Net_rmf, self).__init__()
        self.hid1 = T.nn.Linear(10, 10)
        self.hid2 = T.nn.Linear(10, 20)
        self.oupt = T.nn.Linear(20, 32)

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)
        self.relu = T.nn.ReLU()

    def forward(self, x):
        z = self.hid1(x)
        z = self.relu(z)
        z = self.hid2(z)
        z = self.relu(z)
        z = self.oupt(z)
        return z


# ----------------------------------------------------------

def train(run_pre_filter=False,
          run_semi_sup=True,
          run_rmf = True,
          train_data_ratio = 1, dur=1000):
    """Full HERQULES training pipeline.

    Orchestrates the end-to-end HERQULES workflow:

    1. **Pre-classification** (always): Runs :class:`preclassifier` to obtain
       clean |0⟩ / |1⟩ reference trace sets and saves state to
       ``preclassifier_state.pkl``.
    2. **Feature extraction** (conditional):

       - *run_pre_filter=True*: Compute standard MF envelopes from
         pre-filtered (geometrically purified) traces.
       - *run_semi_sup=True*: Compute MF envelopes from traces filtered by the
         semi-supervised preclassifier (default).
       - Else: Compute MF envelopes directly from the demodulated traces
         (``search_matched_filter_for_all_qubits_demux``).

    3. **RMF computation** (if *run_rmf=True*): Fits :class:`relaxation_mf_classifier`,
       saves state to ``rmf.pkl``, and applies the RMF to train / val / test
       splits.
    4. **Dataset assembly**: Builds :class:`MFOutputDataset` objects from the
       MF and/or RMF scalar features.  If both MF and RMF are used, their
       outputs are concatenated along the feature axis, yielding a
       5-dimensional (MF only) or 10-dimensional (MF + RMF) feature vector.
    5. **Neural network training**: Trains :class:`Net` or :class:`Net_rmf`
       for 100 epochs with Adam (lr = 0.01) and Cross-Entropy loss.  A
       step-decay LR schedule (epochs 30 / 60 / 90) is applied.  PyTorch
       profiler traces are written to ``logs/tb_nn`` for TensorBoard.
    6. **Evaluation**: Loads the best checkpoint (by validation accuracy) and
       reports overall test accuracy + per-qubit accuracy.

    Args:
        run_pre_filter (bool): Use geometrically pre-filtered traces for MF
            computation.  Default: ``False``.
        run_semi_sup (bool): Use the semi-supervised preclassifier output for
            MF computation.  Default: ``True``.
        run_rmf (bool): Include the relaxation matched filter (RMF) features.
            Default: ``True``.
        train_data_ratio (float): Fraction of the training data to use
            (reserved for future ablation experiments). Default: 1.
        dur (int): Readout window duration hint in ADC samples, used to
            compute the fast-readout offset ``fast_readout = 20 - dur/50``.
            Default: 1000.

    Returns:
        list[float]: Per-qubit accuracy on the test set (5 values).

    Side Effects:
        - Writes ``preclassifier_state.pkl`` and ``rmf.pkl``.
        - Writes ``checkpoints/mf_nn/best_epoch.pth``.
        - Creates ``accuracy/``, ``stats/``, ``logs/`` directories.
        - Writes TensorBoard profiler traces to ``logs/tb_nn``.
    """
    # 0. get started
    T.manual_seed(1)
    np.random.seed(1)

    # 1. create Dataset and DataLoader objects
    
    semi_sup_classifier = preclassifier()
    
    rmf = relaxation_mf_classifier()
    
    train_rmf_data = None
    val_rmf_data = None
    test_rmf_data = None
    semi_sup_classifier.fit()
    semi_sup_classifier.save_state('preclassifier_state.pkl')      
    if run_semi_sup:
        filtered_train_data, filtered_labels = semi_sup_classifier.predict(train_semi_sup_data)

    fast_readout=20 - int(dur / 50)
    boxcars = [1, 1, 9, 2, 9]
    print('Boxcars: ', boxcars)
    trace_classes = None
    if run_rmf:
        ##Get relaxtional matched filter data
        trace_classes = semi_sup_classifier.get_traces()
        rmf.fit(trace_classes, boxcars=boxcars)
        rmf.save_state('rmf.pkl')  
        #data_type => 0 for train, 1 for val, 2 for test
        train_rmf_data = rmf.predict(data_type=0)
        val_rmf_data = rmf.predict(data_type=1)
        test_rmf_data = rmf.predict(data_type=2)

          
    best_bc = boxcars
    #best_bc=[0, 0, 0, 0, 0]
    demux_data_train, demux_data_val, demux_data_test = mf_demux_data_prep()
    print(np.array(demux_data_train).shape)

    if run_pre_filter:
        mf_envelopes, _ = search_matched_filter_for_all_qubits(filtered_train_data, best_bc=best_bc)
    elif run_semi_sup:
        mf_envelopes, _ = search_matched_filter_for_all_qubits_preclass(filtered_train_data, filtered_labels, best_bc=best_bc)
    else:
        #mf_envelopes, _ = search_matched_filter_for_all_qubits(train_data, best_bc=best_bc)
        mf_envelopes, _ = search_matched_filter_for_all_qubits_demux(demux_data_train, best_bc=best_bc)

    no_mf = False

    if no_mf and run_rmf:
        train_ds = MFOutputDataset(train_rmf_data)
        val_ds = MFOutputDataset(val_rmf_data)
        test_ds = MFOutputDataset(test_rmf_data)
    elif run_rmf:
        train_ds =  MFOutputDataset(np.concatenate((matched_filter_preprocess_demux(demux_data_train, mf_envelopes), train_rmf_data), axis=2))
        val_ds = MFOutputDataset(np.concatenate((matched_filter_preprocess_demux(demux_data_val, mf_envelopes), val_rmf_data), axis=2))
        test_ds = MFOutputDataset(np.concatenate((matched_filter_preprocess_demux(demux_data_test, mf_envelopes), test_rmf_data), axis=2))
    else:
        train_ds = MFOutputDataset(matched_filter_preprocess_demux(demux_data_train, mf_envelopes))
        val_ds = MFOutputDataset(matched_filter_preprocess_demux(demux_data_val, mf_envelopes))
        test_ds = MFOutputDataset(matched_filter_preprocess_demux(demux_data_test, mf_envelopes))


    bat_size = 512
    train_ldr = T.utils.data.DataLoader(train_ds, batch_size=bat_size, shuffle=True)
    val_ldr = T.utils.data.DataLoader(val_ds, batch_size=bat_size, shuffle=False)
    test_ldr = T.utils.data.DataLoader(test_ds, batch_size=bat_size, shuffle=False)

    # 2. create neural network
    # Creating 10-50-25-32 binary NN classifier
    if run_rmf and no_mf == False:
        net = Net_rmf()  # .to(device)
    else:
        net = Net()

    # 3. train network
    net = net.train()  # set training mode
    lrn_rate = 0.01
    loss_obj = T.nn.CrossEntropyLoss()  # cross entropy
    optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate)
    max_epochs = 100
    ep_log_interval = 1



    best_acc = -1
    acc_train_itr, acc_val_itr = [], []
    path = os.path.join(".", "checkpoints/mf_nn")
    os.makedirs(path, exist_ok=True)
    with T.profiler.profile(
        schedule=T.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=T.profiler.tensorboard_trace_handler('./logs/tb_nn'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:
        for epoch in range(0, max_epochs):
            epoch_loss = 0.0  # for one full epoch

            lr = adjust_learning_rate(lrn_rate, optimizer, epoch)

            for (batch_idx, batch) in enumerate(train_ldr):
                X = batch['predictors']
                Y = batch['target']
                # print(X)
                oupt = net(X)

                loss_val = loss_obj(oupt, Y)
                epoch_loss += loss_val.item()

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                prof.step()
            if epoch % ep_log_interval == 0:
                print("epoch = %4d   loss = %0.4f  lr = %0.4f" % \
                    (epoch, epoch_loss, lr))

            acc_train, acc_train_per_qubit = accuracy(net, train_ldr)
            acc_train_itr.append(acc_train_per_qubit)
            acc_val, acc_val_per_qubit = accuracy(net, val_ldr)
            acc_train_itr.append(acc_val_per_qubit)

            if acc_val >= best_acc:
                best_acc = acc_val
                T.save(net.state_dict(), os.path.join(path, 'best_epoch.pth'))

    print("Finished  training")
    print("Best acc on val dataset: {}".format(best_acc))

    net.load_state_dict(T.load(os.path.join(path, 'best_epoch.pth')))
    acc_test, acc_test_per_qubit = accuracy(net, test_ldr)
    print("Acc on test dataset: {}".format(acc_test))
    print("Acc per qubit: {}".format(acc_test_per_qubit))
    #print("Acc per qubit: {}".format(acc_test_per_qubit))
    return acc_test_per_qubit
    

#train(run_semi_sup=False, run_pre_filter=False, run_rmf=True)


def test():
    """Evaluate the trained HERQULES classifier (:class:`Net_rmf`) on the test set.

    Loads the best checkpoint from ``checkpoints/mf_nn/best_epoch.pth``, then
    performs the same feature extraction pipeline used during training:

    1. Prepare demodulated per-qubit data using :func:`mf_demux_data_prep`.
    2. Compute MF envelopes from the test split (with the same boxcar
       widths as training: ``[1, 1, 9, 2, 9]``).
    3. Apply the envelopes via :func:`matched_filter.matched_filter_preprocess_demux`.
    4. Load the saved RMF state from ``rmf.pkl`` and apply it to the test
       split via :meth:`relaxation_mf_classifier.predict`.
    5. Concatenate MF and RMF outputs and feed them into the loaded network.

    Results are printed to stdout (overall + per-qubit accuracy) and also
    returned.

    Returns:
        tuple: ``(overall_acc, acc_per_qubit)`` where

        - **overall_acc** (float): Fraction of test shots for which all
          5 qubit states are correctly predicted simultaneously.
        - **acc_per_qubit** (list[float]): Per-qubit accuracy (5 values).
    """
    model = Net_rmf()
    model.load_state_dict(T.load('checkpoints/mf_nn/best_epoch.pth'))

    data_train, data_val, data_test = mf_demux_data_prep(trace_length=500)
    for i in data_test:
        print(i.shape)

    best_bc = [1, 1, 9, 2, 9]
    mf_envelopes, _ = search_matched_filter_for_all_qubits_demux(data_test, best_bc=best_bc)

    processed_test = matched_filter_preprocess_demux(data_test, mf_envelopes)
    print("Processed test shape after MF:", processed_test.shape)

    rmf = relaxation_mf_classifier()
    rmf.load_state('rmf.pkl')
    #semi_sup_classifier = preclassifier()
    #semi_sup_classifier.load_state('preclassifier_state.pkl')

    #trace_classes = semi_sup_classifier.get_traces()

    test_rmf_data = rmf.predict(data_type=2, trace_length=500)
    print("Finished prediction")

    processed_test = np.concatenate((processed_test, test_rmf_data), axis=2)
    
    test_ds = MFOutputDataset(processed_test)
    test_loader = T.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False)

    all_preds = []
    all_labels = []
    softmax = T.nn.Softmax(dim=-1)

    print("Loaded data")
    
    with T.no_grad():
        for batch in test_loader:
            X = batch['predictors']
            Y = batch['target']
            outputs = model(X)
            outputs = softmax(outputs)
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(Y.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_indices = np.argmax(all_preds, axis=-1)
    
    # Overall accuracy
    overall_acc = np.sum(pred_indices == all_labels) / len(all_labels)
    
    # Per-qubit accuracy
    acc_per_qubit = []
    pred_indices_copy = pred_indices.copy()
    labels_copy = all_labels.copy()
    
    for i in range(5):
        pred_qubit = pred_indices_copy % 2
        label_qubit = labels_copy % 2
        acc_per_qubit.append(np.sum(pred_qubit == label_qubit) / len(label_qubit))
        pred_indices_copy = pred_indices_copy >> 1
        labels_copy = labels_copy >> 1
    
    print(f"Overall Test Accuracy: {overall_acc:.4f}")
    print("Accuracy per qubit:")
    for i, acc in enumerate(acc_per_qubit):
        print(f"Qubit {i+1}: {acc:.4f}")
        
    return overall_acc, acc_per_qubit

test()

