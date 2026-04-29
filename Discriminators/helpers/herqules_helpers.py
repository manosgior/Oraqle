"""
herqules_helpers.py
===================
Helper functions for the HERQULES matched-filter pipeline, including:
  - Trace purification and pre-classification
  - Matched filter computation
  - Data preprocessing and splitting
"""

import numpy as np
import h5py
from scipy.signal import butter, sosfilt
from helpers.data_loader import custom_hdf5_data_loader


# ============================================================================
# Data loading and splitting
# ============================================================================

def get_train_val_and_test_set(
    trace: np.ndarray,
    y: np.ndarray,
    num_qubits: int = 5,
    NUM_TRAIN_VAL: int = 3000,
    NUM_TEST: int = 7000,
    NUM_VAL_RATIO: float = 0.35,
    trace_length: int = 500
):
    """Split the full dataset into balanced train / validation / test subsets.

    Each of the 2**num_qubits basis states contributes exactly *NUM_TRAIN*
    samples to training, *NUM_VAL* to validation, and *NUM_TEST* to testing.
    Within each state class, samples are randomly shuffled before splitting.

    The function is deterministic conditional on the dataset and does not use
    a global random seed (each call uses independent shuffles); set
    ``np.random.seed()`` before calling if reproducibility is needed.

    Args:
        trace (np.ndarray): IQ trace array of shape
            ``(N_total, trace_length, 2)`` where axis-2 is [I, Q].
        y (np.ndarray): State labels of shape ``(N_total,)``, with entries in
            ``{0, 1, ..., 2**num_qubits - 1}``.
        num_qubits (int): Number of qubits (determines number of basis states
            as 2^num_qubits). Default: 5.
        NUM_TRAIN_VAL (int): Total samples per state for train + val combined.
            Default: 3000.
        NUM_TEST (int): Samples per state reserved for testing. Default: 7000.
        NUM_VAL_RATIO (float): Fraction of *NUM_TRAIN_VAL* used for validation
            (0 < ratio < 1). Default: 0.35.
        trace_length (int): Expected trace length in samples. Used for shape
            consistency checks. Default: 500.

    Returns:
        tuple: ``(traces_split, labels_split)`` where

        - **traces_split** is a 3-tuple of arrays:
          ``(train_traces, val_traces, test_traces)`` each with shape
          ``(num_basis_states × num_samples_per_state, trace_length, 2)``.
        - **labels_split** is a 3-tuple of label arrays:
          ``(train_labels, val_labels, test_labels)`` each with shape
          ``(num_basis_states × num_samples_per_state,)``.
    """
    NUM_VAL = int(NUM_TRAIN_VAL * NUM_VAL_RATIO)
    NUM_TRAIN = NUM_TRAIN_VAL - NUM_VAL
    num_states = 2 ** num_qubits

    train_traces = []
    val_traces = []
    test_traces = []
    train_labels = []
    val_labels = []
    test_labels = []

    for state in range(num_states):
        state_mask = y == state
        state_traces = trace[state_mask]
        state_indices = np.where(state_mask)[0]

        # Randomly shuffle within this state
        perm = np.random.permutation(len(state_indices))
        state_traces = state_traces[perm]
        state_indices = state_indices[perm]

        # Split this state into train / val / test
        train_traces.append(state_traces[:NUM_TRAIN])
        val_traces.append(state_traces[NUM_TRAIN:NUM_TRAIN_VAL])
        test_traces.append(state_traces[NUM_TRAIN_VAL:NUM_TRAIN_VAL + NUM_TEST])

        train_labels.append(np.full(NUM_TRAIN, state))
        val_labels.append(np.full(NUM_VAL, state))
        test_labels.append(np.full(NUM_TEST, state))

    train_traces = np.concatenate(train_traces, axis=0)
    val_traces = np.concatenate(val_traces, axis=0)
    test_traces = np.concatenate(test_traces, axis=0)

    train_labels = np.concatenate(train_labels)
    val_labels = np.concatenate(val_labels)
    test_labels = np.concatenate(test_labels)

    traces_split = (train_traces, val_traces, test_traces)
    labels_split = (train_labels, val_labels, test_labels)

    return traces_split, labels_split


def get_data(qubit: int):
    """Load demodulated IQ traces for a single qubit from disk.

    Reads the per-qubit demodulated trace file (assumed to be saved as HDF5
    by :func:`demodulate_multiplexed_traces`).

    Args:
        qubit (int): Qubit index (1-based, i.e., 1, 2, 3, 4, or 5).

    Returns:
        tuple: ``(traces, labels, time_vector)`` where

        - **traces** (np.ndarray): Demodulated IQ traces, shape
          ``(N_shots, trace_length, 2)``.
        - **labels** (np.ndarray): Measurement outcome per shot (integer in
          {0, ..., 31} for 5-qubit system).
        - **time_vector** (np.ndarray): Time points (not populated; reserved
          for future use).
    """
    # Load the demodulated traces for this qubit
    filename = f"demodulated_q{qubit}.h5"
    with h5py.File(filename, "r") as hf:
        dem = np.array(hf["traces"])

    # Extract the labels from the combined measurement outcome
    y = np.zeros(dem.shape[0], dtype=int)
    # For now, return dummy labels; in practice these come from the measurement records
    # This function assumes the labels are stored in the HDF5 file or available separately

    t = np.array([])

    return dem, y, t


# ============================================================================
# Demodulation
# ============================================================================

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


# ============================================================================
# Trace characterization and purification
# ============================================================================

def distance(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Euclidean distance between two points (or arrays of points) in 2-D IQ space.

    Args:
        x0, y0: I and Q coordinates of the first point (scalars or arrays).
        x1, y1: I and Q coordinates of the second point (scalars or arrays).

    Returns:
        float or np.ndarray: ``sqrt((x0−x1)² + (y0−y1)²)``.
    """
    return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)


def get_traces(num_qubits: int = 5, plot: bool = False, rscale: float = 1, data_type: int = 0):
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
    with a set of filtered global indices suitable for pre-classification.

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
          dicts with entries (see below).
        - **filtered_indices** (tuple): ``(indices_0, one_indices, indices_1, indices_y)``
          used for pre-classifier predictions.

    **qubit_traces** dictionary keys:
        - ``'gnd_0'`` : Mean trace for |0⟩ state
        - ``'gnd_1'`` : Mean trace for |1⟩ state
        - ``'relax'`` : Mean relaxation trace (|1⟩→|0⟩)
        - ``'ket2'`` : Array of |2⟩ leakage traces
        - ``'excite'`` : Mean excitation trace
        - ``'mean_0'`` : Raw IQ centroid for |0⟩
        - ``'mean_1'`` : Raw IQ centroid for |1⟩
        - ``'mean_0_filtered'`` : Purified IQ centroid for |0⟩
        - ``'mean_1_filtered'`` : Purified IQ centroid for |1⟩
        - ``'traces_relax'`` : Array of relaxation traces
        - ``'traces_0'`` : Purified |0⟩ traces
        - ``'traces_1'`` : Purified |1⟩ traces
        - ``'traces_excite'`` : Excitation traces
    """
    qubit_traces = {}
    indices_0 = np.arange(10000)
    one_indices = {2 ** (qubit - 1): None for qubit in range(1, num_qubits + 1)}

    for qubit in range(1, num_qubits + 1):
        dem, y, t = get_data(qubit)
        print(dem.shape)

        traces, ys = get_train_val_and_test_set(dem, y)
        dem = traces[data_type]
        y = ys[data_type]
        del traces
        traces = dem

        # Compute the centroid of the clouds corresponding to 0 and 1
        zero = 0
        one = 2 ** (qubit - 1)
        i0 = np.mean(traces[np.where(y == zero)[0], :, 0], axis=1)
        q0 = np.mean(traces[np.where(y == zero)[0], :, 1], axis=1)
        i1 = np.mean(traces[np.where(y == one)[0], :, 0], axis=1)
        q1 = np.mean(traces[np.where(y == one)[0], :, 1], axis=1)
        i = np.mean(traces[:, :, 0], axis=1)
        q = np.mean(traces[:, :, 1], axis=1)
        x0 = np.mean(i0)
        y0 = np.mean(q0)
        x1 = np.mean(i1)
        y1 = np.mean(q1)
        midx = (x0 + x1) / 2
        midy = (y0 + y1) / 2
        radius = rscale * distance(x0, y0, x1, y1) / 2

        traces_0 = traces[np.where(y == zero)[0], :, :]
        traces_1 = traces[np.where(y == one)[0], :, :]
        traces_0 = traces_0[np.where(distance(i0, q0, x0, y0) < radius)[0], :, :]
        traces_1 = traces_1[np.where(distance(i1, q1, x1, y1) < radius)[0], :, :]
        indices_0 = np.intersect1d(indices_0, np.where((y == zero) & (distance(i, q, x0, y0) < radius))[0])
        indices_1 = np.where((y == one) & (distance(i, q, x1, y1) < radius))[0]

        # Find the new centroids with the purified traces
        x0_filtered = np.mean(traces_0[:, :, 0], axis=1)
        y0_filtered = np.mean(traces_0[:, :, 1], axis=1)
        x1_filtered = np.mean(traces_1[:, :, 0], axis=1)
        y1_filtered = np.mean(traces_1[:, :, 1], axis=1)
        one_indices[one] = indices_1

        if plot:
            print('New # traces:%i; old: %i' % (traces_0.shape[0] + traces_1.shape[0], i0.shape[0] + i1.shape[0]))

        new_i0 = np.mean(traces_0[:, :, 0], axis=1)
        new_q0 = np.mean(traces_0[:, :, 1], axis=1)
        new_i1 = np.mean(traces_1[:, :, 0], axis=1)
        new_q1 = np.mean(traces_1[:, :, 1], axis=1)

        # Found correct traces for ground and excited states
        ground_0_trace = np.mean(traces_0, axis=0)
        ground_1_trace = np.mean(traces_1, axis=0)

        # Identify error traces
        incorrect_traces_0 = traces[np.where(y == zero)[0], :, :]
        incorrect_traces_1 = traces[np.where(y == one)[0], :, :]
        incorrect_traces_0 = incorrect_traces_0[np.where(distance(i0, q0, x0, y0) >= rscale * radius)[0], :, :]
        incorrect_traces_1 = incorrect_traces_1[np.where(distance(i1, q1, x1, y1) >= rscale * radius)[0], :, :]

        # Find relaxation traces
        incorrect_i1 = np.mean(incorrect_traces_1[:, :, 0], axis=1)
        incorrect_q1 = np.mean(incorrect_traces_1[:, :, 1], axis=1)
        relax_traces_1 = incorrect_traces_1[np.where(distance(incorrect_i1, incorrect_q1, x0, y0) <= rscale * radius)[0], :, :]
        relax_trace = np.mean(relax_traces_1, axis=0)

        # Find leakage traces
        ket_2_traces = incorrect_traces_1[np.where(distance(incorrect_i1, incorrect_q1, x0, y0) > rscale * radius)[0], :, :]
        ket_2_trace = np.mean(ket_2_traces, axis=0)

        # Excitation traces
        excitation_traces_0 = incorrect_traces_0
        excitation_trace = np.mean(excitation_traces_0, axis=0)

        # Store results
        data = {
            'gnd_0': ground_0_trace,
            'gnd_1': ground_1_trace,
            'relax': relax_trace,
            'ket2': ket_2_traces,
            'excite': excitation_trace,
            'mean_0': tuple((x0, y0)),
            'mean_1': tuple((x1, y1)),
            'mean_0_filtered': tuple((np.mean(x0_filtered), np.mean(y0_filtered))),
            'mean_1_filtered': tuple((np.mean(x1_filtered), np.mean(y1_filtered))),
            'traces_relax': relax_traces_1,
            'traces_0': traces_0,
            'traces_1': traces_1,
            'traces_excite': incorrect_traces_0,
        }
        qubit_traces[qubit] = data

    indices_y = np.zeros(indices_0.shape)
    indices_1 = np.array([])
    for key in one_indices.keys():
        indices_y = np.hstack((indices_y, key * np.ones(one_indices[key].shape)))
        indices_1 = np.hstack((indices_1, one_indices[key]))
    filtered_indices = tuple((indices_0, one_indices, indices_1, indices_y))

    return qubit_traces, filtered_indices


# ============================================================================
# Matched filter computation
# ============================================================================

def get_mf(traces_0: np.ndarray, traces_1: np.ndarray) -> tuple:
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
