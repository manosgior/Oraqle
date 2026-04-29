"""Matched-filter utilities for superconducting qubit state discrimination.

This module provides functions for computing and applying **matched filters** (MFs)
to IQ readout traces obtained from superconducting qubits.  A matched filter is the
Wiener-optimal linear discriminant for two-class problems with Gaussian noise; it
corresponds to computing the inner product of the input trace with a pre-learned
envelope vector.

Pipeline
--------
The typical usage pattern is:

1. **Find the optimal envelope** — call :func:`find_best_matched_filter` (which
   delegates to :func:`MF_single_disc` or :func:`obtain_matched_filter_with_bcub`)
   once per qubit on the training data.  This returns an *envelope* (the filter
   weights) and a *threshold* for binary classification.

2. **Extend to all qubits** — :func:`search_matched_filter_for_all_qubits`,
   :func:`search_matched_filter_for_all_qubits_preclass`, or
   :func:`search_matched_filter_for_all_qubits_demux` iterate step 1 over all
   5 qubits, returning a list of envelopes and thresholds.

3. **Apply to new traces** — :func:`matched_filter_preprocess` or
   :func:`matched_filter_preprocess_demux` compute the MF scalar output for every
   trace in every basis state, producing the feature array that :class:`HERQULES.MFOutputDataset`
   wraps for PyTorch training.

4. **Evaluate accuracy** — :func:`calculate_matched_filter_acc` or
   :func:`calculate_matched_filter_acc_demux` apply per-qubit thresholds to the
   MF outputs, reconstruct 5-qubit state predictions, and report overall and
   per-qubit classification accuracy.

Mathematical Background
-----------------------
For a binary |0⟩ / |1⟩ discrimination task the matched filter envelope is:

    ``h = E[x_0 - x_1] / Var[x_0 - x_1]``

where ``x_0`` and ``x_1`` are the flattened IQ traces (I channel followed by Q
channel) for the two classes.  Classification is then:

    ``y_pred = (x · h) < threshold``

The *boxcar* used in :func:`MF_meas` and :func:`MF_single_disc` is a rectangular
window that zeros the envelope beyond a given time index, effectively shortening
the readout integration window to reduce sensitivity to late-time noise.

Note
----
For best results the majority class is always randomly sub-sampled to match the
minority class size before fitting the envelope, ensuring balanced estimation.

See Also
--------
- :func:`MF_SVM_limit` — SVM-based threshold optimisation.
- ``HERQULES.py`` — top-level pipeline that calls these utilities.
"""
################################################################# LIBRARIES ###################################################################
import numpy as np
import h5py
from sklearn import svm
from sklearn.metrics import accuracy_score
import warnings
import logging
warnings.filterwarnings('ignore', category = FutureWarning)
warnings.filterwarnings('ignore', category = DeprecationWarning)

################### MATCHED ####################

logger = logging.getLogger('matched_filter')

def MF_meas(X_train, X_test, y_train, y_test, stop_index, bcub=0, envelope_print=False, th_limit=0):
    """Summary

    Args:
        X_train (TYPE):						training traces
        X_test (TYPE):						test traces
        y_train (TYPE):						training labels
        y_test (TYPE):						test labels
        stop_index (TYPE): 					measurement window - max time
        bcub (int, optional): 				boxcar window - max time
        envelope_print (bool, optional): 	plot envelope
        th_limit (int, optional): 			decision threshold: True or False [if True: SVM is used to evaluate threshold; if False: threshold is set to 0]

    Returns:
        TYPE: Description
    """
    X_train = X_train[:,0:2*stop_index]
    X_test 	= X_test[:,0:2*stop_index]

    if not(bcub):
        bcub = stop_index
    bc_axis = np.arange(2*stop_index)

    # X_gnd 	= np.mean(X_train[np.where(y_train==0)[0],:], axis=0)
    # X_ext 	= np.mean(X_train[np.where(y_train==1)[0],:], axis=0)


    ## threshold method
    #match_F		= X_gnd-X_ext
    match_F			= np.mean(X_train[np.where(y_train==0)[0],:]-X_train[np.where(y_train==1)[0],:], axis=0)/np.var(X_train[np.where(y_train==0)[0],:]-X_train[np.where(y_train==1)[0],:], axis=0)
    boxcar_F 		= np.heaviside(2*bcub-bc_axis, 1)
    envelope 		= match_F*boxcar_F

    filtered_train	= np.sum(X_train*envelope, axis=1)
    filtered_test	= np.sum(X_test*envelope, axis=1)

    if th_limit:
        th_limit 	= MF_SVM_limit(filtered_train, y_train)

    y_train_fit 	= (filtered_train < th_limit)+np.zeros_like(filtered_train)
    y_test_fit 		= (filtered_test < th_limit)+np.zeros_like(filtered_test)

    if envelope_print:
        return match_F, boxcar_F, th_limit
    else:
        return accuracy_score(y_test_fit, y_test), y_test_fit, accuracy_score(y_train_fit, y_train), y_train_fit, th_limit

### THRESHOLD

def MF_SVM_limit(X, y):
    """after matched filter, threshold optimization using a linar SVM

    Args:
        X (TYPE): matched filter output
        y (TYPE): labels

    Returns:
        TYPE: threshold position
    """
    x_th		= np.linspace(min(X), max(X), 10000).reshape(-1,1)
    clf 		= svm.LinearSVC(dual=False, C=1000, max_iter=1000)
    clf.fit(X.reshape(-1,1), y)
    y_th 		= clf.predict(x_th)
    th_limit 	= x_th[np.where(y_th[:-1] != y_th[1:])[0]][0,0]
    return th_limit

### EVALUATION 

def MF_single_disc(X, y, stop_index, th_limit_C=0):
    """Summary

    Args:
        X (TYPE):			training traces
        y (TYPE):			training labels
        stop_index (TYPE): 	measurement window - max time
        th_limit_C (TYPE): 	decision threshold: True or False [if True: SVM is used to evaluate threshold; if False: threshold is set to 0]
    Returns:
        TYPE: Description
    """
    X = X[:,0:2*stop_index]

    bc_axis 	= np.arange(2*stop_index)

    X_gnd 		= np.mean(X[np.where(y==0)[0],:], axis=0)
    X_ext 		= np.mean(X[np.where(y==1)[0],:], axis=0)

    match_F		= np.mean(X[np.where(y==0)[0],:]-X[np.where(y==1)[0],:], axis=0)/np.var(X[np.where(y==0)[0],:]-X[np.where(y==1)[0],:], axis=0)
    filtered	= np.sum(X*match_F, axis=1)

    if th_limit_C:
        th_limit = MF_SVM_limit(filtered, y)
    else:
        th_limit = 0

    y_fit 		= (filtered < th_limit)+np.zeros_like(filtered)
    thresh 		= 0.9*accuracy_score(y_fit, y)

    method_T 	= list([])
    bcub 		= 0
    while accuracy_score(y_fit, y) > thresh:
        boxcar_F 	= np.heaviside(2*(stop_index-bcub)-bc_axis, 1)
        envelope 	= match_F*boxcar_F
        filtered	= np.sum(X*envelope, axis=1)

        if th_limit_C:
            th_limit = MF_SVM_limit(filtered, y)

        y_fit 		= (filtered < th_limit)+np.zeros_like(filtered)
        bcub 		+= 1
        method_T.append(accuracy_score(y_fit, y))

    bcub 		= np.argmax(np.array(method_T))
    print(max(method_T))
    print(bcub)
    envelope 	= match_F*np.heaviside(2*(stop_index-bcub)-bc_axis, 1)

    if th_limit_C:
        th_limit = MF_SVM_limit(np.sum(X*envelope, axis=1), y)

    return envelope, th_limit


def obtain_matched_filter_with_bcub(X, y, stop_index, th_limit_C, best_bc):
    X = X[:, 0:2 * stop_index]

    bc_axis = np.arange(2 * stop_index)
    
    zero_traces = X[np.where(y == 0)[0], :]
    one_traces = X[np.where(y == 1)[0], :]
    
    match_F = None
    if one_traces.shape[0] == zero_traces.shape[0]:
        match_F = np.mean(zero_traces - one_traces, axis=0) / np.var(zero_traces - one_traces, axis=0)
    elif one_traces.shape[0] > zero_traces.shape[0]:
        indices = np.random.choice(one_traces.shape[0], zero_traces.shape[0], replace=False)
        match_F = np.mean(zero_traces - one_traces[indices], axis=0) / np.var(zero_traces - one_traces[indices], axis=0)
    else:
        indices = np.random.choice(zero_traces.shape[0], one_traces.shape[0], replace=False)
        match_F = np.mean(zero_traces[indices] - one_traces, axis=0) / np.var(zero_traces[indices] - one_traces, axis=0)
        
    X_gnd = np.mean(X[np.where(y == 0)[0], :], axis=0)
    X_ext = np.mean(X[np.where(y == 1)[0], :], axis=0)

    # match_F = np.mean(X[np.where(y == 0)[0], :] - X[np.where(y == 1)[0], :], axis=0) / np.var(
    #     X[np.where(y == 0)[0], :] - X[np.where(y == 1)[0], :], axis=0)

    boxcar_F = np.heaviside(2 * (stop_index - best_bc) - bc_axis, 1)
    envelope = match_F * boxcar_F
    filtered = np.sum(X * envelope, axis=1)

    if th_limit_C:
        th_limit = MF_SVM_limit(filtered, y)
    else:
        th_limit = 0
    return envelope, th_limit


def find_best_matched_filter(train_gnd, train_ext, best_bc=None):
    X_train = np.concatenate([train_gnd, train_ext], axis=0)
    
    if len(X_train.shape) > 2:
        X_train = X_train.reshape([X_train.shape[0], X_train.shape[1] * X_train.shape[2]])
    num_bins = int(X_train.shape[1] / 2)
    Y_train = np.concatenate([np.zeros(len(train_gnd)), np.ones(len(train_ext))], axis=0)

    if best_bc != None:
        return obtain_matched_filter_with_bcub(X_train, Y_train, num_bins, True, best_bc)
    return MF_single_disc(X_train, Y_train, num_bins, th_limit_C=True)


def search_matched_filter_for_all_qubits(train_data, best_bc=None):
    """Compute matched-filter envelopes for all 5 qubits from a single stacked array.

    Iterates over the 5 qubits, treating qubit *k* as the excited state (label
    ``1 << k``) and index 0 as the ground state, then calls
    :func:`find_best_matched_filter` for each pair.

    Args:
        train_data (np.ndarray or list): Shape
            ``(num_basis_states, num_samples, trace_length, 2)`` — all 32 basis
            states stacked along the first axis.
        best_bc (list[int] or None): Pre-determined best boxcar widths (one per
            qubit).  If ``None``, the optimal boxcar is searched via
            :func:`MF_single_disc`. Default: ``None``.

    Returns:
        tuple: ``(all_envelopes, all_thresholds)``

        - **all_envelopes** (list[np.ndarray]): 5 envelope arrays.
        - **all_thresholds** (list[float]): 5 decision thresholds.
    """
    gnd_idx = 0
    all_envelopes, all_thres = [], []
    for ext_qubit_idx in range(5):
        ext_idx = 1 << ext_qubit_idx
        if best_bc:
            envelope, thres = find_best_matched_filter(train_data[gnd_idx], train_data[ext_idx], best_bc[ext_qubit_idx])
        else:
            envelope, thres = find_best_matched_filter(train_data[gnd_idx], train_data[ext_idx])
        all_envelopes.append(envelope)
        all_thres.append(thres)
    return all_envelopes, all_thres

def search_matched_filter_for_all_qubits_demux(data, best_bc=None):
    """Compute matched-filter envelopes for all 5 qubits from per-qubit demultiplexed data.

    Similar to :func:`search_matched_filter_for_all_qubits`, but expects the
    data to be pre-organised in a per-qubit list structure (the output of
    :func:`HERQULES.mf_demux_data_prep`).

    Args:
        data (list): Length-5 list; ``data[k]`` is an array of shape
            ``(num_basis_states, num_samples, trace_length, 2)`` containing the
            demodulated traces for qubit *k*.
        best_bc (list[int] or None): Pre-determined best boxcar widths (one per
            qubit).  ``None`` triggers an automatic search. Default: ``None``.

    Returns:
        tuple: ``(all_envelopes, all_thresholds)`` — same structure as
        :func:`search_matched_filter_for_all_qubits`.
    """
    gnd_idx = 0
    all_envelopes, all_thres = [], []
    for ext_qubit_idx in range(5):
        train_data = data[ext_qubit_idx]
        ext_idx = 1 << ext_qubit_idx
        if best_bc != None:
            envelope, thres = find_best_matched_filter(train_data[gnd_idx], train_data[ext_idx], best_bc[ext_qubit_idx])
        else:
            envelope, thres = find_best_matched_filter(train_data[gnd_idx], train_data[ext_idx])
        all_envelopes.append(envelope)
        all_thres.append(thres)
    return all_envelopes, all_thres

def search_matched_filter_for_all_qubits_preclass(train_data, ytrain, best_bc=None):
    """Compute MF envelopes for all qubits from a pre-classified (label-filtered) flat array.

    Variant of :func:`search_matched_filter_for_all_qubits` intended for use
    after :class:`HERQULES.preclassifier` has produced a purified flat dataset.
    Ground-state and excited-state traces are selected by label indexing into
    *train_data* rather than by positional indexing.

    Args:
        train_data (np.ndarray): Flat array of shape
            ``(N_filtered, trace_length, 2)`` containing the purified traces.
        ytrain (np.ndarray): Integer labels of shape ``(N_filtered,)``.
        best_bc (list[int] or None): Pre-determined best boxcar widths.
            Default: ``None``.

    Returns:
        tuple: ``(all_envelopes, all_thresholds)``.
    """
    gnd_idx = 0
    all_envelopes, all_thres = [], []
    for ext_qubit_idx in range(5):
        ext_idx = 1 << ext_qubit_idx
        if best_bc:
            envelope, thres = find_best_matched_filter(train_data[np.where(ytrain==gnd_idx)[0]], train_data[np.where(ytrain==ext_idx)[0]], best_bc[ext_qubit_idx])
        else:
            envelope, thres = find_best_matched_filter(train_data[np.where(ytrain==gnd_idx)[0]], train_data[np.where(ytrain==ext_idx)[0]])
        all_envelopes.append(envelope)
        all_thres.append(thres)
    return all_envelopes, all_thres

def matched_filter_preprocess(data, envelopes):
    """Apply a set of MF envelopes to a stacked multi-qubit dataset.

    For each of the 5 MF envelopes (one per qubit) the function:

    1. Finds the minimum number of samples across all 32 basis states and
       sub-samples the others to match (balancing).
    2. Flattens each trace ``(trace_length, 2) → (trace_length * 2,)``.
    3. Computes the MF scalar: ``output = np.dot(trace, envelope)``.

    The results are assembled into an array of shape
    ``(num_basis_states, num_samples, num_qubits)`` and then transposed to
    ``(num_basis_states, num_samples, num_qubits)`` (basis-state-major order).

    Args:
        data (list): Length-32 list; ``data[i]`` is an array of shape
            ``(num_samples_i, trace_length, 2)`` for basis state *i*.
        envelopes (list[np.ndarray]): 5 MF envelopes, each of shape
            ``(trace_length * 2,)``.

    Returns:
        np.ndarray: Shape ``(num_basis_states, num_samples_min, num_qubits)``
        — MF scalar output per state, per sample, per qubit.
    """
    all_data_after_mf = []  # num_qubits(MFs) * num_basis_state * num_records_per_state
    for envelope in envelopes:
        data_per_envelope = []  # num_basis_state * num_records_per_state
        min_ind = min([traces.shape[0] for traces in data])
        for data_per_state in data:
            indices = np.random.choice(data_per_state.shape[0], min_ind, replace=False)
            data_per_state = data_per_state[indices]
            data_per_state = data_per_state.reshape([data_per_state.shape[0], -1])
            data_filtered = np.sum(data_per_state * envelope, axis=1)
            data_per_envelope.append(data_filtered)
        all_data_after_mf.append(np.array(data_per_envelope))
    all_data_after_mf = np.array(all_data_after_mf)
    all_data_after_mf = all_data_after_mf.transpose([1, 2, 0])  # num_basis_state * num_records_per_state * num_qubits(MFs)
    logger.debug("matched filter data")
    logger.debug(all_data_after_mf.shape)
    logger.debug(all_data_after_mf[0][0])
    return all_data_after_mf

def matched_filter_preprocess_demux(data, envelopes):
    """Apply a set of MF envelopes to per-qubit demultiplexed data.

    Functionally identical to :func:`matched_filter_preprocess` but operates on
    the per-qubit data structure produced by :func:`HERQULES.mf_demux_data_prep`
    (each qubit's traces stored separately).

    Args:
        data (list): Length-5 list; ``data[k]`` is a length-32 list of arrays
            ``(num_samples_i, trace_length, 2)`` for qubit *k*.
        envelopes (list[np.ndarray]): 5 MF envelopes.

    Returns:
        np.ndarray: Shape ``(num_basis_states, num_samples_min, num_qubits)``.
    """
    all_data_after_mf = []  # num_qubits(MFs) * num_basis_state * num_records_per_state
    for idx, envelope in enumerate(envelopes):
        data_per_envelope = []  # num_basis_state * num_records_per_state
        min_ind = min([traces.shape[0] for traces in data[idx]])
        for data_per_state in data[idx]:
            indices = np.random.choice(data_per_state.shape[0], min_ind, replace=False)
            data_per_state = data_per_state[indices]
            data_per_state = data_per_state.reshape([data_per_state.shape[0], -1])
            data_filtered = np.sum(data_per_state * envelope, axis=1)
            data_per_envelope.append(data_filtered)
        all_data_after_mf.append(np.array(data_per_envelope))
    all_data_after_mf = np.array(all_data_after_mf)
    all_data_after_mf = all_data_after_mf.transpose([1, 2, 0])  # num_basis_state * num_records_per_state * num_qubits(MFs)
    logger.debug("matched filter data")
    logger.debug(all_data_after_mf.shape)
    logger.debug(all_data_after_mf[0][0])
    return all_data_after_mf


def calculate_matched_filter_acc(data, all_mfs, all_thres):
    """Evaluate MF classifier accuracy on a stacked multi-qubit dataset.

    Applies the matched-filter envelopes, assembles per-qubit binary predictions
    using the stored thresholds, reconstructs the 5-qubit integer state
    prediction by bit-packing, and reports both overall and per-qubit accuracy.

    Args:
        data (list): Length-32 list of trace arrays (same format as
            :func:`matched_filter_preprocess`).
        all_mfs (list[np.ndarray]): 5 MF envelopes.
        all_thres (list[float]): 5 per-qubit decision thresholds.

    Side Effects:
        Prints cumulative accuracy and per-qubit accuracy to stdout.
    """
    all_data_after_mf = matched_filter_preprocess(data, all_mfs)  # num_basis_state * num_records_per_state * num_qubits(MFs)

    num_basis_state = all_data_after_mf.shape[0]
    num_samples_per_state = all_data_after_mf.shape[1]

    all_labels = []
    for i in range(num_basis_state):
        all_labels.append(np.array([i for _ in range(num_samples_per_state)]))

    all_data = all_data_after_mf.reshape((num_basis_state * num_samples_per_state, -1))
    all_labels = np.array(all_labels).reshape((-1))

    all_preds = np.zeros(len(all_labels), dtype=int)
    shift = 0
    for qubit_idx in range(len(all_thres)):
        thres = all_thres[qubit_idx]
        data_per_qubit = all_data[:, qubit_idx]
        preds = data_per_qubit < thres + np.zeros_like(data_per_qubit)
        all_preds += preds << shift
        shift += 1

    cumulative_acc = np.sum(all_preds == all_labels) / len(all_labels)

    acc_per_qubit = []
    for _ in range(5):
        pred_qubit = all_preds % 2
        label_qubit = all_labels % 2
        acc_per_qubit.append(np.sum(pred_qubit == label_qubit) / len(label_qubit))
        all_preds = all_preds >> 1
        all_labels = all_labels >> 1

    print('Cumulative acc: {}'.format(cumulative_acc))
    print('Acc for each qubit: {}'.format(acc_per_qubit))

def calculate_matched_filter_acc_demux(data, all_mfs, all_thres):
    """Evaluate MF classifier accuracy on per-qubit demultiplexed data.

    Variant of :func:`calculate_matched_filter_acc` for demultiplexed data.
    Saves prediction and label arrays to ``mf_preds.pkl`` for offline analysis
    in addition to printing accuracy metrics.

    Args:
        data (list): Per-qubit data structure (same format as
            :func:`matched_filter_preprocess_demux`).
        all_mfs (list[np.ndarray]): 5 MF envelopes.
        all_thres (list[float]): 5 per-qubit decision thresholds.

    Side Effects:
        - Prints cumulative and per-qubit accuracy to stdout.
        - Writes ``mf_preds.pkl`` containing ``{'labels': ..., 'preds': ...}``.
    """
    all_data_after_mf = matched_filter_preprocess_demux(data, all_mfs)  # num_basis_state * num_records_per_state * num_qubits(MFs)

    num_basis_state = all_data_after_mf.shape[0]
    num_samples_per_state = all_data_after_mf.shape[1]

    all_labels = []
    for i in range(num_basis_state):
        all_labels.append(np.array([i for _ in range(num_samples_per_state)]))

    all_data = all_data_after_mf.reshape((num_basis_state * num_samples_per_state, -1))
    all_labels = np.array(all_labels).reshape((-1))

    all_preds = np.zeros(len(all_labels), dtype=int)
    shift = 0
    for qubit_idx in range(len(all_thres)):
        thres = all_thres[qubit_idx]
        data_per_qubit = all_data[:, qubit_idx]
        preds = data_per_qubit < thres + np.zeros_like(data_per_qubit)
        all_preds += preds << shift
        shift += 1

    cumulative_acc = np.sum(all_preds == all_labels) / len(all_labels)
    import pickle
    with open('mf_preds.pkl', 'wb') as file:
        pickle.dump({'labels':all_labels, 'preds':all_preds}, file)
    acc_per_qubit = []
    for _ in range(5):
        pred_qubit = all_preds % 2
        label_qubit = all_labels % 2
        acc_per_qubit.append(np.sum(pred_qubit == label_qubit) / len(label_qubit))
        all_preds = all_preds >> 1
        all_labels = all_labels >> 1

    print('Cumulative acc: {}'.format(cumulative_acc))
    print('Acc for each qubit: {}'.format(acc_per_qubit))

    