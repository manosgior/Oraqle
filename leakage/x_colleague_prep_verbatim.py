"""Colleague's EXACT preprocessing (FFT freq-calibration + phase rotation +
sparse windowed-sinc FIR), TF-free so the PyTorch CNN.py re-runs use identical
features. Copied verbatim from her cnn_model_fir.py (numpy/scipy only).
"""
import h5py
import numpy as np
from scipy.signal import firwin

SAMPLING_RATE = 500e6
NUM_QUBITS = 5
NUM_STATES = 32
CHUNK_SIZE = 5000


def calibrate(file_path, dataset_X, dataset_y, samples_per_state, time_slice):
    with h5py.File(file_path, "r") as hf:
        X_cal_list, y_cal_list = [], []
        for s in range(NUM_STATES):
            start_idx = s * samples_per_state + 5000
            X_tmp = hf[dataset_X][start_idx: start_idx + 500]
            s0, s1 = time_slice
            X_tmp = X_tmp[:, s0:s1, :]
            X_cal_list.append(X_tmp)
            y_cal_list.append(hf[dataset_y][start_idx: start_idx + 500])
    X_cal = np.concatenate(X_cal_list, axis=0)
    y_cal = np.concatenate(y_cal_list).astype(int)
    Xc = X_cal[:, :, 0] + 1j * X_cal[:, :, 1]
    freqs = np.fft.fftfreq(Xc.shape[1], d=1 / SAMPLING_RATE)
    params = []
    for i in range(NUM_QUBITS):
        mask0 = ((y_cal >> i) & 1) == 0
        mask1 = ((y_cal >> i) & 1) == 1
        fft0 = np.mean(np.abs(np.fft.fft(Xc[mask0], axis=1)), axis=0)
        fft1 = np.mean(np.abs(np.fft.fft(Xc[mask1], axis=1)), axis=0)
        best_f = freqs[np.argmax(np.abs(fft1 - fft0))]
        m0 = np.mean(Xc[mask0]); m1 = np.mean(Xc[mask1])
        theta = -np.angle(m1 - m0)
        params.append({"f": best_f, "theta": theta})
    return params


def preprocess(file_path, dataset_X, dataset_y, params, ds_factor, time_slice):
    numtaps = 101; cutoff_freq = 5e6
    taps = firwin(numtaps, cutoff_freq, fs=SAMPLING_RATE)
    half_tap = numtaps // 2
    with h5py.File(file_path, "r") as hf:
        N = hf[dataset_X].shape[0]
        s0, s1 = time_slice
        actual_s1 = s1 if s1 is not None else hf[dataset_X].shape[1]
        target_indices = np.arange(0, (actual_s1 - s0), ds_factor)
        T_ds = len(target_indices)
        X_out = np.zeros((N, T_ds, 1, NUM_QUBITS * 2), dtype=np.float32)
        y_out = hf[dataset_y][:]
        for start in range(0, N, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, N)
            X_chunk = hf[dataset_X][start:end, s0:actual_s1, :]
            Xc = X_chunk[:, :, 0] + 1j * X_chunk[:, :, 1]
            t = np.arange(Xc.shape[1]) / SAMPLING_RATE
            channels = []
            for p in params:
                mixed = Xc * np.exp(-1j * 2 * np.pi * p["f"] * t)
                filtered_rot = np.zeros((mixed.shape[0], T_ds), dtype=complex)
                for i, idx in enumerate(target_indices):
                    w_start = idx - half_tap; w_end = idx + half_tap + 1
                    if w_start < 0 or w_end > mixed.shape[1]:
                        chunk_segment = np.pad(
                            mixed[:, max(0, w_start): min(mixed.shape[1], w_end)],
                            ((0, 0), (max(0, -w_start), max(0, w_end - mixed.shape[1]))))
                    else:
                        chunk_segment = mixed[:, w_start:w_end]
                    filtered_rot[:, i] = np.dot(chunk_segment, taps[::-1])
                rot = filtered_rot * np.exp(1j * p["theta"])
                channels.append(rot)
            X_stack = np.stack(channels, axis=2)
            X_out[start:end, :, 0, :] = np.concatenate([X_stack.real, X_stack.imag], axis=2)
    return X_out, y_out.astype(np.int64)


def to_torch_layout(X_out):
    """(N, T_ds, 1, 10) -> (N, 10, T_ds) for CNN.py / Conv1d."""
    N, T_ds = X_out.shape[0], X_out.shape[1]
    return X_out[:, :, 0, :].transpose(0, 2, 1).astype(np.float32)  # (N, 10, T_ds)


def bits(y_packed):
    return np.stack([(y_packed >> q) & 1 for q in range(NUM_QUBITS)], axis=1).astype(np.float32)
