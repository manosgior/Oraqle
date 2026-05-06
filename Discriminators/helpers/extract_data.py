import h5py
import numpy as np
import os
from scipy.signal import butter, sosfilt

# ===================== CONFIG =====================
sampling_rate = 500e6
num_qubits = 5
num_states = 32
chunk_size = 5000

# Control behavior
downsample_factor = 40          # <-- change this
process_train = True
process_test = True

# Dataset-specific configs
configs = {
    "train": {
        "file_path": "/share/manos/DRaw_C_Tr_v0-001",
        "dataset_X": "X_train",
        "dataset_y": "y_train",
        "samples_per_state": 15000,
        "dtype": "float16"
    },
    "test": {
        "file_path": "/share/manos/DRaw_C_Te_v0-002",
        "dataset_X": "X_test",
        "dataset_y": "y_test",
        "samples_per_state": 35000,
        "dtype": "float16"
    }
}

# Output naming
def get_output_path(mode):
    return f"/data/cnn/Qubit_{num_qubits}Channel_ds{downsample_factor}_{mode}.h5"


# ===================== CORE FUNCTIONS =====================
def calibrate(file_path, dataset_X, dataset_y, samples_per_state):
    print("Step 1: Calibrating Frequencies...")

    with h5py.File(file_path, "r") as hf:
        print(f"File Keys: {list(hf.keys())}")
        print(f"Input Shape: {hf[dataset_X].shape}")

        X_cal_list, y_cal_list = [], []

        for s in range(num_states):
            start_idx = s * samples_per_state + 5000
            X_cal_list.append(hf[dataset_X][start_idx:start_idx + 500])
            y_cal_list.append(hf[dataset_y][start_idx:start_idx + 500])

    X_cal = np.concatenate(X_cal_list, axis=0)
    y_cal = np.concatenate(y_cal_list, axis=0).astype(int)

    X_cal_c = X_cal[:, :, 0] + 1j * X_cal[:, :, 1]
    t_raw = np.arange(X_cal_c.shape[1]) / sampling_rate
    freqs = np.fft.fftfreq(X_cal_c.shape[1], d=1 / sampling_rate)

    sos = butter(4, 5e6, btype='low', fs=sampling_rate, output='sos')
    qubit_params = []

    for i in range(num_qubits):
        mask0 = (y_cal >> i) & 1 == 0
        mask1 = (y_cal >> i) & 1 == 1

        fft0 = np.mean(np.abs(np.fft.fft(X_cal_c[mask0], axis=1)), axis=0)
        fft1 = np.mean(np.abs(np.fft.fft(X_cal_c[mask1], axis=1)), axis=0)
        best_f = freqs[np.argmax(np.abs(fft1 - fft0))]

        sig_dc = sosfilt(sos, X_cal_c * np.exp(-1j * 2 * np.pi * best_f * t_raw), axis=1)
        sig_ds = sig_dc[:, ::downsample_factor]

        m0 = np.mean(sig_ds[mask0, 20:80])
        m1 = np.mean(sig_ds[mask1, 20:80])
        theta = -np.angle(m1 - m0)

        qubit_params.append({'f': best_f, 'theta': theta})
        print(f" > Q{i}: {best_f/1e6:.2f} MHz | θ={theta:.2f}")

        # qubit_params.append({'f': best_f})
        # print(f" > Q{i}: {best_f/1e6:.2f} MHz")
    return qubit_params, sos


def process_dataset(mode, cfg):
    file_path = cfg["file_path"]
    dataset_X = cfg["dataset_X"]
    dataset_y = cfg["dataset_y"]
    samples_per_state = cfg["samples_per_state"]
    dtype_storage = cfg["dtype"]

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

    output_path = get_output_path(mode)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Calibration ---
    qubit_params, sos = calibrate(file_path, dataset_X, dataset_y, samples_per_state)

    print(f"\nStep 2: Processing {mode} → {output_path}")

    with h5py.File(file_path, "r") as h_in, h5py.File(output_path, "w") as h_out:
        N_total = h_in[dataset_X].shape[0]
        T_ds = h_in[dataset_X].shape[1] // downsample_factor

        dset_X = h_out.create_dataset(
            dataset_X,
            (N_total, T_ds, num_qubits, 2),
            dtype=dtype_storage
        )
        dset_y = h_out.create_dataset(dataset_y, (N_total,), dtype='int32')

        for start in range(0, N_total, chunk_size):
            end = min(start + chunk_size, N_total)

            X_chunk = h_in[dataset_X][start:end]
            X_chunk_c = X_chunk[:, :, 0] + 1j * X_chunk[:, :, 1]
            t_seg = np.arange(X_chunk_c.shape[1]) / sampling_rate

            qubit_channels = []
            for p in qubit_params:
                mixed = X_chunk_c * np.exp(-1j * 2 * np.pi * p['f'] * t_seg)
                filtered = sosfilt(sos, mixed, axis=1)[:, ::downsample_factor]
                rotated = filtered * np.exp(1j * p['theta'])
                qubit_channels.append(rotated)

            # for p in qubit_params:
            #     mixed = X_chunk_c * np.exp(-1j * 2 * np.pi * p['f'] * t_seg)
            #     filtered = sosfilt(sos, mixed, axis=1)[:, ::downsample_factor]
            #     qubit_channels.append(filtered)

            X_stacked = np.stack(qubit_channels, axis=2)
            X_final = np.stack([X_stacked.real, X_stacked.imag], axis=-1)

            dset_X[start:end] = X_final.astype(dtype_storage)
            dset_y[start:end] = h_in[dataset_y][start:end]

            print(f"{mode} Progress: {end/N_total:6.1%}", end="\r")

    print(f"\nFinished {mode} → {output_path}\n")


# ===================== RUN =====================
if process_train:
    process_dataset("train", configs["train"])

if process_test:
    process_dataset("test", configs["test"])