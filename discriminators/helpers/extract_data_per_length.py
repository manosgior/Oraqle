"""
extract_data_per_length.py
==========================
Generates one CNN-ready HDF5 file per trace length, all targeting the same
number of downsampled time steps (`TARGET_STEPS`). This is the right setup for
a clean "accuracy vs. readout duration" sweep: the CNN sees the same input
shape regardless of trace length, only the physical readout window changes.

For each trace length L (in raw-sample units), the script:
  1. Truncates each raw trace to the first L samples.
  2. Demodulates with the per-qubit carrier frequency (from a one-time
     calibration on the full trace).
  3. Low-pass filters at 5 MHz (causal Butterworth) at the raw 500 MHz fs.
  4. Decimates by `ds_factor = L // TARGET_STEPS`, yielding exactly
     `TARGET_STEPS` time bins.
  5. Phase-rotates with the per-qubit theta from calibration.

Output: one HDF5 per (trace_length, train|test) at
    /data/cnn/Qubit_5Channel_target{TARGET_STEPS}_len{L}_{train|test}.h5
shape (N, TARGET_STEPS, num_qubits, 2). Float16 storage.

Notes
-----
* Each output file is ~1.5 GB train / ~3 GB test (same size as the existing
  ds=20 file, because TARGET_STEPS is fixed). 10 lengths × 2 splits ≈ 30 GB.
  Existing files are skipped on rerun.
* Calibration uses the full 1000-sample traces; this is intentional so that
  every length shares the same demod frequency / phase reference.
* `TRACE_LENGTHS` must each be divisible by TARGET_STEPS for clean decimation.
"""

import h5py
import numpy as np
import os
from scipy.signal import butter, sosfilt
import sys as _sys  # [unify]
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from disc_paths import RAW_TRAIN, RAW_TEST, CNN_DATA, require_dataset  # [unify] replaces the old docker mount paths

# ===================== CONFIG =====================
sampling_rate = 500e6           # Hz
num_qubits = 5
num_states = 32
chunk_size = 5000               # traces per chunk during processing

TARGET_STEPS = 50               # CNN input time-axis length (after decimation)
TRACE_LENGTHS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

process_train = True
process_test = True
overwrite_existing = False      # set True to force regenerate

configs = {
    "train": {
        "file_path": str(RAW_TRAIN),
        "dataset_X": "X_train",
        "dataset_y": "y_train",
        "samples_per_state": 15000,
        "dtype": "float16",
    },
    "test": {
        "file_path": str(RAW_TEST),
        "dataset_X": "X_test",
        "dataset_y": "y_test",
        "samples_per_state": 35000,
        "dtype": "float16",
    },
}


def get_output_path(mode: str, trace_length: int, target_steps: int) -> str:
    return (
        f"{CNN_DATA}/Qubit_{num_qubits}Channel_target{target_steps}"  # [unify]
        f"_len{trace_length}_{mode}.h5"
    )


# ===================== CALIBRATION (one-time) =====================
def calibrate(file_path: str, dataset_X: str, dataset_y: str, samples_per_state: int):
    """Estimate per-qubit demod frequency and phase, using full-length traces.

    Returns (qubit_params, sos_filter). The same calibration is reused for
    every trace_length in the sweep.
    """
    print("Step 1: Calibrating Frequencies (full-length traces)...")

    with h5py.File(file_path, "r") as hf:
        print(f"  File Keys: {list(hf.keys())}")
        print(f"  Input Shape: {hf[dataset_X].shape}")

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

    sos = butter(4, 5e6, btype="low", fs=sampling_rate, output="sos")
    qubit_params = []

    for i in range(num_qubits):
        mask0 = (y_cal >> i) & 1 == 0
        mask1 = (y_cal >> i) & 1 == 1

        fft0 = np.mean(np.abs(np.fft.fft(X_cal_c[mask0], axis=1)), axis=0)
        fft1 = np.mean(np.abs(np.fft.fft(X_cal_c[mask1], axis=1)), axis=0)
        best_f = freqs[np.argmax(np.abs(fft1 - fft0))]

        # Decimation factor used only to localize a stable phase reference
        # window (samples 20:80 in the ds=20 frame); does not affect output.
        sig_dc = sosfilt(sos, X_cal_c * np.exp(-1j * 2 * np.pi * best_f * t_raw), axis=1)
        sig_ds = sig_dc[:, ::20]

        m0 = np.mean(sig_ds[mask0, 20:80])
        m1 = np.mean(sig_ds[mask1, 20:80])
        theta = -np.angle(m1 - m0)

        qubit_params.append({"f": best_f, "theta": theta})
        print(f"   > Q{i}: f={best_f/1e6:.2f} MHz | θ={theta:.2f}")

    return qubit_params, sos


# ===================== PER-LENGTH PROCESSING =====================
def process_dataset_at_length(
    mode: str,
    cfg: dict,
    trace_length: int,
    target_steps: int,
    qubit_params: list,
    sos: np.ndarray,
):
    """Demod + filter + decimate raw data to (N, target_steps, num_qubits, 2)."""
    if trace_length % target_steps != 0:
        raise ValueError(
            f"trace_length={trace_length} must be divisible by target_steps={target_steps}"
        )

    ds_factor = trace_length // target_steps
    file_path = cfg["file_path"]
    dataset_X = cfg["dataset_X"]
    dataset_y = cfg["dataset_y"]
    dtype_storage = cfg["dtype"]

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing raw file: {file_path}")

    output_path = get_output_path(mode, trace_length, target_steps)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path) and not overwrite_existing:
        print(f"  [skip] exists: {output_path}")
        return

    print(
        f"\nProcessing {mode} | len={trace_length} (=={trace_length*2}ns) "
        f"| ds={ds_factor} -> {target_steps} steps -> {output_path}"
    )

    t_seg = np.arange(trace_length) / sampling_rate

    with h5py.File(file_path, "r") as h_in, h5py.File(output_path, "w") as h_out:
        N_total = h_in[dataset_X].shape[0]
        T_raw_available = h_in[dataset_X].shape[1]
        if trace_length > T_raw_available:
            raise ValueError(
                f"trace_length={trace_length} exceeds raw length {T_raw_available}"
            )

        dset_X = h_out.create_dataset(
            dataset_X,
            (N_total, target_steps, num_qubits, 2),
            dtype=dtype_storage,
        )
        dset_y = h_out.create_dataset(dataset_y, (N_total,), dtype="int32")

        # Stash provenance for debugging / reproducibility.
        h_out.attrs["trace_length_raw"] = trace_length
        h_out.attrs["target_steps"] = target_steps
        h_out.attrs["downsample_factor"] = ds_factor
        h_out.attrs["sampling_rate_hz"] = sampling_rate
        h_out.attrs["filter_cutoff_hz"] = 5e6
        h_out.attrs["qubit_frequencies_hz"] = np.array(
            [p["f"] for p in qubit_params], dtype=np.float64
        )
        h_out.attrs["qubit_thetas_rad"] = np.array(
            [p["theta"] for p in qubit_params], dtype=np.float64
        )

        for start in range(0, N_total, chunk_size):
            end = min(start + chunk_size, N_total)

            X_chunk = h_in[dataset_X][start:end, :trace_length, :]
            X_chunk_c = X_chunk[:, :, 0] + 1j * X_chunk[:, :, 1]

            qubit_channels = []
            for p in qubit_params:
                mixed = X_chunk_c * np.exp(-1j * 2 * np.pi * p["f"] * t_seg)
                filtered = sosfilt(sos, mixed, axis=1)[:, ::ds_factor]
                rotated = filtered * np.exp(1j * p["theta"])
                qubit_channels.append(rotated)

            X_stacked = np.stack(qubit_channels, axis=2)         # (n, T_ds, Q)
            X_final = np.stack([X_stacked.real, X_stacked.imag], axis=-1)
            assert X_final.shape[1] == target_steps, (
                f"unexpected T_ds={X_final.shape[1]} != target={target_steps}"
            )

            dset_X[start:end] = X_final.astype(dtype_storage)
            dset_y[start:end] = h_in[dataset_y][start:end]

            print(
                f"  {mode} L={trace_length} | {end/N_total:6.1%}",
                end="\r",
            )

    print(f"\n  Done: {output_path}")


# ===================== RUN =====================
def main():
    # Calibrate once per split using the full-length traces.
    cal_cache = {}
    for mode in ("train", "test"):
        if (mode == "train" and process_train) or (mode == "test" and process_test):
            cfg = configs[mode]
            cal_cache[mode] = calibrate(
                cfg["file_path"],
                cfg["dataset_X"],
                cfg["dataset_y"],
                cfg["samples_per_state"],
            )

    # Process every trace length, reusing the calibration.
    for length in TRACE_LENGTHS:
        if length % TARGET_STEPS != 0:
            print(
                f"[warn] skipping length={length}: not divisible by "
                f"TARGET_STEPS={TARGET_STEPS}"
            )
            continue
        if process_train:
            qubit_params, sos = cal_cache["train"]
            process_dataset_at_length(
                "train", configs["train"], length, TARGET_STEPS, qubit_params, sos
            )
        if process_test:
            qubit_params, sos = cal_cache["test"]
            process_dataset_at_length(
                "test", configs["test"], length, TARGET_STEPS, qubit_params, sos
            )


if __name__ == "__main__":
    main()
