"""
synth_to_cnn.py
===============
Convert a synthetic raw-IQ dataset (from runners/generate_synthetic.py) into
the CNN input format, matching helpers/extract_data.py's preprocessing
(per-qubit digital down-conversion -> Butterworth LPF -> downsample) BUT
skipping calibration: for synthetic data we already know the qubit
frequencies (FREQ_READOUT) and the cluster phase (0, so rotation theta = 0).
This also sidesteps extract_data.py's per-state-block / fixed-count
assumptions, which don't hold for the shuffled synthetic files.

Output HDF5 has X_train / X_test of shape (N, T_ds, 5, 2) and y_train /
y_test, exactly what helpers.cnn_helpers.prepare_cnn_data expects when called
with original_length = raw trace length and the same downsample_factor.

    python -m runners.synth_to_cnn \
        --in  /data/synthetic_data/Synth_F0.98_p_t1_0.1.hdf5 \
        --out /data/synthetic_data/Synth_F0.98_p_t1_0.1_cnn.hdf5
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
from scipy.signal import butter, sosfilt

NUM_QUBITS    = 5
SAMPLING_RATE = 500e6
# Same IF frequencies the generator uses (sign convention matches).
FREQ_READOUT  = -np.array([-64.729e6, -25.366e6, 24.79e6, 70.269e6, 127.282e6])
# Matches extract_data.py's calibrate() filter (order 4, 5 MHz).
FILTER_CUTOFF = 5e6
FILTER_ORDER  = 4


def _demod_downsample(X: np.ndarray, downsample_factor: int) -> np.ndarray:
    """Raw IQ (N, T, 2) -> per-qubit demod + LPF + downsample -> (N, T_ds, 5, 2)."""
    Xc = X[:, :, 0].astype(np.float64) + 1j * X[:, :, 1].astype(np.float64)  # (N, T)
    T = Xc.shape[1]
    t = np.arange(T) / SAMPLING_RATE
    sos = butter(FILTER_ORDER, FILTER_CUTOFF, btype="low",
                 fs=SAMPLING_RATE, output="sos")

    chans = []
    for q in range(NUM_QUBITS):
        mixed = Xc * np.exp(-1j * 2 * np.pi * FREQ_READOUT[q] * t)
        filt = sosfilt(sos, mixed, axis=1)[:, ::downsample_factor]  # (N, T_ds)
        chans.append(filt)   # theta = 0 -> no rotation

    stacked = np.stack(chans, axis=2)                    # (N, T_ds, 5) complex
    return np.stack([stacked.real, stacked.imag], axis=-1).astype(np.float32)


def convert(in_path: str, out_path: str, downsample_factor: int, chunk: int):
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)
    with h5py.File(in_path, "r") as fin, h5py.File(out_path, "w") as fout:
        for split in ("train", "test"):
            Xk, yk = f"X_{split}", f"y_{split}"
            if Xk not in fin:
                continue
            N = fin[Xk].shape[0]
            T = fin[Xk].shape[1]
            T_ds = T // downsample_factor
            dX = fout.create_dataset(
                Xk, (N, T_ds, NUM_QUBITS, 2), dtype="float32",
                compression="gzip", compression_opts=4,
                chunks=(min(1024, N), T_ds, NUM_QUBITS, 2),
            )
            fout.create_dataset(yk, data=fin[yk][:])
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                dX[s:e] = _demod_downsample(fin[Xk][s:e], downsample_factor)
            print(f"  {split}: {N} traces  raw_T={T} -> T_ds={T_ds}")
        fout.attrs["downsample_factor"] = downsample_factor
        fout.attrs["source"] = os.path.basename(in_path)
    print(f"Wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    p.add_argument("--in", dest="in_path", required=True,
                   help="Synthetic raw-IQ HDF5 (X_train/X_test/y_*).")
    p.add_argument("--out", dest="out_path", required=True,
                   help="Output CNN-format HDF5.")
    p.add_argument("--downsample-factor", type=int, default=20)
    p.add_argument("--chunk", type=int, default=5000)
    p.add_argument("--force", action="store_true",
                   help="Overwrite output if it exists.")
    args = p.parse_args()
    if os.path.exists(args.out_path) and not args.force:
        print(f"{args.out_path} exists; use --force to overwrite.")
        sys.exit(2)
    convert(args.in_path, args.out_path, args.downsample_factor, args.chunk)


if __name__ == "__main__":
    main()
