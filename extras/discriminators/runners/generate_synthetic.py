"""
generate_synthetic.py
=====================
Synthetic 5-qubit multiplexed dispersive-readout dataset with controlled
mid-readout T1 events.

Each shot is generated as follows:

  1. Pick a 5-bit basis state s in [0, 32).
  2. For each qubit q in 0..4, with bit b = bit_q(s):
       - Choose IQ amplitude / phase for state |b>.
       - With probability P_T1 (only meaningful if b == 1), pick a uniformly
         random sample t_decay in [0, T) and switch to the |0> parameters
         from t_decay onward — i.e., the qubit relaxes mid-readout.
  3. Modulate each qubit's IQ at its IF (matches the real chip's FREQ_READOUT)
     and sum across all 5 qubits to form the raw multiplexed IQ trace.
  4. Add per-sample Gaussian noise.
  5. Stack into shape (T, 2) and append to the output.

The resulting HDF5 has the exact dataset layout the rest of the pipeline
expects: X_train, y_train, X_test, y_test. By default both halves are drawn
from the *same* distribution (so HERQULES_original's combined-pool split is
statistically valid), and the file is gzip-compressed.

Run:
    python -m runners.generate_synthetic \
        --out /data/five_qubit_data/Synth_p_t1_0.10.hdf5 \
        --per-class 2000 --p-t1 0.10 --noise-std 1.0

To use with the rest of the codebase, point both train and test file paths at
this single file (HERQULES_original.py concatenates them anyway).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import h5py
import numpy as np
from scipy.stats import norm
from loguru import logger


# ============================================================================
# Physical constants — match the real chip
# ============================================================================

NUM_QUBITS    = 5
NUM_CLASSES   = 2 ** NUM_QUBITS
TRACE_LENGTH  = 500
SAMPLING_RATE = 500e6
DT            = 1.0 / SAMPLING_RATE

# Same IF frequencies as the real dataset (so HERQULES's demodulation just works)
FREQ_READOUT = -np.array(
    [-64.729e6, -25.366e6, 24.79e6, 70.269e6, 127.282e6]
)


def noise_std_for_fidelity(target_fidelity: float, amp: float = 1.0,
                           trace_length: int = TRACE_LENGTH) -> float:
    """Back-solve the per-sample noise std for a target single-qubit fidelity.

    For two antipodal cluster signals (|0> at phase 0, |1> at phase pi, equal
    amplitude `amp`) in white Gaussian noise, the matched-filter deflection is

        d = ||s0 - s1|| / sigma = 2 * amp * sqrt(T) / noise_std

    and the per-qubit error rate is Q(d/2). Setting Q(d/2) = 1 - F and solving:

        noise_std = amp * sqrt(T) / Q^{-1}(1 - F)

    where Q^{-1} is the upper-tail inverse of the standard normal
    (scipy.stats.norm.isf).

    This is the IDEAL matched-filter value. HERQULES's actual pipeline
    (demod + LPF + per-qubit boxcar + Wiener envelope estimated from finite
    data) loses a little signal, so the achieved fidelity lands slightly BELOW
    the target. Treat `target_fidelity` as an upper bound / calibration knob and
    verify the realized fidelity empirically.
    """
    if not (0.5 < target_fidelity < 1.0):
        raise ValueError("target_fidelity must be in (0.5, 1.0)")
    z = norm.isf(1.0 - target_fidelity)  # Q^{-1}(1 - F)
    return float(amp * np.sqrt(trace_length) / z)


# ============================================================================
# Generation
# ============================================================================

def _qubit_cluster_params(num_qubits: int):
    """Per-qubit (amp, phase) for |0>, |1>, and the leakage state |2>.

    Uniform across qubits (same SNR) so any model-vs-model gap is
    interpretable. |0> at phase 0 and |1> at phase pi are antipodal on the I
    axis, so the matched filter discriminates along I.

    |2> is placed on the Q axis (phase pi/2): its projection onto the |0>-|1>
    (I) axis is ~0, i.e. EXACTLY at the MF decision boundary. HERQULES's signed
    linear MF therefore cannot place a leaked trace, while a model that sees the
    Q channel (CNN) can. This is the crux of the leakage transferability test.
    """
    amp_0   = np.ones(num_qubits, dtype=np.float32)
    amp_1   = np.ones(num_qubits, dtype=np.float32)
    amp_2   = np.ones(num_qubits, dtype=np.float32)
    phase_0 = np.zeros(num_qubits, dtype=np.float32)
    phase_1 = np.full(num_qubits, np.pi, dtype=np.float32)
    phase_2 = np.full(num_qubits, np.pi / 2, dtype=np.float32)
    return amp_0, amp_1, amp_2, phase_0, phase_1, phase_2


def _phasor(amp, phase, cos_t, sin_t):
    """A*cos(2*pi*f*t + phase) and A*sin(2*pi*f*t + phase) given cos/sin of 2*pi*f*t."""
    cp, sp = np.cos(phase), np.sin(phase)
    return amp * (cp * cos_t - sp * sin_t), amp * (cp * sin_t + sp * cos_t)


def _gen_per_state(
    state: int,
    n_shots: int,
    p_t1: float,
    p_leakage: float,
    noise_std: float,
    params,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate (n_shots, T, 2) for one basis state.

    For each qubit in |1>, shots are partitioned (mutually exclusively) into:
      - leaked  -> static |2> signal (probability p_leakage)
      - T1      -> piecewise |1>-then-|0> at a uniform random decay time (p_t1)
      - normal  -> static |1>
    Requires p_t1 + p_leakage <= 1. Qubits in |0> are always static |0>.
    The label (basis state) is unchanged: a leaked/decayed |1> is still |1>.
    """
    amp_0, amp_1, amp_2, phase_0, phase_1, phase_2 = params
    bits = np.array([(state >> q) & 1 for q in range(NUM_QUBITS)], dtype=np.int64)

    t = np.arange(TRACE_LENGTH, dtype=np.float32) * DT
    I = np.zeros((n_shots, TRACE_LENGTH), dtype=np.float32)
    Q = np.zeros((n_shots, TRACE_LENGTH), dtype=np.float32)

    for q in range(NUM_QUBITS):
        freq = float(FREQ_READOUT[q])
        cos_t = np.cos(2 * np.pi * freq * t).astype(np.float32)
        sin_t = np.sin(2 * np.pi * freq * t).astype(np.float32)

        if bits[q] == 0:
            Iq, Qq = _phasor(amp_0[q], phase_0[q], cos_t, sin_t)
            I += Iq
            Q += Qq
            continue

        # bits[q] == 1: partition shots by mode using one uniform draw.
        u = rng.random(n_shots)
        leak_mask = u < p_leakage
        t1_mask = (u >= p_leakage) & (u < p_leakage + p_t1)
        normal_mask = ~(leak_mask | t1_mask)

        if leak_mask.any():
            Iq, Qq = _phasor(amp_2[q], phase_2[q], cos_t, sin_t)
            I[leak_mask] += Iq
            Q[leak_mask] += Qq
        if normal_mask.any():
            Iq, Qq = _phasor(amp_1[q], phase_1[q], cos_t, sin_t)
            I[normal_mask] += Iq
            Q[normal_mask] += Qq
        if t1_mask.any():
            I1, Q1 = _phasor(amp_1[q], phase_1[q], cos_t, sin_t)
            I0, Q0 = _phasor(amp_0[q], phase_0[q], cos_t, sin_t)
            for k in np.where(t1_mask)[0]:
                td = int(rng.integers(0, TRACE_LENGTH))
                I[k, :td] += I1[:td]; Q[k, :td] += Q1[:td]
                I[k, td:] += I0[td:]; Q[k, td:] += Q0[td:]

    I += rng.normal(0.0, noise_std, size=I.shape).astype(np.float32)
    Q += rng.normal(0.0, noise_std, size=Q.shape).astype(np.float32)
    return np.stack([I, Q], axis=-1)


def generate_dataset(
    per_class: int,
    p_t1: float,
    p_leakage: float,
    noise_std: float,
    seed: int,
):
    """Generate a balanced dataset of `per_class` shots per basis state.

    Returns (X, y) with X.shape == (NUM_CLASSES * per_class, T, 2) and
    y.shape == (NUM_CLASSES * per_class,).
    """
    if p_t1 + p_leakage > 1.0:
        raise ValueError(f"p_t1 + p_leakage = {p_t1 + p_leakage} > 1")
    rng = np.random.default_rng(seed)
    params = _qubit_cluster_params(NUM_QUBITS)

    total = NUM_CLASSES * per_class
    X = np.empty((total, TRACE_LENGTH, 2), dtype=np.float32)
    y = np.empty(total, dtype=np.int32)

    t0 = time.perf_counter()
    for s in range(NUM_CLASSES):
        sl = slice(s * per_class, (s + 1) * per_class)
        X[sl] = _gen_per_state(
            s, per_class, p_t1, p_leakage, noise_std, params, rng,
        )
        y[sl] = s
        if s % 8 == 7:
            elapsed = time.perf_counter() - t0
            logger.info(
                f"  generated state {s + 1}/{NUM_CLASSES}  "
                f"({elapsed:.1f}s elapsed)"
            )
    return X, y


# ============================================================================
# I/O
# ============================================================================

def _estimate_size_mb(per_class: int, train_frac: float) -> float:
    total = NUM_CLASSES * per_class
    bytes_per_shot = TRACE_LENGTH * 2 * 4   # float32
    raw_bytes = total * bytes_per_shot
    # Noisy IQ compresses poorly; assume gzip gives ~1.2x
    return raw_bytes / (1024 ** 2) / 1.2


def write_hdf5(
    path: str,
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float,
    compression: str = "gzip",
    compression_opts: int = 4,
):
    """Split into train/test and write to one HDF5 with the standard layout."""
    N = X.shape[0]
    n_train = int(round(N * train_frac))

    perm = np.random.RandomState(0).permutation(N)
    train_idx = np.sort(perm[:n_train])
    test_idx  = np.sort(perm[n_train:])

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "X_train", data=X[train_idx],
            compression=compression, compression_opts=compression_opts,
            chunks=(min(1024, n_train), TRACE_LENGTH, 2),
        )
        f.create_dataset("y_train", data=y[train_idx])
        f.create_dataset(
            "X_test", data=X[test_idx],
            compression=compression, compression_opts=compression_opts,
            chunks=(min(1024, N - n_train), TRACE_LENGTH, 2),
        )
        f.create_dataset("y_test", data=y[test_idx])

        f.attrs["num_qubits"]    = NUM_QUBITS
        f.attrs["trace_length"]  = TRACE_LENGTH
        f.attrs["sampling_rate"] = SAMPLING_RATE
        f.attrs["freq_readout"]  = FREQ_READOUT
        f.attrs["generated_at"]  = datetime.now().isoformat()

    on_disk_mb = os.path.getsize(path) / (1024 ** 2)
    logger.info(
        f"Wrote {path}  ({on_disk_mb:.1f} MB on disk; "
        f"X_train={X[train_idx].shape}, X_test={X[test_idx].shape})"
    )


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic 5-qubit readout traces with controlled "
            "mid-readout T1 events."
        )
    )
    parser.add_argument(
        "--out", required=True,
        help="Output HDF5 path (will contain X_train/y_train/X_test/y_test).",
    )
    parser.add_argument(
        "--per-class", type=int, default=2000,
        help=("Shots per basis state. Total shots = per-class * 32. "
              "Default 2000 (~256 MB on disk before compression)."),
    )
    parser.add_argument(
        "--p-t1", type=float, default=0.0,
        help=("Probability per (shot, |1>-qubit) of a mid-readout T1 event. "
              "Default 0.0 (no T1 events, clean Gaussian baseline)."),
    )
    parser.add_argument(
        "--p-leakage", type=float, default=0.0,
        help=("Probability per (shot, |1>-qubit) of leakage to |2> (placed on "
              "the Q axis, orthogonal to the |0>-|1> MF axis). Label stays |1>. "
              "Tests whether a model can recover an off-MF-axis cluster that "
              "HERQULES's linear matched filter cannot. p_t1 + p_leakage <= 1."),
    )
    parser.add_argument(
        "--noise-std", type=float, default=1.0,
        help=("Per-sample Gaussian noise std. With unit signal amplitude, "
              "this controls SNR. Default 1.0. NOTE: 1.0 is far too clean — "
              "the coherent signal integrates to an absurd matched-filter SNR "
              "and Net_rmf training diverges. Prefer --target-fidelity."),
    )
    parser.add_argument(
        "--target-fidelity", type=float, default=None,
        help=("If set (e.g. 0.98), OVERRIDES --noise-std: back-solves the "
              "noise level so the ideal per-qubit matched-filter fidelity is "
              "this value. Realized fidelity lands slightly below (demod/boxcar "
              "losses). This is the recommended knob."),
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.5,
        help=("Fraction of shots placed in X_train/y_train. Remainder goes to "
              "X_test/y_test. Default 0.5 (so both halves are statistically "
              "equivalent — what HERQULES_original would re-split anyway)."),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Skip the size warning prompt.",
    )
    args = parser.parse_args()

    # Resolve the noise level. --target-fidelity wins if given.
    noise_std = args.noise_std
    if args.target_fidelity is not None:
        noise_std = noise_std_for_fidelity(args.target_fidelity)
        d = 2.0 * np.sqrt(TRACE_LENGTH) / noise_std  # ideal deflection, amp=1
        logger.info(
            f"target_fidelity={args.target_fidelity} -> noise_std={noise_std:.3f} "
            f"(ideal matched-filter deflection d={d:.2f})"
        )

    if args.p_t1 + args.p_leakage > 1.0:
        logger.error(
            f"p_t1 ({args.p_t1}) + p_leakage ({args.p_leakage}) > 1 — "
            f"they partition the |1> shots and must sum to <= 1."
        )
        sys.exit(2)

    est_mb = _estimate_size_mb(args.per_class, args.train_frac)
    logger.info(
        f"Plan: per_class={args.per_class}  total={NUM_CLASSES * args.per_class:,}  "
        f"p_t1={args.p_t1}  p_leakage={args.p_leakage}  noise_std={noise_std:.3f}  "
        f"-> ~{est_mb:.0f} MB on disk (gzip)"
    )
    if est_mb > 2000 and not args.force:
        logger.warning(
            f"Estimated size {est_mb:.0f} MB > 2 GB. Re-run with --force to confirm."
        )
        sys.exit(2)
    if os.path.exists(args.out) and not args.force:
        logger.warning(
            f"{args.out} already exists. Re-run with --force to overwrite."
        )
        sys.exit(2)

    X, y = generate_dataset(
        per_class=args.per_class,
        p_t1=args.p_t1,
        p_leakage=args.p_leakage,
        noise_std=noise_std,
        seed=args.seed,
    )
    logger.info(f"Generated array: X={X.shape}  y={y.shape}")

    write_hdf5(args.out, X, y, train_frac=args.train_frac)


if __name__ == "__main__":
    main()
