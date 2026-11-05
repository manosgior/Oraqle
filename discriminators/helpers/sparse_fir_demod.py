"""
sparse_fir_demod.py
===================
Sparse-FIR (windowed-sinc) per-qubit demodulation. Implements the preprocessing
front-end of the hybrid signal-processing + DL discriminator:

  1. Complex mix: x_c = I + j*Q, shifted to baseband per qubit by multiplying
     with exp(-j * 2*pi * f_q * t) (f_q = the calibrated qubit IF).
  2. Low-pass with a windowed-sinc FIR (101 taps, Hamming window, 5 MHz cutoff
     by default).
  3. Sparse evaluation: output only at regularly spaced sample indices
     [0, downsample, 2*downsample, ...]. The FIR kernel is conceptually
     centred at each evaluation index and the dot product computed locally.

Mathematically the sparse-FIR-at-spaced-points formulation is equivalent to
full convolution followed by decimation. For software training we apply the
filter densely with ``scipy.signal.lfilter`` and shift by the FIR group delay
to obtain a zero-phase (centred) output, then keep every ``downsample``-th
sample. The two routes produce identical values at the kept points (modulo
edge effects within ``fir_taps//2`` samples of either end). The "sparse"
designation matters for hardware (FPGA) deployment, where you would skip the
wasted multiplies — for training there is no functional difference.

Output shape: ``(N, T_ds, NUM_QUBITS, 2)`` with the trailing ``2`` carrying
(real, imag) of the baseband signal — same layout produced by
``runners/synth_to_cnn._demod_downsample``, so downstream code can treat the
two preprocessing options interchangeably.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin, lfilter

# Defaults match the rest of the codebase's acquisition setup; override per call
# when working with a different rig.
SAMPLING_RATE = 500e6
FREQ_READOUT = -np.array([-64.729e6, -25.366e6, 24.79e6, 70.269e6, 127.282e6])
NUM_QUBITS = 5

# Filter design (windowed sinc): 101 taps, Hamming window, 5 MHz cutoff.
FIR_TAPS = 101
FIR_CUTOFF = 5e6


def design_sinc_fir(fir_taps: int = FIR_TAPS, fir_cutoff: float = FIR_CUTOFF,
                    fs: float = SAMPLING_RATE) -> np.ndarray:
    """Return the windowed-sinc FIR coefficients (symmetric, sum to ~1)."""
    return firwin(fir_taps, fir_cutoff, window="hamming", fs=fs).astype(np.float64)


def windowed_sinc_demod(X: np.ndarray, downsample: int = 40,
                        fir_taps: int = FIR_TAPS, fir_cutoff: float = FIR_CUTOFF,
                        fs: float = SAMPLING_RATE,
                        freqs: np.ndarray = FREQ_READOUT,
                        chunk: int = 20_000) -> np.ndarray:
    """Per-qubit mix + windowed-sinc LPF + sparse evaluation -> (N, T_ds, Q, 2).

    Parameters
    ----------
    X : (N, T, 2) raw IQ.
    downsample : int
        Spacing between kept evaluation points (e.g. 40 -> T_ds = ceil(T/40)).
    fir_taps : int
        Length of the windowed-sinc kernel. Must be odd for a well-defined
        centring (group delay = (fir_taps-1)/2).
    fir_cutoff : float
        Low-pass cutoff in Hz.
    fs : float
        Sampling rate in Hz.
    freqs : (Q,) array
        Per-qubit IF frequencies (sign convention: positive = positive shift).
    chunk : int
        Process this many traces at a time to bound peak memory.

    Returns
    -------
    np.ndarray, float32, shape (N, T_ds, NUM_QUBITS, 2)
    """
    if fir_taps % 2 == 0:
        raise ValueError("fir_taps must be odd (symmetric centring).")
    fir = design_sinc_fir(fir_taps, fir_cutoff, fs)
    shift = (fir_taps - 1) // 2  # group delay in samples to undo

    N = X.shape[0]
    T = X.shape[1]
    t_axis = np.arange(T) / fs
    T_ds = (T + downsample - 1) // downsample  # ceil

    parts = []
    for s0 in range(0, N, chunk):
        s1 = min(s0 + chunk, N)
        Xc = (X[s0:s1, :, 0].astype(np.float64)
              + 1j * X[s0:s1, :, 1].astype(np.float64))   # (n_chunk, T)

        # Per-qubit mix + filter + decimate.
        chans = []
        for q in range(NUM_QUBITS):
            lo = np.exp(-1j * 2 * np.pi * freqs[q] * t_axis)            # (T,)
            mixed = Xc * lo                                              # (n_chunk, T) complex
            # FIR causal filter; group-delay-shift to centre the window at the
            # evaluation index, then take every downsample-th sample.
            filt_causal = lfilter(fir, [1.0], mixed, axis=-1)
            # Shift left by `shift` so y_centered[i] = full convolution centred at i.
            # Tail (last `shift` samples) gets zeroed — those would be edge
            # samples whose window extends past T anyway.
            y_centered = np.zeros_like(filt_causal)
            if T - shift > 0:
                y_centered[:, :T - shift] = filt_causal[:, shift:]
            y_ds = y_centered[:, ::downsample][:, :T_ds]                 # (n_chunk, T_ds)
            chans.append(y_ds)

        stacked = np.stack(chans, axis=2)                                # (n, T_ds, Q) complex
        out = np.stack([stacked.real, stacked.imag], axis=-1).astype(np.float32)
        parts.append(out)

    return np.concatenate(parts, axis=0)
