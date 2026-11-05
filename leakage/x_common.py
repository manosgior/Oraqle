"""x_common.py -- shared constants and per-chunk feature computation for the
leakage feature dataset (agent tag "x").

Signal chain per qubit q (identical to MCMit _colleague_prep.preprocess, except
that we bin-average instead of decimating by DS=40):
    Xc      = I + iQ                                   (raw ADC trace, 1000 samples)
    mixed   = Xc * exp(-i 2 pi f_q t),  t = n / 500e6   (n = 0..999, t=0 at sample 0)
    filt    = 'same' convolution of mixed with the 101-tap firwin(5 MHz, fs=500e6)
              (zero padding outside the trace, centred => no group delay)
    rot     = filt * exp(i theta_q)
    feat    = mean of rot over 20 contiguous bins of 50 samples (100 ns at 500 MS/s)
    feat[..., 0] = Re(rot) = component along the |0>->|1> axis
    feat[..., 1] = Im(rot) = component perpendicular to it
Box variant: same but without the FIR (plain boxcar mean of mixed*exp(i theta_q)).
1us variant: FIR applied to samples 0..499 only (zero padding at sample 500,
exactly what the CNN sees with time_slice=(0,500)), then 10 bins of 50.
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import numpy as np
from scipy.signal import firwin, fftconvolve

FS = 500e6                 # ADC rate assumed by every pipeline (2 ns / sample)
NUM_QUBITS = 5
N_SAMPLES = 1000           # raw trace length
BIN = 50                   # samples per bin  -> 100 ns
N_BINS = N_SAMPLES // BIN  # 20 bins over the full 2 us trace
N_BINS_1US = 500 // BIN    # 10 bins over the first 1 us (paper's "1 us" window)
NUMTAPS = 101
CUTOFF = 5e6
TAPS = firwin(NUMTAPS, CUTOFF, fs=FS)

FILES = {
    "train": (str(LP.RAW_TRAIN), "X_train", "y_train", 15000),
    "test":  (str(LP.RAW_TEST), "X_test", "y_test", 35000),
}


def fir_same(x, axis=1):
    """'same'-mode convolution with the 101-tap FIR along `axis` (zero padded).
    Equivalent to the per-index np.dot(segment, taps[::-1]) loop in
    _colleague_prep.preprocess (verified numerically in x_crosscheck.py)."""
    shape = [1] * x.ndim
    shape[axis] = NUMTAPS
    return fftconvolve(x, TAPS.reshape(shape), mode="same", axes=axis)


def chunk_features(X, params):
    """X: (n, 1000, 2) float64 raw traces.  params: list of 5 dicts {f, theta}.
    Returns float32 arrays feat (n,5,20,2), featbox (n,5,20,2), feat1us (n,5,10,2)."""
    n, L, _ = X.shape
    assert L == N_SAMPLES
    Xc = X[:, :, 0] + 1j * X[:, :, 1]
    t = np.arange(L) / FS
    feat = np.empty((n, NUM_QUBITS, N_BINS, 2), np.float32)
    featbox = np.empty((n, NUM_QUBITS, N_BINS, 2), np.float32)
    feat1us = np.empty((n, NUM_QUBITS, N_BINS_1US, 2), np.float32)
    for q, p in enumerate(params):
        rot = np.exp(1j * p["theta"])
        mixed = Xc * np.exp(-1j * 2 * np.pi * p["f"] * t)[None, :]
        filt = fir_same(mixed) * rot
        b = filt.reshape(n, N_BINS, BIN).mean(axis=2)
        feat[:, q, :, 0] = b.real; feat[:, q, :, 1] = b.imag
        bb = (mixed * rot).reshape(n, N_BINS, BIN).mean(axis=2)
        featbox[:, q, :, 0] = bb.real; featbox[:, q, :, 1] = bb.imag
        f1 = fir_same(mixed[:, :500]) * rot
        b1 = f1.reshape(n, N_BINS_1US, BIN).mean(axis=2)
        feat1us[:, q, :, 0] = b1.real; feat1us[:, q, :, 1] = b1.imag
    return feat, featbox, feat1us


def bits(y):
    """(N,) packed labels -> (N,5) prepared bits; bit q = (y >> q) & 1 = code qubit q = paper Q(q+1)."""
    return np.stack([(y >> q) & 1 for q in range(NUM_QUBITS)], axis=1).astype(np.int8)
