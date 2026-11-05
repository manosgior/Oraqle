"""Quick sanity check for the synthetic generator.

Loads a generated .hdf5, runs HERQULES's demodulator on a small sample, and
prints the per-qubit cluster positions for |0> vs |1>. If they're meaningfully
different, the generator works and the problem is somewhere else
(training, file path, label mismatch). If they're identical or zero, the
modulation has a bug.

Run inside the container:
    python -m runners._debug_synth /data/synthetic_data/Synth_p_t1_0.00.hdf5
"""

import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import os
import sys

import h5py
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainers.HERQULES_original import (
    demodulate_multiplexed_traces,
    freq_readout as FREQ_READOUT,
    sampling_rate as SAMPLING_RATE,
)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]

    with h5py.File(path, "r") as f:
        n = f["X_train"].shape[0]
        # Sample ACROSS the whole array, not the first N rows: X_train is
        # written sorted by state, so a head slice only covers the lowest
        # states (missing |1> for the high-bit qubits).
        rng = np.random.RandomState(0)
        idx = np.sort(rng.choice(n, size=min(20000, n), replace=False))
        X = f["X_train"][idx]
        y = f["y_train"][idx]

    print(f"Loaded {path}: X={X.shape}  y range=[{y.min()}, {y.max()}]")
    print(f"  X stats: mean={X.mean():.3f}  std={X.std():.3f}")
    print()

    # Demodulate using HERQULES's own function
    demod = demodulate_multiplexed_traces(
        X, FREQ_READOUT, sampling_rate=SAMPLING_RATE,
    )

    print("Per-qubit demodulated cluster check:")
    print(f"  (each row: time-averaged I/Q for traces where that qubit's bit = b)")
    print()
    for q in range(5):
        demod_q = demod[q]  # (N, T, 2)
        mask_0 = ((y >> q) & 1) == 0
        mask_1 = ((y >> q) & 1) == 1
        i0 = demod_q[mask_0, :, 0].mean()
        q0 = demod_q[mask_0, :, 1].mean()
        i1 = demod_q[mask_1, :, 0].mean()
        q1 = demod_q[mask_1, :, 1].mean()
        sep = np.sqrt((i0 - i1) ** 2 + (q0 - q1) ** 2)
        # std within each cluster
        si = demod_q[mask_0, :, 0].std()
        sq = demod_q[mask_0, :, 1].std()
        snr = sep / (0.5 * (si + sq) + 1e-12)
        print(
            f"  qubit {q}:  |0>=({i0:+.3f}, {q0:+.3f})  |1>=({i1:+.3f}, {q1:+.3f})  "
            f"sep={sep:.3f}  intra_std~{0.5*(si+sq):.3f}  SNR~{snr:.2f}"
        )
    print()
    print("Interpreting the numbers:")
    print("  - sep ~ 1-2 per qubit (signal-determined; ~0 means modulation is broken).")
    print("  - intra_std grows with noise_std. For a REALISTIC dataset")
    print("    (--target-fidelity ~0.98, noise_std~11) expect intra_std ~2-3 and")
    print("    per-sample SNR < 1 — that is CORRECT. The discrimination comes from")
    print("    integrating ~500 samples, not from a high per-sample SNR.")
    print("  - The old 'too clean' bug showed intra_std~0.23, SNR~8.5: that integrates")
    print("    to an absurd matched-filter deflection (~45) and diverges Net_rmf.")


if __name__ == "__main__":
    main()
