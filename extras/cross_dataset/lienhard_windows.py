"""Sliding-window fidelity on the Lienhard channels without the low-pass filter, all 1.6 M shots.

The published sliding-window CSVs used the causal 10 MHz filter while the published truncation sweep did not
(check_linear_filter.py). This recomputes the windows the unfiltered way, so both panels of the paper's readout
figure share one preprocessing. Same functions and split as readout/run_baseline_sweeps.py.

    python -I lienhard_windows.py        # needs $ORAQLE_DATA; writes $ORAQLE_WORK/cross_dataset/linear_lienhard_windows.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from check_linear_filter import B, mix  # noqa: E402  (stubs plotting, imports baseline_no_ML)
from common import XD  # noqa: E402


def main():
    B.require_dataset()
    X, y = B.load_and_merge(B.RAW_TRAIN_FILE, B.RAW_TEST_FILE, max_samples=None)
    y = np.asarray(y)
    perm = np.random.default_rng(B.SAMPLE_SEED).permutation(len(y))
    out = {}
    for q in range(5):
        I, Q = mix(X, B.FREQ_READOUT[q])
        demod = np.stack([I, Q], -1).astype(np.float32)[perm][:, :, None, :]
        del I, Q
        yq = ((y[perm] >> q) & 1).astype(int) if y.ndim == 1 else y[perm][:, q]
        out[f"Q{q + 1}"] = {}
        for w in (250, 500):
            c, f = B.evaluate_sliding_window_fidelity(demod, yq, window_ns=w)
            out[f"Q{q + 1}"][f"window_{w}ns"] = dict(center_us=np.asarray(c).tolist(), F=f[0].tolist())
            print(f"Q{q + 1} {w} ns: first {f[0][0]:.4f} best {max(f[0]):.4f}", flush=True)
        del demod
    (XD / "linear_lienhard_windows.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
