"""Which filter reproduces the published Linear Threshold row, and does skipping early samples help?

The current readout/baseline_no_ML.demodulate applies a causal 10 MHz Butterworth filter. This compares it
with a zero-phase version and with no filter, on 40,000 raw shots (1,250 per state from the test file, 70/30
split), against the published Baseline_Threshold_len*.csv. It also scans the start of the integration window
at a fixed end time, on the unfiltered channels.

    python -I check_linear_filter.py      # needs $ORAQLE_DATA; writes $ORAQLE_WORK/cross_dataset/linear_filter_check.json
"""
import csv
import glob
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from common import RESULTS, XD  # noqa: E402

for _m in ("matplotlib", "matplotlib.pyplot", "matplotlib.gridspec", "seaborn"):
    sys.modules.setdefault(_m, types.ModuleType(_m))  # plotting only; never called here
sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]
sys.modules["matplotlib"].gridspec = sys.modules["matplotlib.gridspec"]
import baseline_no_ML as B  # noqa: E402
import h5py  # noqa: E402
from scipy.signal import butter, sosfilt, sosfiltfilt  # noqa: E402

TS = [(200, 100), (400, 200), (600, 300), (1000, 500)]


def mix(X, fq, dt=2e-9):
    """The mixing in baseline_no_ML.demodulate, stopped before its filter."""
    I = X[:, :, 0] - X[:, :, 0].mean(1, keepdims=True)
    Q = X[:, :, 1] - X[:, :, 1].mean(1, keepdims=True)
    Q = Q * I.std(1, keepdims=True) / (Q.std(1, keepdims=True) + 1e-10)
    t = np.arange(X.shape[1]) * dt
    lo_i, lo_q = np.cos(2 * np.pi * fq * t), -np.sin(2 * np.pi * fq * t)
    return I * lo_i - Q * lo_q, I * lo_q + Q * lo_i


def by_file(X, Y, sos):
    """Every published sweep and sliding-window file, against each variant, at the short durations where
    the filter matters. The files come from six partial runs on 2026-05-14 and the code of that day is not
    in git, so this is the only way to tell which code wrote which file."""
    chans = {q: mix(X, B.FREQ_READOUT[q]) for q in range(5)}
    ys = {q: ((Y >> q) & 1).astype(int) for q in range(5)}
    perm = np.random.default_rng(1).permutation(len(Y))
    n = int(0.7 * len(Y))
    tr, te = perm[:n], perm[n:]

    def fid(q, i0, i1, lp):
        a, b = chans[q]
        if lp:
            a, b = sosfilt(sos, a, axis=1), sosfilt(sos, b, axis=1)
        F = np.c_[np.trapezoid(a[:, i0:i1], axis=1), np.trapezoid(b[:, i0:i1], axis=1)]
        c = B.fit_threshold(F[tr], ys[q][tr])
        return float(B.assignment_fidelity(ys[q][te], c.predict(F[te])))

    rows = []
    for L in (50, 100, 150, 200, 250):
        for f in sorted(glob.glob(str(RESULTS / "discriminators" / f"Baseline_Threshold_len{L}_*.csv"))):
            r = list(csv.DictReader(open(f)))[0]
            pub = [float(r[f"qubit_{q}_accuracy"]) / 100 for q in range(5)]
            d = {v: float(np.mean([abs(fid(q, 0, L + 1, v == "causal_lp") - pub[q]) for q in range(5)])) for v in ("causal_lp", "no_lp")}
            rows.append(dict(file=os.path.basename(f), **d))
    for w, name in ((250, "Baseline_Threshold_sliding_window_250ns_20260514_204419.csv"), (500, "Baseline_Threshold_sliding_window_500ns.csv")):
        rs = list(csv.DictReader(open(RESULTS / "discriminators" / name)))
        for k in range(3):  # the first three windows: starts at 0, 100, 200 ns
            start, size = k * 50, w // 2
            pub = [float(rs[k][f"qubit_{q}_fidelity"]) for q in range(5)]
            d = {v: float(np.mean([abs(fid(q, start, start + size, v == "causal_lp") - pub[q]) for q in range(5)])) for v in ("causal_lp", "no_lp")}
            rows.append(dict(file=f"{name} window {k}", **d))
    for r in rows:
        print(f"{r['file'][:66]:66s} causal_lp {100 * r['causal_lp']:5.2f} pp   no_lp {100 * r['no_lp']:5.2f} pp   -> {'no filter' if r['no_lp'] < r['causal_lp'] else 'filter'}", flush=True)
    return rows


def main():
    B.require_dataset()
    X, Y = [], []
    with h5py.File(B.RAW_TEST_FILE, "r") as f:
        for s in range(32):
            a = s * 35000
            X.append(f["X_test"][a:a + 1250])
            Y.append(f["y_test"][a:a + 1250])
    X, Y = np.concatenate(X).astype(np.float64), np.concatenate(Y)
    sos = butter(3, 10e6, btype="low", fs=500e6, output="sos")
    published = {}
    for T, L in TS:
        f = sorted(glob.glob(str(RESULTS / "discriminators" / f"Baseline_Threshold_len{L}_*.csv")))[-1]
        r = list(csv.DictReader(open(f)))[0]
        published[T] = [float(r[f"qubit_{q}_accuracy"]) / 100 for q in range(5)]
    rng = np.random.default_rng(0)
    out = dict(published=published, variants={}, start_scan={})
    for q in range(5):
        y = ((Y >> q) & 1).astype(int)
        I, Q = mix(X, B.FREQ_READOUT[q])
        perm = rng.permutation(len(y))
        n = int(0.7 * len(y))
        tr, te = perm[:n], perm[n:]
        for name, (a, b) in dict(causal_lp=(sosfilt(sos, I, axis=1), sosfilt(sos, Q, axis=1)),
                                 zero_phase_lp=(sosfiltfilt(sos, I, axis=1), sosfiltfilt(sos, Q, axis=1)),
                                 no_lp=(I, Q)).items():
            row = {}
            for T, L in TS:
                F = np.c_[np.trapezoid(a[:, :L + 1], axis=1), np.trapezoid(b[:, :L + 1], axis=1)]
                c = B.fit_threshold(F[tr], y[tr])
                row[T] = float(B.assignment_fidelity(y[te], c.predict(F[te])))
            out["variants"].setdefault(name, {})[f"Q{q + 1}"] = row
            print(f"Q{q + 1} {name:14s}", " ".join(f"{row[T]:.4f}" for T, _ in TS),
                  " | published", " ".join(f"{published[T][q]:.4f}" for T, _ in TS), flush=True)
        scan = {}
        for T in (600, 1000):
            r = {}
            for s in (0, 50, 100, 200, 300, 400, 500):
                i0, i1 = int(s / 2), int(T / 2)
                F = np.c_[np.trapezoid(I[:, i0:i1 + 1], axis=1), np.trapezoid(Q[:, i0:i1 + 1], axis=1)]
                c = B.fit_threshold(F[tr], y[tr])
                r[s] = float(B.assignment_fidelity(y[te], c.predict(F[te])))
            scan[T] = r
        out["start_scan"][f"Q{q + 1}"] = scan
    out["by_file"] = by_file(X, Y, sos)
    (XD / "linear_filter_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
