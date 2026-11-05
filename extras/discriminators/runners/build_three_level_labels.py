"""
build_three_level_labels.py
===========================
Derive per-qubit three-level (|0>/|1>/|2>) ground-truth labels from the real
two-level dataset, following Mude et al. (2025): the rare, naturally-occurring
leaked state |2> shows up as traces that sit FAR from both computational
clusters in the per-trace mean-IQ (Mean Trace Value, MTV) plane.

Labeler (semi-supervised, NOT pure unsupervised clustering). A plain 3-way
Gaussian mixture allocates components by mass and cannot isolate a <1% cluster
- it just splits the two dominant |0>/|1> blobs (we measured 6-22% "leaked").
A "distance from both centroids" rule is also wrong: it conflates genuine |2>
with T1 decay (prepared |1> relaxing to |0> mid-readout), which smears ALONG
the |0>-|1> axis and also lands "far from both" (we saw qubit 2 at 4% with a
prepared-bit mean of 0.95 - almost all from |1>, the T1 signature).

Instead: base labels = the KNOWN prepared bits (exact, matches the 2-level
benchmark, no clustering / no T1 mislabeling); then OVERRIDE to |2> only the
traces displaced PERPENDICULAR to the |0>-|1> axis - the off-axis leg of the
|0>/|1>/|2> IQ triangle. T1 decay rides along the axis (perp ~ 0) and stays
|1>; only real leakage has large perp:
    c_s    = median(MTV | prepared_bit == s)
    u      = (c1 - c0)/|c1 - c0|
    perp   = |(MTV - c0) - ((MTV - c0).u) u|
    leaked = perp > k * sigma_perp     (sigma_perp = median(perp)/0.6745)
The script prints the perp percentiles per qubit so k can be tuned to land the
paper's ~0.03-1.1% leaked range.

Outputs an HDF5 with y3_train / y3_test, each shape (N, 5) ternary, aligned
row-for-row with the original X_train / X_test. All six models train and
evaluate on these SAME labels so the 3-level comparison is consistent.

Run inside the container:
    python -m runners.build_three_level_labels \
        --out /data/five_qubit_data/three_level_labels.hdf5 \
        --k 3.0 --plots
"""

from __future__ import annotations

import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import argparse
import os
import sys

import h5py
import numpy as np
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainers.train_HERQULES import demodulate_all_qubits

NUM_QUBITS  = 5
MAX_LENGTH  = 500
SAMPLE_SEED = 42

RAW_TRAIN_FILE = os.environ.get(
    "HERQULES_TRAIN_FILE", str(RAW_TRAIN))
RAW_TEST_FILE = os.environ.get(
    "HERQULES_TEST_FILE", str(RAW_TEST))


def _load(path, is_test):
    key = "test" if is_test else "train"
    with h5py.File(path, "r") as hf:
        X = hf[f"X_{key}"][:, :MAX_LENGTH, :]
        y = hf[f"y_{key}"][:]
    logger.info(f"Loaded {key}: X={X.shape} y={y.shape}")
    return X, y


def _mtv(demod_q: np.ndarray) -> np.ndarray:
    """Mean Trace Value: time-average of the demodulated IQ -> (N, 2)."""
    return demod_q.mean(axis=1).astype(np.float64)


def _label_qubit(mtv: np.ndarray, prepared_bit: np.ndarray, k: float):
    """Semi-supervised ternary labeling of one qubit's MTV.

    Base labels are the KNOWN prepared bits (exact, no clustering error, matches
    the 2-level benchmark). We then override to |2> only the traces displaced
    PERPENDICULAR to the |0>-|1> axis - the off-axis leg of the |0>/|1>/|2> IQ
    triangle. This is the crucial distinction from "far from both centroids":
    a T1 decay (prepared |1> relaxing to |0> mid-readout) smears ALONG the axis,
    so its perpendicular component is ~0 and it stays labeled |1>; genuine |2>
    leakage sits OFF the axis and is the only thing with large perp.

        c_s    = median(MTV | prepared_bit == s)      # robust to outliers
        u      = (c1 - c0)/|c1 - c0|                   # axis unit vector
        perp   = |(MTV - c0) - ((MTV - c0).u) u|       # distance from the axis
        leaked = perp > k * sigma_perp                 # sigma_perp from median|perp|

    sigma_perp = median(perp)/0.6745 (folded-normal robust std). It is set by
    the cluster width PERPENDICULAR to the axis, so the axial T1 tail does not
    inflate it. For a Gaussian the off-axis tail is 2*(1-Phi(k)) (k=3 -> 0.27%,
    k=3.5 -> 0.047%, k=4 -> 0.006%) PLUS the genuine |2> blob.

    Returns (labels (N,) int8 in {0,1,2}, info dict).
    """
    m0, m1 = prepared_bit == 0, prepared_bit == 1
    c0 = np.median(mtv[m0], axis=0)
    c1 = np.median(mtv[m1], axis=0)
    axis = c1 - c0
    sep = float(np.linalg.norm(axis))
    u = axis / (sep or 1e-12)

    rel = mtv - c0
    t = rel @ u                                    # projection along the axis
    perp = np.linalg.norm(rel - np.outer(t, u), axis=1)  # distance from axis
    sigma_perp = float(np.median(perp)) / 0.6745 or 1e-12

    leaked = perp > k * sigma_perp
    labels = prepared_bit.astype(np.int8)          # base = known prepared bit
    labels[leaked] = 2                             # override off-axis -> |2>

    # Diagnostics. perp/sigma_perp percentiles say where to set k. For leaked
    # traces we also report the along-axis projection t/sep: genuine |2> sits at
    # a distinct off-axis location, NOT strung along 0<t<1 like a T1 smear.
    zperp = perp / sigma_perp
    pct = {f"p{p}": float(np.percentile(zperp, p)) for p in (99, 99.9, 99.99)}
    if leaked.any():
        leaked_prepared_mean = float(prepared_bit[leaked].mean())
        leaked_t_med = float(np.median(t[leaked]) / (sep or 1e-12))
        leaked_perp_med = float(np.median(zperp[leaked]))
        c2 = np.median(mtv[leaked], axis=0).tolist()
    else:
        leaked_prepared_mean = leaked_t_med = leaked_perp_med = float("nan")
        c2 = None
    info = {
        "sizes_by_state": {0: int((labels == 0).sum()),
                           1: int((labels == 1).sum()),
                           2: int((labels == 2).sum())},
        "leaked_prepared_bit_mean": leaked_prepared_mean,
        "separation": sep,
        "sigma_perp": sigma_perp,
        "leaked_t_over_sep_med": leaked_t_med,
        "leaked_perp_z_med": leaked_perp_med,
        "zperp_pct": pct,
        "centroids": {"c0": c0.tolist(), "c1": c1.tolist(), "c2": c2},
    }
    return labels, info


def _plot_qubit(q, mtv, labels, out_dir, tag):
    import matplotlib.pyplot as plt
    sub = np.random.RandomState(0).choice(mtv.shape[0], size=min(8000, mtv.shape[0]), replace=False)
    fig, ax = plt.subplots(figsize=(5, 5))
    for s, c, name in [(0, "C0", "|0>"), (1, "C1", "|1>"), (2, "C3", "|2>")]:
        m = labels[sub] == s
        ax.scatter(mtv[sub][m, 0], mtv[sub][m, 1], s=4, alpha=0.3, color=c, label=name)
    ax.set_title(f"Qubit {q} MTV clusters ({tag})")
    ax.set_xlabel("mean I"); ax.set_ylabel("mean Q"); ax.legend()
    p = os.path.join(out_dir, f"three_level_clusters_q{q}_{tag}.png")
    plt.tight_layout(); plt.savefig(p, dpi=110); plt.close(fig)
    logger.info(f"  saved {p}")


def run(out_path: str, make_plots: bool, k: float):
    Xtr, ytr = _load(RAW_TRAIN_FILE, is_test=False)
    Xte, yte = _load(RAW_TEST_FILE, is_test=True)
    n_tr = Xtr.shape[0]

    # Label on the combined pool (more |2> examples), then split labels back.
    X_all = np.concatenate([Xtr, Xte], axis=0)
    y_all = np.concatenate([ytr, yte], axis=0).astype(np.int64)
    del Xtr, Xte

    logger.info("Demodulating combined pool (per qubit)...")
    demod = demodulate_all_qubits(X_all)
    del X_all

    y3 = np.zeros((y_all.shape[0], NUM_QUBITS), dtype=np.int8)
    logger.info("=" * 64)
    logger.info(f"Semi-supervised |2> labeling, k = {k} * sigma_perp (off-axis)")
    for q in range(NUM_QUBITS):
        mtv = _mtv(demod[q])
        prepared_bit = ((y_all >> q) & 1).astype(np.float64)
        labels, info = _label_qubit(mtv, prepared_bit, k)
        y3[:, q] = labels
        pct = info["zperp_pct"]
        logger.info(
            f"Qubit {q}: |0|1|2 = {info['sizes_by_state']}  "
            f"leaked-prepared-bit mean = {info['leaked_prepared_bit_mean']:.3f}"
        )
        logger.info(
            f"         sep={info['separation']:.4f} sigma_perp={info['sigma_perp']:.4f}  "
            f"leaked: t/sep_med={info['leaked_t_over_sep_med']:.2f} "
            f"perp_z_med={info['leaked_perp_z_med']:.2f}  "
            f"(off-axis => |2>; t/sep in 0..1 + low perp_z => T1 leftover)"
        )
        logger.info(
            f"         perp/sigma_perp pct: p99={pct['p99']:.2f} "
            f"p99.9={pct['p99.9']:.2f} p99.99={pct['p99.99']:.2f}  "
            f"(raise --k above these to shrink |2>)"
        )
        if make_plots:
            _plot_qubit(q, mtv, labels, os.path.dirname(os.path.abspath(out_path)) or ".", "pool")
    logger.info("=" * 64)

    y3_tr, y3_te = y3[:n_tr], y3[n_tr:]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("y3_train", data=y3_tr)
        f.create_dataset("y3_test", data=y3_te)
        f.create_dataset("y_train", data=ytr)   # original 2-level labels, for reference
        f.create_dataset("y_test", data=yte)
        f.attrs["num_qubits"] = NUM_QUBITS
        f.attrs["train_file"] = RAW_TRAIN_FILE
        f.attrs["test_file"] = RAW_TEST_FILE
    logger.info(f"Saved ternary labels: {out_path}  (y3_train={y3_tr.shape}, y3_test={y3_te.shape})")
    # Per-qubit leaked fraction summary
    for q in range(NUM_QUBITS):
        frac = float((y3[:, q] == 2).mean())
        logger.info(f"  qubit {q}: leaked fraction = {frac*100:.3f}%  ({int((y3[:,q]==2).sum())} traces)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    p.add_argument("--out", required=True, help="Output HDF5 for ternary labels.")
    p.add_argument("--k", type=float, default=3.0,
                   help="A trace is labeled |2> iff its PERPENDICULAR distance "
                        "from the |0>-|1> axis exceeds k * sigma_perp. Isolates "
                        "off-axis leakage from on-axis T1 decay. Higher => fewer "
                        "|2> (Gaussian off-axis tail: k=3 ~0.27%%, 3.5 ~0.05%%, "
                        "4 ~0.006%%). Tune from the logged perp percentiles.")
    p.add_argument("--plots", action="store_true", help="Save per-qubit MTV cluster plots.")
    args = p.parse_args()
    run(args.out, args.plots, args.k)
