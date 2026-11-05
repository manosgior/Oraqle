"""
inflate_leakage.py
==================
Artificially elevate the per-qubit |2> rate in the REAL labeled dataset to a
target value, by oversampling |2>-containing rows (with Gaussian noise jitter on
the raw IQ to avoid exact-copy memorization) and uniformly evicting non-|2> rows
to preserve total dataset size.

This is the "real-data inflation" stress test (Path B). At the natural prevalence
(~1%/qubit on average) the 2-level SOTA ranking is preserved on plain accuracy -
the leakage gap shows only on balanced fidelity / |2> recall. Elevating |2> to
e.g. 10-25% makes the gap show up on PLAIN accuracy too: HERQULES is structurally
blind to |2>, so missing X% of test traces docks accuracy by ~X%, while
trace-based models hold.

Per-qubit complication, flagged honestly: a row carries 5 labels, so duplicating
a row where (say) q2 is |2> also duplicates whatever the other qubits are doing.
Per-qubit |2> rates scale UNIFORMLY by the multiplier K = target / current_mean
(each qubit's pre-inflation rate is multiplied by K). Qubit rates therefore stay
PROPORTIONAL to the original (q0 stays low, q2 stays high) - good for fidelity
to real-qubit variation, less good if you wanted equal per-qubit rates.

Layout. Outputs three HDF5 files mirroring the codebase's existing schema:

    <out_dir>/raw_train.hdf5     X_train (N, T, 2),  y_train (N,)  packed 2-level
    <out_dir>/raw_test.hdf5      X_test  (N, T, 2),  y_test  (N,)  packed 2-level
    <out_dir>/labels.hdf5        y3_train (N, 5), y3_test (N, 5),
                                  y_train (N,),    y_test  (N,)

Point the 3-level trainers at them via env vars + --labels:

    HERQULES_TRAIN_FILE=<out_dir>/raw_train.hdf5 \
    HERQULES_TEST_FILE=<out_dir>/raw_test.hdf5 \
    python -m runners.train_three_level_cnn \
        --labels <out_dir>/labels.hdf5 ...

    HERQULES_TRAIN_FILE=<out_dir>/raw_train.hdf5 \
    HERQULES_TEST_FILE=<out_dir>/raw_test.hdf5 \
    python -m runners.train_three_level_herqules \
        --labels <out_dir>/labels.hdf5 ...

Run inside the container, from /app/Discriminators:
    python -m runners.inflate_leakage \
        --in-labels /data/five_qubit_data/three_level_labels.hdf5 \
        --target-leak 0.10 --jitter 0.10 \
        --out-dir /data/five_qubit_data/inflated_p10
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

NUM_QUBITS = 5
SAMPLE_SEED = 42

RAW_TRAIN_FILE = os.environ.get(
    "HERQULES_TRAIN_FILE", str(RAW_TRAIN))
RAW_TEST_FILE = os.environ.get(
    "HERQULES_TEST_FILE", str(RAW_TEST))
LABELS_FILE = os.environ.get(
    "THREE_LEVEL_LABELS", str(THREE_LEVEL_LABELS))


def _per_qubit_leak(y3):
    return np.array([(y3[:, q] == 2).mean() for q in range(NUM_QUBITS)])


def _channel_std(X, rng):
    """Per-channel (I, Q) std for setting the noise-jitter scale, from a fast
    random subsample of X (avoiding the full 6+ GB sweep)."""
    n = min(50_000, X.shape[0])
    idx = np.sort(rng.choice(X.shape[0], size=n, replace=False))
    return X[idx].astype(np.float32).std(axis=(0, 1))  # (2,)


def _inflate_split(X, y2, y3, target_mean_leak, jitter, rng, split_name):
    """Inflate one (X, y2, y3) split to a target mean per-qubit |2> rate.

    Identifies leaked rows (y3 == 2 on any qubit), keeps all of them, samples
    K-1 extra copies from them with jitter*sigma Gaussian noise on the raw IQ,
    and uniformly subsamples non-|2> rows to keep the total size constant.
    """
    n = X.shape[0]
    mask_leak = (y3 == 2).any(axis=1)
    n_leak = int(mask_leak.sum())
    if n_leak == 0:
        raise ValueError(f"{split_name}: no |2> rows in input labels.")

    cur_pq = _per_qubit_leak(y3)
    cur_mean = float(cur_pq.mean())
    K = target_mean_leak / cur_mean
    if K < 1.0:
        raise ValueError(
            f"{split_name}: target_mean_leak {target_mean_leak:.4f} is BELOW the "
            f"current {cur_mean:.4f}; nothing to inflate. Pick a higher target.")

    # Final count of leaked-row instances; per-qubit rates scale by ~K.
    n_leak_final = int(round(K * n_leak))
    n_dup = n_leak_final - n_leak
    n_clean_keep = max(0, n - n_leak_final)

    leak_idx = np.flatnonzero(mask_leak)
    clean_idx = np.flatnonzero(~mask_leak)
    if n_clean_keep > clean_idx.size:
        # Not enough clean rows to fill - dataset will end up smaller than n.
        n_clean_keep = clean_idx.size

    kept_clean = rng.choice(clean_idx, size=n_clean_keep, replace=False)
    dup_src = rng.choice(leak_idx, size=n_dup, replace=True)

    sigma = _channel_std(X, rng)
    logger.info(f"  {split_name}: per-channel sigma=({sigma[0]:.3f}, {sigma[1]:.3f}); "
                f"jitter={jitter} * sigma")

    # Duplicates with Gaussian jitter on raw IQ (preserves the underlying signal,
    # adds a new noise realization so the model can't exactly memorize).
    X_dup = X[dup_src].astype(np.float32).copy()
    if jitter > 0:
        noise = rng.normal(0.0, 1.0, size=X_dup.shape).astype(np.float32)
        X_dup[:, :, 0] += jitter * sigma[0] * noise[:, :, 0]
        X_dup[:, :, 1] += jitter * sigma[1] * noise[:, :, 1]

    X_out = np.concatenate([X[leak_idx], X_dup, X[kept_clean]], axis=0)
    y2_out = np.concatenate([y2[leak_idx], y2[dup_src], y2[kept_clean]], axis=0)
    y3_out = np.concatenate([y3[leak_idx], y3[dup_src], y3[kept_clean]], axis=0)

    perm = rng.permutation(X_out.shape[0])
    X_out, y2_out, y3_out = X_out[perm], y2_out[perm], y3_out[perm]

    new_pq = _per_qubit_leak(y3_out)
    logger.info(
        f"  {split_name}: n {n}->{X_out.shape[0]} (n_leak {n_leak}->{n_leak_final}, "
        f"K={K:.2f})\n"
        f"    per-qubit |2> rate:  before = {np.round(cur_pq * 100, 3)}%\n"
        f"                         after  = {np.round(new_pq * 100, 3)}%  "
        f"(mean {cur_mean*100:.3f}% -> {new_pq.mean()*100:.3f}%)"
    )
    return X_out, y2_out, y3_out


def _load_split(raw_path, labels_path, split):
    """Load X (N, T, 2), y2 (N,) packed, y3 (N, 5) for one split."""
    key = "train" if split == "train" else "test"
    with h5py.File(raw_path, "r") as f:
        X = f[f"X_{key}"][:]
        y2 = f[f"y_{key}"][:]
    with h5py.File(labels_path, "r") as f:
        y3 = f[f"y3_{key}"][:]
    if y3.shape[0] != X.shape[0]:
        raise ValueError(
            f"{split}: raw rows {X.shape[0]} != y3 rows {y3.shape[0]}; "
            f"labels file was not built from this raw file.")
    return X, y2, y3


def _write_raw(out_path, X, y, split):
    key = "train" if split == "train" else "test"
    with h5py.File(out_path, "w") as f:
        # Match input raw layout: X_{split} (N, T, 2), y_{split} (N,) packed.
        # gzip-4 keeps file size manageable without crushing write throughput.
        f.create_dataset(
            f"X_{key}", data=X.astype(np.float32),
            chunks=(min(1024, X.shape[0]), X.shape[1], X.shape[2]),
            compression="gzip", compression_opts=4)
        f.create_dataset(f"y_{key}", data=y)
    logger.info(f"  wrote {out_path}  ({os.path.getsize(out_path) / 1e9:.2f} GB)")


def _write_labels(out_path, y3_tr, y3_te, y2_tr, y2_te):
    with h5py.File(out_path, "w") as f:
        f.create_dataset("y3_train", data=y3_tr)
        f.create_dataset("y3_test", data=y3_te)
        f.create_dataset("y_train", data=y2_tr)
        f.create_dataset("y_test", data=y2_te)
    logger.info(f"  wrote {out_path}")


def run(in_labels, target_leak, jitter, out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    logger.info(f"Inflating to target mean per-qubit |2> rate = {target_leak*100:.2f}%  "
                f"(jitter = {jitter} * per-channel sigma)")

    logger.info(f"Loading TRAIN from raw {RAW_TRAIN_FILE} + labels {in_labels} ...")
    Xtr, y2tr, y3tr = _load_split(RAW_TRAIN_FILE, in_labels, "train")
    logger.info(f"  shapes: X={Xtr.shape} y2={y2tr.shape} y3={y3tr.shape}")
    Xtr_i, y2tr_i, y3tr_i = _inflate_split(
        Xtr, y2tr, y3tr, target_leak, jitter, rng, "train")
    del Xtr, y2tr, y3tr

    logger.info(f"Loading TEST from raw {RAW_TEST_FILE} + labels {in_labels} ...")
    Xte, y2te, y3te = _load_split(RAW_TEST_FILE, in_labels, "test")
    logger.info(f"  shapes: X={Xte.shape} y2={y2te.shape} y3={y3te.shape}")
    Xte_i, y2te_i, y3te_i = _inflate_split(
        Xte, y2te, y3te, target_leak, jitter, rng, "test")
    del Xte, y2te, y3te

    raw_train_out = os.path.join(out_dir, "raw_train.hdf5")
    raw_test_out = os.path.join(out_dir, "raw_test.hdf5")
    labels_out = os.path.join(out_dir, "labels.hdf5")

    logger.info("Writing outputs...")
    _write_raw(raw_train_out, Xtr_i, y2tr_i, "train")
    _write_raw(raw_test_out, Xte_i, y2te_i, "test")
    _write_labels(labels_out, y3tr_i, y3te_i, y2tr_i, y2te_i)

    logger.info("=" * 64)
    logger.info("Done. Point the 3-level trainers at these via:")
    logger.info(f"  HERQULES_TRAIN_FILE={raw_train_out}")
    logger.info(f"  HERQULES_TEST_FILE={raw_test_out}")
    logger.info(f"  --labels {labels_out}")
    logger.info("=" * 64)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    p.add_argument("--in-labels", default=LABELS_FILE,
                   help=f"Original three_level_labels.hdf5 (row-aligned with "
                        f"the raw files pointed to by HERQULES_TRAIN_FILE/"
                        f"TEST_FILE). Default: {LABELS_FILE} (override globally "
                        "via the THREE_LEVEL_LABELS env var).")
    p.add_argument("--target-leak", type=float, required=True,
                   help="Target MEAN per-qubit |2> rate after inflation, e.g. "
                        "0.10 = 10%%. Per-qubit rates scale by K = target / current_mean.")
    p.add_argument("--jitter", type=float, default=0.10,
                   help="Gaussian noise added to duplicated raw IQ, in units of "
                        "the per-channel std. Default 0.1 = 10%% of sigma -- "
                        "breaks exact-copy memorization while staying within the "
                        "cluster. Set to 0 for pure duplication (not recommended).")
    p.add_argument("--out-dir", required=True,
                   help="Directory for raw_train.hdf5 / raw_test.hdf5 / labels.hdf5.")
    p.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = p.parse_args()
    run(args.in_labels, args.target_leak, args.jitter, args.out_dir, args.seed)
