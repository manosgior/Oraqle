"""
train_three_level_fnn.py
========================
3-level (qutrit) FNN benchmark model, and the pipeline SANITY CHECK. Mirrors
the 2-level joint FNN exactly, only ternarized:

    raw multiplexed trace (N, 2*L)  ->  z-norm  ->  SingleQubitFNN(2L, 243)
    ->  243-way softmax  ->  argmax  ->  base-3 decode  ->  per-qubit ternary.

At 2-level this same FNN (joint 32-class) scores ~0.909 mean-5. As a trace-based
model it should "hold" at 3-level (paper ~0.898 F_5Q) - in contrast to HERQULES,
whose tiny Net_rmf collapses on the 243-class joint task. Getting ~0.90 here
validates the labels + harness before we run the other five models.

Headline metric = per-qubit ternary ACCURACY + geometric mean (F_5Q), identical
to the 2-level table. Balanced (macro-recall) fidelity is reported alongside to
show whether |2> is actually recovered (it is rare, so plain accuracy alone can
hide a model that never predicts |2>).

Run inside the container, from /app/Discriminators:
    python -m runners.train_three_level_fnn \
        --labels /data/five_qubit_data/three_level_labels.hdf5 \
        --length 500 --epochs 40 --out /app/optimization_reports/fnn_3level.json
"""

from __future__ import annotations

import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import argparse
import json
import os
import sys

import numpy as np
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.SingleQubitFNN import SingleQubitFNN
from runners.three_level_common import (
    DEVICE, LABELS_FILE, MAX_LENGTH, NUM_LEVELS, NUM_QUBITS,
    confusion_per_qubit, decode_base3, encode_base3, f5q, fit_classifier,
    load_raw_and_labels, make_val_split, per_qubit_balanced_fidelity,
    per_qubit_fidelity, per_qubit_leakage_recall, predict_logits,
)

N_CLASSES = NUM_LEVELS ** NUM_QUBITS  # 243


def run(labels_path, length, epochs, max_train, max_test, out_path,
        class_weight, lr, batch_size):
    Xtr, y3tr, Xte, y3te = load_raw_and_labels(
        labels_path, trace_length=length, max_train=max_train, max_test=max_test)

    # Flatten raw IQ -> (N, 2L), z-norm with TRAIN stats (matches 2-level FNN).
    Xtr = Xtr.reshape(Xtr.shape[0], -1)
    Xte = Xte.reshape(Xte.shape[0], -1)
    mean = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True) + 1e-10
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std

    yj_tr = encode_base3(y3tr)
    yj_te = encode_base3(y3te)

    tr_idx, val_idx = make_val_split(Xtr.shape[0], val_frac=0.1)
    logger.info(f"FNN(2L={2*length} -> {N_CLASSES}); train={len(tr_idx)} "
                f"val={len(val_idx)} test={Xte.shape[0]}")

    cw = None
    if class_weight:
        # Inverse-frequency weights over the 243 joint classes actually seen.
        counts = np.bincount(yj_tr[tr_idx], minlength=N_CLASSES).astype(np.float64)
        cw = np.where(counts > 0, counts.sum() / (counts + 1e-9), 0.0)
        cw = cw / cw[cw > 0].mean()  # normalize so mean weight ~1
        logger.info("Using inverse-frequency class weights over joint classes.")

    model = SingleQubitFNN(input_size=2 * length, output_size=N_CLASSES)
    fit_classifier(
        model, Xtr[tr_idx], yj_tr[tr_idx], Xtr[val_idx], yj_tr[val_idx],
        epochs=epochs, batch_size=batch_size, lr=lr, patience=8,
        class_weight=cw, log_tag="FNN",
    )

    logits = predict_logits(model, Xte)
    y3_pred = decode_base3(logits.argmax(axis=1))

    fid = per_qubit_fidelity(y3te, y3_pred)
    bal = per_qubit_balanced_fidelity(y3te, y3_pred)
    leak = per_qubit_leakage_recall(y3te, y3_pred)
    cms = confusion_per_qubit(y3te, y3_pred)
    F5 = f5q(fid)
    F5_bal = f5q(bal)

    logger.info("=" * 64)
    logger.info(f"FNN 3-level | per-qubit acc = {['%.4f' % x for x in fid]}")
    logger.info(f"            | per-qubit bal = {['%.4f' % x for x in bal]}")
    logger.info(f"            | |2> recall    = {['%.4f' % x for x in leak]}")
    logger.info(f"            | F_5Q (acc) = {F5:.4f}   F_5Q (balanced) = {F5_bal:.4f}")
    for q in range(NUM_QUBITS):
        # Row = true state, so cms[q][2] shows where true |2> traces are sent.
        logger.info(f"   q{q} confusion (rows=true 0/1/2):\n{cms[q]}")
    logger.info("=" * 64)

    result = {
        "model": "FNN", "level": 3, "length": length,
        "per_qubit_acc": fid, "per_qubit_balanced": bal,
        "per_qubit_leakage_recall": leak,
        "f5q_acc": F5, "f5q_balanced": F5_bal,
        "confusion_per_qubit": cms.tolist(),
        "n_train": int(len(tr_idx)), "n_test": int(Xte.shape[0]),
    }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved {out_path}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    p.add_argument("--labels", default=LABELS_FILE,
                   help=f"Ternary labels HDF5. Default: {LABELS_FILE} "
                        "(override globally via the THREE_LEVEL_LABELS env var).")
    p.add_argument("--length", type=int, default=MAX_LENGTH)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--class-weight", action="store_true",
                   help="Inverse-frequency weighting to push the model to "
                        "actually predict the rare |2> joint states.")
    p.add_argument("--out", default=None, help="Output JSON for the metrics.")
    args = p.parse_args()
    run(args.labels, args.length, args.epochs, args.max_train, args.max_test,
        args.out, args.class_weight, args.lr, args.batch_size)
