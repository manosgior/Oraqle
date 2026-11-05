"""
train_three_level_transformer.py
================================
Transformer (ViT-style patch encoder) on the 3-level task. Joint 3^5=243-class
head (mirrors the 2-level joint-32 setup). Input = raw IQ trace (N, L, 2),
per-channel z-normed; the model patches it, attends, and classifies the CLS
token over the 243 joint ternary states.

As a trace-based model it should hold at 3-level (high F_5Q AND recover |2>).

Run inside the container, from /app/Discriminators (GPU strongly recommended):
    python -m runners.train_three_level_transformer \
        --labels /data/five_qubit_data/three_level_labels.hdf5 \
        --length 500 --epochs 30 --max-train 300000 \
        --out /app/optimization_reports/transformer_3level.json
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
from networks.Transformer import QubitClassifierTransformer
from runners.three_level_common import (
    LABELS_FILE, MAX_LENGTH, NUM_LEVELS, NUM_QUBITS,
    confusion_per_qubit, decode_base3, encode_base3, f5q, fit_classifier,
    load_raw_and_labels, make_val_split, per_qubit_balanced_fidelity,
    per_qubit_fidelity, per_qubit_leakage_recall, predict_logits,
)

N_CLASSES = NUM_LEVELS ** NUM_QUBITS  # 243


def run(labels_path, length, patch_size, embedding_dim, num_heads, num_layers,
        epochs, max_train, max_test, out_path, lr, batch_size):
    Xtr, y3tr, Xte, y3te = load_raw_and_labels(
        labels_path, trace_length=length, max_train=max_train, max_test=max_test)

    # Per-channel (I, Q) z-norm with train stats; keep the (N, L, 2) shape the
    # patch embedder expects.
    mean = Xtr.mean(axis=(0, 1), keepdims=True)
    std = Xtr.std(axis=(0, 1), keepdims=True) + 1e-10
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std

    yj_tr = encode_base3(y3tr)
    tr_idx, val_idx = make_val_split(Xtr.shape[0], val_frac=0.1)
    logger.info(f"Transformer(patch={patch_size}, dim={embedding_dim}, "
                f"heads={num_heads}, layers={num_layers} -> {N_CLASSES}); "
                f"train={len(tr_idx)} val={len(val_idx)} test={Xte.shape[0]}")

    model = QubitClassifierTransformer(
        num_classes=N_CLASSES, patch_size=patch_size, embedding_dim=embedding_dim,
        num_heads=num_heads, num_layers=num_layers, dropout=0.1)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Transformer params = {n_params}")
    fit_classifier(
        model, Xtr[tr_idx], yj_tr[tr_idx], Xtr[val_idx], yj_tr[val_idx],
        epochs=epochs, batch_size=batch_size, lr=lr, patience=8, log_tag="Transformer",
    )

    logits = predict_logits(model, Xte)
    y3_pred = decode_base3(logits.argmax(axis=1))

    fid = per_qubit_fidelity(y3te, y3_pred)
    bal = per_qubit_balanced_fidelity(y3te, y3_pred)
    leak = per_qubit_leakage_recall(y3te, y3_pred)
    cms = confusion_per_qubit(y3te, y3_pred)
    F5, F5_bal = f5q(fid), f5q(bal)

    logger.info("=" * 64)
    logger.info(f"Transformer 3-level | per-qubit acc = {['%.4f' % x for x in fid]}")
    logger.info(f"                    | per-qubit bal = {['%.4f' % x for x in bal]}")
    logger.info(f"                    | |2> recall    = {['%.4f' % x for x in leak]}")
    logger.info(f"                    | F_5Q (acc) = {F5:.4f}   F_5Q (balanced) = {F5_bal:.4f}")
    for q in range(NUM_QUBITS):
        logger.info(f"   q{q} confusion (rows=true 0/1/2):\n{cms[q]}")
    logger.info("=" * 64)

    result = {
        "model": "Transformer", "level": 3, "length": length,
        "n_params": int(n_params),
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
    p.add_argument("--patch-size", type=int, default=10)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", default=None, help="Output JSON for the metrics.")
    args = p.parse_args()
    run(args.labels, args.length, args.patch_size, args.embedding_dim,
        args.num_heads, args.num_layers, args.epochs, args.max_train,
        args.max_test, args.out, args.lr, args.batch_size)
