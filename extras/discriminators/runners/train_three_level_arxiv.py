"""
train_three_level_arxiv.py
==========================
Arxiv:2406.18807 FNN on the 3-level task. Per-qubit model (5 trained
independently, then assembled). At 2-level it maps the time-averaged 2-D mean-IQ
point -> binary; here the head becomes 3-way (fc3: 4 -> 3, logits + CE).

KEY CONTRAST with HERQULES: this model's input is the raw 2-D mean-IQ, which
RETAINS the off-axis |2> displacement (it is literally the quantity we cluster
to label |2>). HERQULES instead projects the trace onto the binary matched-
filter axis, discarding the perpendicular component - which is why it cannot see
|2>. So this "averaging" model should still recover |2>, while HERQULES does not.

Run inside the container, from /app/Discriminators:
    python -m runners.train_three_level_arxiv \
        --labels /data/five_qubit_data/three_level_labels.hdf5 \
        --length 500 --epochs 40 \
        --out /app/optimization_reports/arxiv_3level.json
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
import torch.nn as nn
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.Qubic import Arxiv240618807FNN
from runners.three_level_common import (
    LABELS_FILE, MAX_LENGTH, NUM_LEVELS, NUM_QUBITS, demodulate_all_qubits,
    confusion_per_qubit, f5q, fit_classifier, load_raw_and_labels, make_val_split,
    per_qubit_balanced_fidelity, per_qubit_fidelity, per_qubit_leakage_recall,
    predict_logits,
)


class Arxiv3(Arxiv240618807FNN):
    """Arxiv FNN with the binary sigmoid head replaced by a NUM_LEVELS-way
    logit head (CrossEntropy). 2 -> 8 -> 4 -> n_levels."""

    def __init__(self, n_levels=NUM_LEVELS):
        super().__init__()
        self.fc3 = nn.Linear(4, n_levels)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)  # raw logits (no sigmoid)


def run(labels_path, length, epochs, max_train, max_test, out_path, lr, batch_size):
    Xtr, y3tr, Xte, y3te = load_raw_and_labels(
        labels_path, trace_length=length, max_train=max_train, max_test=max_test)

    logger.info("Arxiv: demodulating train/test (mean-IQ per qubit)...")
    demod_tr = demodulate_all_qubits(Xtr)
    demod_te = demodulate_all_qubits(Xte)
    del Xtr, Xte

    tr_idx, val_idx = make_val_split(y3tr.shape[0], val_frac=0.1)
    y3_pred = np.zeros_like(y3te)
    n_params_each = None

    for q in range(NUM_QUBITS):
        # Time-averaged 2-D mean-IQ point for this qubit.
        Xq_tr = demod_tr[q].mean(axis=1)   # (N, 2)
        Xq_te = demod_te[q].mean(axis=1)
        # Min-max normalise to [0, 1] per column with TRAIN stats (paper recipe).
        lo = Xq_tr[tr_idx].min(axis=0, keepdims=True)
        rng = Xq_tr[tr_idx].max(axis=0, keepdims=True) - lo
        rng[rng == 0] = 1e-10
        Xq_tr = (Xq_tr - lo) / rng
        Xq_te = (Xq_te - lo) / rng

        model = Arxiv3()
        if n_params_each is None:
            n_params_each = sum(p.numel() for p in model.parameters())
        fit_classifier(
            model, Xq_tr[tr_idx], y3tr[tr_idx, q], Xq_tr[val_idx], y3tr[val_idx, q],
            epochs=epochs, batch_size=batch_size, lr=lr, patience=8,
            log_tag=f"Arxiv-q{q}",
        )
        y3_pred[:, q] = predict_logits(model, Xq_te).argmax(axis=1)

    fid = per_qubit_fidelity(y3te, y3_pred)
    bal = per_qubit_balanced_fidelity(y3te, y3_pred)
    leak = per_qubit_leakage_recall(y3te, y3_pred)
    cms = confusion_per_qubit(y3te, y3_pred)
    F5, F5_bal = f5q(fid), f5q(bal)
    total_params = int(n_params_each * NUM_QUBITS)

    logger.info("=" * 64)
    logger.info(f"Arxiv 3-level | per-qubit acc = {['%.4f' % x for x in fid]}")
    logger.info(f"              | per-qubit bal = {['%.4f' % x for x in bal]}")
    logger.info(f"              | |2> recall    = {['%.4f' % x for x in leak]}")
    logger.info(f"              | F_5Q (acc) = {F5:.4f}   F_5Q (balanced) = {F5_bal:.4f}")
    for q in range(NUM_QUBITS):
        logger.info(f"   q{q} confusion (rows=true 0/1/2):\n{cms[q]}")
    logger.info("=" * 64)

    result = {
        "model": "Arxiv240618807FNN", "level": 3, "length": length,
        "n_params": total_params,
        "per_qubit_acc": fid, "per_qubit_balanced": bal,
        "per_qubit_leakage_recall": leak,
        "f5q_acc": F5, "f5q_balanced": F5_bal,
        "confusion_per_qubit": cms.tolist(),
        "n_train": int(len(tr_idx)), "n_test": int(y3te.shape[0]),
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
    p.add_argument("--out", default=None, help="Output JSON for the metrics.")
    args = p.parse_args()
    run(args.labels, args.length, args.epochs, args.max_train, args.max_test,
        args.out, args.lr, args.batch_size)
