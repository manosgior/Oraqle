"""
train_three_level_klinq.py
==========================
KLiNQ student on the 3-level task. Per-qubit model (5 trained independently).
Its compact input is [binned time-averaged demod IQ | binary matched-filter
scalar] (= 2*n_bins + 1 features); at 2-level the head is a single binary logit,
here widened to NUM_LEVELS (fc3: 8 -> 3, CrossEntropy).

NOTE: we train the student directly on hard ternary labels (no teacher / no
knowledge distillation) - the same hard-label regime as every other survey
model, which keeps the 6-way comparison uniform. The MF scalar is the binary
(|0> vs |1>) HERQULES matched filter, so it is |2>-blind; but the binned average
RETAINS the off-axis |2> displacement, so the student can still pick up some |2>
(a useful intermediate between HERQULES's pure-MF blindness and the fully
trace-based models).

Run inside the container, from /app/Discriminators:
    python -m runners.train_three_level_klinq \
        --labels /data/five_qubit_data/three_level_labels.hdf5 \
        --length 500 --epochs 40 \
        --out /app/optimization_reports/klinq_3level.json
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
from networks.KLiNQ_StudentModel import KLiNQStudentModel
from trainers.train_HERQULES import compute_all_envelopes
from runners.three_level_common import (
    LABELS_FILE, MAX_LENGTH, NUM_LEVELS, NUM_QUBITS, SAMPLE_SEED, demodulate_all_qubits,
    confusion_per_qubit, f5q, fit_classifier, load_raw_and_labels, make_val_split,
    per_qubit_balanced_fidelity, per_qubit_fidelity, per_qubit_leakage_recall,
    predict_logits,
)

TARGET_LENGTH = 5  # bins for the time-averaged summary (paper convention)


class KLiNQ3(KLiNQStudentModel):
    """KLiNQ student with the single binary logit replaced by a NUM_LEVELS-way
    head (CrossEntropy). Inherited forward already returns fc3(x), now (N, 3)."""

    def __init__(self, input_size, n_levels=NUM_LEVELS):
        super().__init__(input_size)
        self.fc3 = nn.Linear(8, n_levels)


def _klinq_features(demod_q, length, mf_env_q, bin_size, n_bins):
    """[binned avg I/Q | binary MF scalar] for one qubit -> (N, 2*n_bins + 1)."""
    N = demod_q.shape[0]
    binned = (demod_q[:, :n_bins * bin_size, :]
              .reshape(N, n_bins, bin_size, 2).mean(axis=2)
              .reshape(N, 2 * n_bins))
    flat = demod_q[:, :length, :].reshape(N, 2 * length)
    mf_scalar = (flat @ mf_env_q[:2 * length].astype(np.float32)).reshape(N, 1)
    return np.concatenate([binned, mf_scalar], axis=1).astype(np.float32)


def run(labels_path, length, epochs, max_train, max_test, out_path, lr, batch_size):
    Xtr, y3tr, y2tr, Xte, y3te, y2te = load_raw_and_labels(
        labels_path, trace_length=length, max_train=max_train,
        max_test=max_test, with_2level=True)

    logger.info("KLiNQ: demodulating train/test...")
    demod_tr = demodulate_all_qubits(Xtr)
    demod_te = demodulate_all_qubits(Xte)
    del Xtr, Xte

    # Binary MF envelopes (|0> vs |1>), same as HERQULES uses.
    np.random.seed(SAMPLE_SEED)
    logger.info("KLiNQ: computing binary MF envelopes...")
    mf_envs, _ = compute_all_envelopes(demod_tr, y2tr)

    bin_size = max(1, length // TARGET_LENGTH)
    n_bins = length // bin_size
    input_size = 2 * n_bins + 1

    tr_idx, val_idx = make_val_split(y3tr.shape[0], val_frac=0.1)
    y3_pred = np.zeros_like(y3te)
    n_params_each = None

    for q in range(NUM_QUBITS):
        Xq_tr = _klinq_features(demod_tr[q], length, mf_envs[q], bin_size, n_bins)
        Xq_te = _klinq_features(demod_te[q], length, mf_envs[q], bin_size, n_bins)
        # z-norm columns with train stats (heterogeneous feature scales).
        mu = Xq_tr[tr_idx].mean(axis=0, keepdims=True)
        sd = Xq_tr[tr_idx].std(axis=0, keepdims=True) + 1e-9
        Xq_tr = (Xq_tr - mu) / sd
        Xq_te = (Xq_te - mu) / sd

        model = KLiNQ3(input_size=input_size)
        if n_params_each is None:
            n_params_each = sum(p.numel() for p in model.parameters())
        fit_classifier(
            model, Xq_tr[tr_idx], y3tr[tr_idx, q], Xq_tr[val_idx], y3tr[val_idx, q],
            epochs=epochs, batch_size=batch_size, lr=lr, patience=8,
            log_tag=f"KLiNQ-q{q}",
        )
        y3_pred[:, q] = predict_logits(model, Xq_te).argmax(axis=1)

    fid = per_qubit_fidelity(y3te, y3_pred)
    bal = per_qubit_balanced_fidelity(y3te, y3_pred)
    leak = per_qubit_leakage_recall(y3te, y3_pred)
    cms = confusion_per_qubit(y3te, y3_pred)
    F5, F5_bal = f5q(fid), f5q(bal)
    total_params = int(n_params_each * NUM_QUBITS)

    logger.info("=" * 64)
    logger.info(f"KLiNQ 3-level | input_size={input_size} (n_bins={n_bins})")
    logger.info(f"              | per-qubit acc = {['%.4f' % x for x in fid]}")
    logger.info(f"              | per-qubit bal = {['%.4f' % x for x in bal]}")
    logger.info(f"              | |2> recall    = {['%.4f' % x for x in leak]}")
    logger.info(f"              | F_5Q (acc) = {F5:.4f}   F_5Q (balanced) = {F5_bal:.4f}")
    for q in range(NUM_QUBITS):
        logger.info(f"   q{q} confusion (rows=true 0/1/2):\n{cms[q]}")
    logger.info("=" * 64)

    result = {
        "model": "KLiNQ_Student", "level": 3, "length": length,
        "input_size": int(input_size), "n_params": total_params,
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
