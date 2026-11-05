"""
train_three_level_herqules.py
=============================
HERQULES on the 3-level (qutrit) task - the HEADLINE comparison. We keep the
HERQULES feature pipeline EXACTLY as the authors designed it:

  * per-qubit Butterworth demodulation,
  * BINARY (|0> vs |1>) matched filter (MF) + relaxation MF (RMF) envelopes
    computed from the prepared bits  ->  a 10-D feature vector (5 MF + 5 RMF).

The ONLY change is the classifier head: Net_rmf's 10->10->20->32 trunk is
widened to 10->10->20->243 (3^5 joint ternary states), mirroring the 2-level
joint-32 setup. Everything HERQULES knows about a trace is still those 10 binary
matched-filter scalars - which carry no |2> information and only ~5 bits of
|0>/|1> sign. This is precisely the stress test: can hand-built binary MF
features + a tiny net handle a third state?

Features are z-scored before training (a linear rescale - adds no information -
that just keeps the tiny net's optimisation stable; the established fair-
comparison choice from the synthetic sweeps).

Reports per-qubit ternary accuracy + F_5Q (headline, same metric as 2-level) and
balanced/confusion diagnostics so we can SEE the failure mode (e.g. every |2>
called |1>, or |0>/|1> themselves degrading under the 243-way head).

Run inside the container, from /app/Discriminators:
    python -m runners.train_three_level_herqules \
        --labels /data/five_qubit_data/three_level_labels.hdf5 \
        --length 500 --epochs 80 \
        --out /app/optimization_reports/herqules_3level.json
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
import torch
import torch.nn as nn
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainers.train_HERQULES import (
    build_features, compute_all_envelopes, demodulate_all_qubits,
)
from runners.three_level_common import (
    LABELS_FILE, MAX_LENGTH, NUM_LEVELS, NUM_QUBITS, SAMPLE_SEED,
    confusion_per_qubit, decode_base3, encode_base3, f5q, fit_classifier,
    load_raw_and_labels, make_val_split, per_qubit_balanced_fidelity,
    per_qubit_fidelity, per_qubit_leakage_recall, predict_logits,
)

N_CLASSES = NUM_LEVELS ** NUM_QUBITS  # 243


class Net_rmf3(nn.Module):
    """HERQULES Net_rmf with an identical 10->10->20 trunk and the head widened
    to N_CLASSES (243). Same binary MF/RMF inputs - that is the whole point."""

    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.hid1 = nn.Linear(10, 10)
        self.hid2 = nn.Linear(10, 20)
        self.oupt = nn.Linear(20, n_classes)
        for lyr in (self.hid1, self.hid2, self.oupt):
            nn.init.xavier_uniform_(lyr.weight)
            nn.init.zeros_(lyr.bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        z = self.relu(self.hid1(x))
        z = self.relu(self.hid2(z))
        return self.oupt(z)


def run(labels_path, length, epochs, max_train, max_test, out_path, lr, batch_size):
    Xtr, y3tr, y2tr, Xte, y3te, y2te = load_raw_and_labels(
        labels_path, trace_length=length, max_train=max_train,
        max_test=max_test, with_2level=True)

    logger.info("HERQULES: demodulating train traces (per qubit)...")
    demod_tr = demodulate_all_qubits(Xtr)
    logger.info("HERQULES: demodulating test traces...")
    demod_te = demodulate_all_qubits(Xte)
    del Xtr, Xte

    # BINARY MF/RMF envelopes from the 2-level prepared bits - HERQULES as built.
    # We compute the envelopes from PURE prepared traces only (no |2> on any
    # qubit), per HERQULES's design intent: its preclassifier accepts traces
    # near the per-qubit |0>/|1> centroids and uses them to fit the Wiener MF.
    # At natural ~1% leakage including the |2> traces is harmless; at the
    # elevated rates produced by inflate_leakage.py (5-26% per qubit), those
    # off-axis traces pull the centroids enough to collapse the preclassifier's
    # radius test on low-fidelity qubits (q1: pure|0>=0, pure|1>=0). Excluding
    # |2> rows fixes that AND is the right thing to do anyway.
    np.random.seed(SAMPLE_SEED)  # deterministic class-balancing in _get_mf_core
    pure_mask = ~(y3tr == 2).any(axis=1)
    n_pure = int(pure_mask.sum())
    logger.info(f"HERQULES: computing binary MF/RMF envelopes on {n_pure} pure "
                f"prepared traces (excluded {y3tr.shape[0] - n_pure} |2>-containing rows)")
    demod_pure = {q: demod_tr[q][pure_mask] for q in range(NUM_QUBITS)}
    y2_pure = y2tr[pure_mask]

    # Pre-flight diagnostic: replicate preclassify_qubit's geometry per qubit so
    # we can SEE which qubits will fail before _get_mf_core throws on empty.
    logger.info(f"PRE-FLIGHT  y2_pure range=[{int(y2_pure.min())}, {int(y2_pure.max())}]")
    for q in range(NUM_QUBITS):
        one_state = 2 ** q
        m0 = (y2_pure == 0)
        m1 = (y2_pure == one_state)
        if m0.sum() < 5 or m1.sum() < 5:
            logger.warning(
                f"  q{q+1}: too few prep traces m0={int(m0.sum())} m1={int(m1.sum())}")
            continue
        mtv = demod_pure[q].mean(axis=1)        # (n_pure, 2)
        c0 = mtv[m0].mean(axis=0)
        c1 = mtv[m1].mean(axis=0)
        sep = float(np.linalg.norm(c1 - c0))
        d0 = np.linalg.norm(mtv[m0] - c0, axis=1)
        d1 = np.linalg.norm(mtv[m1] - c1, axis=1)
        r = 0.5 * sep
        logger.info(
            f"  q{q+1}: m0={int(m0.sum())} m1={int(m1.sum())} "
            f"sep={sep:.4f} radius={r:.4f}  "
            f"d0_med={np.median(d0):.4f} d1_med={np.median(d1):.4f}  "
            f"pure_in0={int((d0 < r).sum())} pure_in1={int((d1 < r).sum())}  "
            f"c0=({c0[0]:.3f},{c0[1]:.3f}) c1=({c1[0]:.3f},{c1[1]:.3f})")

    mf_envs, rmf_envs = compute_all_envelopes(demod_pure, y2_pure)
    Ftr = build_features(demod_tr, mf_envs, rmf_envs, length)  # (N, 10)
    Fte = build_features(demod_te, mf_envs, rmf_envs, length)

    # z-score (linear; stabilises the tiny net's training, adds no information).
    mu = Ftr.mean(axis=0, keepdims=True)
    sd = Ftr.std(axis=0, keepdims=True) + 1e-9
    Ftr = ((Ftr - mu) / sd).astype(np.float32)
    Fte = ((Fte - mu) / sd).astype(np.float32)

    yj_tr = encode_base3(y3tr)
    tr_idx, val_idx = make_val_split(Ftr.shape[0], val_frac=0.1)
    logger.info(f"Net_rmf3 (10 -> 10 -> 20 -> {N_CLASSES}); "
                f"train={len(tr_idx)} val={len(val_idx)} test={Fte.shape[0]}")

    model = Net_rmf3()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"HERQULES Net_rmf3 params = {n_params}")
    fit_classifier(
        model, Ftr[tr_idx], yj_tr[tr_idx], Ftr[val_idx], yj_tr[val_idx],
        epochs=epochs, batch_size=batch_size, lr=lr, patience=15, log_tag="HERQULES",
    )

    logits = predict_logits(model, Fte)
    y3_pred = decode_base3(logits.argmax(axis=1))

    fid = per_qubit_fidelity(y3te, y3_pred)
    bal = per_qubit_balanced_fidelity(y3te, y3_pred)
    leak = per_qubit_leakage_recall(y3te, y3_pred)
    cms = confusion_per_qubit(y3te, y3_pred)
    F5, F5_bal = f5q(fid), f5q(bal)

    logger.info("=" * 64)
    logger.info(f"HERQULES 3-level | per-qubit acc = {['%.4f' % x for x in fid]}")
    logger.info(f"                 | per-qubit bal = {['%.4f' % x for x in bal]}")
    logger.info(f"                 | |2> recall    = {['%.4f' % x for x in leak]}")
    logger.info(f"                 | F_5Q (acc) = {F5:.4f}   F_5Q (balanced) = {F5_bal:.4f}")
    for q in range(NUM_QUBITS):
        logger.info(f"   q{q} confusion (rows=true 0/1/2):\n{cms[q]}")
    logger.info("=" * 64)

    result = {
        "model": "HERQULES_Net_rmf", "level": 3, "length": length,
        "n_params": int(n_params),
        "per_qubit_acc": fid, "per_qubit_balanced": bal,
        "per_qubit_leakage_recall": leak,
        "f5q_acc": F5, "f5q_balanced": F5_bal,
        "confusion_per_qubit": cms.tolist(),
        "n_train": int(len(tr_idx)), "n_test": int(Fte.shape[0]),
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
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--out", default=None, help="Output JSON for the metrics.")
    args = p.parse_args()
    run(args.labels, args.length, args.epochs, args.max_train, args.max_test,
        args.out, args.lr, args.batch_size)
