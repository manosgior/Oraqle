"""
train_three_level_cnn.py
========================
CNN on the 3-level (qutrit) task. The CNN is per-qubit multi-task at 2-level
(5 binary heads, BCE); here each head becomes a 3-way softmax (CrossEntropy),
matching Mude et al.'s per-qubit k-output design and the survey decision to keep
per-qubit heads per-qubit.

Input representation = the same as the 2-level CNN: per-qubit digital
down-conversion -> Butterworth LPF (order 4, 5 MHz) -> 20x downsample ->
(N, T_ds, 5, 2), reshaped to (N, 10, T_ds) for Conv1d. Built straight from the
raw traces (reusing synth_to_cnn._demod_downsample, theta=0 - a retrained CNN
sees both I and Q so the per-qubit calibration rotation is irrelevant).

As a trace-based model the CNN should hold at 3-level (high F_5Q AND recover
|2>), in contrast to HERQULES whose binary matched filters are blind to |2>.

Run inside the container, from /app/Discriminators:
    python -m runners.train_three_level_cnn \
        --labels /data/five_qubit_data/three_level_labels.hdf5 \
        --length 500 --m-param 16 --epochs 40 \
        --out /app/optimization_reports/cnn_3level.json
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
import torch.nn.functional as F
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.CNN import CNN
from networks.HybridCNN import HybridCNN
from runners.synth_to_cnn import _demod_downsample
from helpers.sparse_fir_demod import windowed_sinc_demod
from runners.three_level_common import (
    LABELS_FILE, MAX_LENGTH, NUM_LEVELS, NUM_QUBITS,
    confusion_per_qubit, f5q, fit_multihead, load_raw_and_labels, make_val_split,
    per_qubit_balanced_fidelity, per_qubit_fidelity, per_qubit_leakage_recall,
    predict_logits,
)


class CNN3(CNN):
    """2-level CNN backbone with per-qubit heads widened from 1 logit to
    NUM_LEVELS. forward returns (N, num_qubits, n_levels)."""

    def __init__(self, in_channels=10, m_param=16, num_qubits=NUM_QUBITS,
                 n_levels=NUM_LEVELS):
        super().__init__(in_channels=in_channels, m_param=m_param, num_qubits=num_qubits)
        self.n_levels = n_levels
        self.output_layers = nn.ModuleList(
            [nn.Linear(m_param, n_levels) for _ in range(num_qubits)])

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.res_block(x)
        x = x.mean(dim=2)                       # global average pool over time
        outs = [layer(x) for layer in self.output_layers]  # each (N, n_levels)
        return torch.stack(outs, dim=1)         # (N, num_qubits, n_levels)


def build_cnn_input(X, downsample=20, preprocessing="butter", chunk=50_000):
    """Raw (N, T, 2) -> (N, 10, T_ds) CNN input, chunked to bound memory.

    ``preprocessing="butter"`` (default) reproduces the classic CNN's
    per-qubit Butterworth-LPF + decimate front-end (synth_to_cnn._demod_downsample).
    ``preprocessing="sinc"`` swaps in the windowed-sinc sparse-FIR front-end
    described for the HybridCNN: 101-tap Hamming-windowed sinc, group-delay-
    corrected, evaluated at every ``downsample``-th sample.
    """
    if preprocessing == "butter":
        demod = lambda chunk_X: _demod_downsample(chunk_X, downsample)
    elif preprocessing == "sinc":
        demod = lambda chunk_X: windowed_sinc_demod(chunk_X, downsample=downsample)
    else:
        raise ValueError(f"unknown preprocessing {preprocessing!r}")
    parts = []
    for s in range(0, len(X), chunk):
        parts.append(demod(X[s:s + chunk]))                          # (n, T_ds, 5, 2)
    arr = np.concatenate(parts, axis=0)
    N, T_ds = arr.shape[0], arr.shape[1]
    arr = arr.reshape(N, T_ds, NUM_QUBITS * 2).transpose(0, 2, 1)    # (N, 10, T_ds)
    return arr.astype(np.float32)


def run(labels_path, length, m_param, epochs, downsample, max_train, max_test,
        out_path, lr, batch_size, arch, preprocessing):
    Xtr, y3tr, Xte, y3te = load_raw_and_labels(
        labels_path, trace_length=length, max_train=max_train, max_test=max_test)

    logger.info(f"CNN: preprocessing={preprocessing}, downsample={downsample} (train)...")
    Xtr = build_cnn_input(Xtr, downsample=downsample, preprocessing=preprocessing)
    logger.info(f"CNN: preprocessing={preprocessing}, downsample={downsample} (test)...")
    Xte = build_cnn_input(Xte, downsample=downsample, preprocessing=preprocessing)

    # z-norm per channel using train stats (over samples + time).
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-10
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    tr_idx, val_idx = make_val_split(Xtr.shape[0], val_frac=0.1)
    logger.info(f"arch={arch} (in=10, m={m_param}, T_ds={Xtr.shape[2]}); "
                f"train={len(tr_idx)} val={len(val_idx)} test={Xte.shape[0]}")

    if arch == "classic":
        model = CNN3(in_channels=10, m_param=m_param)
    elif arch == "hybrid":
        # Per-qubit ternary heads (n_levels=3) -> output (batch, 5, 3).
        # HybridCNN takes the same (batch, 10, T_ds) layout the trainer feeds.
        model = HybridCNN(n_channels=10, m_param=m_param, num_qubits=NUM_QUBITS,
                          n_levels=NUM_LEVELS)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{arch} CNN params = {n_params}")
    fit_multihead(
        model, Xtr[tr_idx], y3tr[tr_idx], Xtr[val_idx], y3tr[val_idx],
        epochs=epochs, batch_size=batch_size, lr=lr, patience=10, log_tag="CNN",
    )

    y3_pred = predict_logits(model, Xte).argmax(axis=2)  # (N, 5)

    fid = per_qubit_fidelity(y3te, y3_pred)
    bal = per_qubit_balanced_fidelity(y3te, y3_pred)
    leak = per_qubit_leakage_recall(y3te, y3_pred)
    cms = confusion_per_qubit(y3te, y3_pred)
    F5, F5_bal = f5q(fid), f5q(bal)

    logger.info("=" * 64)
    logger.info(f"CNN[{arch}] 3-level | per-qubit acc = {['%.4f' % x for x in fid]}")
    logger.info(f"            | per-qubit bal = {['%.4f' % x for x in bal]}")
    logger.info(f"            | |2> recall    = {['%.4f' % x for x in leak]}")
    logger.info(f"            | F_5Q (acc) = {F5:.4f}   F_5Q (balanced) = {F5_bal:.4f}")
    for q in range(NUM_QUBITS):
        logger.info(f"   q{q} confusion (rows=true 0/1/2):\n{cms[q]}")
    logger.info("=" * 64)

    result = {
        "model": f"CNN_{arch}", "level": 3, "length": length, "m_param": m_param,
        "arch": arch, "preprocessing": preprocessing, "downsample": downsample,
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
    p.add_argument("--m-param", type=int, default=16, help="CNN filter width.")
    p.add_argument("--downsample", type=int, default=20,
                   help="Time decimation factor. 20 is the classic default; "
                        "use 40 with --arch hybrid to match the published "
                        "hybrid front-end (~12 samples after sparse FIR @ L=500).")
    p.add_argument("--arch", default="classic", choices=["classic", "hybrid"],
                   help="classic = Conv1d-based CNN (existing). "
                        "hybrid = (3,1)-kernel Conv2d, per-channel-independent "
                        "temporal conv body + per-qubit heads on a shared flat "
                        "embedding.")
    p.add_argument("--preprocessing", default="butter", choices=["butter", "sinc"],
                   help="butter = Butterworth LPF + decimate (current). "
                        "sinc = windowed-sinc 101-tap sparse FIR (matches the "
                        "hybrid description). Pair --arch hybrid --preprocessing sinc "
                        "for the full hybrid setup.")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--out", default=None, help="Output JSON for the metrics.")
    args = p.parse_args()
    run(args.labels, args.length, args.m_param, args.epochs, args.downsample,
        args.max_train, args.max_test, args.out, args.lr, args.batch_size,
        args.arch, args.preprocessing)
