"""
synth_transfer_sweep.py
=======================
Transferability sweep: generate synthetic datasets at a fixed target fidelity
while varying the mid-readout T1 event probability, train HERQULES on each, and
collect one comparison CSV (p_t1 vs per-qubit fidelity + gmean).

This tests whether HERQULES's hand-built MF/RMF machinery degrades gracefully
as the readout acquires non-stationary structure (T1 decays) that the global
matched filter wasn't designed for. The interesting result is the SHAPE of the
degradation curve — to be compared against a trace-based model (Transformer /
sub-window HERQULESPlus) trained on the same files.

Each (generate, train) step runs in its own subprocess, because
HERQULES_original loads + demodulates its data at import time (driven by the
HERQULES_TRAIN_FILE env var) — so a fresh process per dataset is the clean way
to repoint it.

Run inside the container, from /app/Discriminators:
    python -m runners.synth_transfer_sweep \
        --out-dir /data/synthetic_data \
        --p-t1 0.0 0.05 0.1 0.2 \
        --target-fidelity 0.98 --per-class 2000

`--out-dir` MUST be writable (the real /data/five_qubit_data mount is read-only;
use a dedicated synthetic mount, or an existing writable mount like
/app/optimization_reports).
"""

from __future__ import annotations

import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
from loguru import logger

DISCRIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM_QUBITS = 5
CSV_DIR = (
    str(REPORTS_OUT)
)


def _gmean(xs) -> float:
    a = np.clip(np.asarray(xs, dtype=np.float64), 1e-9, 1.0)
    return float(np.exp(np.mean(np.log(a))))


def _run(cmd, env=None):
    logger.info("  $ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=DISCRIM_DIR, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"subprocess exited {r.returncode}: {' '.join(cmd)}")


def _train_one(model: str, data_file: str, trace_length: int):
    """Train one model on data_file (fresh subprocess); return per-qubit list."""
    fd, res_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        if model == "herqules":
            env = dict(os.environ)
            env["HERQULES_TRAIN_FILE"] = data_file
            env["HERQULES_TEST_FILE"] = data_file
            env["HERQULES_TRACE_LENGTH"] = str(trace_length)
            _run([sys.executable, "-m", "runners._train_herqules_once", res_path], env=env)
        elif model == "cnn":
            # CNN needs its own demod+downsample format; build it once per file.
            cnn_file = data_file.replace(".hdf5", "_cnn.hdf5")
            if not os.path.exists(cnn_file):
                _run([sys.executable, "-m", "runners.synth_to_cnn",
                      "--in", data_file, "--out", cnn_file, "--force"])
            _run([sys.executable, "-m", "runners._train_cnn_once",
                  cnn_file, res_path, str(trace_length)])
        else:
            raise ValueError(f"unknown model {model!r} (expected herqules|cnn)")
        with open(res_path) as f:
            return json.load(f)["per_qubit"]
    finally:
        if os.path.exists(res_path):
            os.remove(res_path)


def run(out_dir, values, perturbation, target_fidelity, per_class, trace_length,
        regenerate, models):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    knob = {"t1": "--p-t1", "leakage": "--p-leakage"}[perturbation]

    logger.info("=" * 70)
    logger.info(f"Synthetic transferability sweep: {models} vs {perturbation}")
    logger.info(
        f"  target_fidelity={target_fidelity}  per_class={per_class}  "
        f"trace_length={trace_length}  {perturbation} values={values}"
    )
    logger.info(f"  out_dir={out_dir}")
    logger.info("=" * 70)

    rows = []
    for p in values:
        tag = f"F{target_fidelity}_{perturbation}_{p}"
        data_file = os.path.join(out_dir, f"Synth_{tag}.hdf5")

        # 1. Generate the dataset once (skip if it already exists).
        if regenerate or not os.path.exists(data_file):
            logger.info(f"[{perturbation}={p}] generating {data_file} ...")
            _run([
                sys.executable, "-m", "runners.generate_synthetic",
                "--out", data_file,
                "--per-class", str(per_class),
                "--target-fidelity", str(target_fidelity),
                knob, str(p),
                "--force",
            ])
        else:
            logger.info(f"[{perturbation}={p}] reusing existing {data_file}")

        # 2. Train each model on the SAME data file.
        for model in models:
            logger.info(f"[{perturbation}={p}] training {model} ...")
            t0 = time.perf_counter()
            per_q = _train_one(model, data_file, trace_length)
            gm = _gmean(per_q)
            logger.success(
                f"[{perturbation}={p}] {model}: per_qubit={['%.4f' % x for x in per_q]}  "
                f"gmean={gm:.4f}  ({time.perf_counter() - t0:.0f}s)"
            )
            rows.append({
                "model": model,
                "perturbation": perturbation,
                "value": p,
                "target_fidelity": target_fidelity,
                "per_class": per_class,
                "trace_length": trace_length,
                "gmean": gm,
                **{f"qubit_{q}_acc": per_q[q] for q in range(NUM_QUBITS)},
            })

    # ---- summary + CSV ----
    logger.info("=" * 70)
    logger.info(f"Degradation vs {perturbation}  (target_fidelity={target_fidelity})")
    for model in models:
        mrows = [r for r in rows if r["model"] == model]
        logger.info(f"  --- {model} ---")
        for r in mrows:
            logger.info(
                f"    {perturbation}={r['value']:<6} gmean={r['gmean']:.4f}  "
                f"per_q={['%.3f' % r[f'qubit_{q}_acc'] for q in range(NUM_QUBITS)]}"
            )
        if len(mrows) > 1:
            drop = mrows[0]["gmean"] - mrows[-1]["gmean"]
            logger.info(
                f"    gmean drop {perturbation}={mrows[0]['value']}->{mrows[-1]['value']}: {drop:+.4f}"
            )
    logger.info("=" * 70)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(
        CSV_DIR, f"synth_transfer_sweep_{perturbation}_F{target_fidelity}_{ts}.csv"
    )
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Saved comparison CSV: {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    p.add_argument("--out-dir", required=True,
                   help="Writable directory for the synthetic HDF5 files.")
    p.add_argument("--perturbation", choices=["t1", "leakage"], default="leakage",
                   help="Which perturbation the swept values apply to. "
                        "'leakage' (|2> off the MF axis) is the discriminating "
                        "test; 't1' is the secondary (RMF-handled) one.")
    p.add_argument("--values", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2],
                   help="Probabilities to sweep for the chosen perturbation.")
    p.add_argument("--target-fidelity", type=float, default=0.98,
                   help="Fixed per-qubit target fidelity for all datasets.")
    p.add_argument("--per-class", type=int, default=2000,
                   help="Shots per basis state (total = 32 x per_class).")
    p.add_argument("--trace-length", type=int, default=500,
                   help="Trace length (1 us = 500 samples).")
    p.add_argument("--regenerate", action="store_true",
                   help="Regenerate datasets even if the files already exist.")
    p.add_argument("--models", default="herqules,cnn",
                   help="Comma-separated models to compare on each dataset. "
                        "Supported: herqules, cnn. Default: herqules,cnn.")
    args = p.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    run(args.out_dir, args.values, args.perturbation, args.target_fidelity,
        args.per_class, args.trace_length, args.regenerate, models)
