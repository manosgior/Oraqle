"""
herqules_subset_variance.py
===========================
Subset-stability study for HERQULES at a FIXED split size.

The HERQULES paper already shows fidelity vs. training-set *size* (it plateaus
by a few thousand shots per state). This script asks a different question:
**at a fixed size, how much does the reported fidelity depend on WHICH shots
you happen to draw?** We hold the per-class train+val / test sizes fixed
(default: the paper's 3000 / 7000) and vary only the random draw, via a
per-trial seed that permutes the per-class shot indices before slicing. Each
trial retrains HERQULES from scratch on its draw and evaluates per-qubit
fidelity on that draw's own test split; we report the spread across trials.

Cost: the demodulation (the expensive step) runs once when HERQULES_original is
imported. Each trial only re-does the cheap part — split + MF/RMF envelope fit
+ Net_rmf training — so N trials cost roughly N x (training time), not
N x (demod + training).

Run inside the container (1 us only, as requested):
    python -m runners.herqules_subset_variance \
        --trials 10 --num-train-val 3000 --num-test 7000
"""

from __future__ import annotations

import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import argparse
import csv
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Importing this runs the one-time data load + demodulation (cached in
# _DEMOD_CACHE). With the __main__ guards we added, it no longer auto-trains.
import trainers.HERQULES_original as H

NUM_QUBITS = 5
CSV_DIR = (
    str(REPORTS_OUT)
)
os.makedirs(CSV_DIR, exist_ok=True)


def _gmean(per_q) -> float:
    a = np.clip(np.asarray(per_q, dtype=np.float64), 1e-9, 1.0)
    return float(np.exp(np.mean(np.log(a))))


# Fixed seed for everything that is NOT the subset draw (Net_rmf init,
# MF class-balancing, cross-shot mixing). Held constant across trials so the
# only thing varying is which shots are drawn -> measures PURE subset variance.
_NUISANCE_SEED = 0


def run(trials: int, num_train_val: int, num_test: int, base_seed: int,
        vary_all: bool = False):
    # Pin the split sizes for every trial; vary only the draw.
    H.SUBSET_NUM_TRAIN_VAL = num_train_val
    H.SUBSET_NUM_TEST = num_test

    logger.info("=" * 70)
    logger.info("HERQULES subset-stability study (fixed size, varying draw)")
    logger.info(
        f"  trace_length = {H.TRACE_LENGTH}  |  per-class: "
        f"train+val={num_train_val}, test={num_test}  |  trials={trials}"
    )
    logger.info("=" * 70)

    logger.info(
        f"  mode = {'VARY-ALL (subset + init + balancing all tied to trial seed)' if vary_all else 'PURE-SUBSET (init/balancing fixed, only draw varies)'}"
    )

    rows = []
    for t in range(trials):
        seed = base_seed + t
        H.SUBSET_SEED = seed
        # Fix (or vary) the non-subset randomness. In pure-subset mode the
        # nuisance seed is constant, so Net_rmf init + MF balancing + cross-shot
        # mixing are identical every trial and the spread is attributable to the
        # subset draw alone. In vary-all mode everything is tied to the trial
        # seed (total run-to-run variance).
        nuisance = seed if vary_all else _NUISANCE_SEED
        np.random.seed(nuisance)
        torch.manual_seed(nuisance)
        logger.info(f"--- trial {t + 1}/{trials}  (SUBSET_SEED={seed}, nuisance_seed={nuisance}) ---")
        t0 = time.perf_counter()
        # save=False: don't clobber the real checkpoint, skip disk I/O.
        # Same flags the module-level call uses (demux MF + RMF, no presel).
        per_q = H.train(
            run_semi_sup=False, run_pre_filter=False, run_rmf=True, save=False,
        )
        per_q = [float(a) for a in per_q]
        gm = _gmean(per_q)
        logger.success(
            f"  trial {t + 1}: per_qubit={['%.4f' % a for a in per_q]}  "
            f"gmean={gm:.4f}  ({time.perf_counter() - t0:.0f}s)"
        )
        rows.append({
            "trial": t,
            "seed": seed,
            "num_train_val": num_train_val,
            "num_test": num_test,
            "trace_length": H.TRACE_LENGTH,
            "gmean": gm,
            **{f"qubit_{q}_acc": per_q[q] for q in range(NUM_QUBITS)},
        })

    # ---- summary ----
    arr = np.array([[r[f"qubit_{q}_acc"] for q in range(NUM_QUBITS)] for r in rows])
    gm = np.array([r["gmean"] for r in rows])
    logger.info("=" * 70)
    logger.info("SUMMARY  (mean ± std over trials; min..max)")
    for q in range(NUM_QUBITS):
        logger.info(
            f"  qubit {q}:  {arr[:, q].mean():.4f} ± {arr[:, q].std():.4f}   "
            f"[{arr[:, q].min():.4f} .. {arr[:, q].max():.4f}]"
        )
    logger.info(
        f"  gmean  :  {gm.mean():.4f} ± {gm.std():.4f}   "
        f"[{gm.min():.4f} .. {gm.max():.4f}]   peak-to-peak={gm.ptp():.4f}"
    )
    logger.info("=" * 70)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(
        CSV_DIR,
        f"herqules_subset_variance_len{H.TRACE_LENGTH}"
        f"_tv{num_train_val}_te{num_test}_{ts}.csv",
    )
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Saved per-trial results: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    p.add_argument("--trials", type=int, default=10,
                   help="Number of independent random draws to evaluate.")
    p.add_argument("--num-train-val", type=int, default=3000,
                   help="Per-class train+val size (paper default 3000).")
    p.add_argument("--num-test", type=int, default=7000,
                   help="Per-class test size (paper default 7000).")
    p.add_argument("--base-seed", type=int, default=1000,
                   help="Trial i uses SUBSET_SEED = base_seed + i.")
    p.add_argument("--vary-all", action="store_true",
                   help=("Tie ALL randomness (Net_rmf init, MF balancing, "
                         "cross-shot mixing) to the trial seed instead of "
                         "holding it fixed. Measures total run-to-run variance "
                         "rather than pure subset variance."))
    args = p.parse_args()
    run(args.trials, args.num_train_val, args.num_test, args.base_seed,
        vary_all=args.vary_all)
