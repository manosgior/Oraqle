"""Multi-length, multi-seed no-mix HERQULES sweep -> per-seed CSV.

Reproduces the 21-seed no-mix (shared-choice) snapshot methodology at SHORT
lengths (200/400/600/800 ns) so the no-mix readout-length curve has full
seed statistics at every length, matching the existing 1000 ns datapoint.

Each (length, seed) RETRAINS the network from scratch (faithful published
protocol, scramble=False). Same data/split every run; only the seed changes.

Writes one row per (length, seed) and flushes after every run, so partial
progress survives an interruption.

Env:
  SWEEP_N        : number of seeds, runs 1..N            (default 21)
  SWEEP_SAMPLES  : comma list of trace_sample lengths    (default 100,200,300,400)
                   ns = 2 * samples -> 200,400,600,800 ns
  SWEEP_SCRAMBLE : "1" mixed / "0" no-mix                (default 0)
  FAIL_GMEAN     : gmean below this flagged failed=1     (default 0.85)
  SWEEP_OUT      : output csv name in reports dir        (default herqules_nomix_seedsweep_200_800ns.csv)

NOTE: TRACE_LENGTH is read by HERQULES_original at IMPORT time, so we cannot
change length within one live interpreter. Each length is run as a fresh
subprocess via the __main__ guard below (re-exec with HERQULES_TRACE_LENGTH set).
"""
import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import csv
import os
import subprocess
import sys

REPORTS = ensure_dir(REPORTS_OUT)
N = int(os.environ.get("SWEEP_N", "21"))
SAMPLES = [int(x) for x in os.environ.get("SWEEP_SAMPLES", "100,200,300,400").split(",")]
SCRAMBLE = os.environ.get("SWEEP_SCRAMBLE", "0") == "1"
FAIL = float(os.environ.get("FAIL_GMEAN", "0.85"))
MODE = "mixed" if SCRAMBLE else "no-mix"
METHOD = "scramble-original" if SCRAMBLE else "shared-choice"
OUT = os.path.join(REPORTS, os.environ.get("SWEEP_OUT", "herqules_nomix_seedsweep_200_800ns.csv"))
FIELDS = ["config", "method", "length_ns", "trace_samples", "seed",
          "Q1", "Q2", "Q3", "Q4", "Q5", "gmean", "arith_mean", "failed"]


def run_one_length(L):
    """Re-exec this script for a single length so HERQULES_original imports at L."""
    disc_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /app/Discriminators
    env = dict(os.environ)
    env["HERQULES_TRACE_LENGTH"] = str(L)
    env["_SWEEP_SINGLE_L"] = str(L)
    env["PYTHONPATH"] = disc_dir + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, os.path.abspath(__file__)], env=env,
                   cwd=disc_dir, check=True)


def sweep_single_length(L):
    import numpy as np
    import trainers.HERQULES_original as H
    H.SUBSET_NUM_TRAIN_VAL = 3000
    H.SUBSET_NUM_TEST = 7000
    _d = [0.35 if (isinstance(x, float) and abs(x - 0.53) < 1e-9) else x
          for x in H.get_train_val_and_test_set.__defaults__]
    H.get_train_val_and_test_set.__defaults__ = tuple(_d)

    print(f"[sweep] mode={MODE} L={L} ({L*2}ns) seeds=1..{N}", flush=True)
    new_file = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader(); f.flush()
        for s in range(1, N + 1):
            acc = [float(a) for a in H.train(run_semi_sup=False, run_pre_filter=False,
                                             run_rmf=True, save=False,
                                             scramble=SCRAMBLE, seed=s)]
            g = float(np.prod(acc) ** 0.2)
            am = float(np.mean(acc))
            failed = int(g < FAIL)
            w.writerow(dict(config=MODE, method=METHOD, length_ns=L * 2, trace_samples=L,
                            seed=s, Q1=round(acc[0], 4), Q2=round(acc[1], 4),
                            Q3=round(acc[2], 4), Q4=round(acc[3], 4), Q5=round(acc[4], 4),
                            gmean=round(g, 4), arith_mean=round(am, 4), failed=failed))
            f.flush()
            flag = "  <-- FAIL" if failed else ""
            print(f"[sweep:{MODE}] L={L} ({L*2}ns) seed {s:2d}  "
                  f"Q1-5 {[round(a,3) for a in acc]}  gmean {g:.4f}{flag}", flush=True)


if __name__ == "__main__":
    single = os.environ.get("_SWEEP_SINGLE_L")
    if single is not None:
        sweep_single_length(int(single))
    else:
        for L in SAMPLES:
            run_one_length(L)
        print(f"[sweep] DONE -> {OUT}", flush=True)
