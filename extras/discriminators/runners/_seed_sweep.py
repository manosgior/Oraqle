"""Seed sweep of faithful published HERQULES at one length, one mixing mode.
Characterises run-to-run training variance and the catastrophic-failure rate.
Same data/split every run; only the training seed changes.

Env:
  HERQULES_TRACE_LENGTH : length (default 500)
  SWEEP_N               : number of seeds, runs seeds 1..N (default 7)
  SWEEP_SCRAMBLE        : "1" = mixed (published), "0" = no-mix (default 0)
  FAIL_GMEAN            : gmean below this counts as a catastrophic failure (default 0.85)
"""
import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import os
import numpy as np
import trainers.HERQULES_original as H

N = int(os.environ.get("SWEEP_N", "7"))
SCRAMBLE = os.environ.get("SWEEP_SCRAMBLE", "0") == "1"
FAIL = float(os.environ.get("FAIL_GMEAN", "0.85"))
MODE = "MIXED" if SCRAMBLE else "NO-MIX"

H.SUBSET_NUM_TRAIN_VAL = 3000
H.SUBSET_NUM_TEST = 7000
_d = [0.35 if (isinstance(x, float) and abs(x - 0.53) < 1e-9) else x
      for x in H.get_train_val_and_test_set.__defaults__]
H.get_train_val_and_test_set.__defaults__ = tuple(_d)

print(f"[sweep] mode={MODE} length={H.TRACE_LENGTH} seeds=1..{N}", flush=True)
rows = []
for s in range(1, N + 1):
    acc = [float(a) for a in H.train(run_semi_sup=False, run_pre_filter=False,
                                     run_rmf=True, save=False, scramble=SCRAMBLE, seed=s)]
    g = float(np.prod(acc) ** 0.2)
    rows.append(acc + [g])
    flag = "  <-- FAIL" if g < FAIL else ""
    print(f"[sweep:{MODE}] seed {s:2d}  Q1-5 {[round(a,3) for a in acc]}  gmean {g:.4f}{flag}", flush=True)

arr = np.array(rows)
ok = arr[arr[:, 5] >= FAIL]
n_fail = len(arr) - len(ok)
lab = ["Q1", "Q2", "Q3", "Q4", "Q5", "gmean"]
print(f"[sweep:{MODE}] failures: {n_fail}/{len(arr)}", flush=True)
if len(ok):
    m, sd = ok.mean(0), ok.std(0)
    print(f"[sweep:{MODE}] SUCCESS mean: " + "  ".join(f"{lab[i]}={m[i]:.4f}" for i in range(6)), flush=True)
    print(f"[sweep:{MODE}] SUCCESS std : " + "  ".join(f"{lab[i]}={sd[i]:.4f}" for i in range(6)), flush=True)
