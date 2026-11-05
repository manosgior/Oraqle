"""Run HERQULES exactly as the published notebook intends, then again with the
ONLY change being the shot-mixing turned OFF -- an apples-to-apples mixed-vs-not.

Faithful to satvikmaurya/HERQULES/HERQULES.ipynb:
  - invocation train(run_semi_sup=False, run_pre_filter=False, run_rmf=True)
  - MF envelope fit on TRAIN; test accuracy = accuracy(net, test_ldr) inside
    train() (no separate test() refit)
  - published split: 1950 train / 1050 val / 7000 test per class (10000/class)

Two runs in one process (one demodulation):
  - scramble=True  : published behaviour (shot-mixing ON)
  - scramble=False : identical pipeline, shot-mixing OFF (per-shot features)

Writes herqules_faithful_published.csv (master_fidelity schema) to the reports
dir. Env: HERQULES_TRACE_LENGTH.
"""
import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import os
import csv
import numpy as np
import trainers.HERQULES_original as H

REPORTS = (str(REPORTS_OUT))
L = H.TRACE_LENGTH
NS = L * 2
PARAMS = 1002  # Net_rmf parameter count

# ---- match the published split exactly ----
H.SUBSET_NUM_TRAIN_VAL = 3000
H.SUBSET_NUM_TEST = 7000
_def = [0.35 if (isinstance(x, float) and abs(x - 0.53) < 1e-9) else x
        for x in H.get_train_val_and_test_set.__defaults__]
H.get_train_val_and_test_set.__defaults__ = tuple(_def)

rows = []
for scramble in (True, False):
    tag = "mixed (published)" if scramble else "no-mix (per-shot)"
    print(f"[pub] === length={L} ({NS} ns) | shot-mixing {'ON' if scramble else 'OFF'} ===", flush=True)
    acc = H.train(run_semi_sup=False, run_pre_filter=False, run_rmf=True,
                  save=False, scramble=scramble)
    acc = [float(a) for a in acc]
    gmean = float(np.prod(acc) ** (1.0 / len(acc)))
    print(f"[pub] {tag}: per-qubit {[round(a,4) for a in acc]}  F5Q_gmean={gmean:.4f}", flush=True)
    rows.append({
        "model": f"HERQULES (faithful, {tag})",
        "duration_ns": NS, "trace_samples": L, "parameters": PARAMS,
        **{f"Q{i+1}_fidelity": f"{acc[i]:.4f}" for i in range(5)},
        "F5Q_gmean": f"{gmean:.4f}", "shot_mixing": "on" if scramble else "off",
    })

out = os.path.join(REPORTS, f"herqules_faithful_published_len{L}.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"[pub] wrote {out}", flush=True)
