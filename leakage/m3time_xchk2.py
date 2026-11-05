#!/usr/bin/env python3
"""m3time_xchk2.py -- extra lag-locking checks for prepared-|0> Q1/Q5 entries into L (agent tag m3time).
Reuses m3time_xchk.lag_lift on the filtered canonical Viterbi paths; writes m3time_xchk2.json.
Run (graham): docker run --rm --user $(id -u):$(id -g) -v ~/oraqle-ics-sims/leakage/features:/feat:ro \
  -v ~/oraqle-ics-sims/leakage/m3time:/m3 oraqle-torch:latest python /m3/scripts/m3time_xchk2.py --run /m3/out_re/feat_xt1
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3time_xchk import lag_lift, trans, NAMES, CODE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
a = ap.parse_args()
y = np.load(str(LP.FEAT / "y_test.npy")).astype(np.int64)
bits = (y[:, None] >> np.arange(5)) & 1
P = [np.load(os.path.join(a.run, f"m3time_re_q{q}_arrays.npz"))["path"].astype(np.int8) for q in range(5)]
res = {}
for q in (0, 4):
    for (a0, a1) in (("0", "L"), ("L", "0")):
        for (b0, b1) in (("0", "1"), ("1", "0"), ("1", "L"), ("0", "L")):
            Eq, Tr = trans(P[q], CODE[a0], CODE[a1]), trans(P[2], CODE[b0], CODE[b1])
            for br in (0, 1):
                cls = (bits[:, q] == 0) & (bits[:, 2] == br)
                res[f"{NAMES[q]} {a0}->{a1} vs Q3 {b0}->{b1} | prepq0_prepr{br}"] = lag_lift(Eq, Tr, cls)
with open(os.path.join(a.run, "m3time_xchk2.json"), "w") as f:
    json.dump(res, f, indent=1)
print("[m3time-xchk2] done")
