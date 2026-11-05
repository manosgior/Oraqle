"""s2cluster_xtalk_probe.py -- is a qubit's "|2>" label driven by ANOTHER qubit?

Uses s2cluster_perp_z.npz (per-shot z_perp, t/sep, MTV from
s2cluster_recompute_perp.py; first 500 samples = 1 us; labels = z_perp > 3).
For a target qubit A and a source qubit B, restricted to shots with A and B both
prepared in |1>, bin by B's own along-axis position t_B/sep_B:
   t_B < 0.5           -> B reads like |0> (B decayed / prep error)
   0.5 <= t_B < 1.5    -> B reads like |1>
   t_B >= 1.5          -> B beyond |1> (B's own displaced arc, e.g. Q3)
and report P(z_A > 3) per bin (Wilson 95% CI). If A's labels were A's own
leakage, P should not depend on B's bin.

Run (graham):
  docker run --rm --user $(id -u):$(id -g) -v ~/oraqle-ics-sims/leakage/s2cluster:/out \
     oraqle-torch:latest python /out/s2cluster_xtalk_probe.py
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import math

import numpy as np

PAPER = {q: f"Q{q+1}" for q in range(5)}


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [float("nan")] * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [p, max(0.0, c - h), min(1.0, c + h)]


d = np.load(str(LP.stage("s2cluster") / "s2cluster_perp_z.npz"))
z, t, y = d["z_perp"], d["t_over_sep"], d["y"].astype(np.int64)
bits = np.stack([(y >> q) & 1 for q in range(5)], 1)
lab = z > 3.0
bins = [("B_reads_0(t<0.5)", -np.inf, 0.5), ("B_reads_1(0.5<=t<1.5)", 0.5, 1.5),
        ("B_beyond_1(t>=1.5)", 1.5, np.inf)]
out = {"definition": "P(z_perp_A>3 | A prep 1, B prep 1, B's t/sep bin); also A prep 0 rows",
       "pairs": {}}
for A in range(5):
    for B in range(5):
        if A == B:
            continue
        row = {}
        for a_prep in (1, 0):
            sel0 = (bits[:, A] == a_prep) & (bits[:, B] == 1)
            for name, lo, hi in bins:
                sel = sel0 & (t[:, B] >= lo) & (t[:, B] < hi)
                k, n = int((lab[sel, A]).sum()), int(sel.sum())
                row[f"A_prep{a_prep}|{name}"] = {"k": k, "n": n, "rate_ci95": wilson(k, n)}
            selb0 = (bits[:, A] == a_prep) & (bits[:, B] == 0)
            k, n = int(lab[selb0, A].sum()), int(selb0.sum())
            row[f"A_prep{a_prep}|B_prep0"] = {"k": k, "n": n, "rate_ci95": wilson(k, n)}
        out["pairs"][f"A={PAPER[A]}(q{A}) B={PAPER[B]}(q{B})"] = row
# fraction of each qubit's prepared-|1> shots in each t/sep bin (own position)
own = {}
for q in range(5):
    m = bits[:, q] == 1
    own[PAPER[q]] = {name: float(((t[m, q] >= lo) & (t[m, q] < hi)).mean()) for name, lo, hi in bins}
out["own_tsep_bin_fraction_prep1"] = own
with open(str(LP.stage("s2cluster") / "s2cluster_xtalk_probe.json"), "w") as f:
    json.dump(out, f, indent=1)
for key, row in out["pairs"].items():
    s = " | ".join(f"{k.split('|')[1][:10]} {100*v['rate_ci95'][0]:.2f}% (n={v['n']})"
                   for k, v in row.items() if k.startswith("A_prep1"))
    print(key, "::", s)
print(json.dumps(own, indent=0))
