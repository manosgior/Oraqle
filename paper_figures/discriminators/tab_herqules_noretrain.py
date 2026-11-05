"""tab:herqules-noretrain, "Retraining across durations" (appendix_herqules.tex:199-217). [unify, new file]

No-retraining HERQULES (one model trained at 1 us, evaluated on truncated inputs) from
results/discriminators/herqules_truncation_pertrace_deployment.csv, policy=freeze
(runners/_herqules_truncation_pertrace.py).

Known discrepancy: the paper's 1000 ns row is copied from tab:fidelity (HERQULES,
1000 ns: 0.969 0.740 0.924 0.944 0.970 0.905); the freeze run's own 1000 ns row (printed
here) has Q3 0.925 and Q4 0.942. Pass --paper-1000 to print the paper's row instead.
"""
import csv
import os
from decimal import ROUND_HALF_UP, Decimal
import sys

from _common import FIG_OUT, RESULTS_DISC


def r3(x):
    """Round the 4-decimal CSV string half-up to 3 decimals, as printed in the paper."""
    return str(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

r = {(x["policy"], int(x["length_ns"])): x for x in
     csv.DictReader(open(os.path.join(str(RESULTS_DISC), "herqules_truncation_pertrace_deployment.csv")))}
m = {int(x["duration_ns"]): x for x in csv.DictReader(open(os.path.join(str(RESULTS_DISC), "master_fidelity.csv")))
     if x["model"] == "HERQULES"}
L = []
for ns in (1000, 800, 600, 400, 200):
    if ns == 1000 and "--paper-1000" in sys.argv:
        v = [m[ns][f"Q{i}_fidelity"] for i in range(1, 6)] + [m[ns]["F5Q_gmean"]]
    else:
        x = r[("freeze", ns)]
        v = [x[f"Q{i}"] for i in range(1, 6)] + [x["gmean5"]]
    pad = "" if ns == 1000 else " "
    L.append(f"{ns}\\,ns{pad} & " + " & ".join(r3(a) for a in v) + " \\\\")
out = os.path.join(FIG_OUT, "tab_herqules_noretrain.tex")
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("wrote", out)
