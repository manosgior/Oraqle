"""tab:herqules-pertrace (appendix_herqules.tex:147-174). [unify, new file]

Per-trace vs released (mixed-shot) HERQULES, F5Q and F4Q, from
results/discriminators/herqules_length_faithful_vs_demux.csv (runners/_herqules_length_both.py).
"""
import csv
import os
from decimal import ROUND_HALF_UP, Decimal

from _common import FIG_OUT, RESULTS_DISC


def r3(x):
    """Round the 4-decimal CSV string half-up to 3 decimals, as printed in the paper."""
    return str(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

r = {(x["variant"], int(x["length_ns"])): x
     for x in csv.DictReader(open(os.path.join(str(RESULTS_DISC), "herqules_length_faithful_vs_demux.csv")))}
L = []
for ns in (1000, 800, 600, 400, 200):
    a, b = r[("deployment_pertrace", ns)], r[("demux_subsample", ns)]
    pad = "" if ns == 1000 else " "
    L.append(f"{ns}\\,ns{pad} & {r3(a['gmean5'])} & {r3(a['gmean4_no_q2'])} & "
             f"{r3(b['gmean5'])} & {r3(b['gmean4_no_q2'])} \\\\")
out = os.path.join(FIG_OUT, "tab_herqules_pertrace.tex")
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("wrote", out)
