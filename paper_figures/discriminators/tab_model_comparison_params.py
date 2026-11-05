"""tab:model_comparison, row "Est. Parameters" (overview.tex:31-35, prose overview.tex:68). [unify, new file]

Prints the parameter counts behind the row from master_fidelity.csv. The block diagrams
(figures/oraqle-models.pdf) and the feature rows are hand-made.
"""
import csv
import os

from _common import FIG_OUT, RESULTS_DISC

rows = list(csv.DictReader(open(os.path.join(str(RESULTS_DISC), "master_fidelity.csv"))))
p = {(r["model"], int(r["duration_ns"])): int(r["parameters"]) for r in rows}
lines = ["model,parameters_at_1000ns,min_over_durations(ns),max_over_durations(ns),paper"]
paper = {"Baseline FNN": "~635K", "HERQULES": "~1K", "QubiCML": "~65 (per qubit)", "KLiNQ": "~9K",
         "MCMit-T": "~0.1-4.8M", "MCMit-CNN": "~3K"}
for m in paper:
    v = {ns: p[(m, ns)] for ns in (200, 400, 600, 800, 1000)}
    lo, hi = min(v, key=v.get), max(v, key=v.get)
    per_q = " (65 per qubit x 5)" if m == "QubiCML" else ""
    lines.append(f"{m},{v[1000]}{per_q},{v[lo]} ({lo}),{v[hi]} ({hi}),{paper[m]}")
out = os.path.join(FIG_OUT, "tab_model_comparison_params.csv")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("wrote", out)
