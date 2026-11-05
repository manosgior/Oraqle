"""fig:pareto (ml_discriminators.tex:264) -> plots/pareto_params_vs_fidelity_600ns_1000ns.pdf.  [unify, new file]

Runs plot_fidelities.plot_pareto_params_vs_fidelity() on master_fidelity.csv as rebuilt
by build_table2.py from the per-model CSVs (identical to the published one; pass
--published to plot the committed file directly). No dataset needed.
"""
import os
import subprocess
import sys

from _common import FIG_OUT, PY, REPO, RESULTS_DISC

csv_path = os.path.join(str(RESULTS_DISC), "master_fidelity.csv")
if "--published" not in sys.argv:
    tables = os.path.join(FIG_OUT, "tables")
    subprocess.run([PY, os.path.join(REPO, "discriminators", "tables", "build_table2.py"), "--check", "--out", tables],
                   check=True, stdout=subprocess.DEVNULL)
    csv_path = os.path.join(tables, "master_fidelity.csv")
import plot_fidelities  # noqa: E402  (discriminators/ is on sys.path via _common)

out = plot_fidelities.plot_pareto_params_vs_fidelity(csv_path=csv_path, out_dir=FIG_OUT)
print("wrote", out)
