"""fig:readout_sweeps (qubit_readout.tex:8) -> plots/readout_sweeps.pdf.  [unify, new file]

Re-plots the integration-and-threshold baseline sweeps from the published CSVs
(results/discriminators/Baseline_Threshold_*; pinned files as in the paper) with
readout/plot_readout_sweeps.py. No dataset needed. To regenerate the CSVs from the
raw traces, see readout/run_baseline_sweeps.py.
"""
import os
import subprocess

from _common import FIG_OUT, PY, REPO

out = os.path.join(FIG_OUT, "readout_sweeps.pdf")
subprocess.run([PY, os.path.join(REPO, "readout", "plot_readout_sweeps.py"), out], check=True)
print("wrote", out)
