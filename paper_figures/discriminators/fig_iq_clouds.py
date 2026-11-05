"""fig:iq_clouds (qubit_readout.tex:34) -> plots/iq_clouds_vs_length_q5.pdf.  [unify, new file]

Re-plots from the published extraction results/discriminators/iq_clouds_vs_length_q5.npz
(IQ_REPLOT=1). No dataset needed. To redo the extraction from the raw traces run
readout/iq_clouds_vs_length.py without IQ_REPLOT (needs ORAQLE_DATA, ~30 GB RAM).
"""
import os
import shutil
import subprocess

from _common import FIG_OUT, PY, REPO, RESULTS_DISC

work = os.path.join(FIG_OUT, "_iq_clouds_work")
os.makedirs(work, exist_ok=True)
shutil.copy(os.path.join(str(RESULTS_DISC), "iq_clouds_vs_length_q5.npz"), work)
env = dict(os.environ, IQ_REPLOT="1", IQ_QUBIT="5", IQ_REPORTS=work)
subprocess.run([PY, os.path.join(REPO, "readout", "iq_clouds_vs_length.py")], check=True, env=env)
for ext in ("pdf", "png"):
    shutil.move(os.path.join(work, f"iq_clouds_vs_length_q5.{ext}"), os.path.join(FIG_OUT, f"iq_clouds_vs_length_q5.{ext}"))
shutil.rmtree(work)
print("wrote", os.path.join(FIG_OUT, "iq_clouds_vs_length_q5.pdf"))
