"""Regenerate every discriminator/readout paper figure and table into
$ORAQLE_OUT/paper_figures/discriminators (no dataset needed). [unify, new file]

    Discriminators/.venv/bin/python paper_figures/discriminators/make_all.py [--paper-root ../oraqle-ics]
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["fig_readout_sweeps.py", "fig_iq_clouds.py", "fig_pareto.py", "tab_fidelity.py",
           "tab_model_comparison_params.py", "tab_herqules_pertrace.py", "tab_herqules_noretrain.py",
           "tab_fpga_resources.py"]
rc = 0
for s in SCRIPTS:
    extra = sys.argv[1:] if s == "tab_fidelity.py" else []
    print(f"=== {s}", flush=True)
    rc |= subprocess.run([sys.executable, os.path.join(HERE, s)] + extra).returncode
sys.exit(rc)
