"""tab:fidelity (ml_discriminators.tex:145-191) and tab:fidelity_linear (appendix_herqules.tex:224-240).
[unify, new file]

Rebuilds master_fidelity.csv from the per-model CSVs (byte-identical check against
results/), writes both tables as LaTeX, and, if the paper checkout is given, diffs every
cell (value, bold, shading):

    python paper_figures/discriminators/tab_fidelity.py [--paper-root ../oraqle-ics]
"""
import os
import subprocess
import sys

from _common import FIG_OUT, PY, REPO

args = [PY, os.path.join(REPO, "discriminators", "tables", "build_table2.py"), "--check",
        "--out", os.path.join(FIG_OUT, "tables")]
if "--paper-root" in sys.argv:
    root = sys.argv[sys.argv.index("--paper-root") + 1]
    args += ["--paper-tex", os.path.join(root, "chapters", "ml_discriminators.tex"),
             "--paper-appendix", os.path.join(root, "chapters", "appendix_herqules.tex")]
sys.exit(subprocess.run(args).returncode)
