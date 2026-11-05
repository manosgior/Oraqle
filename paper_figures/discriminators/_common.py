"""Shared paths for the discriminator paper-figure scripts. [unify, new file]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "discriminators"))
sys.path.insert(0, os.path.join(REPO, "discriminators", "tables"))
from disc_paths import OUT, RESULTS_DISC, ensure_dir  # noqa: E402,F401

FIG_OUT = ensure_dir(os.environ.get("ORAQLE_FIG_OUT", str(OUT / "paper_figures" / "discriminators")))
PY = sys.executable
