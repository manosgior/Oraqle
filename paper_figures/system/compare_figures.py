"""Compare regenerated §7 figures ($ORAQLE_OUT/paper_figures/system) with the paper's PDFs
(results/system/reference_figures): page size, text, raster difference and drawing streams.

Same checks as paper_figures/qec/compare_figures.py (reused here with the §7 directories).
Usage: python paper_figures/system/compare_figures.py [--out DIR] [--ref DIR] [--max-diff 0.01]
"""
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper_figures" / "qec"))
sys.path.insert(0, str(REPO))
import compare_figures as C  # noqa: E402  (paper_figures/qec/compare_figures.py)
from oraqle_env import OUT, RESULTS  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--out" not in argv:
        argv += ["--out", str(OUT / "paper_figures" / "system")]
    if "--ref" not in argv:
        argv += ["--ref", str(RESULTS / "system" / "reference_figures")]
    sys.argv = [sys.argv[0]] + argv
    C.main()
