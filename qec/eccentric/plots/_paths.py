"""Path plumbing for the vendored ECCentric plotting scripts (Oraqle artifact).

The original scripts ran from the root of an ECCentric checkout and used relative paths
("experiment_results/..."). Here the inputs live in the Oraqle repository and the figures
go to the gitignored output tree:

    ECC_RESULTS   directory that holds discriminators/, noise_models/, readout_compare/
                  default: $ORAQLE_RESULTS/qec/experiment_results  (= <repo>/results/qec/experiment_results)
    ECC_FIG_OUT   where figures are written
                  default: $ORAQLE_OUT/paper_figures/qec           (= <repo>/out/paper_figures/qec)
"""
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from oraqle_env import OUT, RESULTS  # noqa: E402

ECC_RESULTS = os.environ.get("ECC_RESULTS") or str(RESULTS / "qec" / "experiment_results")
FIG_OUT = os.environ.get("ECC_FIG_OUT") or str(OUT / "paper_figures" / "qec")


def res(*parts):
    """Input file under ECC_RESULTS (replaces the original 'experiment_results/<parts>')."""
    return os.path.join(ECC_RESULTS, *parts)


def out(name):
    """Output file under FIG_OUT (created on demand)."""
    os.makedirs(FIG_OUT, exist_ok=True)
    return os.path.join(FIG_OUT, name)
