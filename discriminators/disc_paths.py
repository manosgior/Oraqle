"""Path plumbing for discriminators/, readout/ and fpga/.

The original code ran inside a Docker container with fixed bind mounts. Each
mount now resolves through ../oraqle_env.py (override any of them with the
environment variables listed there, or with the ORAQLE_* variables below):

    old container path            new location
    ----------------------------  ---------------------------------------------
    /data/five_qubit_data         DATA          ($ORAQLE_DATA, raw Lienhard set)
    /data/cnn                     CNN_DATA      ($ORAQLE_WORK/cnn)
    /app/optimization_reports     REPORTS_OUT   ($ORAQLE_OUT/discriminators/reports)
      (for writing)                 re-runs write here, never into results/
    /app/optimization_reports     RESULTS_DISC  ($ORAQLE_RESULTS/discriminators)
      (for reading published runs)  the committed paper results
    /app/saved_models             MODELS_OUT    ($ORAQLE_WORK/checkpoints) for new runs,
                                  CHECKPOINTS   ($ORAQLE_CHECKPOINTS) for the published ones

Import it from any script after putting discriminators/ on sys.path:

    sys.path.insert(0, <repo>/discriminators)
    from disc_paths import RAW_TRAIN, RAW_TEST, REPORTS_OUT, ...
"""
import os
import pathlib
import sys

DISC_ROOT = pathlib.Path(__file__).resolve().parent
REPO_ROOT = DISC_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from oraqle_env import (  # noqa: E402
    CHECKPOINTS, DATA, OUT, RAW_TEST, RAW_TRAIN, RESULTS, WORK, require_dataset,
)


def _env_path(var, default):
    return pathlib.Path(os.environ.get(var, str(default))).expanduser()


RESULTS_DISC = RESULTS / "discriminators"
REPORTS_OUT = _env_path("ORAQLE_REPORTS_OUT", OUT / "discriminators" / "reports")
MODELS_OUT = _env_path("ORAQLE_MODELS_OUT", WORK / "checkpoints")
CNN_DATA = _env_path("ORAQLE_CNN_DATA", WORK / "cnn")
THREE_LEVEL_LABELS = _env_path("ORAQLE_THREE_LEVEL_LABELS", WORK / "three_level_labels.hdf5")


def ensure_dir(p):
    """Create directory p (and parents) and return it as a str."""
    p = pathlib.Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)
