"""Central path configuration for the Oraqle artifact.

Every script resolves its inputs and outputs through these variables instead of
hard-coded absolute paths. Override any of them with an environment variable.

    ORAQLE_DATA         Directory with the raw Lienhard et al. five-qubit traces
                        (DRaw_C_Tr_v0-001, DRaw_C_Te_v0-002). NOT distributed
                        with this repository; see dataset/README.md.
    ORAQLE_WORK         Scratch space for regenerable derived data (demodulated
                        traces, features, downsampled training sets). Gitignored.
    ORAQLE_RESULTS      Small result files used by the paper (tracked in git).
    ORAQLE_CHECKPOINTS  Trained model checkpoints (tracked in git).
    ORAQLE_OUT          Regenerated figures/tables. Gitignored.

Usage from any script inside the repository:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[N]))  # repo root
    from oraqle_env import DATA, WORK, RESULTS, CHECKPOINTS, OUT
"""
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent


def _path(var: str, default: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(os.environ.get(var, str(default))).expanduser()


DATA = _path("ORAQLE_DATA", ROOT / "dataset")
WORK = _path("ORAQLE_WORK", ROOT / "work")
RESULTS = _path("ORAQLE_RESULTS", ROOT / "results")
CHECKPOINTS = _path("ORAQLE_CHECKPOINTS", ROOT / "checkpoints")
OUT = _path("ORAQLE_OUT", ROOT / "out")

RAW_TRAIN = DATA / "DRaw_C_Tr_v0-001"
RAW_TEST = DATA / "DRaw_C_Te_v0-002"


def require_dataset() -> None:
    """Fail with a clear message when the raw dataset is not plugged in."""
    missing = [p for p in (RAW_TRAIN, RAW_TEST) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Raw five-qubit dataset not found: "
            + ", ".join(str(p) for p in missing)
            + ". Set ORAQLE_DATA to the directory holding the Lienhard et al. "
            "files (see dataset/README.md)."
        )
