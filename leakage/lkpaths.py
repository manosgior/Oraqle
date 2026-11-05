"""lkpaths.py -- path plumbing for the readout-leakage / error-mode analysis.

The original scripts ran either in docker on graham with per-run bind mounts
(/data, /feat, /reports, /rep, /r, /soft, /models, /work, /out, /lk, /m3, /e4,
/e1hmm, /e2seg, /xs, ...) or on the laptop with absolute /home/manosgior/...
paths. Every such constant now resolves through this module, which derives
everything from the repository-wide ../oraqle_env.py. Any location can be
overridden with the environment variable given in brackets.

    original location                                  here
    -------------------------------------------------  ---------------------------------------------
    /data, ~/qubit_readout_klinq/data/five_qubit_data  DATA, RAW_TRAIN, RAW_TEST   [ORAQLE_DATA]
    graham ~/oraqle-ics-sims/leakage  (/lk)            LK = $ORAQLE_WORK/leakage   [ORAQLE_LEAK_WORK]
      .../features  (/feat, /lk/features, x_* /out)    FEAT = LK/features          [ORAQLE_LEAK_FEAT]
      .../<stage>   (/work of that stage's container)  stage("em/e4recon") = LK/em/e4recon, ...
    ~/oraqle_reports  (/reports, /rep, /r)             REPORTS = results/discriminators [ORAQLE_REPORTS]
    ~/oraqle_models   (/models)                        MODELS = checkpoints        [ORAQLE_MODELS]
    oraqle-ics/data/sims/soft, leakage/f/in,
      em/e5vrepro/in (/soft, /work/in; byte copies)    SOFT (first dir holding cnn_soft_1000ns.npz) [ORAQLE_SOFT]
    oraqle-ics/data/leakage                            RES = results/leakage       [ORAQLE_LEAK_RESULTS]
    Oraqle/Discriminators (PYTHONPATH of e3herq runs)  DISCRIMINATORS = discriminators [ORAQLE_DISCRIMINATORS]
    oraqle-ics/plots (figure outputs)                  FIGS = $ORAQLE_OUT/leakage  [ORAQLE_LEAK_OUT]

Stage directories mirror the cluster tree, so a stage's outputs land where the
next stage expects them (e.g. em_e4recon_glrt.py writes LK/em/e4recon/out and
em_e5vstat_check.py reads it back). The local assembly scripts read and write
RES in place, as they did in oraqle-ics/data/leakage: point ORAQLE_LEAK_RESULTS
at a scratch copy before re-running them (verify/leakage.py --regen does that).

No analysis constant lives here: only locations.
"""
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent          # <repo>/leakage
ROOT = HERE.parent                                       # <repo>
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from oraqle_env import (  # noqa: E402
    CHECKPOINTS, DATA, OUT, RAW_TEST, RAW_TRAIN, RESULTS, WORK, require_dataset,
)


def _env(var, default):
    return pathlib.Path(os.environ.get(var, str(default))).expanduser()


def _first_with(var, fname, candidates):
    """$var if set, else the first candidate directory that holds fname, else the first candidate."""
    if os.environ.get(var):
        return pathlib.Path(os.environ[var]).expanduser()
    for c in candidates:
        if (c / fname).exists():
            return c
    return candidates[0]


LK = _env("ORAQLE_LEAK_WORK", WORK / "leakage")
FEAT = _env("ORAQLE_LEAK_FEAT", LK / "features")
RES = _env("ORAQLE_LEAK_RESULTS", RESULTS / "leakage")
FIGS = _env("ORAQLE_LEAK_OUT", OUT / "leakage")
DISCRIMINATORS = _env("ORAQLE_DISCRIMINATORS", ROOT / "discriminators")
REPORTS = _first_with("ORAQLE_REPORTS", "preds_MCMitCNN_pb.npy",
                      [RESULTS / "discriminators"])
MODELS = _first_with("ORAQLE_MODELS", "HERQULES_len500.pth",
                     [CHECKPOINTS, CHECKPOINTS / "herqules", CHECKPOINTS / "discriminators"])
SOFT = _first_with("ORAQLE_SOFT", "cnn_soft_1000ns.npz",
                   [RESULTS / "system" / "soft", RESULTS / "system" / "sims" / "soft",
                    RESULTS / "system", RESULTS / "sims" / "soft", RESULTS / "qec" / "soft",
                    RESULTS / "discriminators"])
# HERQULES pre-classifier state used only by s1local_survey_numbers.py (19.6 MB, not in git)
PRECLASSIFIER_PKL = _env("ORAQLE_PRECLASSIFIER_PKL", DISCRIMINATORS / "preclassifier_state.pkl")
# Fig. iq_taxonomy generator (other owner: discriminators/runners)
IQ_GALLERY_GEN = _env("ORAQLE_IQ_GALLERY_GEN", DISCRIMINATORS / "runners" / "iq_error_gallery.py")


def stage(name):
    """Work directory of one analysis stage, = graham ~/oraqle-ics-sims/leakage/<name>."""
    return LK / name


def outdir(p):
    """Create an output directory (formerly provided by a docker bind mount) and return it as str."""
    p = pathlib.Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


if __name__ == "__main__":
    for k in ("DATA", "RAW_TRAIN", "RAW_TEST", "LK", "FEAT", "RES", "REPORTS", "MODELS", "SOFT",
              "DISCRIMINATORS", "FIGS", "PRECLASSIFIER_PKL", "IQ_GALLERY_GEN"):
        p = globals()[k]
        print(f"{k:18s} {p}  {'(exists)' if p.exists() else '(missing)'}")
