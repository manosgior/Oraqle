"""
_train_herqules_once.py
=======================
Train HERQULES once on the dataset pointed to by the HERQULES_TRAIN_FILE /
HERQULES_TEST_FILE (and optional HERQULES_TRACE_LENGTH) environment variables,
then dump the per-qubit test fidelity to a JSON file.

Meant to be invoked as a SUBPROCESS (one fresh interpreter per dataset) so that
HERQULES_original's module-level data load + demodulation picks up the right
file — those run at import time and can't be repointed within a live process.

    HERQULES_TRAIN_FILE=... HERQULES_TEST_FILE=... \
        python -m runners._train_herqules_once /tmp/result.json
"""

import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trainers.HERQULES_original as H  # noqa: E402  (import triggers data load)


def main():
    if len(sys.argv) < 2:
        print("usage: python -m runners._train_herqules_once <result.json>")
        sys.exit(1)
    out_path = sys.argv[1]

    # save=False: don't clobber the real HERQULES_len*.pth checkpoint.
    # standardize=True: z-score the MF/RMF features so Net_rmf trains stably on
    # synthetic data (large MF magnitudes otherwise diverge it at lr=0.01).
    # This matches the CNN baseline's internal BatchNorm — a fair comparison.
    per_q = H.train(
        run_semi_sup=False, run_pre_filter=False, run_rmf=True,
        save=False, standardize=True,
    )

    result = {
        "per_qubit": [float(x) for x in per_q],
        "trace_length": int(H.TRACE_LENGTH),
        "train_file": os.environ.get("HERQULES_TRAIN_FILE", "default"),
    }
    with open(out_path, "w") as f:
        json.dump(result, f)
    print("WROTE_RESULT:" + out_path)


if __name__ == "__main__":
    main()
