#!/usr/bin/env python3
"""Tier-2 smoke test: run every Table 2 / readout producer for a TINY config on the
real raw data, from the repo layout.  [unify, new file]

The goal is only to prove that each entry point imports, finds the dataset through
ORAQLE_DATA, trains, evaluates and writes its output file after the repository
reorganisation. Numbers from these runs are meaningless (1 epoch / 1 trial / few shots).

The repository code is NOT modified. Each case shrinks the run in memory:
  * module-style runners: set module constants (epochs, trials, lengths, subsample),
  * script-style runners: exec the source with exact-string substitutions, each of
    which must match exactly once (so the harness fails loudly if the code changes).

    python discriminators/smoke/smoke_table2.py <case> [<case> ...]
    Table 2 cases: fnn_qubicml transformer klinq herqules cnn baseline   (or: table2)
    appendix/figures: herqules_noretrain herqules_full iq_clouds              (all: everything)

Outputs land in $ORAQLE_OUT/discriminators/reports and $ORAQLE_WORK/checkpoints as for a
real run. Set ORAQLE_OUT / ORAQLE_WORK to a scratch location.
"""
import importlib
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
DISC = os.path.dirname(HERE)
REPO = os.path.dirname(DISC)
sys.path.insert(0, DISC)
from disc_paths import REPORTS_OUT, require_dataset  # noqa: E402


def patched_source(path, subs):
    src = open(path).read()
    for old, new in subs:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"smoke patch for {path}: expected 1x {old!r}, found {n}")
        src = src.replace(old, new)
    return src


def load_patched_module(name, path, subs):
    """Import `name` from `path` with in-memory substitutions and register it in sys.modules."""
    pkg = name.rpartition(".")[0]
    if pkg:
        importlib.import_module(pkg)
    mod = types.ModuleType(name)
    mod.__file__ = path
    sys.modules[name] = mod
    exec(compile(patched_source(path, subs), path, "exec"), mod.__dict__)
    if pkg:
        setattr(sys.modules[pkg], name.rpartition(".")[2], mod)
    return mod


def run_patched_script(path, subs):
    ns = {"__name__": "__main__", "__file__": path}
    exec(compile(patched_source(path, subs), path, "exec"), ns)


# ---------------------------------------------------------------- cases
def case_fnn_qubicml():
    """Baseline FNN + QubiCML rows: hyper_optimize_paper.py with its own CLI knobs."""
    import runners.hyper_optimize_paper as HP
    HP.main(["--models", "FNN", "Arxiv240618807FNN", "--lengths", "100", "--trials", "1", "--epochs", "1",
             "--max-total-samples", "8000"])


def case_transformer():
    """MCMit-T rows: hyper_optimize_transformer.optimize_transformer() with shrunk constants."""
    import runners.hyper_optimize_transformer as HT
    HT.TRACE_LENGTHS = [100]
    HT.N_OPTUNA_TRIALS = 1
    HT.EPOCHS_PER_TRIAL = 1
    HT.N_PARALLEL_TRIALS = 1
    HT.MAX_TRAIN_SAMPLES = 6000
    HT.MAX_TEST_SAMPLES = 3000
    HT.optimize_transformer()


def case_klinq():
    """KLiNQ rows: reproduce_klinq_paper.main() with 1 epoch, one duration, and a strided
    subset reader (every 40th train / 80th test shot) in place of load_full()."""
    import h5py
    import numpy as np
    import runners.reproduce_klinq_paper as K
    K.DURATIONS = [(200, 100)]
    K.TEACHER_EPOCHS = 1
    K.STUDENT_EPOCHS = 1

    def load_strided(filepath, is_test):
        key = "test" if is_test else "train"
        step = 80 if is_test else 40
        with h5py.File(filepath, "r") as hf:
            X = hf[f"X_{key}"][::step, :K.FULL_SAMPLES, :]
            y = hf[f"y_{key}"][::step]
        return X.astype(np.float32), y
    K.load_full = load_strided
    K.main()


def case_herqules():
    """HERQULES rows: _herqules_length_both.py, 10% subsample at import, one length, 1 epoch."""
    os.environ.setdefault("HERQULES_TRACE_LENGTH", "100")
    load_patched_module("trainers.HERQULES_original", os.path.join(DISC, "trainers", "HERQULES_original.py"), [
        ("MAX_TRAIN_SAMPLES = None", "MAX_TRAIN_SAMPLES = 48000  # smoke"),
        ("MAX_TEST_SAMPLES = None", "MAX_TEST_SAMPLES = 112000  # smoke"),
    ])
    run_patched_script(os.path.join(DISC, "runners", "_herqules_length_both.py"), [
        ("H.SUBSET_NUM_TRAIN_VAL = 20000; H.SUBSET_NUM_TEST = 30000",
         "H.SUBSET_NUM_TRAIN_VAL = 1500; H.SUBSET_NUM_TEST = 3000  # smoke"),
        ("LENGTHS_NS = [200, 250, 400, 500, 600, 750, 800, 1000]", "LENGTHS_NS = [200]  # smoke"),
        ("    for ep in range(100):", "    for ep in range(1):  # smoke"),
    ])


def case_cnn():
    """MCMit-CNN rows: _cnn_length_sweep.py at CNN_LEN=100 with 1 epoch (full data)."""
    os.environ["CNN_LEN"] = os.environ.get("CNN_LEN", "100")
    run_patched_script(os.path.join(DISC, "runners", "_cnn_length_sweep.py"), [
        ("LR, DECAY, STEP, BS, EPOCHS = 5e-4, 0.95, 3, 64, 30", "LR, DECAY, STEP, BS, EPOCHS = 5e-4, 0.95, 3, 64, 1  # smoke"),
    ])


def case_baseline():
    """Linear Threshold rows + Sec. 4 sweeps: readout/run_baseline_sweeps.py on 4000 shots per file."""
    sys.path.insert(0, os.path.join(REPO, "readout"))
    import run_baseline_sweeps as RB
    RB.main(["--max-samples", "4000"])


def _herq_subsample():
    load_patched_module("trainers.HERQULES_original", os.path.join(DISC, "trainers", "HERQULES_original.py"), [
        ("MAX_TRAIN_SAMPLES = None", "MAX_TRAIN_SAMPLES = 48000  # smoke"),
        ("MAX_TEST_SAMPLES = None", "MAX_TEST_SAMPLES = 112000  # smoke"),
    ])


def case_herqules_noretrain():
    """tab:herqules-noretrain: _herqules_truncation_pertrace.py, subsample, two lengths, 1 epoch."""
    os.environ.setdefault("HERQULES_TRACE_LENGTH", "500")
    _herq_subsample()
    run_patched_script(os.path.join(DISC, "runners", "_herqules_truncation_pertrace.py"), [
        ("H.SUBSET_NUM_TRAIN_VAL = 20000; H.SUBSET_NUM_TEST = 30000",
         "H.SUBSET_NUM_TRAIN_VAL = 1500; H.SUBSET_NUM_TEST = 3000  # smoke"),
        ("LENGTHS_NS = [200, 250, 400, 500, 600, 750, 800, 1000]", "LENGTHS_NS = [400, 1000]  # smoke"),
        ("for ep in range(100):", "for ep in range(1):  # smoke"),
    ])


def case_herqules_full():
    """App. seed-to-seed variance: _herqules_full.py, subsample, 2 seeds, 1 epoch."""
    os.environ.setdefault("HERQULES_TRACE_LENGTH", "100")
    os.environ.setdefault("NSEED", "2")
    _herq_subsample()
    run_patched_script(os.path.join(DISC, "runners", "_herqules_full.py"), [
        ("H.SUBSET_NUM_TRAIN_VAL = 20000\nH.SUBSET_NUM_TEST = 30000",
         "H.SUBSET_NUM_TRAIN_VAL = 1500\nH.SUBSET_NUM_TEST = 3000  # smoke"),
        ("    for ep in range(100):", "    for ep in range(1):  # smoke"),
    ])


def case_iq_clouds():
    """fig:iq_clouds extraction: readout/iq_clouds_vs_length.py extract()+plot() on a subsample."""
    os.environ.setdefault("HERQULES_TRACE_LENGTH", "500")
    _herq_subsample()
    run_patched_script(os.path.join(REPO, "readout", "iq_clouds_vs_length.py"), [
        ("N_PER_STATE = 3000", "N_PER_STATE = 500  # smoke"),
    ])


CASES = {"fnn_qubicml": case_fnn_qubicml, "transformer": case_transformer, "klinq": case_klinq,
         "herqules": case_herqules, "cnn": case_cnn, "baseline": case_baseline,
         "herqules_noretrain": case_herqules_noretrain, "herqules_full": case_herqules_full,
         "iq_clouds": case_iq_clouds}
TABLE2 = ["fnn_qubicml", "transformer", "klinq", "herqules", "cnn", "baseline"]


def main():
    names = sys.argv[1:] or ["all"]
    if names == ["all"]:
        names = list(CASES)
    elif names == ["table2"]:
        names = TABLE2
    require_dataset()
    for n in names:
        t0 = time.time()
        before = set(os.listdir(REPORTS_OUT)) if os.path.isdir(REPORTS_OUT) else set()
        print(f"[smoke] ===== {n} =====", flush=True)
        CASES[n]()
        new = sorted(set(os.listdir(REPORTS_OUT)) - before)
        print(f"[smoke] {n}: OK in {time.time() - t0:.0f} s; new files in {REPORTS_OUT}: {new}", flush=True)


if __name__ == "__main__":
    main()
