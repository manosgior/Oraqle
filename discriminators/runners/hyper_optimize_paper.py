"""Paper protocol for the Baseline FNN and QubiCML rows of tab:fidelity.  [unify, new file]

runners/hyper_optimize.py is kept exactly as it was last used on the cluster
(2026-06-01). In that state its constants target a later KLiNQ-student run:
TRACE_LENGTHS=[500], N_OPTUNA_TRIALS=15, EPOCHS_PER_TRIAL=30, full data, and
__main__ runs only "KLiNQ_student".

The published FNN_len*.csv / Arxiv240618807FNN_q*_len*.csv files (results/
discriminators/, May 6-14 2026) record n_optuna_trials=10 and epochs=25. That
is the configuration of Oraqle git commits 7d3b10a/b41eeb7: all ten lengths
50..500 samples, 10 Optuna trials, 25 epochs per trial, and a random subsample
of MAX_TOTAL_SAMPLES=400_000 shots (320k from the train file, 80k from the test
file; SAMPLE_SEED=42). The FNN and QubiCML objectives, data loading and
evaluation code are byte-identical between b41eeb7 and the current file; only
these module constants differ (b41eeb7 itself says N_OPTUNA_TRIALS=30, but the
CSVs say 10, so the runs used 10).

This wrapper imports hyper_optimize as a module, sets those constants, and calls
its optimize_models(). It changes no algorithm, search space, seed or split.

    python discriminators/runners/hyper_optimize_paper.py                  # paper protocol
    python discriminators/runners/hyper_optimize_paper.py --lengths 100 200 300 400 500
    python discriminators/runners/hyper_optimize_paper.py --models FNN

Outputs: CSVs to $ORAQLE_OUT/discriminators/reports, checkpoints to
$ORAQLE_WORK/checkpoints (see discriminators/disc_paths.py). Optuna's TPE sampler
is not seeded in hyper_optimize.py, so a rerun gives statistically similar, not
bit-identical, numbers.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PAPER = dict(lengths=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500], trials=10, epochs=25,
             max_total_samples=400_000, models=["FNN", "Arxiv240618807FNN"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=PAPER["models"],
                    help="hyper_optimize model keys (default: FNN Arxiv240618807FNN)")
    ap.add_argument("--lengths", nargs="+", type=int, default=PAPER["lengths"],
                    help="trace lengths in samples (2 ns each)")
    ap.add_argument("--trials", type=int, default=PAPER["trials"])
    ap.add_argument("--epochs", type=int, default=PAPER["epochs"])
    ap.add_argument("--max-total-samples", type=int, default=PAPER["max_total_samples"],
                    help="0 = use all shots; default 400000 (80/20 train/test file split)")
    a = ap.parse_args(argv)

    from disc_paths import require_dataset
    require_dataset()
    import runners.hyper_optimize as HO

    HO.TRACE_LENGTHS = list(a.lengths)
    HO.N_OPTUNA_TRIALS = a.trials
    HO.EPOCHS_PER_TRIAL = a.epochs
    total = a.max_total_samples or None
    HO.MAX_TOTAL_SAMPLES = total
    HO.MAX_TRAIN_SAMPLES = int(total * 0.8) if total else None
    HO.MAX_TEST_SAMPLES = int(total * 0.2) if total else None
    print(f"[hyper_optimize_paper] models={a.models} lengths={HO.TRACE_LENGTHS} trials={HO.N_OPTUNA_TRIALS} "
          f"epochs={HO.EPOCHS_PER_TRIAL} train/test subsample={HO.MAX_TRAIN_SAMPLES}/{HO.MAX_TEST_SAMPLES} "
          f"csv->{HO.CSV_DIR} ckpt->{HO.SAVE_DIR}", flush=True)
    HO.optimize_models(models_to_run=list(a.models))


if __name__ == "__main__":
    main()
