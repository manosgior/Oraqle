#!/usr/bin/env python3
"""Run the integration-and-threshold baseline sweeps of Sec. 4 and the Linear
Threshold rows (tab:fidelity_linear) in one go.  [unify, new file]

baseline_no_ML.py's __main__ was edited by hand between runs (exit() calls after
each step), so the published CSVs carry six timestamps from 2026-05-14. This
driver calls the same functions of baseline_no_ML.py, in the order of its
__main__, without changing any of them:

  load_and_merge(RAW_TRAIN, RAW_TEST, MAX_SAMPLES) -> decode_labels -> demodulate
  -> shuffle with SAMPLE_SEED (=42)
  sweep  : sweep_fidelity_incremental over T = 100..1900 ns (T_END capped at
           T_max-1 = 999 samples) -> save_sweep_results -> Baseline_Threshold_len{L}_*.csv
  dfdt   : derivative_critical_window -> save_derivative_results -> Baseline_Threshold_dFdT_analysis_*.csv
  window : evaluate_sliding_window_fidelity(window_ns=250 and 500) -> save_sliding_window_results
           -> Baseline_Threshold_sliding_window_{250,500}ns_*.csv

Outputs go to $ORAQLE_OUT/discriminators/reports. MAX_SAMPLES defaults to the value
in baseline_no_ML.py (None = all 1.6M shots; ~130 GB RAM for the demodulated
array). The value used for the published May-14 runs is not recorded.

    python readout/run_baseline_sweeps.py                    # all steps, all shots
    python readout/run_baseline_sweeps.py --max-samples 20000 --steps sweep   # quick check
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_no_ML as B  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-samples", type=int, default=B.MAX_SAMPLES,
                    help="per-file subsample passed to load_and_merge (default: baseline_no_ML.MAX_SAMPLES)")
    ap.add_argument("--steps", nargs="+", default=["sweep", "dfdt", "window"], choices=["sweep", "dfdt", "window"])
    ap.add_argument("--windows", nargs="+", type=float, default=[250, 500])
    a = ap.parse_args(argv)
    B.require_dataset()

    X, y = B.load_and_merge(B.RAW_TRAIN_FILE, B.RAW_TEST_FILE, max_samples=a.max_samples)
    y_binary = B.decode_labels(y)
    T_max = X.shape[1]
    print("Demodulating...", flush=True)
    demod = B.demodulate(X, B.FREQ_READOUT, dt=B.DT)
    del X
    rng = np.random.default_rng(B.SAMPLE_SEED)
    perm = rng.permutation(len(y_binary))
    demod, y_binary = demod[perm], y_binary[perm]

    if "sweep" in a.steps or "dfdt" in a.steps:
        T_START = int(B.T_START_NS / (B.DT * 1e9))
        T_END = min(int(B.T_END_NS / (B.DT * 1e9)), T_max - 1)
        T_STRIDE = int(B.T_STEP_NS / (B.DT * 1e9))
        T_indices = np.arange(T_START, T_END + 1, T_STRIDE)
        T_values_s = T_indices * B.DT
        print(f"Sweeping F(T) at {len(T_indices)} points ({T_indices[0]}..{T_indices[-1]} samples)", flush=True)
        fidelities = B.sweep_fidelity_incremental(demod, y_binary, T_indices, dt=B.DT)
        if "sweep" in a.steps:
            B.save_sweep_results(T_indices, fidelities)
        if "dfdt" in a.steps:
            dFdT = B.derivative_critical_window(fidelities, T_values_s)
            B.save_derivative_results(T_indices, dFdT)
    if "window" in a.steps:
        for w in a.windows:
            print(f"Sliding window {w:g} ns ...", flush=True)
            centers, fids = B.evaluate_sliding_window_fidelity(demod, y_binary, window_ns=w)
            B.save_sliding_window_results(centers, fids, window_ns=w)
    print(f"done; CSVs in {B.REPORTS_OUT}", flush=True)


if __name__ == "__main__":
    main()
