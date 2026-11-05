"""
[Oraqle artifact, audit 2026-11-05] Complete the lattice-sim half of the CNN-readout simulator comparison.

results/qec/experiment_results/synchronization-artifact/artifact_readout_ler_cnn.csv (eccentric @da7164c,
2026-06-13) has 49 of its 50 points: the run was stopped before its slowest job, d=13 / 200 ns / current, finished
(the CSV is still in imap_unordered order, i.e. the driver's final sort never ran). This script computes that one
point with the UNCHANGED driver qec/lattice-sim/compare_readout_artifact.py (byte-identical to eccentric @da7164c
synchronization-artifact/compare_readout_artifact.py; its sim/ is byte-identical too): it calls the driver's own
_run_one((13, 200, "current")) with READOUT_MODE="cnn" (the driver default), in N_CHUNKS independent processes of
CHUNK shots each, and adds up the failures. The estimator (failures / shots, any of the 3 logical observables wrong,
PyMatching MWPM) and the total of 500,000 shots are the driver's; only the shots are split across processes, because
one 500k-shot call at d=13 with a 20 % readout flip takes hours on one core.

Run (steve, 2026-11-05), from a directory holding a copy of qec/lattice-sim and this script:
  docker run --rm --name audit-qec-latsim-d13 -v $PWD:/w -w /w oraqle-latsim \
      python -u run_latsim_cnn_missing_point.py lattice-sim latsim_cnn_d13_200ns_current.csv
Output CSV: the driver's columns plus shots, failures, chunks, seconds.
"""
import csv
import multiprocessing as mp
import os
import sys
import time

LATSIM = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "lattice-sim")
OUT = sys.argv[2] if len(sys.argv) > 2 else "latsim_cnn_d13_200ns_current.csv"
CONFIG = (13, 200, "current")
TOTAL_SHOTS = 500_000
N_CHUNKS = int(os.environ.get("N_CHUNKS", "50"))
CHUNK = TOTAL_SHOTS // N_CHUNKS
assert CHUNK * N_CHUNKS == TOTAL_SHOTS


def _chunk(i):
    sys.path.insert(0, LATSIM)
    import compare_readout_artifact as M  # noqa: PLC0415  (unchanged driver)
    assert M.READOUT_MODE == "cnn"
    M.NUM_SHOTS = CHUNK
    tool, d, ro, nm, meas, ler = M._run_one(CONFIG)
    return i, meas, int(round(ler * CHUNK))


def main():
    t0 = time.time()
    procs = int(os.environ.get("PROCS", str(N_CHUNKS)))
    with mp.Pool(procs) as pool:
        res = pool.map(_chunk, range(N_CHUNKS))
    fails = sum(r[2] for r in res)
    meas = res[0][1]
    ler = fails / TOTAL_SHOTS
    secs = time.time() - t0
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tool", "distance", "readout_ns", "noise_model", "measure_error", "logical_error_rate",
                    "shots", "failures", "chunks", "seconds"])
        w.writerow(["artifact", CONFIG[0], CONFIG[1], CONFIG[2], meas, ler, TOTAL_SHOTS, fails, N_CHUNKS,
                    round(secs, 1)])
    print(f"d={CONFIG[0]} ro={CONFIG[1]} {CONFIG[2]}: {fails}/{TOTAL_SHOTS} = {ler:.6f} ({secs:.0f}s) -> {OUT}",
          flush=True)


if __name__ == "__main__":
    main()
