"""One-point smoke test of the Oraqle ECCentric drivers (Oraqle artifact; no change to driver logic).

Imports a driver from the ECCentric checkout (cwd) and calls its own run_one() for ONE configuration instead of
its full multiprocessing sweep, then compares the LER with the paper's CSV at the same point (binomial z-score).
fetch_eccentric.sh copies this file into the checkout root; run it there (e.g. in the eccentric image, cwd /qec):

  python smoke_point.py discriminators [code] [discriminator] [readout_ns]   # run_ler_discriminators.py, LER_DIST env
  python smoke_point.py noise_models  [code] [noise_model]   [readout_ns]   # run_ler_noise_models.py,  LER_MODE env

Defaults: surface, MCMit-T / heron_cnn, 600 ns, d = $LER_DIST (default 5). SMOKE_SHOTS overrides the driver's
MWPM/BP-OSD shot budget (default: the driver's own, 20,000 MWPM shots under current noise). Reference CSVs are read
from $ECC_RESULTS (default $ORAQLE_RESULTS/qec/experiment_results) when present.
"""
import csv
import importlib
import json
import math
import os
import sys


def reference(kind, code, d, line, T):
    root = os.environ.get("ECC_RESULTS") or os.path.join(os.environ.get("ORAQLE_RESULTS", "/results"), "qec", "experiment_results")
    if kind == "discriminators":
        path, col = os.path.join(root, "discriminators", f"ler_discriminators_d{d}.csv"), "discriminator"
    else:
        suffix = {"current": ""}.get(os.environ.get("LER_MODE", "current"), "_" + os.environ.get("LER_MODE", ""))
        path, col = os.path.join(root, "noise_models", f"ler_noise_models{suffix}.csv"), "noise_model"
    if not os.path.exists(path):
        return None
    for r in csv.DictReader(open(path)):
        if r["code"] == code and int(r["distance"]) == d and r[col] == line and int(r["readout_ns"]) == T:
            return float(r["logical_error_rate"]), int(r["num_shots"])
    return None


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "discriminators"
    code = sys.argv[2] if len(sys.argv) > 2 else "surface"
    line = sys.argv[3] if len(sys.argv) > 3 else ("MCMit-T" if kind == "discriminators" else "heron_cnn")
    T = int(sys.argv[4]) if len(sys.argv) > 4 else 600
    M = importlib.import_module("run_ler_discriminators" if kind == "discriminators" else "run_ler_noise_models")
    if os.environ.get("SMOKE_SHOTS"):
        M.MWPM_SHOTS = M.BPOSD_SHOTS = int(os.environ["SMOKE_SHOTS"])
    if kind == "discriminators":
        d = M.code_distance(code)
        print("master_fidelity.csv:", M.MASTER_CSV, flush=True)
    else:
        d = int(os.environ.get("LER_DIST", "5"))
    res = M.run_one((code, d, line, T))
    ler, shots = res[7], res[6]
    out = dict(driver=M.__name__, code=code, distance=d, line=line, readout_ns=T, measure_error=res[5],
               shots=shots, ler=ler, seconds=round(res[8], 1))
    ref = reference(kind, code, d, line, T)
    if ref and ler is not None:
        p0, n0 = ref
        se = math.sqrt(max(p0 * (1 - p0), 1e-12) / shots + max(p0 * (1 - p0), 1e-12) / n0)
        out.update(paper_ler=p0, paper_shots=n0, z=round((ler - p0) / se, 2))
    print(json.dumps(out), flush=True)
    if ler is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
