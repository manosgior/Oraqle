#!/usr/bin/env python3
"""m3time_examples_re.py -- pull the raw TEST traces of the example shots chosen by m3time_reanalyze.py (canonical labels)
and recompute their trajectories from scratch (agent tag m3time).

For each qubit q and each example (raw_idx into X_test):
  * the FIR chain of x_common (mix at f_q, 101-tap FIR 'same', rotate by theta_q) on the raw trace,
  * 100 ns bin means (must equal feat_test[row, q] -> checked, max |diff| reported),
  * the FIR-filtered rotated signal every 5 samples (10 ns) for a smooth time trace,
  * 100 ns boxcar bins (no FIR) as a cross-check,
  * the neighbour-configuration crosstalk offsets used by the HMM are subtracted (per 100 ns bin).
Writes OUT/m3time_re_examples_raw.npz.

Run (graham):
  docker run --rm --user $(id -u):$(id -g) -v ~/qubit_readout_klinq/data/five_qubit_data:/data:ro \
    -v ~/oraqle-ics-sims/leakage/features:/feat:ro -v ~/oraqle-ics-sims/leakage/scripts:/xs:ro \
    -v ~/oraqle-ics-sims/leakage/m3time:/m3 oraqle-torch:latest \
    python /m3/scripts/m3time_examples_re.py --run /m3/out_re/feat_xt1
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import argparse, json, os, sys
import numpy as np
import h5py

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # was /xs (x_common.py)
from x_common import FS, fir_same  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    calib = json.load(open(str(LP.FEAT / "calib.json")))["used_for_features"]
    feat = np.load(str(LP.FEAT / "feat_test.npy"), mmap_mode="r")
    hf = h5py.File(str(LP.RAW_TEST), "r")
    X = hf["X_test"]
    out = {}
    t = np.arange(1000) / FS
    P_all = [np.load(os.path.join(a.run, f"m3time_re_q{r}_arrays.npz"))["path"] for r in range(5)]
    for q in range(5):
        d = json.load(open(os.path.join(a.run, f"m3time_re_q{q}.json")))
        ex = d["examples"]
        offs = np.array(d["offsets_xt"])               # (16,20,2)
        xt = d["xt_correction"]
        rows = [e["raw_idx"] for e in ex]
        urows, inv = np.unique(np.array(rows), return_inverse=True)
        R = np.asarray(X[urows])[inv]                 # h5py needs strictly increasing indices
        f, th = calib[q]["f_Hz"], calib[q]["theta_rad"]
        Xc = R[:, :, 0] + 1j * R[:, :, 1]
        mixed = Xc * np.exp(-1j * 2 * np.pi * f * t)[None]
        filt = fir_same(mixed) * np.exp(1j * th)
        b100 = filt.reshape(len(rows), 20, 50).mean(2)
        box = (mixed * np.exp(1j * th)).reshape(len(rows), 20, 50).mean(2)
        F = np.stack([np.asarray(feat[e["test_row"], q]) for e in ex])  # (n,20,2)
        diff = np.abs(np.stack([b100.real, b100.imag], -1) - F).max()
        cfg = np.array([e["cfg_other"] for e in ex])
        o = offs[cfg] if xt else np.zeros((len(ex), 20, 2))
        b100c = np.stack([b100.real, b100.imag], -1) - o
        boxc = np.stack([box.real, box.imag], -1) - o
        fine = filt[:, ::5]                             # 200 points, 10 ns
        ofine = np.repeat(o, 10, axis=1)                # each 100 ns bin = 10 fine points
        finec = np.stack([fine.real, fine.imag], -1) - ofine
        out[f"q{q}_b100"] = b100c
        out[f"q{q}_box100"] = boxc
        out[f"q{q}_fine10ns"] = finec
        out[f"q{q}_rows"] = np.array(rows)
        out[f"q{q}_cats"] = np.array([e["cat"] for e in ex])
        out[f"q{q}_paths"] = np.array([e["path"] for e in ex])
        out[f"q{q}_paths_raw"] = np.array([e["path_raw"] for e in ex])
        out[f"q{q}_y"] = np.array([e["y"] for e in ex])
        out[f"q{q}_postL"] = np.array([e["postL"] for e in ex])
        out[f"q{q}_maxdiff_vs_feat"] = np.array(diff)
        # context: the other qubits' per-bin features (feat_test, no xt correction) and Viterbi paths
        trows = np.array([e["test_row"] for e in ex])
        out[f"q{q}_feat_all"] = np.stack([np.asarray(feat[i]) for i in trows])          # (n,5,20,2)
        out[f"q{q}_paths_all"] = np.stack([P_all[r][trows] for r in range(5)], 1)      # (n,5,20)
        print(f"q{q}: {len(ex)} examples, max |recomputed - feat_test| = {diff:.2e}", flush=True)
    np.savez_compressed(os.path.join(a.run, "m3time_re_examples_raw.npz"), **out)


if __name__ == "__main__":
    main()
