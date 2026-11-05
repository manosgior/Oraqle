"""s2cluster_cnn_err_vs_label.py -- MCMit-CNN errors vs. the off-axis (|2>-like) flag.

Inputs (read-only):
  ~/oraqle_reports/preds_MCMitCNN_{pb,tb}.npy  (runners/_xtalk_cnn.py, 2026-06-13):
      the paper's MCMit-CNN (networks/CNN.py m=16, colleague prep, DS=40,
      samples [0,500) = 1 us), predicted / true bits on the FULL test file in
      file order (1,120,000 x 5)
  three_level_labels.hdf5 y3_test / y_test (same order)
  s2cluster_perp_z.npz (z_perp, t/sep from s2cluster_recompute_perp.py)

Outputs per qubit and prepared bit: CNN error rate on flagged (z>3) vs unflagged
shots, the share of CNN errors that land on flagged shots, and the same by z bins.

Run (graham):
  docker run --rm --user $(id -u):$(id -g) -v ~/oraqle_reports:/r:ro \
    -v ~/qubit_readout_klinq/data/five_qubit_data:/data:ro \
    -v ~/oraqle-ics-sims/leakage/s2cluster:/out oraqle-torch:latest \
    python /out/s2cluster_cnn_err_vs_label.py
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import math

import h5py
import numpy as np

PAPER = {q: f"Q{q+1}" for q in range(5)}


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [float("nan")] * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [p, max(0.0, c - h), min(1.0, c + h)]


pb = np.load(str(LP.REPORTS / "preds_MCMitCNN_pb.npy")).astype(np.int64)
tb = np.load(str(LP.REPORTS / "preds_MCMitCNN_tb.npy")).astype(np.int64)
with h5py.File(str(LP.DATA / "three_level_labels.hdf5"), "r") as f:
    y3 = f["y3_test"][:]
    y = f["y_test"][:].astype(np.int64)
bits = np.stack([(y >> q) & 1 for q in range(5)], 1)
d = np.load(str(LP.stage("s2cluster") / "s2cluster_perp_z.npz"))
split = d["split"]
z = d["z_perp"][split == 1]
assert z.shape[0] == y.shape[0]
res = {"aligned_tb_equals_prepared_bits": bool(np.array_equal(tb, bits)),
       "n_test": int(len(y)), "window": "samples [0,500) = 1 us", "per_qubit": {}}
zbins = [(0, 3), (3, 4), (4, 6), (6, 1e9)]
for q in range(5):
    err = pb[:, q] != tb[:, q]
    flag = y3[:, q] == 2
    r = {"paper": PAPER[q], "cnn_error_rate_all": wilson(int(err.sum()), len(err)),
         "n_err": int(err.sum()),
         "share_of_errors_on_flagged": wilson(int((err & flag).sum()), int(err.sum())),
         "n_err_on_flagged": int((err & flag).sum())}
    for b in (0, 1):
        m = bits[:, q] == b
        r[f"prep{b}"] = {
            "err_rate_flagged": {"k": int((err & flag & m).sum()), "n": int((flag & m).sum()),
                                 "rate_ci95": wilson(int((err & flag & m).sum()), int((flag & m).sum()))},
            "err_rate_unflagged": {"k": int((err & ~flag & m).sum()), "n": int((~flag & m).sum()),
                                   "rate_ci95": wilson(int((err & ~flag & m).sum()), int((~flag & m).sum()))},
            "err_rate_by_zbin": {f"{lo}-{hi if hi < 1e8 else 'inf'}": {
                "k": int((err & m & (z[:, q] >= lo) & (z[:, q] < hi)).sum()),
                "n": int((m & (z[:, q] >= lo) & (z[:, q] < hi)).sum()),
                "rate_ci95": wilson(int((err & m & (z[:, q] >= lo) & (z[:, q] < hi)).sum()),
                                    int((m & (z[:, q] >= lo) & (z[:, q] < hi)).sum()))}
                for lo, hi in zbins},
        }
    res["per_qubit"][f"q{q}"] = r
    print(f"{PAPER[q]}(q{q}) CNN err={100*r['cnn_error_rate_all'][0]:.3f}%  "
          f"errors on flagged={r['n_err_on_flagged']}/{r['n_err']}="
          f"{100*r['share_of_errors_on_flagged'][0]:.2f}%  "
          + " ".join(f"prep{b}: flagged {100*r[f'prep{b}']['err_rate_flagged']['rate_ci95'][0]:.1f}% "
                     f"(n={r[f'prep{b}']['err_rate_flagged']['n']}) vs unflagged "
                     f"{100*r[f'prep{b}']['err_rate_unflagged']['rate_ci95'][0]:.2f}%" for b in (0, 1)))
tot_err = sum(r["n_err"] for r in res["per_qubit"].values())
tot_fl = sum(r["n_err_on_flagged"] for r in res["per_qubit"].values())
res["pooled_share_of_errors_on_flagged"] = wilson(tot_fl, tot_err)
print("pooled share", tot_fl, tot_err, res["pooled_share_of_errors_on_flagged"],
      "aligned", res["aligned_tb_equals_prepared_bits"])
with open(str(LP.stage("s2cluster") / "s2cluster_cnn_err_vs_label.json"), "w") as f:
    json.dump(res, f, indent=1)
