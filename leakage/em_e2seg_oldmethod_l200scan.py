"""em_e2seg_oldmethod_l200scan.py (tag e2seg): find the preprocessing of HybridCNN_len200.pth.

em_e2seg_oldmethod.py reproduces the 1 us (L=500) CSV to within 0.05% but HybridCNN_len200.pth gives 3-5x the
CSV error counts with DS=40. Scan the sparse-FIR downsample factor and report HybridCNN error counts vs the CSV;
for the DS that matches, recompute the old masks and apply them to the soft400 MCMit-CNN model.
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import csv
import glob
import json
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # was /work/scripts
import em_e2seg_oldmethod as OM

L = 200
res = {"note": __doc__, "scan": {}}
with h5py.File(OM.RAW_TR, "r") as hf:
    Xtr_all = hf["X_train"][:, :L, :].astype(np.float32)
    ytr = hf["y_train"][:].astype(np.int64)
with h5py.File(OM.RAW_TE, "r") as hf:
    Xte_all = hf["X_test"][:, :L, :].astype(np.float32)
    yte = hf["y_test"][:].astype(np.int64)
Ytr, Yte = OM.bitsf(ytr), OM.bitsf(yte)
old = list(csv.DictReader(open(sorted(glob.glob(f"{LP.REPORTS}/herqules_error_modes_len200_hybridcnn_*.csv"))[-1])))
st = torch.load(str(LP.MODELS / "HybridCNN_len200.pth"), map_location="cpu")
model = OM.HybridCNN(n_channels=10, m_param=int(st["conv1.weight"].shape[0]), num_qubits=5, n_levels=2)
model.load_state_dict(st)
model.eval()
torch.set_num_threads(16)
soft400 = np.load(str(LP.SOFT / "cnn_soft_400ns.npz"))["prob"].astype(np.float32)
best = None
for ds in (40, 20, 10, 8, 5, 4, 2, 1):
    OM.DS = ds
    Xtr = OM.build(np.ascontiguousarray(Xtr_all))
    Xte = OM.build(np.ascontiguousarray(Xte_all))
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-10
    Xn = (Xte - mu) / sd
    Ps = []
    with torch.no_grad():
        Xt = torch.tensor(Xn, dtype=torch.float32)
        for s in range(0, len(Xt), 8192):
            Ps.append(torch.sigmoid(model(Xt[s:s + 8192])).numpy())
    P = np.concatenate(Ps, 0)
    nerr = ((P > 0.5).astype(np.int64) != Yte.astype(np.int64)).sum(0).tolist()
    dev = float(np.sum(np.abs(np.array(nerr) - np.array([int(o["n_error"]) for o in old]))))
    res["scan"][str(ds)] = {"T_ds": int(Xtr.shape[2]), "n_error": nerr, "csv": [int(o["n_error"]) for o in old], "abs_dev": dev}
    OM.log("ds", ds, nerr, dev)
    if best is None or dev < best[0]:
        M = OM.masks(Xtr, Xte, Ytr)
        best = (dev, ds, OM.categorise(M, P, Yte, hard=False), OM.categorise(M, soft400, Yte, hard=False))
    json.dump(res, open(f"{OM.O}/em_e2seg_oldmethod_l200scan.json", "w"), indent=1)
res["best_ds"] = best[1]
res["hybridcnn_reproduced_best"] = best[2]
res["soft400_MCMitCNN_oldmasks_best_ds"] = best[3]
json.dump(res, open(f"{OM.O}/em_e2seg_oldmethod_l200scan.json", "w"), indent=1)
OM.log("done best ds", best[1])
