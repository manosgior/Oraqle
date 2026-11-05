"""x_crosscheck.py -- verify the vectorised feature chain against the VERBATIM
MCMit _colleague_prep.preprocess() (per-index FIR dot-product loop), and verify
that the stored feature files match a fresh recomputation from the raw HDF5.

  python /scripts/x_crosscheck.py /out
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json, os, sys, tempfile
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x_colleague_prep_verbatim as P
from x_common import FILES, BIN, N_BINS, N_BINS_1US, chunk_features, fir_same, FS

OUT = sys.argv[1] if len(sys.argv) > 1 else str(LP.FEAT)
calib = json.load(open(os.path.join(OUT, "calib.json")))
params = [{"f": u["f_Hz"], "theta": u["theta_rad"]} for u in calib["used_for_features"]]
rng = np.random.default_rng(7)
res = {}

# --- (1) chain vs verbatim preprocess() on 96 random train shots ---------------
path, kx, ky, _ = FILES["train"]
rows = np.sort(rng.choice(480000, 96, replace=False))
with h5py.File(path, "r") as hf:
    X = np.stack([hf[kx][r] for r in rows]); y = np.array([hf[ky][r] for r in rows])
tmp = os.path.join(tempfile.mkdtemp(), "x_sub.h5")
with h5py.File(tmp, "w") as hf:
    hf["X_train"] = X; hf["y_train"] = y
feat, featbox, feat1us = chunk_features(X, params)
for name, sl, ours, nb in [("full_0_1000", (0, 1000), feat, N_BINS), ("first_0_500", (0, 500), feat1us, N_BINS_1US)]:
    Xo, _ = P.preprocess(tmp, "X_train", "y_train", params, 1, sl)       # (n, T, 1, 10) float32, per-sample
    re, im = Xo[:, :, 0, :5], Xo[:, :, 0, 5:]                              # (n, T, 5)
    binned = np.stack([re, im], -1).reshape(len(rows), nb, BIN, 5, 2).mean(2).transpose(0, 2, 1, 3)  # (n,5,nb,2)
    # per-sample comparison too
    Xc = X[:, sl[0]:sl[1], 0] + 1j * X[:, sl[0]:sl[1], 1]
    t = np.arange(Xc.shape[1]) / FS
    maxdiff_sample = 0.0
    for q, p in enumerate(params):
        z = fir_same(Xc * np.exp(-1j * 2 * np.pi * p["f"] * t)) * np.exp(1j * p["theta"])
        maxdiff_sample = max(maxdiff_sample, float(np.max(np.abs(z.real - re[:, :, q]))),
                             float(np.max(np.abs(z.imag - im[:, :, q]))))
    d = float(np.max(np.abs(binned - ours)))
    scale = float(np.std(ours))
    res[f"vs_colleague_preprocess_{name}"] = {"max_abs_diff_binned": d, "max_abs_diff_per_sample": maxdiff_sample,
                                             "feature_std": scale}
    print(f"[xcheck] {name}: max|ours - colleague preprocess| binned={d:.3e} per-sample={maxdiff_sample:.3e} "
          f"(feature std {scale:.2f}; float32 output of preprocess)", flush=True)

# --- (2) boxcar variant equals a direct mean of mixed*rot -----------------------
Xc = X[:, :, 0] + 1j * X[:, :, 1]; t = np.arange(1000) / FS
mb = 0.0
for q, p in enumerate(params):
    z = (Xc * np.exp(-1j * 2 * np.pi * p["f"] * t) * np.exp(1j * p["theta"])).reshape(len(rows), N_BINS, BIN).mean(2)
    mb = max(mb, float(np.max(np.abs(z.real - featbox[:, q, :, 0]))), float(np.max(np.abs(z.imag - featbox[:, q, :, 1]))))
res["boxcar_direct_max_abs_diff"] = mb
print(f"[xcheck] boxcar direct max diff {mb:.3e}", flush=True)

# --- (3) stored files vs fresh recomputation (random rows of each split) --------
for split in ["train", "test"]:
    fp = os.path.join(OUT, f"feat_{split}.npy")
    if not os.path.exists(fp):
        continue
    F = np.load(fp, mmap_mode="r"); FB = np.load(os.path.join(OUT, f"featbox_{split}.npy"), mmap_mode="r")
    F1 = np.load(os.path.join(OUT, f"feat1us_{split}.npy"), mmap_mode="r")
    Y = np.load(os.path.join(OUT, f"y_{split}.npy")); I = np.load(os.path.join(OUT, f"idx_{split}.npy"))
    path, kx, ky, _ = FILES[split]
    rr = np.sort(rng.choice(F.shape[0], 300, replace=False))
    with h5py.File(path, "r") as hf:
        Xs = np.stack([hf[kx][I[r]] for r in rr]); ys = np.array([hf[ky][I[r]] for r in rr])
        yall = hf[ky][:len(Y)]
    f, fb, f1 = chunk_features(Xs, params)
    dd = [float(np.max(np.abs(f - F[rr]))), float(np.max(np.abs(fb - FB[rr]))), float(np.max(np.abs(f1 - F1[rr])))]
    res[f"stored_{split}"] = {"max_abs_diff_feat_featbox_feat1us": dd, "labels_equal_sample": bool((ys == Y[rr]).all()),
                              "labels_equal_all": bool((yall == Y).all()), "idx_is_arange": bool((I == np.arange(len(I))).all())}
    print(f"[xcheck] stored {split}: max diff feat/featbox/feat1us = {dd}; labels equal: {(ys == Y[rr]).all()} "
          f"(all rows: {(yall == Y).all()}); idx==arange: {(I == np.arange(len(I))).all()}", flush=True)

json.dump(res, open(os.path.join(OUT, "crosscheck.json"), "w"), indent=1)
print("[xcheck] wrote crosscheck.json", flush=True)
