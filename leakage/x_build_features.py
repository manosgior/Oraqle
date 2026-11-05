"""x_build_features.py -- step 2: per-shot binned IQ features for EVERY shot.

For each split (train 480000, test 1120000 shots) writes to OUT:
  feat_<split>.npy     float32 (N, 5, 20, 2)  mixed -> 101-tap FIR ('same') -> rotate -> mean of 20 x 50-sample bins
  featbox_<split>.npy  float32 (N, 5, 20, 2)  mixed -> rotate -> mean of 20 x 50-sample bins (no FIR)
  feat1us_<split>.npy  float32 (N, 5, 10, 2)  FIR applied to samples 0..499 only (zero pad at 500, = CNN 1us input)
  y_<split>.npy        int32   (N,)           prepared 5-bit label, copied from y_<split> of the raw file
  idx_<split>.npy      int64   (N,)           row index into the raw HDF5 (X_<split>[idx])
[..., 0] = along the |0>->|1> axis (Re), [..., 1] = perpendicular (Im).
Parameters come from calib.json "used_for_features" (or --params colleague for the
raw calibrate(0,1000) output with the colleague theta, written to a separate dir).

Run (inside oraqle-torch:latest, data at /data, OUT writable):
  python /scripts/x_build_features.py --out /out --workers 24
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import argparse, json, os, sys, time, hashlib
from multiprocessing import Pool
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x_common import FILES, NUM_QUBITS, N_BINS, N_BINS_1US, chunk_features

_H = {}


def _work(task):
    split, a, b, params = task
    path, kx, ky, _ = FILES[split]
    if split not in _H:
        _H[split] = h5py.File(path, "r")
    X = _H[split][kx][a:b]
    f, fb, f1 = chunk_features(X, params)
    return split, a, b, f, fb, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(LP.FEAT))
    ap.add_argument("--calib", default=None, help="calib.json (default OUT/calib.json)")
    ap.add_argument("--params", default="used", choices=["used", "colleague"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first LIMIT rows per split")
    a = ap.parse_args()
    calib = json.load(open(a.calib or os.path.join(a.out, "calib.json")))
    if a.params == "used":
        params = [{"f": u["f_Hz"], "theta": u["theta_rad"]} for u in calib["used_for_features"]]
    else:
        params = [{"f": u["f_Hz"], "theta": u["theta_colleague_rad"]} for u in calib["colleague_calibrate"]["full_0_1000"]]
    print(f"[build] params ({a.params}): " + ", ".join(f"q{q}: f={p['f']/1e6:+.4f} MHz th={p['theta']:+.4f}"
                                                    for q, p in enumerate(params)), flush=True)
    os.makedirs(a.out, exist_ok=True)
    meta = {"params": params, "params_source": a.params, "chunk": a.chunk, "splits": {}}
    t0 = time.time()
    tasks, mm = [], {}
    for split in a.splits.split(","):
        path, kx, ky, _ = FILES[split]
        with h5py.File(path, "r") as hf:
            N = hf[kx].shape[0] if not a.limit else a.limit
            y = hf[ky][:N].astype(np.int32)
        np.save(os.path.join(a.out, f"y_{split}.npy"), y)
        np.save(os.path.join(a.out, f"idx_{split}.npy"), np.arange(N, dtype=np.int64))
        O = lambda nm, shp: np.lib.format.open_memmap(os.path.join(a.out, f"{nm}_{split}.npy"), mode="w+",
                                                     dtype=np.float32, shape=shp)
        mm[split] = (O("feat", (N, NUM_QUBITS, N_BINS, 2)), O("featbox", (N, NUM_QUBITS, N_BINS, 2)),
                     O("feat1us", (N, NUM_QUBITS, N_BINS_1US, 2)), np.zeros(N, bool))
        meta["splits"][split] = {"N": int(N), "raw_file": path, "raw_X": kx, "raw_y": ky}
        tasks += [(split, s, min(s + a.chunk, N), params) for s in range(0, N, a.chunk)]
    print(f"[build] {len(tasks)} chunks, {a.workers} workers", flush=True)
    done = 0
    with Pool(a.workers) as pool:
        for split, s, e, f, fb, f1 in pool.imap_unordered(_work, tasks):
            F, FB, F1, seen = mm[split]
            F[s:e] = f; FB[s:e] = fb; F1[s:e] = f1; seen[s:e] = True
            done += 1
            if done % 40 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"[build] {done}/{len(tasks)} chunks  {el:.0f}s  eta {el/done*(len(tasks)-done):.0f}s", flush=True)
    for split, (F, FB, F1, seen) in mm.items():
        F.flush(); FB.flush(); F1.flush()
        assert seen.all(), f"{split}: {np.sum(~seen)} rows not written"
        for nm, arr in [("feat", F), ("featbox", FB), ("feat1us", F1)]:
            bad = 0; zero = 0
            for s in range(0, arr.shape[0], 100000):
                blk = np.asarray(arr[s:s + 100000])
                bad += int((~np.isfinite(blk)).sum())
                zero += int((np.abs(blk).reshape(blk.shape[0], -1).max(1) == 0).sum())
            meta["splits"][split][f"{nm}_nonfinite"] = bad
            meta["splits"][split][f"{nm}_allzero_rows"] = zero
            meta["splits"][split][f"{nm}_shape"] = list(arr.shape)
            print(f"[build] {split} {nm} {arr.shape}: nonfinite={bad} allzero_rows={zero}", flush=True)
    sd = os.path.dirname(os.path.abspath(__file__))
    meta["script_md5"] = {fn: hashlib.md5(open(os.path.join(sd, fn), "rb").read()).hexdigest()
                          for fn in ["x_build_features.py", "x_common.py", "x_calib.py", "x_colleague_prep_verbatim.py"]}
    meta["elapsed_s"] = time.time() - t0
    meta["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(meta, open(os.path.join(a.out, "build_meta.json"), "w"), indent=1)
    print(f"[build] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
