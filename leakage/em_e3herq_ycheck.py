import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import h5py, numpy as np, glob
f = h5py.File(str(LP.RAW_TEST), "r")
print("keys", list(f.keys()))
ky = [k for k in f.keys() if k.lower().startswith("y")]
y = np.asarray(f[ky[0]][()]).astype(np.int64).ravel() if ky else None
print("ykey", ky, y.shape, y[:10], np.bincount(y)[:5])
tbits = np.stack([(y >> q) & 1 for q in range(5)], 1)
tb = np.load(str(LP.REPORTS / "preds_MCMitCNN_tb.npy")).astype(np.int64)
print("CNN tb == raw y bits:", tb.shape, bool(np.array_equal(tb, tbits)))
for f2 in sorted(glob.glob(f"{LP.stage('em/e3herq')}/out/em_e3herq_[BC]_*ns.npz")):
    d = np.load(f2)
    print(f2.split("/")[-1], "test_state==raw y:", bool(np.array_equal(d["test_state"].astype(np.int64), y)),
          "rawrow identity:", bool(np.array_equal(d["test_rawrow"], np.arange(len(y)))))
for f2 in sorted(glob.glob(f"{LP.stage('em/e3herq')}/out/em_e3herq_A_*ns.npz")):
    d = np.load(f2)
    print(f2.split("/")[-1], "A test_state==y[test_rawrow]:", bool(np.array_equal(d["test_state"].astype(np.int64), y[d["test_rawrow"]])))
