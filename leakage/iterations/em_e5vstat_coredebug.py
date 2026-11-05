"""em_e5vstat_coredebug.py (tag e5vstat): per-cfg comparison of the histogram-global-mode late cores used in
em_e5vstat_check.py with median-seeded mean-shift cores (Q3 prepared |1> discrepancy)."""
import sys
import numpy as np
sys.path.insert(0, "/work/scripts")
import em_e5vstat_check as C  # noqa: E402

q = int(sys.argv[1]) if len(sys.argv) > 1 else 2
D = "/feat"
ytr = np.load(D + "/y_train.npy")
Btr = C.bits(ytr)
cft, ptr = C.cfg_of(Btr, q), Btr[:, q]
Xtr = np.asarray(np.load(f"{D}/featbox_train.npy", mmap_mode="r")[:, q], dtype=np.float64)
Ltr, S59 = Xtr[:, 10:20].mean(1), Xtr[:, 5:10].mean(1)
for c in range(16):
    m0, m1 = (cft == c) & (ptr == 0), (cft == c) & (ptr == 1)
    s = C.mad(Ltr[m0, 1])
    c0h, a0h, a1h = C.mode2d(Ltr[m0], s), C.mode2d(S59[m0], s), C.mode2d(S59[m1], s)
    c0m = C.mode2d(Ltr[m0], s, seed=np.median(Ltr[m0], 0))
    a0m = C.mode2d(S59[m0], s, seed=np.median(S59[m0], 0))
    a1m = C.mode2d(S59[m1], s, seed=np.median(S59[m1], 0))
    c1h = C.mode2d(Ltr[m1], s, seed=c0h + (a1h - a0h))
    c1m = C.mode2d(Ltr[m1], s, seed=c0m + (a1m - a0m))
    sep = np.linalg.norm(c1m - c0m)
    print(c, "c0 diff/s %.3f" % (np.linalg.norm(c0h - c0m) / s), "a0 %.3f" % (np.linalg.norm(a0h - a0m) / s),
          "a1 %.3f" % (np.linalg.norm(a1h - a1m) / s), "c1 diff/s %.3f" % (np.linalg.norm(c1h - c1m) / s),
          "c1h-c0m in sep units", np.round((c1h - c0m) / sep, 3), "c1m-c0m", np.round((c1m - c0m) / sep, 3), "sep/s %.2f" % (sep / s), flush=True)
