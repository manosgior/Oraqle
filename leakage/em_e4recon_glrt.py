"""em_e4recon_glrt.py (tag e4recon): re-run the e2seg v3 step detector (GLRT, alpha-calibrated null) on ALL TEST shots
and attribute several prediction sets with the SAME detector, so that HERQULES errors can be attributed exactly like
MCMit-CNN errors.

Nothing in em_e2seg_attr_v3.py is changed: it is imported read-only as a module and its worker() is called verbatim.
Trick: the first prediction set of every window is "allshots" = 1 - prepared bit (every shot is an "error"), so the
per-shot store that worker() writes covers every TEST shot (Sd/Su, thresholds, FP at alpha = 0.05/0.01/0.002, mhat,
CNN proxy, z, cfg, prep). The same seeds/RNG order as e2seg v3 are used, so the canonical attribution must reproduce
em_e2seg_v3_raw_main.json exactly (checked in em_e4recon_analyze.py).

Extra outputs written here (deterministic, no RNG):
  * late-window core membership per shot (featbox bins 10-19, the e2seg v3 seeded mean-shift core, radius 1.5 MAD),
    used as the CNN-native / HERQULES-native "stationary" overlap null in em_e4recon_analyze.py;
  * the line-frame leakage z for 1 us (feat1us 0-9), 600 ns (featbox 0-5), 400 ns (featbox 0-3) for all shots.

Docker (graham):
  docker run -d --name leak-em-e4recon-glrt --user $(id -u):$(id -g) \
    -v ~/oraqle-ics-sims/leakage/features:/feat:ro -v ~/oraqle_reports:/reports:ro -v ~/oraqle-ics-sims/leakage/f/in:/soft:ro \
    -v ~/oraqle-ics-sims/leakage/em/e2seg/scripts:/e2seg:ro -v ~/oraqle-ics-sims/leakage/em/e4recon:/work \
    oraqle-torch:latest python /work/scripts/em_e4recon_glrt.py
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # was /e2seg (em_e2seg_attr_v3.py)
import em_e2seg_attr_v3 as E  # noqa: E402  (read-only import; module creates /work/out, which is our own dir)

OUT = str(LP.stage("em/e4recon") / "out")
os.makedirs(OUT, exist_ok=True)
E.O = OUT
HERQ = str(LP.stage("em/e4recon") / "in" / "em_e4recon_herq_in.npz")


def latecore_flags(Btr, Bte, q):
    """Replicates the e2seg v3 late-window core selection (deterministic), applied to TEST shots."""
    D = E.D
    cft, cfe = E.cfg_of(Btr, q), E.cfg_of(Bte, q)
    Xtr = np.asarray(np.load(f"{D}/featbox_train.npy", mmap_mode="r")[:, q, :10, :], dtype=np.float64)
    A1tr = Xtr.mean(1)
    Btr_late = np.asarray(np.load(f"{D}/featbox_train.npy", mmap_mode="r")[:, q, 10:20, :], dtype=np.float64).mean(1)
    Bte_late = np.asarray(np.load(f"{D}/featbox_test.npy", mmap_mode="r")[:, q, 10:20, :], dtype=np.float64).mean(1)
    lc = np.zeros((Bte.shape[0], 2), bool)
    dist = np.zeros((Bte.shape[0], 2), np.float32)
    info = []
    for c in range(16):
        s_late = float(E.mad(Btr_late[(cft == c) & (Btr[:, q] == 0), 1]))
        s_e = float(E.mad(A1tr[(cft == c) & (Btr[:, q] == 0), 1]))
        me = cfe == c
        for b in (0, 1):
            mb = np.where((cft == c) & (Btr[:, q] == b))[0]
            Pl = Btr_late[mb]
            ce = E.meanshift(A1tr[mb], 0.5 * s_e)
            ecore = np.linalg.norm(A1tr[mb] - ce, axis=1) < 1.0 * s_e
            seed = np.median(Pl[ecore], 0)
            cb = E.meanshift(Pl, 0.5 * s_late, seed=seed)
            d = np.linalg.norm(Bte_late[me] - cb, axis=1) / s_late
            dist[me, b] = d
            lc[me, b] = d < E.LATE_R
            info.append({"c": c, "b": b, "core": cb.tolist(), "s_late": s_late,
                         "frac_test_prepb_in_core": float(lc[me & (Bte[:, q] == b), b].mean())})
    return lc, dist, info


def main():
    t0 = time.time()
    D = E.D
    ytr, yte = np.load(D + "/y_train.npy"), np.load(D + "/y_test.npy")
    Btr, Bte = E.bits(ytr), E.bits(yte)
    E.G["Btr"], E.G["Bte"] = Btr, Bte
    LW = {"1us": ("feat1us", 0, 10), "600ns": ("featbox", 0, 6), "400ns": ("featbox", 0, 4)}
    E.G["z"] = {}
    for wn, (src, b0, b1) in LW.items():
        Atr, Ate = E.wmean(src, "train", b0, b1), E.wmean(src, "test", b0, b1)
        fr = E.fit_line(Atr, Btr)
        E.G["z"][wn] = E.zline(Ate, Bte, fr)
        E.log("frame", wn)
    np.savez_compressed(f"{OUT}/em_e4recon_z.npz", **{f"z_{wn}": E.G["z"][wn].astype(np.float32) for wn in LW})
    pb = np.load(E.R + "/preds_MCMitCNN_pb.npy").astype(np.int8)
    tb = np.load(E.R + "/preds_MCMitCNN_tb.npy").astype(np.int8)
    assert np.array_equal(tb, Bte)
    soft = {T: np.load(f"{E.S}/cnn_soft_{T}ns.npz") for T in (1000, 600, 400)}
    E.G["soft"] = {T: soft[T]["prob"].astype(np.float32) for T in soft}
    H = np.load(HERQ)
    chk = {}
    for w in ("1us", "400ns"):
        chk[f"herq_y_{w}_equals_y_test"] = bool(np.array_equal(H[f"y_{w}"].astype(np.int64), yte.astype(np.int64)))
        assert chk[f"herq_y_{w}_equals_y_test"]
        assert (H[f"pt_pred_{w}"] != 255).all() and (H[f"scr_pred_{w}"] != 255).all()
    allshots = (1 - Bte).astype(np.int8)
    E.G["preds"] = {
        "1us": {"allshots": allshots, "canonical": pb, "soft1000": (soft[1000]["prob"] > 0.5).astype(np.int8),
                "herq_pertrace": H["pt_pred_1us"].astype(np.int8), "herq_scrambled_mfsrc": H["scr_pred_1us"].astype(np.int8)},
        "600ns": {"allshots": allshots, "soft600": (soft[600]["prob"] > 0.5).astype(np.int8)},
        "400ns": {"allshots": allshots, "soft400": (soft[400]["prob"] > 0.5).astype(np.int8),
                  "herq_pertrace": H["pt_pred_400ns"].astype(np.int8), "herq_scrambled_mfsrc": H["scr_pred_400ns"].astype(np.int8)},
    }
    E.G["variants"] = {
        "box1us": {"K": 10, "src": "featbox", "win": "1us", "pred_win": "1us", "fir": False, "soft": 1000},
        "box600": {"K": 6, "src": "featbox", "win": "600ns", "pred_win": "600ns", "fir": False, "soft": 600},
        "box400": {"K": 4, "src": "featbox", "win": "400ns", "pred_win": "400ns", "fir": False, "soft": 400},
    }
    qs = [int(x) for x in os.environ.get("E4_QUBITS", "0,1,2,3,4").split(",")]
    vl = os.environ.get("E4_VARIANTS", "box1us,box600,box400").split(",")
    tasks = [(v, q) for v in vl for q in qs]
    with Pool(min(int(os.environ.get("E4_NPROC", "15")), len(tasks))) as pool:
        outs = pool.map(E.worker, tasks)
    # rename the per-shot stores (our own files, written by worker() under the e2seg name into OUT)
    for v, q in tasks:
        src = f"{OUT}/em_e2seg_v3_shots_{v}_q{q}.npz"
        if os.path.exists(src):
            os.replace(src, f"{OUT}/em_e4recon_allshots_{v}_q{q}.npz")
    lcs, lcinfo = {}, {}
    for q in qs:
        lc, dist, info = latecore_flags(Btr, Bte, q)
        lcs[f"latecore_q{q}"] = lc
        lcs[f"latedist_q{q}"] = dist
        lcinfo[f"q{q}"] = info
    np.savez_compressed(f"{OUT}/em_e4recon_latecore.npz", **lcs)
    res = {"note": __doc__, "checks": chk, "alpha0": E.ALPHA0, "alphas": E.ALPHAS, "NB": E.NB, "latecore_info": lcinfo,
           "variants": {}}
    for o in outs:
        res["variants"].setdefault(o["variant"], {})[f"q{o['q']}"] = o
    json.dump(res, open(f"{OUT}/em_e4recon_glrt_raw.json", "w"), indent=1, default=float)
    E.log("all done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
