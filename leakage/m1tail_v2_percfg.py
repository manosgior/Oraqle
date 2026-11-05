#!/usr/bin/env python
"""m1tail_v2_percfg.py (agent tag m1tail): per-neighbour-configuration consistency of the one-sided
off-axis excess. For every window / qubit / prepared bit and each of the 16 configurations of the
other four qubits' prepared bits: mirror excess E(3) and E(4) of z_b, count of d_seg > 5, and the
centroid (u, v) in separation units of the off-segment shots (d_seg > 5, perp > 0).
Also the Q3-conditioned split: E(3) of each qubit vs the prepared bit of every other qubit."""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m1tail_v2 as V  # noqa: E402


def main():
    feat, outdir, split = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "test"
    y, F, F1 = V.load(feat, split)
    wm = V.window_means(F, F1)
    del F, F1
    res = {}
    for w in ("early", "1us", "2us"):
        for q in range(V.NQ):
            fr = V.build_frame(wm[w][:, q, 0], wm[w][:, q, 1], y, q)
            R = {}
            for bb in (0, 1):
                m = fr["b"] == bb
                zb, d, u, v, cfg = fr["zb"][m], fr["d"][m], fr["u"][m], fr["v"][m], fr["cfg"][m]
                yy = fr["y"][m]
                rows = []
                for c in range(16):
                    mc = cfg == c
                    n = int(mc.sum())
                    p3, m3 = int((zb[mc] > 3).sum()), int((zb[mc] < -3).sum())
                    p4, m4 = int((zb[mc] > 4).sum()), int((zb[mc] < -4).sum())
                    ex = mc & (d > 5) & (yy > 0)
                    ne = int(ex.sum())
                    rows.append(dict(cfg=c, N=n, E3=(p3 - m3) / n, E3_se=np.sqrt(p3 + m3) / n, E4=(p4 - m4) / n,
                                     n_dseg5_plus=ne,
                                     centroid_uv=([float(u[ex].mean()), float(v[ex].mean())] if ne >= 10 else None)))
                E3 = np.array([r["E3"] for r in rows]); se = np.array([r["E3_se"] for r in rows])
                wts = 1 / np.maximum(se, 1e-12) ** 2
                mean = float((E3 * wts).sum() / wts.sum())
                chi2 = float((((E3 - mean) / np.maximum(se, 1e-12)) ** 2).sum())
                cents = np.array([r["centroid_uv"] for r in rows if r["centroid_uv"] is not None])
                # dependence on each other qubit's prepared bit
                dep = {}
                for j in range(V.NQ):
                    if j == q:
                        continue
                    bj = (y[m] >> j) & 1
                    out = {}
                    for vb in (0, 1):
                        mj = bj == vb
                        p3, m3 = int((zb[mj] > 3).sum()), int((zb[mj] < -3).sum())
                        out[f"E3_given_q{j}={vb}"] = (p3 - m3) / int(mj.sum())
                        out[f"E3_se_given_q{j}={vb}"] = float(np.sqrt(p3 + m3) / int(mj.sum()))
                    dep[f"q{j}"] = out
                R[f"prep{bb}"] = dict(rows=rows, E3_weighted_mean=mean, E3_cfg_chi2=chi2,
                                      E3_cfg_chi2_p=float(stats.chi2.sf(chi2, 15)),
                                      E3_cfg_min=float(E3.min()), E3_cfg_max=float(E3.max()),
                                      E3_cfg_rel_sd=float(E3.std() / abs(E3.mean())) if E3.mean() != 0 else None,
                                      centroid_uv_cfg_mean=(cents.mean(0).tolist() if len(cents) else None),
                                      centroid_uv_cfg_sd=(cents.std(0).tolist() if len(cents) > 2 else None),
                                      n_cfg_with_centroid=int(len(cents)), by_other_qubit_bit=dep)
            res[f"{w}_q{q}"] = R
            V.log("percfg", w, q)
    with open(f"{outdir}/m1tail_v2_percfg_{split}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    V.log("written")


if __name__ == "__main__":
    main()
