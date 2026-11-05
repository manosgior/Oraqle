#!/usr/bin/env python
"""m1tail_v2_models.py (agent tag m1tail): heavy-tail model comparison on the v2 standardised
perpendicular coordinate z_b (H0 width) and z_a (own-state width), per window / qubit / prepared bit.
Gaussian vs Student-t vs GMM k=1..3, BIC on a random 100k subsample (seeded), parallel over cases.
Uses the frame of m1tail_v2.build_frame (same directory)."""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m1tail_v2 as V  # noqa: E402



def fit_case(task):
    key, zb, za = task
    rng = np.random.default_rng(0)
    out = {}
    for name, z in (("zb", zb), ("za", za)):
        sub = z if len(z) <= 100000 else rng.choice(z, 100000, replace=False)
        out[name] = V.fit_models_1d(sub)
        # heavy-tail summary: weight of the non-core GMM2 component and its mean (sign = side)
        g2 = out[name]["gmm2"]
        out[name]["gmm2_minor_weight"] = g2["weights"][1]
        out[name]["gmm2_minor_mean"] = g2["means"][1]
        out[name]["dBIC_t_minus_gauss"] = out[name]["student_t"]["bic"] - out[name]["gauss"]["bic"]
        out[name]["dBIC_gmm2_minus_gauss"] = g2["bic"] - out[name]["gauss"]["bic"]
    return key, out


def main():
    feat, outdir, split = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "test"
    y, F, F1 = V.load(feat, split)
    wm = V.window_means(F, F1)
    del F, F1
    tasks = []
    for w in ("early", "1us", "2us"):
        for q in range(V.NQ):
            fr = V.build_frame(wm[w][:, q, 0], wm[w][:, q, 1], y, q)
            for bb in (0, 1):
                m = fr["b"] == bb
                tasks.append((f"{w}_q{q}_b{bb}", fr["zb"][m].copy(), fr["za"][m].copy()))
            V.log("frame", w, q)
    del wm
    with Pool(15) as pool:
        res = dict(pool.map(fit_case, tasks, chunksize=1))
    with open(f"{outdir}/m1tail_v2_models_{split}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    V.log("models written")


if __name__ == "__main__":
    main()
