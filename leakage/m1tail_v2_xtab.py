#!/usr/bin/env python
"""m1tail_v2_xtab.py (agent tag m1tail): cross-tabulation of a target qubit's one-sided off-axis flag
(z_b > 3) against the observed category of a source qubit (c0-like: u<0.5 & |z_b|<=3,
c1-like: u>=0.5 & z_b<=3, off-axis: z_b>3), for shots where both are prepared |1> (and target |1>,
source |0>). Tests whether the target's off-axis population coincides with the source being found
in |0> (excitation-exchange-like process) or in its own off-axis state."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m1tail_v2 as V  # noqa: E402

feat, outdir, split = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "test"
y, F, F1 = V.load(feat, split)
wm = V.window_means(F, F1)
del F, F1
res = {}
for w in ("1us", "2us"):
    Z = {}; U = {}
    for q in range(V.NQ):
        fr = V.build_frame(wm[w][:, q, 0], wm[w][:, q, 1], y, q)
        Z[q] = fr["zb"]; U[q] = fr["u"]
    bits = (y[:, None] >> np.arange(V.NQ)) & 1
    out = {}
    for t in range(V.NQ):
        for s in range(V.NQ):
            if s == t:
                continue
            for bs in (0, 1):
                m = (bits[:, t] == 1) & (bits[:, s] == bs)
                cat = np.where(Z[s][m] > 3, 2, np.where(U[s][m] < 0.5, 0, 1))
                ft = Z[t][m] > 3
                ftm = Z[t][m] < -3
                row = {}
                for c, nm in ((0, "src_c0like"), (1, "src_c1like"), (2, "src_offaxis")):
                    mc = cat == c
                    n = int(mc.sum())
                    row[nm] = dict(n=n, frac_of_shots=n / int(m.sum()),
                                   P_target_offaxis=(float(ft[mc].mean()) if n else None),
                                   mirror_E3_target=(float((ft[mc].sum() - ftm[mc].sum()) / n) if n else None))
                row["P_src_cat_given_target_offaxis"] = [float((cat[ft] == c).mean()) for c in (0, 1, 2)]
                row["P_src_cat_given_target_not"] = [float((cat[~ft] == c).mean()) for c in (0, 1, 2)]
                out[f"t{t}_s{s}_srcprep{bs}"] = row
    res[w] = out
    V.log("xtab", w)
json.dump(res, open(f"{outdir}/m1tail_v2_xtab_{split}.json", "w"), indent=1)
V.log("written")
