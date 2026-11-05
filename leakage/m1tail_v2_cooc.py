#!/usr/bin/env python
"""m1tail_v2_cooc.py (agent tag m1tail): shot-level co-occurrence of one-sided off-axis flags
between qubits, conditioned on the prepared bits. Flag_q = z_b(q) > 3 (H0-standardised perp).
For each ordered pair (target q, source j) and prepared bits (bq, bj):
   P(flag_q | flag_j), P(flag_q | not flag_j), lift, and the share of q's off-axis EXCESS
   (mirror: N(z>3) - N(z<-3)) that sits in shots where j is flagged.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m1tail_v2 as V  # noqa: E402


def main():
    feat, outdir, split = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "test"
    y, F, F1 = V.load(feat, split)
    wm = V.window_means(F, F1)
    del F, F1
    res = {}
    for w in ("1us", "2us"):
        Z = np.zeros((len(y), V.NQ), np.float32)
        for q in range(V.NQ):
            fr = V.build_frame(wm[w][:, q, 0], wm[w][:, q, 1], y, q)
            Z[:, q] = fr["zb"]
            V.log("frame", w, q)
        bits = (y[:, None] >> np.arange(V.NQ)) & 1
        out = {}
        for q in range(V.NQ):
            for j in range(V.NQ):
                if j == q:
                    continue
                for bq in (0, 1):
                    for bj in (0, 1):
                        m = (bits[:, q] == bq) & (bits[:, j] == bj)
                        fq_p = Z[m, q] > 3
                        fq_m = Z[m, q] < -3
                        fj = Z[m, j] > 3
                        n_j = int(fj.sum()); n_nj = int((~fj).sum())
                        pq_j = float(fq_p[fj].mean()) if n_j else None
                        pq_nj = float(fq_p[~fj].mean())
                        mir_all = int(fq_p.sum()) - int(fq_m.sum())
                        mir_in_j = int((fq_p & fj).sum()) - int((fq_m & fj).sum())
                        out[f"q{q}<-q{j}_b{bq}{bj}"] = dict(
                            N=int(m.sum()), n_flag_j=n_j, P_flag_q_given_flag_j=pq_j,
                            P_flag_q_given_not_flag_j=pq_nj,
                            lift=(pq_j / pq_nj if pq_j is not None and pq_nj > 0 else None),
                            mirror_excess_q=mir_all / int(m.sum()),
                            mirror_excess_q_given_not_flag_j=(int((fq_p & ~fj).sum()) - int((fq_m & ~fj).sum())) / max(n_nj, 1),
                            share_of_q_excess_in_flag_j_shots=(mir_in_j / mir_all if mir_all > 0 else None))
        res[w] = out
    with open(f"{outdir}/m1tail_v2_cooc_{split}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    V.log("written")


if __name__ == "__main__":
    main()
