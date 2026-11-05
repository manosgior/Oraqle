"""m4conf_pairflip.py -- does qubit j's prepared-|1> 'flip' (reads on the |0> side in 1 us) depend on whether
qubit q is prepared |1>, and does the excess of j-flips match q's off-axis excursions? (agent tag m4conf)
Tests the |1_j 1_q> <-> |0_j 2_q> exchange reading of the Q3-flip / Q5-off-axis coincidence.
Counts: early = flip already in the first 200 ns window (tA_early < 0.5) and in 1 us (tA < 0.5)."""
import json, sys
import numpy as np
OUT = sys.argv[1]
D = np.load(f"{OUT}/m4conf_pershot.npz")
y = D["y"]; tA = D["tA"]; tE = D["tA_early"]; zA0 = D["zA0"]; A = D["A"]
bits = ((y[:, None] >> np.arange(5)) & 1)
res = {}
for j in range(5):
    for q in range(5):
        if j == q:
            continue
        m1 = (bits[:, j] == 1) & (bits[:, q] == 1); m0 = (bits[:, j] == 1) & (bits[:, q] == 0)
        flip = tA[:, j] < 0.5
        early = flip & (tE[:, j] < 0.5)
        offq = np.abs(zA0[:, q]) > 5
        e = {"P_flip_j|q=1": float(flip[m1].mean()), "P_flip_j|q=0": float(flip[m0].mean()),
             "P_earlyflip_j|q=1": float(early[m1].mean()), "P_earlyflip_j|q=0": float(early[m0].mean()),
             "excess_earlyflip_per_shot": float(early[m1].mean() - early[m0].mean()),
             "P_offq_noise0scale|q=1,j=1": float(offq[m1].mean()), "P_offq|q=1,j=0": float(offq[(bits[:, j] == 0) & (bits[:, q] == 1)].mean()),
             "P_Aq|q=1,j=1,earlyflip": float(A[m1 & early, q].mean()) if (m1 & early).any() else None,
             "P_Aq|q=1,j=1,noflip": float(A[m1 & ~flip, q].mean()),
             "n_q1j1": int(m1.sum())}
        res[f"j=q{j}(Q{j+1}) q=q{q}(Q{q+1})"] = e
json.dump(res, open(f"{OUT}/m4conf_pairflip.json", "w"), indent=1)
for k, e in res.items():
    print(k, " ".join(f"{a}={b:.5f}" if isinstance(b, float) else f"{a}={b}" for a, b in e.items()))
