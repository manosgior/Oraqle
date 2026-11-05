"""m2mix_traj2.py -- time-resolved and acquisition-order diagnostics of the mixture components.

For each qubit and window (TEST split), every shot is mapped to its fit bin (same grid and the
same neighbour-shift correction as m2mix_fit2), and gets the per-bin posterior
responsibilities of models M1, M1f, M2f and M1fr12. Then:
  traj_soft[model]  (2 prepared bits, K comps, 20 time bins, 2) responsibility-weighted mean of
                    the 20-bin (along, perp) trace (feat_test, 100 ns bins over the 2 us trace)
  traj_hard[model]  same for shots with responsibility > 0.8 (+ std and counts)
  acquisition-order homogeneity of the extra-component share: each prepared 5-bit state's
  35000 shots (stored contiguously) are cut into 14 chunks of 2500; chi2 of hard counts
  (resp>0.5) against a constant rate, summed over the 16 states with the same own bit.
  per-state share across the 32 prepared states (neighbour dependence).
Usage: python m2mix_traj2.py /in /fit2 /feat /out
"""
import sys, json, numpy as np
from scipy.stats import chi2 as CHI2

IN, FIT, FEAT, OUT = sys.argv[1:5]
Z = np.load(f"{IN}/m2mix_winmeans_test.npz")
y = Z["y"]
F = np.load(f"{FEAT}/feat_test.npy", mmap_mode="r")
assert (np.load(f"{FEAT}/y_test.npy") == y).all()
MODELS = ["M1", "M1f", "M2f", "M1fr12"]
summary = {}
for q in range(5):
    tr = np.asarray(F[:, q]).astype(np.float64)        # (N, 20, 2)
    b = (y >> q) & 1
    for w in ("early", "1us", "2us"):
        J = json.load(open(f"{FIT}/m2mix_fit2_test_q{q}_{w}.json"))
        P = np.load(f"{FIT}/m2mix_plot2_test_q{q}_{w}.npz")
        g = J["grid"]
        x = Z[f"x_{w}"][:, q, :].astype(np.float64) - Z[f"shift_{w}"][y, q, :]
        ix = np.clip(((x - np.asarray(g["lo"])) / g["h"]).astype(np.int64), 0, [g["nx"] - 1, g["ny"] - 1])
        out, S = {}, {}
        for m in MODELS:
            R = P[f"{m}_resp"]                          # (2, K, nx, ny)
            names = [str(s) for s in P[f"{m}_groupnames"]]
            r = R[b, :, ix[:, 0], ix[:, 1]].astype(np.float64)   # (N, K)
            K = len(names)
            soft = np.zeros((2, K, 20, 2)); hard = np.zeros((2, K, 20, 2)); hsd = np.zeros((2, K, 20, 2))
            neff = np.zeros((2, K)); nh = np.zeros((2, K), np.int64)
            for bb in (0, 1):
                mb = b == bb
                rb, tb = r[mb], tr[mb]
                neff[bb] = rb.sum(0)
                soft[bb] = np.einsum("nk,ntc->ktc", rb, tb) / np.maximum(neff[bb], 1e-12)[:, None, None]
                for k in range(K):
                    hm = rb[:, k] > 0.8
                    nh[bb, k] = hm.sum()
                    if hm.sum() > 1:
                        hard[bb, k] = tb[hm].mean(0); hsd[bb, k] = tb[hm].std(0)
            out[f"{m}_soft"], out[f"{m}_hard"], out[f"{m}_hardsd"] = soft, hard, hsd
            out[f"{m}_neff"], out[f"{m}_nhard"], out[f"{m}_names"] = neff, nh, np.array(names)
            # acquisition order + neighbour dependence for the extra components
            res = J["results"][m]
            extras = [n for n in names if n in ("G2", "G3", "R12")]
            sm = {"names": names, "neff": neff.tolist(), "nhard08": nh.tolist()}
            if m == "M2f":
                sm["far"] = res.get("far")
            for nm in extras + ["R"]:
                k = names.index(nm)
                hardc = r[:, k] > 0.5
                per_state = np.array([hardc[y == s].mean() for s in range(32)])
                soft_state = np.array([r[y == s, k].mean() for s in range(32)])
                d = {"share_hard_by_state": per_state.tolist(), "share_soft_by_state": soft_state.tolist()}
                for bb in (0, 1):
                    states = [s for s in range(32) if ((s >> q) & 1) == bb]
                    c2, dof, fr = 0.0, 0, []
                    for s in states:
                        rows = np.nonzero(y == s)[0]
                        ch = hardc[rows].reshape(14, -1).sum(1).astype(float)   # 14 chunks x 2500
                        fr.append(ch / 2500.0)
                        E = ch.mean()
                        if E >= 5:
                            c2 += ((ch - E) ** 2 / E).sum(); dof += 13
                    n_s = np.array([hardc[y == s].sum() for s in states], float)
                    Es = n_s.mean()
                    c2s = float(((n_s - Es) ** 2 / Es).sum()) if Es >= 5 else None
                    d[f"prep{bb}"] = {
                        "chunk_chi2": c2, "chunk_dof": dof,
                        "chunk_p": float(CHI2.sf(c2, dof)) if dof else None,
                        "chunk_overdispersion": c2 / dof if dof else None,
                        "chunk_share_mean_over_states": np.mean(fr, 0).tolist(),
                        "state_chi2": c2s, "state_dof": len(states) - 1,
                        "state_p": float(CHI2.sf(c2s, len(states) - 1)) if c2s is not None else None,
                        "share_hard": float(hardc[b == bb].mean()), "n_hard": int(hardc[b == bb].sum()),
                        "N": int((b == bb).sum())}
                sm[nm] = d
            S[m] = sm
        np.savez_compressed(f"{OUT}/m2mix_traj2_q{q}_{w}.npz", **out)
        summary[f"q{q}_{w}"] = S
        print(q, w, json.dumps({m: {nm: [S[m][nm][f"prep{bb}"]["share_hard"] for bb in (0, 1)]
                                    for nm in S[m] if nm in ("G2", "G3", "R12")} for m in S}), flush=True)
json.dump(summary, open(f"{OUT}/m2mix_traj2_summary.json", "w"), indent=1)
print("done")
