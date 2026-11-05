"""x_xtalk_diag.py -- is the off-axis structure crosstalk or a per-qubit third state?

(a) per-32-state conditional centroids: for each qubit q and full prepared state s,
    median along/perp and MAD sigmas (TEST, 35000 shots per state). If the |b>
    centroid of qubit q moves with the OTHER qubits' prepared bits, that is
    state-dependent crosstalk (would show up as sub-clouds, each ~50/25/..% of shots).
(b) shot-level co-occurrence of off-axis events: flag_q = |perp - med_perp[s,q]| /
    sig_perp[s,q] > K (state-conditional centroid). Lift = P(flag_q & flag_r) /
    (P(flag_q) P(flag_r)). Lift >> 1 means the same physical shot is off-axis in two
    channels (one event seen twice, e.g. mixer-image crosstalk between the mirror IF
    pairs Q2/Q3 (+25.39/-24.76 MHz) and Q1/Q4 (+64.73/-70.26 MHz)).
(c) per-bin cross-channel transfer of qubit r's state into qubit q's channel.
Windows: 1us (feat bins 0..9) and full_2us (bins 0..19). Writes OUT/xtalk.json.
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json, os, sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else str(LP.FEAT)
NQ = 5
F = np.load(os.path.join(OUT, "feat_test.npy"), mmap_mode="r")
y = np.load(os.path.join(OUT, "y_test.npy"))
res = {"note": __doc__, "windows": {}}
for w, (a, b) in {"1us": (0, 10), "full_2us": (0, 20)}.items():
    M = np.asarray(F[:, :, a:b, :], dtype=np.float64).mean(2)        # (N,5,2)
    med = np.zeros((32, NQ, 2)); sig = np.zeros((32, NQ, 2))
    for s in range(32):
        m = M[y == s]
        med[s] = np.median(m, 0)
        sig[s] = 1.4826 * np.median(np.abs(m - med[s]), 0)
    R = {"cond_median_along": med[:, :, 0].tolist(), "cond_median_perp": med[:, :, 1].tolist(),
         "cond_sigma_along_MAD": sig[:, :, 0].tolist(), "cond_sigma_perp_MAD": sig[:, :, 1].tolist(),
         "state_index_note": "row = prepared 5-bit state s (bit q = code qubit q = paper Q(q+1)), col = code qubit"}
    # spread of own-state centroid across the 16 configurations of the other qubits, in units of sigma
    spread = {}
    for q in range(NQ):
        for bit in (0, 1):
            ss = [s for s in range(32) if ((s >> q) & 1) == bit]
            ma = med[ss, q, 0]; mp = med[ss, q, 1]
            sa = np.median(sig[ss, q, 0]); sp = np.median(sig[ss, q, 1])
            # which other qubit's bit explains most of the variation?
            expl = {}
            for r in range(NQ):
                if r == q:
                    continue
                g1 = [i for i, s in enumerate(ss) if (s >> r) & 1]; g0 = [i for i, s in enumerate(ss) if not (s >> r) & 1]
                expl[f"q{r}"] = {"d_along_sigma": float((ma[g1].mean() - ma[g0].mean()) / sa),
                                 "d_perp_sigma": float((mp[g1].mean() - mp[g0].mean()) / sp)}
            spread[f"q{q}_prep{bit}"] = {"range_along_sigma": float((ma.max() - ma.min()) / sa),
                                         "range_perp_sigma": float((mp.max() - mp.min()) / sp),
                                         "effect_of_other_qubit_bit_1_minus_0": expl}
    R["own_centroid_spread_over_other_qubits"] = spread
    # off-axis flags relative to the 32-state conditional centroid
    co = {}
    for K in (4.0, 6.0):
        z = np.abs(M[:, :, 1] - med[y, :, 1]) / sig[y, :, 1]        # (N,5)
        fl = z > K
        p = fl.mean(0)
        lift = np.full((NQ, NQ), np.nan); both = np.zeros((NQ, NQ), int)
        for q in range(NQ):
            for r in range(NQ):
                both[q, r] = int((fl[:, q] & fl[:, r]).sum())
                if q != r and p[q] > 0 and p[r] > 0:
                    lift[q, r] = both[q, r] / len(y) / (p[q] * p[r])
        # fraction of each qubit's flags that coincide with a flag on any other qubit
        anyother = {f"q{q}": float((fl[:, q] & np.delete(fl, q, 1).any(1)).sum() / max(fl[:, q].sum(), 1)) for q in range(NQ)}
        co[f"K{K:g}"] = {"flag_count": fl.sum(0).tolist(), "N": int(len(y)), "flag_frac": p.tolist(),
                         "both_count": both.tolist(), "lift": np.where(np.isnan(lift), None, lift).tolist(),
                         "frac_of_flags_coincident_with_other_qubit_flag": anyother}
    R["offaxis_cooccurrence"] = co
    # per-shot residual correlation between channels (state-conditional centroids removed)
    resid = M - med[y]
    cc = np.corrcoef(np.concatenate([resid[:, :, 0], resid[:, :, 1]], 1).T)
    R["residual_corr_labels"] = [f"q{q}_along" for q in range(NQ)] + [f"q{q}_perp" for q in range(NQ)]
    R["residual_corr"] = cc.tolist()
    res["windows"][w] = R
    print(f"=== window {w}", flush=True)
    for q in range(NQ):
        for bit in (0, 1):
            S = spread[f"q{q}_prep{bit}"]
            big = {k: (round(v["d_along_sigma"], 2), round(v["d_perp_sigma"], 2)) for k, v in S["effect_of_other_qubit_bit_1_minus_0"].items()
                   if abs(v["d_along_sigma"]) > 0.1 or abs(v["d_perp_sigma"]) > 0.1}
            print(f"  q{q}/Q{q+1} prep{bit}: centroid range over other-qubit states: along {S['range_along_sigma']:.2f} sig, "
                  f"perp {S['range_perp_sigma']:.2f} sig; effects >0.1 sig (d_along, d_perp): {big}", flush=True)
    for K in ("K4", "K6"):
        C = co[K]
        print(f"  off-axis {K}: flag frac per qubit {[f'{100*v:.3f}%' for v in C['flag_frac']]}", flush=True)
        print(f"    lift matrix: {np.round(np.array([[np.nan if v is None else v for v in row] for row in C['lift']]), 1).tolist()}", flush=True)
        print(f"    frac of flags coincident with another qubit's flag: {C['frac_of_flags_coincident_with_other_qubit_flag']}", flush=True)
    lab = R["residual_corr_labels"]
    strong = [(lab[i], lab[j], round(cc[i, j], 3)) for i in range(10) for j in range(i + 1, 10) if abs(cc[i, j]) > 0.05]
    print(f"  residual |corr| > 0.05: {strong}", flush=True)
# (c) per-bin cross-channel transfer: complex mean of feat_q for bit_r=1 minus bit_r=0
#     (the dataset is balanced over all 32 states, so this marginal difference is balanced),
#     relative to qubit r's own |0>->|1> response |D_rr| in the same bin, and in units of
#     qubit q's per-bin sigma. Mirror-IF pairs (Q2/Q3, Q1/Q4) are where an IQ-imbalance
#     image would appear.
Fc = np.asarray(F[:, :, :, 0], dtype=np.float64) + 1j * np.asarray(F[:, :, :, 1], dtype=np.float64)   # (N,5,20)
bitsm = np.stack([(y >> r) & 1 for r in range(NQ)], 1).astype(bool)
D = np.zeros((NQ, NQ, 20), complex)
for r in range(NQ):
    D[:, r, :] = Fc[bitsm[:, r]].mean(0) - Fc[~bitsm[:, r]].mean(0)
sig_bin = np.zeros((NQ, 20))
for q in range(NQ):
    for k in range(20):
        v = Fc[~bitsm[:, q], q, k]
        sig_bin[q, k] = 1.4826 * np.median(np.abs(v.real - np.median(v.real)))
xt = {}
for q in range(NQ):
    for r in range(NQ):
        if q == r:
            continue
        ratio = np.abs(D[q, r, 2:]) / np.abs(D[r, r, 2:])
        insig = np.abs(D[q, r, :]) / sig_bin[q]
        xt[f"q{q}<-q{r}"] = {"amp_ratio_bins2_19_median": float(np.median(ratio)), "amp_ratio_bins2_19_max": float(ratio.max()),
                             "abs_shift_in_sigma_q_per_bin": insig.tolist(),
                             "complex_shift_per_bin_re": D[q, r].real.tolist(), "complex_shift_per_bin_im": D[q, r].imag.tolist()}
res["cross_channel_transfer_test"] = xt
res["own_response_per_bin_abs"] = {f"q{r}": np.abs(D[r, r]).tolist() for r in range(NQ)}
print("=== cross-channel transfer |D_q<-r| / |D_r<-r| (bins 2-19), median / max; max shift in sigma_q:", flush=True)
for k, v in xt.items():
    if v["amp_ratio_bins2_19_max"] > 0.01 or max(v["abs_shift_in_sigma_q_per_bin"]) > 0.1:
        print(f"  {k}: {v['amp_ratio_bins2_19_median']:.4f} / {v['amp_ratio_bins2_19_max']:.4f};  max shift "
              f"{max(v['abs_shift_in_sigma_q_per_bin']):.3f} sigma_q", flush=True)
json.dump(res, open(os.path.join(OUT, "xtalk.json"), "w"), indent=1)
print("wrote xtalk.json", flush=True)
