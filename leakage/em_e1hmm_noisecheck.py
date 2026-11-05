#!/usr/bin/env python3
"""em_e1hmm_noisecheck.py (tag e1hmm): is the per-bin readout noise state-independent? (decides the null model)

Transition-robust noise estimate: for 100 ns boxcar bins k = 2..K-2 (featbox, xt-corrected as in m3time),
d_k = (x_{k+1} - x_k)/sqrt(2). A transition moves at most one or two of the differences of a shot, so the
MAD over all shots and bins of d is a robust estimate of the white per-bin noise for each prepared state
(prepared |1> includes decays; the MAD ignores them). Compared with the HMM's tied Sig[k] and with the robust
per-config prepared-|0> covariance. Also: lag-1 / lag-2 noise correlations from prepared |0> cores, and the
across-shot perp width of window means for prepared |1> vs the within-shot (difference-based) prediction
(perp is insensitive to |1>->|0> decays; Q3/Q5/Q2 perp is contaminated by L, so Q1 and Q4 are the clean tests).
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import numpy as np

FEAT, M3 = str(LP.FEAT), str(LP.stage("m3time"))
OUT = str(LP.stage("em/e1hmm") / "noisecheck")


def mad(x, axis=None):
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis)


def cfg_other(y, q):
    return ((y >> (q + 1)) << q) | (y & ((1 << q) - 1))


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    y = np.load(f"{FEAT}/y_test.npy").astype(np.int64)
    F = np.load(f"{FEAT}/featbox_test.npy", mmap_mode="r")
    res = {}
    for q in range(5):
        d = json.load(open(f"{M3}/out/featbox_xt1/m3time_q{q}.json"))
        offs = np.asarray(d["offsets_xt"])
        Sig = np.asarray(d["params_train"]["Sig"])
        b, c = (y >> q) & 1, cfg_other(y, q)
        X = np.asarray(F[:, q], dtype=np.float64) - offs[c]
        R = {"hmm_sig_along_bins2_9": np.sqrt(Sig[2:10, 0, 0]).tolist(),
             "hmm_sig_perp_bins2_9": np.sqrt(Sig[2:10, 1, 1]).tolist()}
        for s in (0, 1):
            Z = X[b == s]
            D = (Z[:, 3:10] - Z[:, 2:9]) / np.sqrt(2.0)          # diffs among bins 2..9 (1 us window)
            R[f"prep{s}_diff_mad_along"] = float(mad(D[..., 0].ravel()))
            R[f"prep{s}_diff_mad_perp"] = float(mad(D[..., 1].ravel()))
            R[f"prep{s}_diff_mad_along_perbin"] = mad(D[..., 0], axis=0).ravel().tolist()
            R[f"prep{s}_diff_mad_perp_perbin"] = mad(D[..., 1], axis=0).ravel().tolist()
            # tail check of the difference distribution (Gaussian: P(|d|>4 sigma) = 6.3e-5)
            sd = R[f"prep{s}_diff_mad_perp"]
            R[f"prep{s}_diff_perp_frac_gt4sig"] = float((np.abs(D[..., 1] - np.median(D[..., 1])) > 4 * sd).mean())
            # across-shot window-mean width (bins 0..9, perp and along), MAD
            W = Z[:, :10].mean(1)
            R[f"prep{s}_wmean_mad_along"] = float(mad(W[:, 0]))
            R[f"prep{s}_wmean_mad_perp"] = float(mad(W[:, 1]))
            # within-shot white-noise prediction for the 10-bin mean (uses HMM Sig per bin, bins 0..9)
        R["pred_wmean_sd_perp_white_hmm"] = float(np.sqrt(Sig[:10, 1, 1].sum()) / 10)
        R["pred_wmean_sd_along_white_hmm"] = float(np.sqrt(Sig[:10, 0, 0].sum()) / 10)
        # lag correlations from prepared-|0> core (bins 2..9)
        Z = X[b == 0][:, 2:10]
        Zc = Z - np.median(Z, 0)
        core = (np.abs(Zc) < 3 * mad(Zc, axis=0)).all((1, 2))
        Zc = Zc[core]
        for ax, nm in ((0, "along"), (1, "perp")):
            R[f"prep0_lag1_corr_{nm}"] = float(np.mean([np.corrcoef(Zc[:, k, ax], Zc[:, k + 1, ax])[0, 1] for k in range(7)]))
            R[f"prep0_lag2_corr_{nm}"] = float(np.mean([np.corrcoef(Zc[:, k, ax], Zc[:, k + 2, ax])[0, 1] for k in range(6)]))
        R["prep0_cross_corr_along_perp_samebin"] = float(np.mean([np.corrcoef(Zc[:, k, 0], Zc[:, k, 1])[0, 1] for k in range(8)]))
        R["ratio_diffnoise_prep1_over_prep0_along"] = R["prep1_diff_mad_along"] / R["prep0_diff_mad_along"]
        R["ratio_diffnoise_prep1_over_prep0_perp"] = R["prep1_diff_mad_perp"] / R["prep0_diff_mad_perp"]
        res[f"q{q}"] = R
        print(q, {k: (round(v, 4) if isinstance(v, float) else None) for k, v in R.items() if isinstance(v, float)}, flush=True)
    json.dump(res, open(f"{OUT}/em_e1hmm_noisecheck.json", "w"), indent=1)


if __name__ == "__main__":
    main()
