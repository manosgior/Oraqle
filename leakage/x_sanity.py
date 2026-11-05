"""x_sanity.py -- steps 4/5: sanity statistics and histogram data for plots.

Reads OUT/{feat,featbox,feat1us,y}_{train,test}.npy, writes OUT/sanity.json and
OUT/x_hist.npz (2-D/1-D histograms; plotted locally by x_plot_iq.py).

Window means (mean over bins of the stored 100 ns bin means; == mean over samples):
  full_2us   = feat bins 0..19  (samples 0..999)
  1us        = feat bins 0..9   (samples 0..499; FIR taps reach 50 samples into 500..549)
  1us_cnn    = feat1us bins 0..9 (FIR applied to samples 0..499 only = CNN input at "1 us")
  400ns      = feat bins 0..3   (samples 0..199)
  early_200ns= feat bins 0..1   (samples 0..99)
  box_full_2us / box_1us = same with featbox (no FIR)
Threshold classifier: 1-D threshold on the along-axis window mean, fitted on TRAIN
to minimise the balanced error 0.5*(P(1|0)+P(0|1)), evaluated on TEST.
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json, os, sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else str(LP.FEAT)
NQ = 5
Z95 = 1.959963984540054
WINDOWS = {"full_2us": ("feat", 0, 20), "1us": ("feat", 0, 10), "1us_cnn": ("feat1us", 0, 10),
           "400ns": ("feat", 0, 4), "early_200ns": ("feat", 0, 2),
           "box_full_2us": ("featbox", 0, 20), "box_1us": ("featbox", 0, 10)}
PAPER_CNN_ERR_1US = {0: 3.2, 1: 25.8, 2: 6.1, 3: 5.6, 4: 3.0}


def wilson(k, n, z=Z95):
    if n == 0:
        return [float("nan")] * 3
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(p), float(c - h), float(c + h)]


def mad_sigma(x):
    m = np.median(x)
    return float(1.4826 * np.median(np.abs(x - m)))


def fit_threshold(a, b):
    """a: along values, b: bits. Returns threshold minimising balanced error for rule pred = a >= th."""
    o = np.argsort(a, kind="stable"); s = a[o]; bb = b[o]
    n1 = bb.sum(); n0 = len(bb) - n1
    c1_below = np.concatenate([[0], np.cumsum(bb)])          # bit-1 shots below candidate i (pred 0)
    c0_below = np.concatenate([[0], np.cumsum(1 - bb)])
    err = 0.5 * (c1_below / n1 + (n0 - c0_below) / n0)        # P(0|1) + P(1|0)
    i = int(np.argmin(err))
    if i == 0:
        return float(s[0] - 1)
    if i == len(s):
        return float(s[-1] + 1)
    return float(0.5 * (s[i - 1] + s[i]))


def load(split):
    d = {nm: np.load(os.path.join(OUT, f"{nm}_{split}.npy"), mmap_mode="r") for nm in ["feat", "featbox", "feat1us"]}
    y = np.load(os.path.join(OUT, f"y_{split}.npy"))
    return d, y


def wmean(d, w):
    nm, a, b = WINDOWS[w]
    return np.asarray(d[nm][:, :, a:b, :], dtype=np.float64).mean(axis=2)     # (N,5,2)


def main():
    dtr, ytr = load("train"); dte, yte = load("test")
    btr = np.stack([(ytr >> q) & 1 for q in range(NQ)], 1); bte = np.stack([(yte >> q) & 1 for q in range(NQ)], 1)
    S = {"windows": {k: {"array": v[0], "bins": [v[1], v[2] - 1], "samples": [v[1] * 50, v[2] * 50 - 1],
                         "ns_at_500MSps": [v[1] * 100, v[2] * 100]} for k, v in WINDOWS.items()},
         "per_qubit": {}}
    H = {}
    means = {}
    for w in WINDOWS:
        means[w] = (wmean(dtr, w), wmean(dte, w))
    for q in range(NQ):
        Q = {"paper_name": f"Q{q+1}", "code_name": f"q{q}", "paper_cnn_error_1us_pct": PAPER_CNN_ERR_1US[q], "windows": {}}
        for w in WINDOWS:
            mtr, mte = means[w][0][:, q], means[w][1][:, q]
            b_tr, b_te = btr[:, q], bte[:, q]
            th = fit_threshold(mtr[:, 0], b_tr)
            pred = (mte[:, 0] >= th).astype(int)
            n0 = int((b_te == 0).sum()); n1 = int((b_te == 1).sum())
            k10 = int(((pred == 1) & (b_te == 0)).sum()); k01 = int(((pred == 0) & (b_te == 1)).sum())
            e0 = wilson(k10, n0); e1 = wilson(k01, n1)
            be = 0.5 * (e0[0] + e1[0]); se = 0.5 * np.sqrt(e0[0] * (1 - e0[0]) / n0 + e1[0] * (1 - e1[0]) / n1)
            ent = {"threshold_along_fit_on_train": th,
                   "test_P_pred1_given_prep0": e0, "test_P_pred0_given_prep1": e1,
                   "test_counts": {"n_prep0": n0, "n_prep1": n1, "k_pred1_prep0": k10, "k_pred0_prep1": k01},
                   "test_balanced_error_pct": [100 * be, 100 * (be - Z95 * se), 100 * (be + Z95 * se)]}
            # centroids & widths on TEST (robust + plain), per prepared bit
            cen = {}
            for bit in (0, 1):
                m = mte[b_te == bit]
                cen[f"prep{bit}"] = {"median_along": float(np.median(m[:, 0])), "median_perp": float(np.median(m[:, 1])),
                                     "mean_along": float(m[:, 0].mean()), "mean_perp": float(m[:, 1].mean()),
                                     "sigma_along_MAD": mad_sigma(m[:, 0]), "sigma_perp_MAD": mad_sigma(m[:, 1]),
                                     "std_along": float(m[:, 0].std()), "std_perp": float(m[:, 1].std()), "n": int(len(m))}
            c0, c1 = cen["prep0"], cen["prep1"]
            sep = c1["median_along"] - c0["median_along"]
            sig = 0.5 * (c0["sigma_along_MAD"] + c1["sigma_along_MAD"])
            ent["centroids_test"] = cen
            ent["separation_median_along"] = sep
            ent["snr_sep_over_mean_MAD_sigma_along"] = sep / sig
            ent["gaussian_overlap_error_pct_from_snr"] = float(100 * 0.5 * (1 + __import__("math").erf(-sep / sig / 2 / np.sqrt(2))))
            ent["centroid_diff_perp_median"] = c1["median_perp"] - c0["median_perp"]
            ent["centroid_diff_angle_deg_median"] = float(np.degrees(np.arctan2(c1["median_perp"] - c0["median_perp"], sep)))
            ent["centroid_diff_angle_deg_mean"] = float(np.degrees(np.arctan2(c1["mean_perp"] - c0["mean_perp"],
                                                                               c1["mean_along"] - c0["mean_along"])))
            # perpendicular tails (descriptive only): |perp - median_perp(bit)| / sigma_perp_MAD(bit), TEST
            tails = {}
            for bit in (0, 1):
                m = mte[b_te == bit]
                z = np.abs(m[:, 1] - cen[f"prep{bit}"]["median_perp"]) / cen[f"prep{bit}"]["sigma_perp_MAD"]
                tails[f"prep{bit}"] = {f"gt{k}": {"count": int((z > k).sum()), "n": int(len(z)),
                                                  "frac": wilson(int((z > k).sum()), len(z)),
                                                  "gauss_expect_frac": float(1 - __import__("math").erf(k / np.sqrt(2)))}
                                       for k in (3, 4, 5, 6)}
            ent["perp_tails_test"] = tails
            Q["windows"][w] = ent
        # featbox vs feat agreement (full window and 1us), TEST
        for wf, wb in [("full_2us", "box_full_2us"), ("1us", "box_1us")]:
            a, b = means[wf][1][:, q], means[wb][1][:, q]
            Q[f"fir_vs_box_{wf}"] = {"corr_along": float(np.corrcoef(a[:, 0], b[:, 0])[0, 1]),
                                     "corr_perp": float(np.corrcoef(a[:, 1], b[:, 1])[0, 1]),
                                     "median_abs_diff_along": float(np.median(np.abs(a[:, 0] - b[:, 0]))),
                                     "median_abs_diff_perp": float(np.median(np.abs(a[:, 1] - b[:, 1]))),
                                     "sigma_along_feat": mad_sigma(a[:, 0])}
        # per-bin centroid trajectories (mean over TEST shots per prepared bit)
        F = np.asarray(dte["feat"][:, q], dtype=np.float64); FB = np.asarray(dte["featbox"][:, q], dtype=np.float64)
        Q["per_bin_mean_test"] = {f"prep{bit}": {"feat_along": F[bte[:, q] == bit, :, 0].mean(0).tolist(),
                                                 "feat_perp": F[bte[:, q] == bit, :, 1].mean(0).tolist(),
                                                 "featbox_along": FB[bte[:, q] == bit, :, 0].mean(0).tolist(),
                                                 "featbox_perp": FB[bte[:, q] == bit, :, 1].mean(0).tolist(),
                                                 "feat_sigma_along_MAD": [mad_sigma(F[bte[:, q] == bit, k, 0]) for k in range(20)],
                                                 "feat_sigma_perp_MAD": [mad_sigma(F[bte[:, q] == bit, k, 1]) for k in range(20)]}
                                  for bit in (0, 1)}
        del F, FB
        S["per_qubit"][f"q{q}"] = Q
        # histograms (POOLED train+test) for full_2us, 1us, early_200ns
        for w in ["full_2us", "1us", "early_200ns"]:
            m = np.concatenate([means[w][0][:, q], means[w][1][:, q]]); b = np.concatenate([btr[:, q], bte[:, q]])
            lo = np.quantile(m, 1e-5, axis=0); hi = np.quantile(m, 1 - 1e-5, axis=0); pad = 0.08 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            # equal aspect: same span on both axes centred on the data centre
            span = float(max(hi - lo)); ctr = 0.5 * (lo + hi)
            ex = [ctr[0] - span / 2, ctr[0] + span / 2]; ey = [ctr[1] - span / 2, ctr[1] + span / 2]
            for bit in (0, 1):
                h, xe, ye = np.histogram2d(m[b == bit, 0], m[b == bit, 1], bins=260, range=[ex, ey])
                H[f"q{q}_{w}_prep{bit}_h2"] = h.astype(np.int32)
                H[f"q{q}_{w}_xe"] = xe; H[f"q{q}_{w}_ye"] = ye
                inside = int(h.sum()); H[f"q{q}_{w}_prep{bit}_outside"] = np.array(int((b == bit).sum()) - inside)
            # 1-D: perp z-score and along in units of separation, per bit
            c = S["per_qubit"][f"q{q}"]["windows"][w]["centroids_test"]
            c0a, c1a = c["prep0"]["median_along"], c["prep1"]["median_along"]
            for bit in (0, 1):
                mm = m[b == bit]
                zp = (mm[:, 1] - c[f"prep{bit}"]["median_perp"]) / c[f"prep{bit}"]["sigma_perp_MAD"]
                ua = (mm[:, 0] - c0a) / (c1a - c0a)
                H[f"q{q}_{w}_prep{bit}_hperp"] = np.histogram(zp, bins=np.linspace(-12, 12, 241))[0]
                H[f"q{q}_{w}_prep{bit}_halong"] = np.histogram(ua, bins=np.linspace(-2.5, 3.5, 241))[0]
            H["perp_edges"] = np.linspace(-12, 12, 241); H["along_edges"] = np.linspace(-2.5, 3.5, 241)
        print(f"[sanity] q{q} done", flush=True)
    json.dump(S, open(os.path.join(OUT, "sanity.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, "x_hist.npz"), **H)
    # compact printout
    for q in range(NQ):
        Q = S["per_qubit"][f"q{q}"]
        line = f"[sanity] q{q}/Q{q+1} paper CNN 1us err {Q['paper_cnn_error_1us_pct']}% | "
        for w in ["1us", "1us_cnn", "full_2us", "400ns", "box_1us"]:
            e = Q["windows"][w]["test_balanced_error_pct"]
            line += f"{w}: {e[0]:.2f}% [{e[1]:.2f},{e[2]:.2f}]  "
        print(line, flush=True)
        for w in ["1us", "full_2us"]:
            E = Q["windows"][w]
            c0, c1 = E["centroids_test"]["prep0"], E["centroids_test"]["prep1"]
            print(f"    {w}: c0=({c0['median_along']:.3f},{c0['median_perp']:.3f}) c1=({c1['median_along']:.3f},{c1['median_perp']:.3f}) "
                  f"sigA={c0['sigma_along_MAD']:.3f}/{c1['sigma_along_MAD']:.3f} sigP={c0['sigma_perp_MAD']:.3f}/{c1['sigma_perp_MAD']:.3f} "
                  f"sep={E['separation_median_along']:.3f} SNR={E['snr_sep_over_mean_MAD_sigma_along']:.2f} "
                  f"angle(med)={E['centroid_diff_angle_deg_median']:+.2f} angle(mean)={E['centroid_diff_angle_deg_mean']:+.2f} deg", flush=True)
            for bit in (0, 1):
                T = E["perp_tails_test"][f"prep{bit}"]
                print(f"      perp tails prep{bit}: " + "  ".join(
                    f">{k}s {T[f'gt{k}']['count']}/{T[f'gt{k}']['n']} ({100*T[f'gt{k}']['frac'][0]:.4f}% vs gauss {100*T[f'gt{k}']['gauss_expect_frac']:.4f}%)"
                    for k in (3, 4, 5, 6)), flush=True)
    print("[sanity] wrote sanity.json and x_hist.npz", flush=True)


if __name__ == "__main__":
    main()
