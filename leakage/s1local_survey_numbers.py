"""s1local: re-derive every EXISTING leakage-related number for the Lienhard 5-qubit
readout dataset from the small local artifacts (no raw-data access), with exact
Clopper-Pearson 95% CIs, and write them to data/leakage/.

Inputs (read-only):
  ~/Documents/oraqle_reports/herqules_error_modes_len{500,200}_hybridcnn_*.csv
      -> source of oraqle-ics Table tab:error_modes ("MCMit-CNN" = HybridCNN run)
  ~/Documents/oraqle_reports/herqules_error_modes_len{500,200}_pertrace_20260602_*.csv
      -> HERQULES per-trace side of the (removed) Fig. cnn_vs_herqules_error_breakdown
  ~/Documents/oraqle_reports/herqules_error_breakdown_full.csv  (HERQULES full-data run)
  ~/Documents/oraqle_reports/iq_error_gallery_q5.npz            (Fig. iq_taxonomy data)
  ~/Documents/Oraqle/Discriminators/preclassifier_state.pkl      (HERQULES preclassifier)

Outputs:
  data/leakage/s1local_existing_leakage_numbers.csv
  data/leakage/s1local_q5_gallery_offaxis.json
Run: <venv-python> -I sims/leakage/s1local_survey_numbers.py
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import csv
import glob
import json
import os
import pickle

import numpy as np
from scipy.stats import beta, skew

REP = str(LP.REPORTS)
OUT = str(LP.RES)
PKL = str(LP.PRECLASSIFIER_PKL)


def cp(k, n, a=0.05):
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def pct(k, n):
    lo, hi = cp(k, n)
    return 100.0 * k / n, 100.0 * lo, 100.0 * hi


rows = []


def add(source, model, window, qubit, quantity, k, n, definition, trust):
    v, lo, hi = pct(k, n)
    rows.append(dict(source=source, model=model, window=window, qubit=qubit,
                     quantity=quantity, k=k, n=n, value_pct=round(v, 5),
                     ci95_lo_pct=round(lo, 5), ci95_hi_pct=round(hi, 5),
                     definition=definition, trust=trust))


AMP_DEF = ("misclassified shot whose time-mean |IQ| (own-qubit demod band) has |z|>3 vs the "
           "train mean/std pooled over ALL shots (both prepared states, all 32 basis states); "
           "exclusive after the T1-pattern test (T1 > amp > near)")
TRUST = "heuristic proxy; counts only MISCLASSIFIED shots; not a leakage measurement"


def qname(q):
    return f"Q{q+1} (code q{q})"


for split, model, files in [
    ("hybridcnn", "HybridCNN (labelled MCMit-CNN in paper Table tab:error_modes)",
     sorted(glob.glob(f"{REP}/herqules_error_modes_len*_hybridcnn_*.csv"))),
    ("pertrace", "HERQULES per-trace (Net_rmf)",
     [f"{REP}/herqules_error_modes_len500_pertrace_20260602_082923.csv",
      f"{REP}/herqules_error_modes_len200_pertrace_20260602_082117.csv"]),
]:
    for f in files:
        L = int(os.path.basename(f).split("_len")[1].split("_")[0])
        win = f"raw samples [0:{L}) = {2*L} ns at 2 ns/sample"
        tot = dict(ex=0, any=0, err=0, n=0)
        for r in csv.DictReader(open(f)):
            q, n, ne = int(r["qubit"]), int(r["n_test"]), int(r["n_error"])
            ex, an = int(r["excl_n_amp_outlier"]), int(r["any_n_amp_outlier"])
            src = os.path.basename(f)
            add(src, model, win, qname(q), "amp_outlier (excl) share of ERRORS", ex, ne, AMP_DEF, TRUST)
            add(src, model, win, qname(q), "amp_outlier (excl) share of SHOTS", ex, n, AMP_DEF, TRUST)
            add(src, model, win, qname(q), "amp_outlier (any, non-exclusive) share of SHOTS", an, n, AMP_DEF, TRUST)
            add(src, model, win, qname(q), "error rate", ne, n, "pred bit != prepared bit", "measurement")
            tot["ex"] += ex; tot["any"] += an; tot["err"] += ne; tot["n"] += n
        add(os.path.basename(f), model, win, "pooled Q1-Q5", "amp_outlier (excl) share of ERRORS",
            tot["ex"], tot["err"], AMP_DEF, TRUST)
        add(os.path.basename(f), model, win, "pooled Q1-Q5", "amp_outlier (excl) share of qubit-SHOTS",
            tot["ex"], tot["n"], AMP_DEF, TRUST)

# HERQULES full-dataset breakdown (different test split: 960000 shots/qubit)
for r in csv.DictReader(open(f"{REP}/herqules_error_breakdown_full.csv")):
    q = int(r["qubit"]) - 1
    add("herqules_error_breakdown_full.csv", "HERQULES full-data Net_rmf (demux, no scramble)",
        "TRACE_LENGTH default 500 samples = 1000 ns (assumed)", qname(q),
        "amp_outlier (excl) share of SHOTS", int(r["amp"]), int(r["n_test"]), AMP_DEF, TRUST)

# HERQULES preclassifier "ket2" bucket
st = pickle.load(open(PKL, "rb"))
for qb, d in st["trace_classes"].items():
    q = qb - 1
    k2 = d["ket2"].shape[0]
    n1 = d["traces_1"].shape[0] + d["traces_relax"].shape[0] + k2
    T = d["ket2"].shape[1]
    add("Oraqle(Documents)/Discriminators/preclassifier_state.pkl", "HERQULES preclassifier (get_traces, rscale=1)",
        f"T={T} demod samples (= {2*T} ns if undecimated); train split, only states y=0 and y=2^q",
        qname(q), "'ket2' share of |1>-prepared (y=2^q) shots", k2, n1,
        "y==2^q shot whose mean-IQ is >= r from c1 AND > r from c0, r = |c1-c0|/2",
        "geometric artifact: circles are tangent at the midpoint, so most Gaussian-tail shots land here")
    ke = d["traces_excite"].shape[0]
    n0 = d["traces_0"].shape[0] + ke
    add("Oraqle(Documents)/Discriminators/preclassifier_state.pkl", "HERQULES preclassifier (get_traces, rscale=1)",
        f"T={T} demod samples; train split", qname(q), "'excite' share of |0>-prepared (y=0) shots", ke, n0,
        "y==0 shot whose mean-IQ is >= r from c0", "geometric artifact (not thermal excitation)")

# Q5 gallery npz: off-axis one-sided excess (cheap test on existing random subsamples)
g = np.load(f"{REP}/iq_error_gallery_q5.npz")
c0, c1, r, mid = g["cen0"], g["cen1"], float(g["radius"]), g["mid"]
ax = (c1 - c0) / np.linalg.norm(c1 - c0)
pdir = np.array([-ax[1], ax[0]])
res = {"source": "iq_error_gallery_q5.npz (HERQULES demod, Q5 = code q4, window T=%d samples = %d ns)"
       % (g["traj_relax"].shape[0], 2 * g["traj_relax"].shape[0]),
       "note": "cloud0/cloud1 are uniform random subsamples (4000 each) of prepared-|0>/|1> Q5 shots; "
               "leak_cloud is capped at 2500 so the full leak_m count is only known to be >= 2500"}
for nm in ("cloud0", "cloud1"):
    c = g[nm]
    perp = (c - mid) @ pdir
    par = (c - mid) @ ax
    s = np.median(np.abs(perp - np.median(perp))) / 0.6745
    z = (perp - np.median(perp)) / s
    kp, km, n = int((z > 3).sum()), int((z < -3).sum()), len(z)
    kp4, km4 = int((z > 4).sum()), int((z < -4).sum())
    dm0 = np.linalg.norm(c - c0, axis=1); dm1 = np.linalg.norm(c - c1, axis=1)
    lm = (dm0 > 1.3 * r) & (dm1 > 1.3 * r) & (np.abs(perp) > 0.8 * r)
    res[nm] = dict(n=n, perp_skew=float(skew(perp)), robust_sigma_perp=float(s),
                   n_z_gt_3=kp, n_z_lt_m3=km, n_z_gt_4=kp4, n_z_lt_m4=km4,
                   gallery_leak_rule_k=int(lm.sum()), gallery_leak_rule_pct_ci=pct(int(lm.sum()), n),
                   z_gt_3_pct_ci=pct(kp, n),
                   par_over_r_quantiles_of_z_gt_3=(np.percentile(par[z > 3] / r, [5, 25, 50, 75, 95]).tolist()
                                                   if kp else None))
lc = g["leak_cloud"]
lp = (lc - mid) @ pdir
res["leak_cloud"] = dict(n=int(len(lc)), frac_on_positive_side=float((lp > 0).mean()))
for nm, k, n, d in [("gallery leak rule, prepared |1>", res["cloud1"]["gallery_leak_rule_k"], 4000,
                     "iq_error_gallery leak_m: d(c0)>1.3r & d(c1)>1.3r & |perp|>0.8r, r=|c1-c0|/2"),
                    ("one-sided off-axis z_perp>+3, prepared |1>", res["cloud1"]["n_z_gt_3"], 4000,
                     "perp distance from the c0-c1 axis > +3 robust sigma (positive side only)"),
                    ("one-sided off-axis z_perp>+3, prepared |0>", res["cloud0"]["n_z_gt_3"], 4000,
                     "same, prepared |0> (Gaussian expectation 0.135% per side)")]:
    add("iq_error_gallery_q5.npz", "HERQULES demod (no classifier)", res["source"].split("window ")[1].rstrip(")"),
        "Q5 (code q4)", nm, k, n, d, "INFERRED from a 4000-shot random subsample; origin (|2> vs other) not established")

os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/s1local_existing_leakage_numbers.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
with open(f"{OUT}/s1local_q5_gallery_offaxis.json", "w") as f:
    json.dump(res, f, indent=2, default=float)
print(f"wrote {len(rows)} rows")
for r_ in rows:
    if "pooled" in r_["qubit"] or "gallery" in r_["source"] or "pkl" in r_["source"]:
        print(r_["source"][:40], r_["qubit"], r_["quantity"][:45], r_["k"], r_["n"], r_["value_pct"],
              r_["ci95_lo_pct"], r_["ci95_hi_pct"])
print(json.dumps(res, indent=1, default=float)[:2500])
