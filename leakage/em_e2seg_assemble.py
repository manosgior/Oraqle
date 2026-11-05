"""em_e2seg_assemble.py (tag e2seg): assemble the HMM-free segment-based error-mode attribution of the canonical
MCMit-CNN errors into data/leakage/em_e2seg_results.json and data/leakage/em_e2seg_summary.md.

Inputs (all under data/leakage/, copied from graham ~/oraqle-ics-sims/leakage/em/e2seg/out/):
  em_e2seg_v3_raw_main.json        detector + attribution counts (sims/leakage/em_e2seg_attr_v3.py)
  em_e2seg_v2_raw_main.json        same with the v2 core-selection bug (for the v2 -> v3 comparison only)
  em_e2seg_oldmethod.json          old tab:error_modes procedure re-run (sims/leakage/em_e2seg_oldmethod.py)
  em_e2seg_oldmethod_l200scan.json 400 ns HybridCNN preprocessing scan (sims/leakage/em_e2seg_oldmethod_l200scan.py)
  f_cnn_leak_xtab.json             reference leakage cross-tab (tag f)

Counting rules (per qubit, prepared bit b; N = errors of that prepared bit):
  Leakage L = N(z > 3) - N(z < -3) among errors (mirror-corrected), var = N+ + N-.
  Transition T (order A, after leakage) = flags - sum FP over the non-leaked pool (+ mirror stand-in),
      var = flags + MC variance of the FP sum.  Relaxation test (down) on prepared-|1> errors, excitation test (up)
      on prepared-|0> errors.  FP = max over the two stationary branches (conservative), alpha = 0.01.
  Order B: T over all errors first, then L among unflagged errors.
  Residual R = N - L - T.  CIs: +/- 1.96 sqrt(var)/N.  Error rates: Wilson.
  A transition number is reported only if (i) the real-data quasi-null (TEST shots in the |b> core of the
  independent 1-2 us window) gives observed/expected flags within [0.6, 1.3] for that test, and (ii) the
  injection efficiency given a CNN-proxy misread is >= 5%.  Otherwise it is n/m (and stays in the residual).
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import math
import os

import numpy as np

ROOT = str(LP.ROOT)
DL = str(LP.RES)
PQ = ["Q1", "Q2", "Q3", "Q4", "Q5"]
N_SHOTS_TOT = 1120000
WIN = {"1us": ("box1us", "canonical"), "600ns": ("box600", "soft600"), "400ns": ("box400", "soft400")}
ROBUST = {"fir1us_canonical": ("fir1us", "canonical"), "box1us_tau0_canonical": ("box1us_tau0", "canonical"),
          "box1us_soft1000": ("box1us", "soft1000")}
SETS = ["alpha0.05", "alpha0.01", "alpha0.002", "alpha0.01_selnull", "alpha0.01_prepbranch"]
PRIMARY = "alpha0.01"
CAL_LO, CAL_HI, EFF_MIN = 0.6, 1.3, 0.05
PAPER = {  # tab:error_modes (HybridCNN, old procedure), shares of errors in %
    "err_rate": [3.2, 25.8, 6.1, 5.6, 3.0], "near": [100.0, 97.5, 99.9, 99.4, 85.7],
    "t1": [0.0, 2.3, 0.0, 0.5, 14.3], "leak": [0.0, 0.2, 0.1, 0.04, 0.0],
    "pooled": {"near": 97.4, "t1": 2.4, "leak": 0.2}}
PAPER_FID = {"1us": [0.970, 0.746, 0.942, 0.947, 0.970], "600ns": [0.953, 0.725, 0.938, 0.937, 0.970],
             "400ns": [0.907, 0.697, 0.926, 0.901, 0.951]}


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [100 * (c - h), 100 * (c + h)]


def share(cnt, var, N):
    if cnt is None or N == 0:
        return None
    h = 1.96 * math.sqrt(max(var, 0.0)) / N
    return {"count": round(cnt, 1), "pct": 100 * cnt / N, "ci95": [100 * (cnt / N - h), 100 * (cnt / N + h)]}


def r2(x, nd=2):
    return None if x is None else round(x, nd)


def fmt(s, nd=2):
    if s is None:
        return "n/m"
    return f"{s['pct']:.{nd}f} [{s['ci95'][0]:.{nd}f}, {s['ci95'][1]:.{nd}f}]"


def calib(v, b):
    """Real-data quasi-null calibration of the relevant test for prepared bit b (alpha0)."""
    key = "down_on_prep1_latecore" if b == 1 else "up_on_prep0_latecore"
    c = v["calibration_all_shots"][key]["0.01"]
    return {"test": key, "observed": c["observed"], "expected": round(c["expected"], 1), "n": c["n"],
            "ratio": c["observed"] / max(c["expected"], 1e-9)}


def efficiency(v, b):
    mode = "down" if b == 1 else "up"
    k = v["sensitivity"][mode]["tau_fit"]
    return {"given_cnn_proxy_misread": k["eff_uniform_ts_given_proxy_misread"],
            "given_mf_misread": k["eff_uniform_ts_given_mf_misread"],
            "p_flag_uniform_ts": k["p_flag_uniform_ts"]}


def t_status(q, v, b):
    if q == 3:
        return "n/m: Q4 |1> response is multimodal; the two-level Gaussian null is invalid (quasi-null obs/exp far from 1)"
    cal = calib(v, b)
    eff = efficiency(v, b)
    if eff["given_cnn_proxy_misread"] < EFF_MIN:
        return f"n/m: not resolvable (injection efficiency given a CNN-proxy misread {eff['given_cnn_proxy_misread']:.3f} < {EFF_MIN})"
    if not (CAL_LO <= cal["ratio"] <= CAL_HI):
        return f"n/m: null calibration fails on real quasi-null shots (obs/exp = {cal['ratio']:.2f})"
    return "measured"


def per_prep(q, v, a, ch, b):
    x = a[f"prep{b}"]
    N = x["n_err"]
    st = t_status(q, v, b)
    tm = st == "measured"
    lm = q != 3
    L = (x["leak_plus"] - x["leak_minus"]) if lm else None
    varL = (x["leak_plus"] + x["leak_minus"]) if lm else 0
    TA = (x["flag_pool"] - x["fp_pool"]) if tm else None
    varTA = x["flag_pool"] + x["fp_pool_var_mc"]
    TB = (x["flag_all"] - x["fp_all"]) if tm else None
    varTB = x["flag_all"] + x["fp_all_var_mc"]
    LB = (x["nP_notF"] - x["nM_notF"]) if (lm and tm) else L
    varLB = (x["nP_notF"] + x["nM_notF"]) if (lm and tm) else varL
    RA = N - (L or 0) - (TA or 0)
    RB = N - (LB or 0) - (TB or 0)
    ov = ((x["flag_leakplus"] - x["fp_leakplus"]) - (x["flag_leakminus"] - x["fp_leakminus"])) if (lm and tm) else None
    return {"n_shots": x["n_shots"], "n_err": N, "transition_status": st,
            "transition_kind": "relaxation |1>->|0>" if b == 1 else "excitation |0>->|1> (or out of |0>)",
            "raw": {k: x[k] for k in ("leak_plus", "leak_minus", "flag_pool", "fp_pool", "fp_pool_var_mc", "flag_all", "fp_all",
                                      "flag_leakplus", "flag_leakminus", "fp_leakplus", "fp_leakminus", "nP_notF", "nM_notF")},
            "orderA": {"leakage": share(L, varL, N), "transition": share(TA, varTA, N),
                       "residual": share(RA, varL + (varTA if tm else 0), N)},
            "orderB": {"transition": share(TB, varTB, N), "leakage": share(LB, varLB, N),
                       "residual": share(RB, varLB + (varTB if tm else 0), N)},
            "overlap_leak_and_transition": share(ov, x["flag_leakplus"] + x["flag_leakminus"], N) if ov is not None else None,
            "transition_raw_flags_upper_bound": share(x["flag_all"], x["flag_all"], N),
            "transition_fp_corrected_ungated": share(x["flag_pool"] - x["fp_pool"], varTA, N),
            "_cnt": {"N": N, "L": L or 0, "varL": varL, "TA": TA or 0, "varTA": varTA if tm else 0,
                     "TB": TB or 0, "varTB": varTB if tm else 0, "LB": LB or 0, "varLB": varLB, "nshots": x["n_shots"],
                     "n_T_measured": int(tm), "n_L_measured": int(lm), "n_parts": 1}}


def combine(parts, n_shots):
    c = {k: sum(p["_cnt"][k] for p in parts) for k in parts[0]["_cnt"]}
    N = c["N"]
    RA = N - c["L"] - c["TA"]
    RB = N - c["LB"] - c["TB"]
    tstat = "measured" if c["n_T_measured"] == c["n_parts"] else ("n/m" if c["n_T_measured"] == 0 else
                                                                   f"partial: {c['n_T_measured']}/{c['n_parts']} (qubit, prepared-bit) tests measurable; others counted as 0 (lower bound)")
    lstat = "measured" if c["n_L_measured"] == c["n_parts"] else ("n/m" if c["n_L_measured"] == 0 else
                                                                   f"partial: {c['n_L_measured']}/{c['n_parts']} measurable (Q4 n/m counted as 0)")
    out = {"n_err": N, "n_shots": n_shots, "transition_status": tstat, "leakage_status_combined": lstat,
           "err_rate_pct": 100 * N / n_shots, "err_rate_ci95": wilson(N, n_shots),
           "orderA": {"leakage": share(c["L"], c["varL"], N) if c["n_L_measured"] else None,
                      "transition": share(c["TA"], c["varTA"], N) if c["n_T_measured"] else None,
                      "residual": share(RA, c["varL"] + c["varTA"], N)},
           "orderB": {"transition": share(c["TB"], c["varTB"], N) if c["n_T_measured"] else None,
                      "leakage": share(c["LB"], c["varLB"], N) if c["n_L_measured"] else None,
                      "residual": share(RB, c["varLB"] + c["varTB"], N)},
           "pct_of_shots": {"leakage": 100 * c["L"] / n_shots, "transition": 100 * c["TA"] / n_shots,
                            "residual": 100 * RA / n_shots,
                            "leakage_ci95": [100 * (c["L"] - 1.96 * math.sqrt(c["varL"])) / n_shots,
                                             100 * (c["L"] + 1.96 * math.sqrt(c["varL"])) / n_shots],
                            "transition_ci95": [100 * (c["TA"] - 1.96 * math.sqrt(c["varTA"])) / n_shots,
                                                100 * (c["TA"] + 1.96 * math.sqrt(c["varTA"])) / n_shots]},
           "_cnt": c}
    return out


def residual_char(ch, b):
    x = ch[f"prep{b}"]
    p = x["proxy"]
    h = p["resid_hist"]
    n = sum(h)
    gp = p["gauss_pred_proxy_err_hist"][1:]
    gtot = sum(gp)
    hm = x["resid_hist"]
    gm = x["gauss_pred_mf_err_hist"][1:]
    return {
        "n_resid": n,
        "proxy_dist_bins": ["proxy reads correct (<0)", "[0,1) sigma", "[1,2)", "[2,3)", ">=3"],
        "resid_hist_proxy": h,
        "frac_within_1sigma": (h[0] + h[1]) / n if n else None, "frac_within_1sigma_ci95": wilson(h[0] + h[1], n),
        "frac_within_2sigma": (h[0] + h[1] + h[2]) / n if n else None, "frac_within_2sigma_ci95": wilson(h[0] + h[1] + h[2], n),
        "frac_beyond_2sigma": (h[3] + h[4]) / n if n else None,
        "gauss_pred_frac_within_1sigma_given_err": gp[0] / gtot if gtot > 0 else None,
        "gauss_pred_frac_beyond_2sigma_given_err": (gp[2] + gp[3]) / gtot if gtot > 0 else None,
        "proxy_errors_all_shots_observed": p["proxy_err_total"], "proxy_errors_gauss_pred": gtot,
        "proxy_obs_over_gauss": p["proxy_err_total"] / gtot if gtot > 0 else None,
        "proxy_err_hist_all_shots_observed": p["proxy_err_hist_all_shots"][1:], "proxy_err_hist_gauss_pred": [round(g, 1) for g in gp],
        "mf_resid_hist": hm, "mf_err_all_shots_observed": x["mf_err_total"], "mf_err_gauss_pred": sum(gm),
        "mf_err_hist_all_shots_observed": x["mf_err_hist_all_shots"][1:], "mf_err_hist_gauss_pred": [round(g, 1) for g in gm],
        "resid_soft_pcorrect_median": x["resid_soft_pcorrect_median"],
        "cnn_errors_this_bit": x["n_err"],
        "cnn_errors_in_excess_of_gauss_overlap_frac": max(0.0, 1 - gtot / x["n_err"]) if x["n_err"] else None,
        "min_frac_cnn_errors_not_gauss_overlap_MF": max(0.0, 1 - sum(gm) / x["n_err"]) if x["n_err"] else None,
        "note": ("MF = per-config GLS level estimator m (near-Bayes linear rule for the stationary two-Gaussian model with the "
                 "measured bin covariance); its Gaussian error count bounds what pure overlap can explain for any classifier. "
                 "Proxy = per-config least-squares linear fit of the soft-CNN logit (CNN-like boundary); distances in proxy noise widths.")}


def hidden_leak(ch, q):
    if q == 3:
        return None
    out = {}
    tot = {"plus": 0, "minus": 0, "n": 0, "err_plus": 0, "err_minus": 0}
    for b in (0, 1):
        la = ch[f"prep{b}"]["leak_all"]
        for k in tot:
            tot[k] += la[k]
        out[f"prep{b}"] = la
    for key, la in list(out.items()) + [("all", tot)]:
        Lk = la["plus"] - la["minus"]
        Ek = la["err_plus"] - la["err_minus"]
        d = {"leaked_pct_of_shots": 100 * Lk / la["n"],
             "leaked_pct_ci95": [100 * (Lk - 1.96 * math.sqrt(la["plus"] + la["minus"])) / la["n"],
                                 100 * (Lk + 1.96 * math.sqrt(la["plus"] + la["minus"])) / la["n"]],
             "leaked_shots": Lk, "leaked_and_misread": Ek,
             "frac_leaked_read_correctly": (1 - Ek / Lk) if Lk > 0 else None,
             "frac_read_correctly_ci95": ([1 - (Ek + 1.96 * math.sqrt(la["err_plus"] + la["err_minus"])) / Lk,
                                           1 - (Ek - 1.96 * math.sqrt(la["err_plus"] + la["err_minus"])) / Lk] if Lk > 0 else None)}
        if key == "all":
            out["all"] = d
        else:
            out[key] = {**la, **d}
    return out


def window_block(R, vname, pname, sname=PRIMARY, full=True):
    W = {"variant": vname, "pred": pname, "set": sname, "qubits": {}, "pooled": {}}
    perq = {}
    for q in range(5):
        v = R["variants"][vname][f"q{q}"]
        a = v["attr"][pname][sname]
        ch = v["attr"][pname]["characterisation"]
        pp = {b: per_prep(q, v, a, ch, b) for b in (0, 1)}
        comb = combine([pp[0], pp[1]], N_SHOTS_TOT)
        perq[q] = (pp, comb)
        Q = {"paper_name": PQ[q], **{k: v_ for k, v_ in comb.items() if k != "_cnt"},
             "by_prep": {f"prep{b}": {k: v_ for k, v_ in pp[b].items() if k != "_cnt"} for b in (0, 1)},
             "leakage_status": "measured" if q != 3 else "n/m: Q4 |1> response is multimodal (sub-clouds); off-axis excess not attributable"}
        if full:
            Q["null_calibration_quasinull"] = {f"prep{b}": calib(v, b) for b in (0, 1)}
            Q["transition_efficiency"] = {f"prep{b}": efficiency(v, b) for b in (0, 1)}
            eff_inf = {}
            for b in (0, 1):
                T = pp[b]["orderA"]["transition"]
                e = efficiency(v, b)
                if T is not None:
                    eff_inf[f"prep{b}"] = {"pct_of_errors_effcorr_proxy": T["pct"] / max(e["given_cnn_proxy_misread"], 1e-9),
                                           "pct_of_errors_effcorr_mf": T["pct"] / max(e["given_mf_misread"], 1e-9),
                                           "note": "INFERRED: FP-corrected detected share / injection efficiency (uniform transition time in the window)"}
            Q["transition_efficiency_corrected_INFERRED"] = eff_inf
            Q["residual_characterisation"] = {f"prep{b}": residual_char(ch, b) for b in (0, 1)}
            Q["hidden_leakage"] = hidden_leak(ch, q)
            Q["flag_time_hist_100ns_flagged_errors"] = {f"prep{b}": ch[f"prep{b}"]["flag_time_hist_100ns"] for b in (0, 1)}
            Q["opposite_direction_flags_secondary"] = {f"prep{b}": ch[f"prep{b}"]["opposite_direction_alpha0"] for b in (0, 1)}
            Q["sigma_m_per_cfg"] = v["sig_m"]
            Q["sigma_m_outlier_cfgs"] = v.get("sig_m_outlier_cfgs")
        W["qubits"][PQ[q]] = Q
    for gname, qs in (("all5", [0, 1, 2, 3, 4]), ("Q1Q3Q4Q5", [0, 2, 3, 4]), ("Q1Q2Q3Q5", [0, 1, 2, 4]), ("Q1Q3Q5", [0, 2, 4])):
        parts = [perq[q][0][b] for q in qs for b in (0, 1)]
        comb = combine(parts, N_SHOTS_TOT * len(qs))
        W["pooled"][gname] = {k: v_ for k, v_ in comb.items() if k != "_cnt"}
        W["pooled"][gname]["note"] = ("Q4 leakage and every n/m transition test contribute 0 (lower bounds); "
                                      "residual therefore includes them")
    return W


def main():
    R = json.load(open(f"{DL}/em_e2seg_v3_raw_main.json"))
    R2 = json.load(open(f"{DL}/em_e2seg_v2_raw_main.json"))
    OM = json.load(open(f"{DL}/em_e2seg_oldmethod.json"))
    SC = json.load(open(f"{DL}/em_e2seg_oldmethod_l200scan.json"))
    F = json.load(open(f"{DL}/f_cnn_leak_xtab.json"))
    res = {"tag": "e2seg", "date": "2026-11-05",
           "method_doc_detector": R["note"], "assembler_doc": __doc__,
           "alpha_primary": 0.01, "null_bank_size": R["NB"], "checks": R["checks"]}
    # error rates vs paper
    er = {}
    for wn, key in (("1us", "canonical_err_per_qubit"), ("1us_soft", "soft1000_err_per_qubit"),
                    ("600ns", "soft600_err_per_qubit"), ("400ns", "soft400_err_per_qubit")):
        er[wn] = {}
        for q in range(5):
            e = R["checks"][key][q]
            k = int(round(e * N_SHOTS_TOT))
            pw = wn.replace("_soft", "")
            er[wn][PQ[q]] = {"err_pct": 100 * e, "ci95": wilson(k, N_SHOTS_TOT),
                             "paper_tab_fidelity_err_pct": round(100 * (1 - PAPER_FID[pw][q]), 1),
                             "paper_tab_error_modes_err_pct": PAPER["err_rate"][q] if pw == "1us" else None}
    res["error_rates"] = er
    # main breakdowns
    res["breakdown"] = {wn: window_block(R, *vp) for wn, vp in WIN.items()}
    # robustness
    rob = {"alpha_and_null_variants_1us": {}, "pipeline_variants_1us": {}, "v2_bug_vs_v3_1us": {}}
    for s in SETS:
        rob["alpha_and_null_variants_1us"][s] = {
            PQ[q]: {k: q_["orderA"][k] for k in ("leakage", "transition", "residual")} | {"by_prep_T": {
                b: q_["by_prep"][b]["orderA"]["transition"] for b in ("prep0", "prep1")}}
            for q, q_ in enumerate(window_block(R, "box1us", "canonical", s, full=False)["qubits"].values())}
    for s in SETS:
        for wn in ("600ns", "400ns"):
            vn, pn = WIN[wn]
            rob.setdefault(f"alpha_and_null_variants_{wn}", {})[s] = {
                PQ[q]: {"by_prep_T": {b: q_["by_prep"][b]["orderA"]["transition"] for b in ("prep0", "prep1")}}
                for q, q_ in enumerate(window_block(R, vn, pn, s, full=False)["qubits"].values())}
    for rn, (vn, pn) in ROBUST.items():
        blk = window_block(R, vn, pn, PRIMARY, full=False)
        rob["pipeline_variants_1us"][rn] = {Q: {"err_rate_pct": x["err_rate_pct"], **{k: x["orderA"][k] for k in ("leakage", "transition", "residual")},
                                                "by_prep_T": {b: x["by_prep"][b]["orderA"]["transition"] for b in ("prep0", "prep1")},
                                                "status": {b: x["by_prep"][b]["transition_status"] for b in ("prep0", "prep1")}}
                                            for Q, x in blk["qubits"].items()}
    for q in range(5):
        a2 = R2["variants"]["box1us"][f"q{q}"]["attr"]["canonical"][PRIMARY]
        a3 = R["variants"]["box1us"][f"q{q}"]["attr"]["canonical"][PRIMARY]
        rob["v2_bug_vs_v3_1us"][PQ[q]] = {
            f"prep{b}": {"v2_flags": a2[f"prep{b}"]["flag_pool"], "v2_fp": round(a2[f"prep{b}"]["fp_pool"], 1),
                         "v3_flags": a3[f"prep{b}"]["flag_pool"], "v3_fp": round(a3[f"prep{b}"]["fp_pool"], 1)} for b in (0, 1)}
        rob["v2_bug_vs_v3_1us"][PQ[q]]["sigma_m1_per_cfg_v2"] = [round(x[1], 3) for x in R2["variants"]["box1us"][f"q{q}"]["sig_m"]]
        rob["v2_bug_vs_v3_1us"][PQ[q]]["sigma_m1_per_cfg_v3"] = [round(x[1], 3) for x in R["variants"]["box1us"][f"q{q}"]["sig_m"]]
    # order robustness summary (primary)
    ordr = {}
    for wn in WIN:
        ordr[wn] = {}
        for Q, x in res["breakdown"][wn]["qubits"].items():
            A, B = x["orderA"], x["orderB"]
            ordr[wn][Q] = {"leak_A": r2(A["leakage"]["pct"]) if A["leakage"] else None, "leak_B": r2(B["leakage"]["pct"]) if B["leakage"] else None,
                           "trans_A": r2(A["transition"]["pct"]) if A["transition"] else None, "trans_B": r2(B["transition"]["pct"]) if B["transition"] else None,
                           "resid_A": r2(A["residual"]["pct"]), "resid_B": r2(B["residual"]["pct"])}
    rob["order_A_vs_B"] = ordr
    # leakage frame cross-check vs f_cnn_leak_xtab
    lx = {}
    for wn in ("1us", "600ns", "400ns"):
        lx[wn] = {}
        for q in range(5):
            for b in (0, 1):
                mine = R["leak_E3_check"][wn][f"q{q}_prep{b}"]
                ref = F["windows"][wn]["qubits"][f"q{q}"][f"prep{b}"]
                lx[wn][f"{PQ[q]}_prep{b}"] = {"E3_pct_e2seg": 100 * mine["E3"], "E3_pct_f": 100 * ref["E3"]}
            if wn == "1us" and q != 3:
                a = R["variants"]["box1us"][f"q{q}"]["attr"]["canonical"][PRIMARY]
                Lp = sum(a[f"prep{b}"]["leak_plus"] for b in (0, 1))
                Lm = sum(a[f"prep{b}"]["leak_minus"] for b in (0, 1))
                N = sum(a[f"prep{b}"]["n_err"] for b in (0, 1))
                lx[wn][f"{PQ[q]}_leaked_share_of_canon_errors"] = {
                    "e2seg_pct": 100 * (Lp - Lm) / N,
                    "f_pct": 100 * F["windows"][wn]["qubits"][f"q{q}"]["overall"]["leaked_share_of_canon_errors"]}
    rob["leakage_frame_vs_f_cnn_leak_xtab"] = lx
    res["robustness"] = rob
    # detector characterisation (null + sensitivity), primary variants
    det = {}
    for wn, (vn, pn) in WIN.items():
        det[wn] = {}
        for q in range(5):
            v = R["variants"][vn][f"q{q}"]
            s = v["sensitivity"]
            det[wn][PQ[q]] = {
                "ringup_fit": v["ringup"], "tau_r_used_ns": v["tau_r_used_ns"],
                "null_calibration_all": {k: (c["0.01"] if "0.01" in c else c) for k, c in v["calibration_all_shots"].items()},
                "carrier_null_real_prep0_core_alpha0.01": v["carrier_null_alpha0"],
                "null_example_cfg0": v["null_example_cfg0"],
                "sensitivity_vs_ts": {mode: {"ts_ns": s[mode]["tau_fit"]["ts_ns"], "p_flag": s[mode]["tau_fit"]["p_flag"],
                                             "p_flag_given_proxy_misread": s[mode]["tau_fit"]["p_flag_given_proxy_misread"],
                                             "p_flag_given_mf_misread": s[mode]["tau_fit"]["p_flag_given_mf_misread"],
                                             "p_proxy_misread": s[mode]["tau_fit"]["p_proxy_misread"],
                                             "p_flag_tau2x": s[mode]["tau2"]["p_flag"], "p_flag_tau0": s[mode]["tau0"]["p_flag"]}
                                       for mode in ("down", "up")},
                "efficiency_by_alpha": v["efficiency_by_alpha_tau_fit"],
                "proxy": v["proxy"], "cov_summary": v["cov"], "seed_diag": v.get("seed_diag")}
    res["detector"] = det
    # old method
    old = {"criteria": {
        "source": "Oraqle/Discriminators/runners/_train_and_analyze_hybridcnn.py (per-shot categories) + herqules_error_report.py (aggregation into herqules_error_breakdown_hybridcnn.csv, 1000 ns rows -> tab:error_modes)",
        "model": "HybridCNN (not MCMit-CNN), sparse-FIR demod (HERQULES IFs, 101-tap Hamming 5 MHz LPF, every DS-th sample; DS=40 at 1 us, T_ds=13; the 400 ns checkpoint matches only with DS=20, T_ds=10)",
        "near_boundary": "P(correct) < 0.60 from the model's own sigmoid output: true for EVERY error (an error has P(correct) < 0.5)",
        "t1_relaxation": "Wiener envelope env = mean(x0-x1)/var(x0-x1) on flattened [I,Q] demod (TRAIN, random equal-size pairing, RandomState(0)); mf1/mf2 = projections of the first/second half of the window (raw, no offset subtracted); flag if |mf1| > 0.5 median|mf1| and |mf2| > 0.5 median|mf2| over all TEST shots and sign(mf1) != sign(mf2)",
        "leakage_column": "amp outlier: |z| > 3 with z = (time-mean |IQ| of the demod - TRAIN mean)/TRAIN std pooled over both states; labelled 'Excitation' by herqules_error_report.py and 'Leakage' in the paper",
        "priority": "t1 > amp > near > other (other always empty because near covers every error)",
        "why_t1_is_not_a_transition_test": "the sign of a raw (un-centred) projection is not a state indicator: on Q1 and Q3 both prepared states give the same sign in both halves (Q3: frac mf>0 = 0 for both), so the test can never fire; on Q5 prepared |0> has frac(mf1>0) = 0.25 vs frac(mf2>0) = 0.75, so 9.5% of ALL prepared-|0> shots are 'T1' flagged (prepared |1>: 1.4%); counts follow the IQ-origin geometry, not transitions"}}
    L500 = OM["L"]["500"]
    old["reproduction_1us"] = {"csv": L500["hybridcnn_vs_csv"]["csv"], "per_qubit_ours_vs_csv": L500["hybridcnn_vs_csv"]["per_qubit_[ours,csv]"],
                               "note": "re-run of HybridCNN_len500.pth reproduces every CSV count to within 0.06% (inference nondeterminism)"}
    old["reproduction_400ns"] = {"scan_downsample": SC["scan"], "best_ds": SC["best_ds"],
                                 "note": "only DS=20 (T_ds=10) reproduces the 400 ns CSV (abs dev 36 errors over 5 qubits)"}
    old["diag_1us_sign_fractions"] = L500["diag"]

    def oldrow(rows):
        o = {}
        for r in rows:
            n = r["n_error"]
            o[PQ[r["qubit"]]] = {"n_err": n, "err_rate_pct": 100 * r["err_rate"], "near_pct": 100 * r["excl_n_near_boundary"] / n,
                                 "t1_pct": 100 * r["excl_n_t1_pattern"] / n, "amp_pct": 100 * r["excl_n_amp_outlier"] / n,
                                 "amp_pct_of_shots": 100 * r["any_n_amp_outlier"] / r["n_test"] if "n_test" in r else 100 * r["any_n_amp_outlier"] / N_SHOTS_TOT,
                                 "t1_count": r["excl_n_t1_pattern"], "amp_count": r["excl_n_amp_outlier"]}
        tot = sum(r["n_error"] for r in rows)
        o["pooled_all5"] = {"near_pct": 100 * sum(r["excl_n_near_boundary"] for r in rows) / tot,
                            "t1_pct": 100 * sum(r["excl_n_t1_pattern"] for r in rows) / tot,
                            "amp_pct": 100 * sum(r["excl_n_amp_outlier"] for r in rows) / tot}
        return o
    old["old_procedure_on_hybridcnn_1us_reproduced"] = oldrow(L500["hybridcnn_reproduced"])
    old["old_procedure_on_canonical_mcmit_1us"] = oldrow(L500["mcmit"]["canonical_MCMitCNN"])
    old["old_procedure_on_soft600_mcmit"] = oldrow(OM["L"]["300"]["mcmit"]["soft600_MCMitCNN"])
    old["old_procedure_on_soft400_mcmit_ds20"] = oldrow(SC["soft400_MCMitCNN_oldmasks_best_ds"])
    old["old_procedure_on_hybridcnn_400ns_ds20"] = oldrow(SC["hybridcnn_reproduced_best"])
    old["paper_tab_error_modes"] = PAPER
    # decomposition at 1 us
    B1 = res["breakdown"]["1us"]
    dec = {}
    for q in range(5):
        Q = PQ[q]
        om = old["old_procedure_on_canonical_mcmit_1us"][Q]
        nw = B1["qubits"][Q]
        dec[Q] = {"paper_err": PAPER["err_rate"][q], "canon_err": round(nw["err_rate_pct"], 2),
                  "paper_t1": PAPER["t1"][q], "oldproc_canon_t1": r2(om["t1_pct"]),
                  "new_transition_A": r2(nw["orderA"]["transition"]["pct"]) if nw["orderA"]["transition"] else None,
                  "paper_leak": PAPER["leak"][q], "oldproc_canon_amp": r2(om["amp_pct"]),
                  "new_leak_A": r2(nw["orderA"]["leakage"]["pct"]) if nw["orderA"]["leakage"] else None,
                  "paper_near": PAPER["near"][q], "oldproc_canon_near": r2(om["near_pct"]),
                  "new_residual_A": r2(nw["orderA"]["residual"]["pct"])}
    old["decomposition_1us"] = dec
    res["old_method"] = old
    json.dump(res, open(f"{DL}/em_e2seg_results.json", "w"), indent=1, default=float)
    print("wrote", f"{DL}/em_e2seg_results.json")
    return res


if __name__ == "__main__":
    main()
