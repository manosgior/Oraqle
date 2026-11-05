"""em_e4recon_assemble.py (tag e4recon): assemble the FINAL error-mode numbers for the paper.

Reads (local):
  data/leakage/em_e4recon_analysis.json   (em_e4recon_analyze.py on graham; per-bit counts/variances for every model)
  data/leakage/em_e1hmm_results.json      (robustness: HMM state change, Gaussian overlap at the CNN surrogate)
  data/leakage/em_e2seg_results.json      (robustness: alpha/null variants, efficiency-corrected shares, Gaussian MF/proxy)
  data/leakage/em_e3herq_summary_v3.json  (HERQULES error rates / co-failure, for cross-reference)
Writes:
  data/leakage/em_e4recon_final.json

Final per-error accounting (count level, per qubit and prepared bit, then summed):
  N = L + T + O + S
  L = leakage (line-frame E3 flag z > 3 minus mirror z < -3)                    MEASURED (k = 3 lower bound)
  T = resolved mid-readout transition (e2seg v3 GLRT, alpha = 0.01, FP-corrected, gated)  MEASURED lower bound
  O = near-boundary overlap = min(R, late-core expectation), R = N - L - T         MEASURED upper bound
      late-core expectation = sum_cfg N(b,cfg) * P(model error | prepared b, cfg, late window 1-2 us in |b> core)
  S = R - O = no resolved event, in excess of overlap                              MEASURED lower bound (by elimination)
      identity INFERRED: qubit in the other computational state for most of the window (transition before
      ~150-250 ns or before the window, preparation/thermal error); supported by the late-window confirmation.
  State change (relaxation/excitation before or during the readout) = T + S = N - L - O   (independent of the step detector)
CIs: var(L) = N+ + N-; var(T) = flags + MC var(FP sum); var(O) = binomial per cfg; var(S) = var(L)+var(T)+var(O);
     var(T+S) = var(L) + var(O); if O is capped at R then O = R, S = 0, var(O) = var(L) + var(T).
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import math
import os

ROOT = str(LP.ROOT)
DL = str(LP.RES)
QN = ["Q1", "Q2", "Q3", "Q4", "Q5"]
POOLS = {"all5": QN, "Q1Q3Q4Q5": ["Q1", "Q3", "Q4", "Q5"], "Q1Q2Q3Q5": ["Q1", "Q2", "Q3", "Q5"], "Q1Q3Q5": ["Q1", "Q3", "Q5"],
         "Q3Q5": ["Q3", "Q5"]}


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0)) / den
    return [100 * (c - h), 100 * (c + h)]


def r(x, nd=2):
    return None if x is None else round(float(x), nd)


def bit_parts(dd):
    N, ns = dd["n_err"], dd["n_shots"]
    Lm = dd["leak"] is not None
    L, Lv = (dd["leak"], dd["leak_var"]) if Lm else (0.0, 0.0)
    Tm = dd["T_measurable"]
    T, Tv = (dd["T"], dd["T_var"]) if Tm else (0.0, 0.0)
    R = N - L - T
    Oe, Oev = dd["overlap_expected_latecore_upper"], dd["overlap_expected_var"]
    capped = Oe >= R
    if capped:
        O, Ov = R, Lv + Tv
        S, Sv = 0.0, 0.0
    else:
        O, Ov = Oe, Oev
        S, Sv = R - O, Lv + Tv + Oev
    TS, TSv = T + S, (Lv + Ov if not capped else Tv)
    return {"N": N, "ns": ns, "L": L, "Lv": Lv, "T": T, "Tv": Tv, "O": O, "Ov": Ov, "S": S, "Sv": Sv, "TS": TS, "TSv": TSv,
            "R": R, "Oe": Oe, "L_meas": Lm, "T_meas": Tm, "O_capped": capped,
            "T_ungated": dd["T_ungated"], "pool_flag": dd["pool_flag"], "pool_fp": dd["pool_fp"]}


def add(a, b):
    o = {k: a[k] + b[k] for k in ("N", "ns", "L", "Lv", "T", "Tv", "O", "Ov", "S", "Sv", "TS", "TSv", "R", "Oe", "T_ungated")}
    for k in ("L_meas", "T_meas", "O_capped"):
        o[k] = (a[k] if isinstance(a[k], list) else [a[k]]) + (b[k] if isinstance(b[k], list) else [b[k]])
    return o


def fmt_mode(x, v, N, ns, measured=True):
    if not measured:
        return None
    h = 1.96 * math.sqrt(max(v, 0.0))
    return {"count": r(x, 1), "pct_err": r(100 * x / N), "ci95_pct_err": [r(100 * (x - h) / N), r(100 * (x + h) / N)],
            "pct_shots": r(100 * x / ns, 4), "ci95_pct_shots": [r(100 * (x - h) / ns, 4), r(100 * (x + h) / ns, 4)]}


def summarize(p, Lmeas=True, Tmeas=True):
    N, ns = p["N"], p["ns"]
    return {"n_err": N, "n_shots": ns, "err_rate_pct": r(100 * N / ns, 3), "err_rate_ci95_pct": [r(x, 3) for x in wilson(N, ns)],
            "leakage": fmt_mode(p["L"], p["Lv"], N, ns, Lmeas),
            "transition_resolved": fmt_mode(p["T"], p["Tv"], N, ns, Tmeas),
            "overlap_near_boundary_upper": fmt_mode(p["O"], p["Ov"], N, ns),
            "state_change_unresolved_lower": fmt_mode(p["S"], p["Sv"], N, ns),
            "state_change_total_T_plus_S": fmt_mode(p["TS"], p["TSv"], N, ns),
            "residual_no_resolved_event": {"pct_err": r(100 * p["R"] / N)},
            "late_core_overlap_expectation_uncapped_pct_err": r(100 * p["Oe"] / N),
            "transition_ungated_pct_err": r(100 * p["T_ungated"] / N)}


def model_block(A, wn, mn):
    M = A["windows"][wn]["models"][mn]
    out, parts = {}, {}
    for q in QN:
        v = M["per_qubit"][q]
        bp = {b: bit_parts(v["per_bit"][b]["detail"]) for b in ("prep0", "prep1")}
        tot = add(bp["prep0"], bp["prep1"])
        parts[q] = tot
        Lmeas = q != "Q4"
        Tm = [bp["prep0"]["T_meas"], bp["prep1"]["T_meas"]]
        blk = summarize(tot, Lmeas, any(Tm))
        blk["transition_status"] = ("measured" if all(Tm) else "partial (" + ", ".join(
            f"{b} n/m" for b, t in zip(("prep0", "prep1"), Tm) if not t) + "; counted as 0 -> in residual)") if any(Tm) else "n/m"
        blk["leakage_status"] = "measured" if Lmeas else "n/m (multimodal |1> response; no off-axis measurement possible)"
        blk["per_bit"] = {}
        for b in ("prep0", "prep1"):
            x = summarize(bp[b], Lmeas, bp[b]["T_meas"])
            d = v["per_bit"][b]["detail"]
            x["overlap_capped_at_residual"] = bp[b]["O_capped"]
            x["late_core_reference"] = {"n_latecore": d["n_latecore"], "n_latecore_err": d["n_latecore_err"],
                                        "latecore_err_rate_pct": r(d["latecore_err_rate_pct"], 4)}
            x["late_window_confirmation"] = d["late_window_confirmation"]
            x["e1hmm_surrogate_depth_of_residual"] = d["e1hmm_surrogate_depth"]
            x["soft_output_zone_split_REJECTED"] = {"pstar_5pct_of_latecore_err_Pcorrect": r(d["pstar"], 4),
                                                   "near_share_pct": r(100 * d["near"] / d["n_err"]) if d["near"] is not None else None}
            x["orderB"] = d["orderB"]
            x["overlap_leak_and_transition_count"] = r(d["overlap_leak_and_transition"], 1) if d["overlap_leak_and_transition"] is not None else None
            blk["per_bit"][b] = x
        blk["hidden_leakage"] = v["hidden_leakage"]
        blk["orderB_pct_err"] = v["all"]["orderB"]
        out[q] = blk
    pooled = {}
    for pn, ks in POOLS.items():
        t = parts[ks[0]]
        for k in ks[1:]:
            t = add(t, parts[k])
        s = summarize(t)
        s["note"] = "leakage counts Q4 as 0; transition counts n/m (qubit, bit) tests as 0 (they stay in the residual)"
        s["orderB_pct_err"] = M["pooled"][pn]["orderB"]
        pooled[pn] = s
    return {"per_qubit": out, "pooled": pooled}, parts


def diff_ci(a, b, key):
    """a - b (percentage points of errors) for two models, independent-sample CI (conservative: error sets overlap)."""
    xa, xb = a.get(key), b.get(key)
    if xa is None or xb is None:
        return None
    ha = (xa["ci95_pct_err"][1] - xa["ci95_pct_err"][0]) / 2
    hb = (xb["ci95_pct_err"][1] - xb["ci95_pct_err"][0]) / 2
    d = xa["pct_err"] - xb["pct_err"]
    h = math.sqrt(ha * ha + hb * hb)
    return {"diff_pp": r(d), "ci95": [r(d - h), r(d + h)], "within_ci": abs(d) <= h}


def main():
    A = json.load(open(f"{DL}/em_e4recon_analysis.json"))
    E1 = json.load(open(f"{DL}/em_e1hmm_results.json"))
    E2 = json.load(open(f"{DL}/em_e2seg_results.json"))
    H3 = json.load(open(f"{DL}/em_e3herq_summary_v3.json"))
    F = {"tag": "e4recon", "date": "2026-11-05",
         "sources": {"analysis": "data/leakage/em_e4recon_analysis.json (graham ~/oraqle-ics-sims/leakage/em/e4recon/out/)",
                     "scripts": ["sims/leakage/em_e4recon_glrt.py", "sims/leakage/em_e4recon_analyze.py", "sims/leakage/em_e4recon_assemble.py"],
                     "inputs": ["e2seg v3 detector (sims/leakage/em_e2seg_attr_v3.py, imported read-only)", "e1hmm v6 results",
                                "e3herq per-trace HERQULES predictions (em_e3herq_preds_{1us,400ns}.npz)",
                                "canonical preds_MCMitCNN_pb.npy (1 us); cnn_soft_{1000,600,400}ns.npz"]},
         "definitions": __doc__}
    F["reproduction_checks"] = {
        "e2seg_v3_bit_identical": "em_e4recon_glrt.py re-ran the e2seg v3 detector on all TEST shots; the alpha=0.01 attribution of "
                                  "canonical, soft1000, soft600, soft400 is byte-identical to em_e2seg_v3_raw_main.json for all 5 qubits (MEASURED)",
        "leakage_frame": "line-frame E3 / leaked shares identical to e2seg (own frame) and within 0.01 pp of f_cnn_leak_xtab "
                         "(Q3 3.144 vs 3.136 %, Q5 1.034 vs 1.043 %, pooled Q1Q2Q3Q5 0.814 vs 0.81 %)",
        "error_rates": "canonical 2.98/25.43/5.84/5.31/2.96 %; soft600 4.71/27.48/6.16/6.31/2.99 %; soft400 9.32/30.28/7.36/9.88/4.87 % "
                       "= paper Tab. fidelity MCMit-CNN rows (soft argmax = P>0.5); HERQULES per-trace 3.15/26.07/7.60/5.86/3.05 % (1 us), "
                       "9.43/30.44/8.38/10.14/5.50 % (400 ns) = paper tab:fidelity HERQULES rows within 0.08 pp except Q4 1 us (-0.26 pp)"}
    W = {}
    parts_all = {}
    for wn, mn in (("1us", "cnn_canonical"), ("1us", "cnn_soft1000"), ("600ns", "cnn_soft600"), ("400ns", "cnn_soft400"),
                   ("1us", "herq_pertrace"), ("400ns", "herq_pertrace"), ("1us", "herq_scrambled_mfsrc"), ("400ns", "herq_scrambled_mfsrc")):
        blk, parts = model_block(A, wn, mn)
        W.setdefault(wn, {})[mn] = blk
        parts_all[(wn, mn)] = parts
    F["table_1us_cnn_canonical"] = W["1us"]["cnn_canonical"]
    F["windows"] = {"600ns_cnn_soft600": W["600ns"]["cnn_soft600"], "400ns_cnn_soft400": W["400ns"]["cnn_soft400"],
                    "1us_cnn_soft1000_crosscheck": W["1us"]["cnn_soft1000"]}
    # gating and sensitivity of the step detector
    gat = {}
    for wn in ("1us", "600ns", "400ns"):
        gat[wn] = {}
        for q in QN:
            gat[wn][q] = {}
            for b in ("prep0", "prep1"):
                g = A["windows"][wn]["gating"][q][b]
                ts, pf = g["sens_ts_ns"], g["sens_p_flag"]
                pick = {str(int(t)): r(p, 3) for t, p in zip(ts, pf) if int(t) in (0, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000)}
                gat[wn][q][b] = {"test": "relaxation (down)" if b == "prep1" else "excitation (up)",
                                 "quasinull_obs_over_exp": r(g["quasinull_ratio"], 3), "eff_given_cnn_proxy_misread": r(g["eff_given_proxy_misread"], 3),
                                 "eff_given_mf_misread": r(g["eff_given_mf_misread"], 3), "measurable": g["measurable"],
                                 "p_flag_vs_ts_ns": pick, "ts_window_p_flag_ge_0.5_ns": g["sens_window_p_ge_0.5_ns"], "p_flag_max": r(g["sens_max"], 3)}
    F["step_detector_gating_and_sensitivity"] = gat
    # leakage vs window (line frame, prepared |1> / |0>)
    F["leakage_E3_by_window"] = {wn: {q: (None if v is None else {b: {"E3_pct": r(v[b]["E3_pct"], 3), "ci95": [r(x, 3) for x in v[b]["ci95"]]}
                                                              for b in ("prep0", "prep1")})
                                      for q, v in A["windows"][wn]["leak_E3"].items()} for wn in ("400ns", "600ns", "1us")}
    # ---- HERQULES comparison (same attribution)
    hc = {}
    for wn, cn in (("1us", "cnn_canonical"), ("400ns", "cnn_soft400")):
        hc[wn] = {}
        C, Hh = W[wn][cn]["per_qubit"], W[wn]["herq_pertrace"]["per_qubit"]
        Sx = W[wn]["herq_scrambled_mfsrc"]["per_qubit"]
        for q in QN:
            c, h, s = C[q], Hh[q], Sx[q]
            hc[wn][q] = {"err_cnn_pct": c["err_rate_pct"], "err_herq_pertrace_pct": h["err_rate_pct"], "err_herq_scrambled_pct": s["err_rate_pct"],
                         "ratio_cnn_over_herq_pertrace": r(c["err_rate_pct"] / h["err_rate_pct"], 3),
                         "ratio_cnn_over_herq_scrambled": r(c["err_rate_pct"] / s["err_rate_pct"], 3),
                         "shares_cnn": {k: (None if c[k] is None else c[k]["pct_err"]) for k in ("leakage", "transition_resolved", "overlap_near_boundary_upper", "state_change_unresolved_lower", "state_change_total_T_plus_S")},
                         "shares_herq_pertrace": {k: (None if h[k] is None else h[k]["pct_err"]) for k in ("leakage", "transition_resolved", "overlap_near_boundary_upper", "state_change_unresolved_lower", "state_change_total_T_plus_S")},
                         "shares_herq_scrambled_mfsrc_INDICATIVE": {k: (None if s[k] is None else s[k]["pct_err"]) for k in ("leakage", "transition_resolved", "overlap_near_boundary_upper", "state_change_unresolved_lower", "state_change_total_T_plus_S")},
                         "diff_cnn_minus_herq_pertrace_pp": {k: diff_ci(c, h, k) for k in ("leakage", "transition_resolved", "overlap_near_boundary_upper", "state_change_unresolved_lower", "state_change_total_T_plus_S")},
                         "mode_rates_pct_shots_cnn": {k: (None if c[k] is None else c[k]["pct_shots"]) for k in ("leakage", "transition_resolved", "overlap_near_boundary_upper", "state_change_total_T_plus_S")},
                         "mode_rates_pct_shots_herq_pertrace": {k: (None if h[k] is None else h[k]["pct_shots"]) for k in ("leakage", "transition_resolved", "overlap_near_boundary_upper", "state_change_total_T_plus_S")}}
    F["herqules"] = {"model_used": "per-trace HERQULES checkpoint (e3herq 'pred_bits_pertrace'): the only single-shot HERQULES; matches paper tab:fidelity; "
                                   "the cut figure panel (b) also used it. The 'demux-pipeline / scrambled' HERQULES combines MF and RMF of DIFFERENT shots "
                                   "(same-shot in 0.0029% of rows, chance), so per-shot attribution of its errors is ill-defined; its numbers are INDICATIVE only "
                                   "(attributed to the MF-source shot, as the original analysis did).",
                     "1us_herq_pertrace": W["1us"]["herq_pertrace"], "400ns_herq_pertrace": W["400ns"]["herq_pertrace"],
                     "1us_herq_scrambled_mfsrc_INDICATIVE": W["1us"]["herq_scrambled_mfsrc"], "400ns_herq_scrambled_mfsrc_INDICATIVE": W["400ns"]["herq_scrambled_mfsrc"],
                     "comparison": hc,
                     "co_failure_from_e3herq": "68.2% [67.9, 68.5] of per-trace HERQULES Q3 errors and 92.1% [91.8, 92.4] of Q5 errors are also CNN errors (1 us); 75.5% / 80.5% at 400 ns (e3herq, MEASURED)"}
    # ---- robustness: e1hmm and e2seg alternatives per qubit/window
    rob = {}
    for wn, es in (("1us", "canon"), ("600ns", "soft"), ("400ns", "soft")):
        rob[wn] = {}
        for q in QN:
            p = E1["per_qubit"][q][wn]["errsets"][es]
            prim = p["primary"]["all"]
            vals = [prim["state_change_detected"]["pct_of_errors"]] + [p["variants"][v]["all"]["state_change_detected"]["pct_of_errors"] for v in p["variants"]]
            vals.append(p["features_feat_FIR_lab_p50"]["all"]["state_change_detected"]["pct_of_errors"])
            vals.append(prim["order_swap_S_then_L"]["state_change_detected"]["pct_of_errors"])
            inw = [prim["of_which_in_window_transition"]["pct_of_errors"]] + [p["variants"][v]["all"]["of_which_in_window_transition"]["pct_of_errors"] for v in ("p50", "viterbi", "p80")]
            g = p["near_boundary"]["gaussian_overlap_prediction_cnn_surrogate"]
            nerr = p["n_err"]
            rc = E2["breakdown"][wn]["qubits"][q].get("residual_characterisation") or {}
            prox = sum(rc[b]["proxy_errors_gauss_pred"] for b in rc) if rc else None
            mf = sum(rc[b]["mf_err_gauss_pred"] for b in rc) if rc else None
            mine = W[wn]["cnn_canonical" if wn == "1us" else ("cnn_soft600" if wn == "600ns" else "cnn_soft400")]["per_qubit"][q]
            rob[wn][q] = {
                "state_change_total_primary_e4recon": mine["state_change_total_T_plus_S"]["pct_err"],
                "e1hmm_state_change_primary": r(prim["state_change_detected"]["pct_of_errors"]),
                "e1hmm_state_change_variant_range": [r(min(vals)), r(max(vals))],
                "e1hmm_in_window_transition_primary": r(prim["of_which_in_window_transition"]["pct_of_errors"]),
                "e1hmm_in_window_transition_range_over_priors": [r(min(inw)), r(max(inw))],
                "e1hmm_no_detected_event": r(prim["no_detected_event"]["pct_of_errors"]),
                "overlap_primary_latecore_upper_e4recon": mine["overlap_near_boundary_upper"]["pct_err"],
                "overlap_gauss_at_cnn_surrogate_e1hmm_pct_err": r(100 * g["expected_overlap_errors"] / nerr),
                "overlap_gauss_at_cnn_proxy_e2seg_pct_err": r(100 * prox / nerr) if prox is not None else None,
                "overlap_gauss_near_bayes_mf_e2seg_pct_err": r(100 * mf / nerr) if mf is not None else None,
                "deconv_chi2_per_dof_e1hmm": r(prim["deconv_chi2"] / prim["deconv_dof"], 1)}
    F["robustness_cross_method"] = rob
    # e2seg alpha/null variants (transition share of errors, 1 us)
    av = {}
    for k, v in E2["robustness"]["alpha_and_null_variants_1us"].items():
        av[k] = {q: (None if v[q]["transition"] is None else r(v[q]["transition"]["pct"])) for q in QN}
    F["robustness_transition_detector_variants_1us_pct_err"] = av
    eff = {}
    for q in QN:
        x = E2["breakdown"]["1us"]["qubits"][q].get("transition_efficiency_corrected_INFERRED")
        if x:
            eff[q] = {b: [r(x[b]["pct_of_errors_effcorr_mf"], 1), r(x[b]["pct_of_errors_effcorr_proxy"], 1)] for b in x}
    F["transition_efficiency_corrected_INFERRED_pct_of_bit_errors"] = eff
    F["robustness_notes"] = {
        "order_swap": "transition before leakage: residual / overlap / unresolved state change move by <= 0.03 pp on every qubit; only the L/T split moves "
                      "(Q3 L 3.14 -> 1.70 %, T 1.75 -> 3.18 %; Q5 L 1.03 -> 0.89 %, T 5.02 -> 5.16 %): 7.7 % of Q3 prepared-|0> errors are both leaked and step-flagged",
        "soft1000_vs_canonical": "same attribution on the soft 1 us model: all shares within 0.6 pp (Q4 overlap +0.62 pp; others <= 0.5 pp) (see windows.1us_cnn_soft1000_crosscheck)",
        "soft_output_depth_split_rejected": "a per-error near-boundary split on the model's own P(correct) (threshold = 5th percentile of P(correct) of the "
                                            "late-core overlap errors, p* = 0.004-0.06) classifies 85-93% of residual errors as 'near' although the count-level "
                                            "late-core expectation allows only 4-12% of prepared-|1> errors on Q1/Q3/Q5: the CNN is NOT confident on early-decay "
                                            "traces (it is calibrated against decays), so model confidence is not a near-boundary criterion. This is also why the old "
                                            "P(correct) < 0.6 rule could not separate anything.",
        "late_core_is_upper_bound": "the late-core expectation multiplies the stationary-shot misread rate by ALL prepared-b shots (incl. shots that decayed or leaked "
                                    "inside the window) and its reference set contains a few shots that changed state and came back; O is therefore an upper bound and "
                                    "S / T+S lower bounds. For Q2 (SNR ~1.2) the late core is impure (|0> and |1> late clouds overlap) and O is loose.",
        "Q4": "Q4 leakage and transitions are n/m (multimodal |1>: a stationary reduced-shift sub-cloud holds ~21% of prepared |1>); errors on that sub-cloud fall "
              "outside the late-core reference and are counted in S, so Q4's S is not purely state change.",
        "Q2": "overlap-dominated in every estimate: late core 98.5% (upper), e2seg near-Bayes Gaussian >= observed count, e1hmm Gaussian at the CNN surrogate 58%; "
              "e1hmm's 51% HMM state change for Q2 is not credible (15% false-alarm rate on stationary prepared-|1> shots)."}
    # ---- old method
    om = E2["old_method"]
    F["old_method"] = {"criteria": om["criteria"], "paper_table": om["paper_tab_error_modes"],
                       "old_procedure_on_canonical_mcmit_1us": om["old_procedure_on_canonical_mcmit_1us"],
                       "old_procedure_on_hybridcnn_1us_reproduced": om["old_procedure_on_hybridcnn_1us_reproduced"],
                       "verified_in_source_by_e4recon": "Oraqle/Discriminators/runners/_train_and_analyze_hybridcnn.py l.28-29 (NEAR_THR=0.60, AMP_Z=3), "
                                                        "l.48-56 (_mf_envelope = mean(z-o)/var(z-o), RandomState(0) pairing), l.146-171 (conf<0.6; "
                                                        "amp z vs pooled TRAIN mean/std; mf1/mf2 un-centred halves, |mf|>0.5 median, sign(mf1)!=sign(mf2); "
                                                        "priority t1 > amp > near)"}
    F["text_numbers"] = text_numbers(F)
    json.dump(F, open(f"{DL}/em_e4recon_final.json", "w"), indent=1)
    print("written", f"{DL}/em_e4recon_final.json")


def text_numbers(F):
    """Every number the Results / Analysis / Takeaway #5 text needs, with CI, status and source path."""
    T1 = F["table_1us_cnn_canonical"]
    W6, W4 = F["windows"]["600ns_cnn_soft600"], F["windows"]["400ns_cnn_soft400"]
    H1, H4 = F["herqules"]["1us_herq_pertrace"], F["herqules"]["400ns_herq_pertrace"]

    def m(blk, q, key, unit="pct_err", pooled=False):
        x = (blk["pooled"] if pooled else blk["per_qubit"])[q][key]
        if x is None:
            return {"value": None, "status": "n/m"}
        ck = "ci95_pct_err" if unit == "pct_err" else "ci95_pct_shots"
        return {"value": x[unit], "ci95": x[ck], "unit": "% of errors" if unit == "pct_err" else "% of shots"}

    def mb(blk, q, b, key, unit="pct_err"):
        x = blk["per_qubit"][q]["per_bit"][b][key]
        if x is None:
            return {"value": None, "status": "n/m"}
        ck = "ci95_pct_err" if unit == "pct_err" else "ci95_pct_shots"
        return {"value": x[unit], "ci95": x[ck], "unit": f"% of {b} errors" if unit == "pct_err" else f"% of {b} shots"}

    def conf(q, b, cls):
        c = T1["per_qubit"][q]["per_bit"][b]["late_window_confirmation"][cls]
        k = round(c["late_in_misread_state_core"] * c["n"])
        return {"value": r(100 * c["late_in_misread_state_core"], 1), "ci95": [r(x, 1) for x in wilson(k, c["n"])], "n": c["n"]}

    tn = {}
    for q in ("Q3", "Q5"):
        for wn, blk in (("1us", T1), ("400ns", W4), ("600ns", W6)):
            tn[f"{q}_{wn}_cnn"] = {k: m(blk, q, k) for k in ("leakage", "transition_resolved", "overlap_near_boundary_upper",
                                                             "state_change_unresolved_lower", "state_change_total_T_plus_S")}
            tn[f"{q}_{wn}_cnn"]["err_rate_pct"] = {"value": blk["per_qubit"][q]["err_rate_pct"], "ci95": blk["per_qubit"][q]["err_rate_ci95_pct"]}
            tn[f"{q}_{wn}_cnn_per_shot"] = {k: m(blk, q, k, "pct_shots") for k in ("leakage", "transition_resolved",
                                                                                   "overlap_near_boundary_upper", "state_change_total_T_plus_S")}
        for wn, blk in (("1us", H1), ("400ns", H4)):
            tn[f"{q}_{wn}_herq_pertrace"] = {k: m(blk, q, k) for k in ("leakage", "transition_resolved", "overlap_near_boundary_upper",
                                                                       "state_change_unresolved_lower", "state_change_total_T_plus_S")}
            tn[f"{q}_{wn}_herq_pertrace"]["err_rate_pct"] = {"value": blk["per_qubit"][q]["err_rate_pct"], "ci95": blk["per_qubit"][q]["err_rate_ci95_pct"]}
            tn[f"{q}_{wn}_herq_per_shot"] = {k: m(blk, q, k, "pct_shots") for k in ("overlap_near_boundary_upper", "state_change_total_T_plus_S")}
    for p in ("Q1Q3Q4Q5", "all5", "Q1Q2Q3Q5"):
        for wn, blk in (("1us", T1), ("600ns", W6), ("400ns", W4)):
            tn[f"pooled_{p}_{wn}_cnn"] = {k: m(blk, p, k, pooled=True) for k in ("leakage", "transition_resolved", "overlap_near_boundary_upper",
                                                                                "state_change_total_T_plus_S")}
    tn["relaxation_side_state_change_share_of_prep1_errors_1us"] = {q: mb(T1, q, "prep1", "state_change_total_T_plus_S") for q in QN}
    tn["resolved_relaxation_share_of_prep1_errors_1us"] = {q: mb(T1, q, "prep1", "transition_resolved") for q in QN}
    tn["excitation_side_1us"] = {q: {"state_change_pct_of_prep0_shots": mb(T1, q, "prep0", "state_change_total_T_plus_S", "pct_shots"),
                                     "resolved_excitation_pct_of_prep0_shots": mb(T1, q, "prep0", "transition_resolved", "pct_shots"),
                                     "leakage_share_of_prep0_errors": mb(T1, q, "prep0", "leakage")} for q in ("Q1", "Q3", "Q5")}
    tn["late_window_confirmation_1us_prep1_errors_in_0_core_vs_correct"] = {
        q: {"errors": conf(q, "prep1", "all_errors"), "correctly_read": conf(q, "prep1", "correct_shots")} for q in ("Q1", "Q3", "Q4", "Q5")}
    tn["late_window_confirmation_1us_prep0_errors_in_1_core_vs_correct"] = {
        q: {"errors": conf(q, "prep0", "all_errors"), "correctly_read": conf(q, "prep0", "correct_shots")} for q in ("Q1", "Q3", "Q5")}
    hl = {}
    for q in ("Q1", "Q2", "Q3", "Q5"):
        h = T1["per_qubit"][q]["hidden_leakage"]
        hl[q] = {"leaked_pct_of_shots": {"value": r(h["all"]["leaked_pct_shots"], 2), "ci95": [r(x, 2) for x in h["all"]["ci95"]]},
                 "leaked_pct_of_prep1_shots": F["leakage_E3_by_window"]["1us"][q]["prep1"],
                 "leaked_pct_of_prep0_shots": F["leakage_E3_by_window"]["1us"][q]["prep0"],
                 "read_correctly_pct": {"value": r(100 * h["all"]["frac_leaked_read_correctly"], 1),
                                        "ci95": [r(100 * x, 1) for x in h["all"]["frac_read_correctly_ci95"]]},
                 "leaked_errors_pct_of_shots": m(T1, q, "leakage", "pct_shots")}
    tn["hidden_leakage_1us"] = hl
    tn["Q3_leakage_vs_window_prep1"] = {wn: F["leakage_E3_by_window"][wn]["Q3"]["prep1"] for wn in ("400ns", "600ns", "1us")}
    tn["Q3_leakage_vs_window_note"] = ("line frame (same frame for all windows): 3.51 / 6.45 / 11.46 %; LEAKAGE_REPORT headline 3.5 / 6.5 / 11.3 % "
                                       "mixes single-frame 400/600 ns values with the 1 us multi-frame central (frames 10.7-11.8 %). Both are fine; "
                                       "use 3.5 / 6.5 / 11.5 % for one consistent frame.")
    tn["Q5_leakage_vs_window_prep1"] = {wn: F["leakage_E3_by_window"][wn]["Q5"]["prep1"] for wn in ("400ns", "600ns", "1us")}
    g = F["step_detector_gating_and_sensitivity"]["1us"]
    tn["step_detector_1us"] = {q: {b: {"measurable": g[q][b]["measurable"], "p_flag_ge_0.5_window_ns": g[q][b]["ts_window_p_flag_ge_0.5_ns"],
                                       "p_flag_max": g[q][b]["p_flag_max"], "eff_given_cnn_proxy_misread": g[q][b]["eff_given_cnn_proxy_misread"],
                                       "quasinull_obs_over_exp": g[q][b]["quasinull_obs_over_exp"]} for b in ("prep0", "prep1")} for q in QN}
    tn["herq_error_ratio_cnn_over_herq"] = {wn: {q: {"pertrace": F["herqules"]["comparison"][wn][q]["ratio_cnn_over_herq_pertrace"],
                                                     "scrambled_two_shot": F["herqules"]["comparison"][wn][q]["ratio_cnn_over_herq_scrambled"]}
                                                 for q in QN} for wn in ("1us", "400ns")}
    tn["herq_state_change_diff_cnn_minus_herq_pp"] = {wn: {q: F["herqules"]["comparison"][wn][q]["diff_cnn_minus_herq_pertrace_pp"]["state_change_total_T_plus_S"]
                                                           for q in ("Q3", "Q5")} for wn in ("1us", "400ns")}
    tn["old_vs_new_pooled_all5_1us"] = {"old_near_boundary": 97.4, "old_T1": 2.4, "old_leakage": 0.2,
                                        "new_overlap_upper": T1["pooled"]["all5"]["overlap_near_boundary_upper"]["pct_err"],
                                        "new_state_change_lower": T1["pooled"]["all5"]["state_change_total_T_plus_S"]["pct_err"],
                                        "new_resolved_step": T1["pooled"]["all5"]["transition_resolved"]["pct_err"],
                                        "new_leakage": T1["pooled"]["all5"]["leakage"]["pct_err"]}
    return tn


if __name__ == "__main__":
    main()
