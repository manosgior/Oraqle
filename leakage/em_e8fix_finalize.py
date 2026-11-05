"""em_e8fix_finalize.py (tag e8fix): final numbers file after the review-fix stage.

Reads existing results only (no new measurement of the raw data):
  data/leakage/em_e6edit_final_numbers.json   (previous final numbers; copied, then extended)
  data/leakage/em_e4recon_final.json          (step-detector gating / sensitivity at 1 us)
  data/leakage/m1tail_v2_percfg_test.json      (per-neighbour-configuration E3, used for the Q3 conditioning of Q1/Q5)
  data/leakage/LEAKAGE_REPORT.md               (multi-frame ranges of the Q3 conditioning, cited, not parsed)
  sims/leakage/em_e2seg_attr_v3.py             (LATE_R, mean-shift bandwidth, MAD width: cited constants)
Writes data/leakage/em_e8fix_final_numbers.json (new file). Nothing existing is modified.
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import math
import os

ROOT = str(LP.ROOT)
DL = str(LP.RES)


def load(name):
    with open(os.path.join(DL, name)) as f:
        return json.load(f)


def ratio_ci(a, a_ci, b, b_ci):
    """b/a with a delta-method (log) 95% CI from two independent 95% CIs."""
    sa = (a_ci[1] - a_ci[0]) / 3.92
    sb = (b_ci[1] - b_ci[0]) / 3.92
    r = b / a
    rel = math.sqrt((sa / a) ** 2 + (sb / b) ** 2)
    return round(r, 3), [round(r * math.exp(-1.96 * rel), 3), round(r * math.exp(1.96 * rel), 3)]


def main():
    base = load("em_e6edit_final_numbers.json")
    rec = load("em_e4recon_final.json")
    pc = load("m1tail_v2_percfg_test.json")
    out = dict(base)
    out["tag"] = "e8fix"
    out["date"] = "2026-11-05"
    out["supersedes"] = "data/leakage/em_e6edit_final_numbers.json (kept unchanged; every e6edit entry is copied here)"
    out["inputs"] = list(base["inputs"]) + [
        "data/leakage/em_e6edit_final_numbers.json",
        "data/leakage/m1tail_v2_percfg_test.json",
        "sims/leakage/em_e2seg_attr_v3.py (constants LATE_R, mean-shift bandwidth, MAD width)",
    ]
    ch = {}

    # ---- 1. rounding convention in the Results paragraph (point estimates of bounded quantities)
    t = base["table_error_modes_1us"]
    r = base["results_per_qubit_window_cnn"]
    ch["results_rounding"] = {
        "text": ">=77% (Q3) and >=90% (Q5) state change vs <=20% and <=9% overlap",
        "Q3_state_change": {"value": r["Q3_1us"]["state_change_pct_err"]["value"], "ci95": r["Q3_1us"]["state_change_pct_err"]["ci95"]},
        "Q5_state_change": {"value": r["Q5_1us"]["state_change_pct_err"]["value"], "ci95": r["Q5_1us"]["state_change_pct_err"]["ci95"]},
        "Q3_overlap": {"value": r["Q3_1us"]["overlap_pct_err"]["value"], "ci95": r["Q3_1us"]["overlap_pct_err"]["ci95"]},
        "Q5_overlap": {"value": r["Q5_1us"]["overlap_pct_err"]["value"], "ci95": r["Q5_1us"]["overlap_pct_err"]["ci95"]},
        "convention": "point estimates rounded to integers; >= / <= mark the systematic bound status (T+S lower bound, O upper bound), "
                      "same as the table headers. e6edit used CI edges for Q3 only (76/21); changed to 77/20 for consistency (review nit).",
        "status": "MEASURED",
    }

    # ---- 2. Q3 conditioning of the Q1/Q5 off-axis populations (per-configuration E3, m1tail frame)
    def cond(key, bit, extra=None):
        rows = pc[key]["prep1"]["rows"]
        res = {}
        for v in (0, 1):
            sel = [x for x in rows if ((x["cfg"] >> bit) & 1) == v and (extra is None or extra(x["cfg"]))]
            n = sum(x["N"] for x in sel)
            e = sum(x["E3"] * x["N"] for x in sel) / n
            se = math.sqrt(sum((x["E3_se"] * x["N"]) ** 2 for x in sel)) / n
            res[f"Q3_prep{v}"] = {"value": round(100 * e, 3), "ci95": [round(100 * (e - 1.96 * se), 3), round(100 * (e + 1.96 * se), 3)],
                                  "se": 100 * se, "n_shots": n}
        a, b = res["Q3_prep1"]["value"], res["Q3_prep0"]["value"]
        sa, sb = res["Q3_prep1"]["se"], res["Q3_prep0"]["se"]
        s = a / (a + b)
        ds = math.sqrt((b / (a + b) ** 2 * sa) ** 2 + (a / (a + b) ** 2 * sb) ** 2)
        for v in res.values():
            v.pop("se")
        res["share_of_population_with_Q3_prep1_pct"] = {"value": round(100 * s, 1),
                                                        "ci95": [round(100 * (s - 1.96 * ds), 1), round(100 * (s + 1.96 * ds), 1)]}
        return res

    # configuration index = bits of the other four qubits in code order; Q3 = code q2 -> bit 1 for q0, bit 2 for q4
    q5 = cond("1us_q4", 2)
    q5["Q4_prep0_only"] = cond("1us_q4", 2, lambda c: ((c >> 3) & 1) == 0)
    q5["Q4_prep1_only"] = cond("1us_q4", 2, lambda c: ((c >> 3) & 1) == 1)
    q1 = cond("1us_q0", 1)
    ch["Q3_conditioning_of_Q1_Q5_leakage_1us"] = {
        "Q5_prep1_pct": q5,
        "Q1_prep1_pct": q1,
        "multi_frame_ranges_LEAKAGE_REPORT_3.2": {
            "Q5": "Q3,Q4 = 00: 0.31-0.42%; 01: 1.3-1.6%; 10: 8.6-9.7%; 11: 6.8-8.2%",
            "Q1": "0.12-0.13% with Q3 prepared |0>, 1.3-1.5% with Q3 prepared |1>",
        },
        "frame": "m1tail per-state-centre frame (mean-vs-core erratum bias <=0.06 pp for Q1/Q5); mirror-subtracted E3, "
                 "CI +/-1.96 sqrt(N+ + N-)/N per configuration, summed",
        "status": "MEASURED (populations and their Q3 dependence); origin unresolved",
        "paper_text": "main text: '4.7% of Q5's (mostly when Q3 is in |1>)'; App.: '4.7% (Q5) and 0.8% (Q1) of prepared |1>, ~90% of it with Q3 prepared in |1> (origin unresolved)'",
    }

    # ---- 3. per-shot state-change growth 400 ns -> 1 us (all three qubits of the pool)
    growth = {}
    for q in ("Q1", "Q3", "Q5"):
        a = r[f"{q}_400ns"]["state_change_pct_shots"]
        b = r[f"{q}_1us"]["state_change_pct_shots"]
        v, ci = ratio_ci(a["value"], a["ci95"], b["value"], b["ci95"])
        growth[q] = {"400ns_pct_shots": [a["value"], a["ci95"]], "1us_pct_shots": [b["value"], b["ci95"]], "ratio": v, "ratio_ci95": ci}
    growth["paper_text"] = "Longer readouts make them at most 1.6x more frequent per shot (Q1; Q3: 4.4->4.5%)"
    growth["status"] = "MEASURED"
    ch["state_change_per_shot_growth_400ns_to_1us"] = growth

    # ---- 4. HERQULES vs CNN, per-shot modes (Q3, 1 us)
    h = base["herqules_pertrace_1us"]["Q3"]
    ch["herqules_vs_cnn_Q3_modes_pct_shots"] = {
        "cnn_state_change": [r["Q3_1us"]["state_change_pct_shots"]["value"], r["Q3_1us"]["state_change_pct_shots"]["ci95"]],
        "herq_state_change": [h["state_change_pct_shots"]["value"], h["state_change_pct_shots"]["ci95"]],
        "cnn_overlap": [r["Q3_1us"]["overlap_pct_shots"]["value"], r["Q3_1us"]["overlap_pct_shots"]["ci95"]],
        "herq_overlap": [h["overlap_pct_shots"]["value"], h["overlap_pct_shots"]["ci95"]],
        "cnn_state_change_share_pct_err": [r["Q3_1us"]["state_change_pct_err"]["value"], r["Q3_1us"]["state_change_pct_err"]["ci95"]],
        "herq_state_change_share_pct_err": [h["state_change_pct_err"]["value"], h["state_change_pct_err"]["ci95"]],
        "paper_text": "also mostly by state change (Q3: 69%); MCMit-CNN's 23% fewer Q3 errors span both modes",
        "status": "MEASURED",
    }

    # ---- 5. step detector T: gating, efficiency, sensitivity at 1 us (from e4recon)
    g = rec["step_detector_gating_and_sensitivity"]["1us"]
    gate = {}
    for q, d in g.items():
        if not isinstance(d, dict):
            continue
        for b, o in d.items():
            if isinstance(o, dict):
                gate[f"{q}_{b}"] = {k: o.get(k) for k in ("test", "quasinull_obs_over_exp", "eff_given_cnn_proxy_misread",
                                                         "measurable", "ts_window_p_flag_ge_0.5_ns", "p_flag_max")}
    used = [v for v in gate.values() if v["measurable"]]
    ch["step_T_gating_1us"] = {
        "per_test": gate,
        "rule": "measurable iff 0.6 <= quasi-null obs/exp <= 1.3 and efficiency given a misread of the linear CNN proxy >= 5% "
                "(sims/leakage/em_e4recon_analyze.py gate_info)",
        "excluded_at_1us": sorted(k for k, v in gate.items() if not v["measurable"]),
        "efficiency_range_of_used_tests_pct": [round(100 * min(v["eff_given_cnn_proxy_misread"] for v in used), 1),
                                               round(100 * max(v["eff_given_cnn_proxy_misread"] for v in used), 1)],
        "p_flag_ge_0.5_window_ns_used_tests": sorted({tuple(v["ts_window_p_flag_ge_0.5_ns"]) for v in used if v["ts_window_p_flag_ge_0.5_ns"]}),
        "Q1_p_flag_max": [gate["Q1_prep0"]["p_flag_max"], gate["Q1_prep1"]["p_flag_max"]],
        "paper_text": "flagged with probability >=0.5 only between ~225 and ~775 ns on Q3/Q5 (Q1 peaks at 0.44); 16-28% of injected steps "
                      "that flip a linear proxy of the CNN are flagged; obs/exp 0.6-1.3, efficiency >=5%; at 1 us excludes Q2, Q4, Q3 relaxation",
        "status": "MEASURED (injection / quasi-null calibration)",
    }

    # ---- 6. definitions now spelled out in App. F (constants of the attribution code)
    ch["attribution_definitions"] = {
        "cores": "Gaussian-kernel mean-shift modes of the window-mean IQ points, bandwidth 0.5 sigma, TRAIN, per neighbour configuration",
        "sigma_leakage": "1.4826 * MAD of the prepared-|0> perpendicular displacement from the core line (per configuration)",
        "z_sign": "normal oriented to the off-axis side (positive Q component of n); leaked if z>3, mirror z<-3 subtracted",
        "late_core_radius_LATE_R": 1.5,
        "late_core_radius_units": "1.4826 * MAD of the prepared-|0> late-window (1-2 us) Q coordinate, per configuration",
        "bins_ns": 100,
        "source": "sims/leakage/em_e2seg_attr_v3.py (LATE_R, meanshift, mad, fit_line); em_e4recon_glrt.py reuses E.LATE_R",
    }

    # ---- 7. T+S absorbs sub-threshold leakage (Q3)
    k = base["leakage"]["Q3_leakage_share_k_range"]
    ch["S_absorbs_subthreshold_leakage_Q3"] = {
        "state_change_k3": r["Q3_1us"]["state_change_pct_err"]["value"],
        "state_change_k1": k["Q3_state_change_with_k1"],
        "difference_pp": round(r["Q3_1us"]["state_change_pct_err"]["value"] - k["Q3_state_change_with_k1"], 2),
        "paper_text": "on Q3, a z>1 cut lowers it by 1.5 pp",
        "source": k["source"],
        "status": "MEASURED",
    }

    # ---- 8. half-plane identity (formula now in App. F)
    hp = base["late_window_half_plane_check_1us"]
    ch["half_plane_identity"] = {
        "formula": "f_err = s + (1 - s) f_corr  ->  s = (f_err - f_corr) / (1 - f_corr)",
        "errors_on_0_side_Q1_Q3_Q5_pct": [hp[q]["errors_on_0_side_pct"]["value"] for q in ("Q1", "Q3", "Q5")],
        "correct_on_0_side_Q1_Q3_Q5_pct": [hp[q]["correct_on_0_side_pct"]["value"] for q in ("Q1", "Q3", "Q5")],
        "pooled_s_pct": hp["pooled_Q1Q3Q5"]["identity_estimate_pct"],
        "pooled_attributed_pct": hp["pooled_Q1Q3Q5"]["attributed_state_change_prep1_pct"],
        "paper_text": "90-95% vs 6-33%; s = 89% (attributed 90%)",
        "note": "e6edit text quoted 89-95% over Q1, Q3-Q5 (89 = Q4); now restricted to Q1/Q3/Q5 to match the conclusion",
        "status": "MEASURED fractions; reading s as state change INFERRED",
    }

    # ---- 9. reference band labels
    ch["reference_band"] = {
        "pooled_state_change_by_reference": "e4 core 81.27, r=1.0 81.36, r=2.0 80.66, r=2.5 80.23, half-plane 78.60, e5vrepro 2-sigma disc 80.73",
        "paper_text": "79-81% across stayed-in-state criteria (main); late-core radius 1-2.5 widths, half-plane (App.)",
    }
    out["e8fix_changes"] = ch

    # ---- final paper_text_map (replaces the e6edit one; e6edit's stays in its own file)
    out["paper_text_map"] = [
        {"where": "ml_discriminators.tex, Sec. 5.2 Analysis (l.89)", "text": "extra length accumulates qubit state changes (was: T_1 events)",
         "numbers": "qualitative; per-shot state change grows on Q1/Q5 (e8fix_changes.state_change_per_shot_growth_400ns_to_1us)"},
        {"where": "ml_discriminators.tex, What counts as a failure", "text": "state changes (relaxation/excitation before or during readout, or state-prep errors), leakage, overlap",
         "numbers": "none"},
        {"where": "Methodology", "text": "overlap = error rate on shots whose 1-2 us tail sits in the prepared state's core (upper bound); MCMit-CNN all qubits 1 us, Q1/Q3/Q5 400 ns; per-trace HERQULES Q3/Q5 1 us",
         "numbers": "scope: results_per_qubit_window_cnn (Q1/Q3/Q5 at 400 ns), herqules_pertrace_1us"},
        {"where": "Results", "text": ">=77% / >=90% state change vs <=20% / <=9% overlap", "numbers": "e8fix_changes.results_rounding"},
        {"where": "Results", "text": "91%/95% of Q3/Q5 prep-|1> errors on the |0> side vs 33%/15% of correct reads",
         "numbers": "Q3 90.97 [90.72, 91.21] vs 32.96 [32.83, 33.09]; Q5 94.66 [94.40, 94.92] vs 15.33 [15.23, 15.42]"},
        {"where": "Results", "text": "only 2-8% of errors show a detectable step inside the window (a lower bound)",
         "numbers": "resolved_step_T: Q1 4.33-7.76, Q5 5.02-7.36, Q3 (excitation only) 1.68-1.75 % of errors; conservative (FP null 24 vs 54.1)"},
        {"where": "Results", "text": "400 ns overlap 40% (Q3), 56% (Q5)", "numbers": "Q3 40.14 [39.35, 40.93]; Q5 55.90 [55.04, 56.76]"},
        {"where": "Results", "text": "11.5% of Q3's prep-|1> leak, as do 4.7% of Q5's (mostly when Q3 is in |1>); 97-99% of these read correctly; Q3 3.5/6.5/11.5%",
         "numbers": "Q3 11.46 [11.38, 11.55]; Q5 4.72 [4.66, 4.78]; Q5 share with Q3=1: e8fix_changes.Q3_conditioning_of_Q1_Q5_leakage_1us; read correctly Q3 97.0, Q5 98.7"},
        {"where": "Table tab:error_modes", "text": "unchanged values; caption: error rate % of shots, n/m defined; footnotes dagger/ddagger, Q4 split indicative",
         "numbers": "table_error_modes_1us; Q4_state_change_reference_band 82.1-88.7"},
        {"where": "Analysis", "text": "pooled Q1,Q3,Q5: 81% / 17% (79-81% and 17-20% across stayed-in-state criteria), leakage 1.8%",
         "numbers": "81.27 [80.90, 81.64] band 78.6-81.4; 16.90 [16.54, 17.27] band 16.8-19.6; 1.83 [1.75, 1.91]"},
        {"where": "Analysis", "text": "only Q2 is SNR-bound (a linear filter's Gaussian overlap alone predicts 28% error)",
         "numbers": "Q2_snr_bound: 28.26% (TRAIN LDA Delta 1.151); CNN 25.43 [25.35, 25.51]; status INFERRED"},
        {"where": "Analysis", "text": "at most 1.6x more frequent per shot (Q1; Q3 4.4->4.5%); 37% -> 81%; overlap falls 2.5-14x; leakage triples on Q3",
         "numbers": "e8fix_changes.state_change_per_shot_growth_400ns_to_1us (Q1 1.566 [1.50, 1.63]); pooled_Q1Q3Q5_window_trend; overlap_per_shot_fall_400ns_to_1us; leakage.Q3_growth_1us_over_400ns 3.27"},
        {"where": "Analysis", "text": "per-trace HERQULES: 68%/92% co-failure; also mostly state change (Q3 69%); CNN 23% fewer Q3 errors span both modes",
         "numbers": "herqules_pertrace_1us; e8fix_changes.herqules_vs_cnn_Q3_modes_pct_shots (SC 4.48 vs 5.27, O 1.17 vs 2.06 % of shots)"},
        {"where": "Takeaway #5", "text": "~80% state changes (relaxation, excitation, or faulty preparation), at most ~20% overlap; 0.4-6% of shots silently leak",
         "numbers": "81.27 (band 78.6-81.4); 16.9 (<=19.6 under every reference); leaked shots Q1 0.41 [0.39, 0.42] .. Q3 6.14 [6.10, 6.19]"},
        {"where": "Fig. iq_taxonomy caption", "text": "Leakage is an off-axis, non-computational population, not necessarily |2>", "numbers": "none"},
        {"where": "system_bottleneck.tex RQ13 hypothesis", "text": "at short readouts most residual misclassifications are low-confidence shots ('near the decision boundary' dropped)",
         "numbers": "hypothesis; no number"},
        {"where": "discussion.tex", "text": "(mostly qubit state changes and SNR overlap)", "numbers": "qualitative (e6edit, unchanged)"},
        {"where": "appendix_herqules.tex, App. error-modes", "text": "models paragraph moved before the procedure; L/T/O/S definitions; validation; leakage",
         "numbers": "e8fix_changes.attribution_definitions, step_T_gating_1us, S_absorbs_subthreshold_leakage_Q3, half_plane_identity, "
                    "Q3_conditioning_of_Q1_Q5_leakage_1us; resolved_step_T.Q3_relaxation_null_by_partners (1.035 / 5.39); appendix_facts.order_swap"},
    ]
    with open(os.path.join(DL, "em_e8fix_final_numbers.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: ch[k] for k in ("Q3_conditioning_of_Q1_Q5_leakage_1us", "state_change_per_shot_growth_400ns_to_1us")}, indent=1)[:3000])
    print("excluded:", ch["step_T_gating_1us"]["excluded_at_1us"], "eff:", ch["step_T_gating_1us"]["efficiency_range_of_used_tests_pct"],
          "windows:", ch["step_T_gating_1us"]["p_flag_ge_0.5_window_ns_used_tests"], "Q1 max:", ch["step_T_gating_1us"]["Q1_p_flag_max"])
    print("S absorbs:", ch["S_absorbs_subthreshold_leakage_Q3"]["difference_pp"])


if __name__ == "__main__":
    main()
