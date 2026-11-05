"""em_e6edit_finalize.py (tag e6edit): the FINAL numbers that appear in the paper.

Reads only existing result files (no recomputation from raw data is needed):
  data/leakage/em_e4recon_final.json     reconciled attribution (L -> T -> O -> S)
  data/leakage/em_e5vstat_summary.json   adversarial re-computation (pools, reference band,
                                          half-plane identity check, LDA, step null on
                                          misread stationary shots, co-failure)
  data/leakage/em_e5vstat_check.json     per-qubit counts (n for Wilson CIs)
  data/leakage/em_e5vrepro_results.json  independent reproduction from raw TEST HDF5
                                          (own detector -> T bracket; empirical null)
  data/leakage/em_e5vrepro_diag_partners.json  Q3 relaxation-null failure by partner state

Applies the verifier corrections judged valid (see VERDICTS below) and writes
  data/leakage/em_e6edit_final_numbers.json
  data/leakage/em_e6edit_final_numbers.md
Every number that goes into the paper is listed with value, 95% CI, status and source.
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import math
import os

ROOT = str(LP.ROOT)
DL = str(LP.RES)
E4 = json.load(open(os.path.join(DL, "em_e4recon_final.json")))
VS = json.load(open(os.path.join(DL, "em_e5vstat_summary.json")))
VC = json.load(open(os.path.join(DL, "em_e5vstat_check.json")))
VR = json.load(open(os.path.join(DL, "em_e5vrepro_results.json")))
VP = json.load(open(os.path.join(DL, "em_e5vrepro_diag_partners.json")))

SRC_E4 = "data/leakage/em_e4recon_final.json"
SRC_VS = "data/leakage/em_e5vstat_summary.json"
SRC_VC = "data/leakage/em_e5vstat_check.json"
SRC_VR = "data/leakage/em_e5vrepro_results.json"
SRC_VP = "data/leakage/em_e5vrepro_diag_partners.json"
SRC_REP = "data/leakage/LEAKAGE_REPORT.md"

Z = 1.959963984540054


def wilson(k, n):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return [100 * (c - h), 100 * (c + h)]


def r(x, nd=2):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [r(v, nd) for v in x]
    if isinstance(x, dict):
        return {k: r(v, nd) for k, v in x.items()}
    return round(float(x), nd)


def entry(value, ci, unit, status, source, key, paper=None, note=None):
    e = {"value": r(value, 3) if isinstance(value, (int, float)) else value,
         "ci95": r(ci, 3) if ci is not None else None,
         "unit": unit, "status": status, "source": source, "key": key}
    if paper is not None:
        e["paper_text"] = paper
    if note is not None:
        e["note"] = note
    return e


OUT = {"tag": "e6edit", "date": "2026-11-05",
       "inputs": [SRC_E4, SRC_VS, SRC_VC, SRC_VR, SRC_VP, SRC_REP],
       "conventions": (
           "Qubits Q1..Q5 = code q0..q4. TEST split, 1.12M shots per qubit (560k per prepared bit). "
           "1 us = samples 0-499 (paper 'full trace'); 600 ns = 0-299; 400 ns = 0-199. "
           "1 us model = canonical MCMit-CNN (preds_MCMitCNN_pb.npy); 600/400 ns = soft MCMit-CNN, same recipe. "
           "HERQULES = per-trace checkpoint (the one in tab:fidelity). Shares are of that qubit's errors "
           "unless marked. CIs: error rates Wilson; L +/-1.96 sqrt(N+ + N-); O binomial per configuration; "
           "var(T+S) = var(L) + var(O). All CIs statistical; reference/detector systematics given as ranges.")}

# --------------------------------------------------------------------------------------
# A. Table tab:error_modes (MCMit-CNN canonical, 1 us)
# --------------------------------------------------------------------------------------
T1 = E4["table_1us_cnn_canonical"]
tab = {}
for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
    e = T1["per_qubit"][q]
    row = {"err_rate_pct": entry(e["err_rate_pct"], e["err_rate_ci95_pct"], "% of shots", "MEASURED",
                                 SRC_E4, f"table_1us_cnn_canonical.per_qubit.{q}.err_rate_pct"),
           "n_err": e["n_err"]}
    if e["leakage"] is None:
        row["leakage"] = {"value": "n/m", "reason": "multimodal |1> response (LEAKAGE_REPORT 3.2)"}
    else:
        row["leakage"] = entry(e["leakage"]["pct_err"], e["leakage"]["ci95_pct_err"], "% of errors",
                               "MEASURED (k=3 lower bound, mirror-corrected)", SRC_E4,
                               f"table_1us_cnn_canonical.per_qubit.{q}.leakage")
    sc = e["state_change_total_T_plus_S"]
    ov = e["overlap_near_boundary_upper"]
    if q == "Q2":
        row["state_change"] = {"value": "n/m",
                               "late_core_value_uninformative": r(sc["pct_err"]),
                               "reason": "late-window |0>/|1> cores only 0.52 sigma apart (e5vstat), so O ~ N and "
                                         "S ~ 0 by construction (same circularity as the old rule)"}
        row["overlap"] = {"value": "n/m", "late_core_value_uninformative": r(ov["pct_err"])}
    else:
        st = "MEASURED lower bound" if q != "Q4" else "INDICATIVE (reference fails noise-independence test)"
        row["state_change"] = entry(sc["pct_err"], sc["ci95_pct_err"], "% of errors", st, SRC_E4,
                                    f"table_1us_cnn_canonical.per_qubit.{q}.state_change_total_T_plus_S")
        st = "MEASURED upper bound" if q != "Q4" else "INDICATIVE"
        row["overlap"] = entry(ov["pct_err"], ov["ci95_pct_err"], "% of errors", st, SRC_E4,
                               f"table_1us_cnn_canonical.per_qubit.{q}.overlap_near_boundary_upper")
    hl = E4["text_numbers"]["hidden_leakage_1us"].get(q)
    if hl:
        row["leaked_shots_pct_of_shots"] = entry(hl["leaked_pct_of_shots"]["value"], hl["leaked_pct_of_shots"]["ci95"],
                                                 "% of all shots", "MEASURED", SRC_E4,
                                                 f"text_numbers.hidden_leakage_1us.{q}.leaked_pct_of_shots")
        row["leaked_read_correctly_pct"] = entry(hl["read_correctly_pct"]["value"], hl["read_correctly_pct"]["ci95"],
                                                 "% of leaked shots", "MEASURED", SRC_E4,
                                                 f"text_numbers.hidden_leakage_1us.{q}.read_correctly_pct")
    else:
        row["leaked_shots_pct_of_shots"] = {"value": "n/m"}
    tab[q] = row

# headline pool: Q1, Q3, Q5 (verifier e5vstat #3: Q4 out of the headline)
p = T1["pooled"]["Q1Q3Q5"]
tab["Q1Q3Q5"] = {
    "err_rate_pct": entry(p["err_rate_pct"], p["err_rate_ci95_pct"], "% of shots", "MEASURED", SRC_E4,
                          "table_1us_cnn_canonical.pooled.Q1Q3Q5.err_rate_pct"),
    "n_err": p["n_err"],
    "leakage": entry(p["leakage"]["pct_err"], p["leakage"]["ci95_pct_err"], "% of errors", "MEASURED", SRC_E4,
                     "table_1us_cnn_canonical.pooled.Q1Q3Q5.leakage"),
    "state_change": entry(p["state_change_total_T_plus_S"]["pct_err"], p["state_change_total_T_plus_S"]["ci95_pct_err"],
                          "% of errors", "MEASURED lower bound", SRC_E4,
                          "table_1us_cnn_canonical.pooled.Q1Q3Q5.state_change_total_T_plus_S"),
    "overlap": entry(p["overlap_near_boundary_upper"]["pct_err"], p["overlap_near_boundary_upper"]["ci95_pct_err"],
                     "% of errors", "MEASURED upper bound", SRC_E4,
                     "table_1us_cnn_canonical.pooled.Q1Q3Q5.overlap_near_boundary_upper"),
}
# reference systematic for the pooled state change (e5vstat own cores + half-plane; e5vrepro own 2-sigma disc)
band = {k: VS["pooled"]["1us|cnn_canonical"]["Q1Q3Q5"][k]["SC_pct_err"] for k in ["e4_lc", "r1.0", "r1.5", "r2.0", "r2.5", "hp"]}
g = VR["attribution"]["1us"]["gauss"]
sc_rep = sum(g[q]["state_change_TS_lower"]["count"] for q in ["Q1", "Q3", "Q5"])
ne_rep = sum(g[q]["n_err"] for q in ["Q1", "Q3", "Q5"])
band["e5vrepro_2sigma_disc"] = 100 * sc_rep / ne_rep
tab["Q1Q3Q5"]["state_change_reference_band"] = {
    "values": r(band), "range": r([min(band.values()), max(band.values())], 1), "status": "systematic (range, not CI)",
    "source": f"{SRC_VS} pooled.1us|cnn_canonical.Q1Q3Q5.*.SC_pct_err; {SRC_VR} attribution.1us.gauss",
    "paper_text": "79--81% across reference definitions"}
ov_band = {k: VS["pooled"]["1us|cnn_canonical"]["Q1Q3Q5"][k]["O_pct_err"] for k in ["e4_lc", "r1.0", "r2.5", "hp"]}
tab["Q1Q3Q5"]["overlap_reference_band"] = {"values": r(ov_band), "range": r([min(ov_band.values()), max(ov_band.values())], 1)}
# other pools, NOT headlined (kept for traceability)
for k in ["Q1Q3Q4Q5", "all5"]:
    pp = T1["pooled"][k]
    tab[k + "_not_headlined"] = {
        "state_change": r(pp["state_change_total_T_plus_S"]["pct_err"]),
        "overlap": r(pp["overlap_near_boundary_upper"]["pct_err"]),
        "leakage": r(pp["leakage"]["pct_err"]),
        "why_not": ("Q4 leakage counted as 0 and Q4 O indicative (e5vstat #3)" if k == "Q1Q3Q4Q5" else
                    "dominated by Q2's 285k errors whose split is circular (e5vstat #2); e5vrepro moves it to 63.6/35.7")}
OUT["table_error_modes_1us"] = tab

# Q4 reference band (indicative)
q4band = {k: VS["pooled"]["1us|cnn_canonical"]["Q4"][k]["SC_pct_err"] for k in ["e4_lc", "r1.0", "r1.5", "r2.5", "hp"]}
OUT["Q4_state_change_reference_band"] = {"values": r(q4band), "range": r([min(q4band.values()), max(q4band.values())], 1),
                                         "source": SRC_VS}
# Q2: SNR-bound evidence
lda2 = VS["lda_vs_cnn_latecore"]["Q2"]["1us"]["prep0"]
OUT["Q2_snr_bound"] = {
    "late_window_core_separation_sigma": r(VS["late_window_separation_sigma"]["Q2"]),
    "other_qubits_separation_sigma": r({q: VS["late_window_separation_sigma"][q] for q in ["Q1", "Q3", "Q4", "Q5"]}),
    "lda_delta": r(lda2["delta"], 3),
    "lda_gaussian_predicted_error_pct": r(lda2["lda_gauss_pred_pct"]),
    "cnn_error_pct": r(T1["per_qubit"]["Q2"]["err_rate_pct"]),
    "status": "MEASURED (Delta from TRAIN LDA on 20 features); 'SNR-bound' = INFERRED from Phi(-Delta/2) >= observed",
    "source": f"{SRC_VS} lda_vs_cnn_latecore.Q2.1us.prep0; late_window_separation_sigma",
    "paper_text": "Q2's states are only 0.5 sigma apart in the unused tail; a linear filter's Gaussian overlap alone predicts 28% error"}

# --------------------------------------------------------------------------------------
# B. Results paragraph: Q3 / Q5 at 1 us and 400 ns (+600 ns)
# --------------------------------------------------------------------------------------
res = {}
for q in ["Q3", "Q5", "Q1"]:
    for w, src in [("1us", ("table_1us_cnn_canonical", None)), ("600ns", ("windows", "600ns_cnn_soft600")),
                   ("400ns", ("windows", "400ns_cnn_soft400"))]:
        e = T1["per_qubit"][q] if w == "1us" else E4["windows"][src[1]]["per_qubit"][q]
        key = (f"table_1us_cnn_canonical.per_qubit.{q}" if w == "1us" else f"windows.{src[1]}.per_qubit.{q}")
        d = {"err_rate_pct": entry(e["err_rate_pct"], e["err_rate_ci95_pct"], "% of shots", "MEASURED", SRC_E4, key)}
        for m, lab in [("state_change_total_T_plus_S", "state_change"), ("overlap_near_boundary_upper", "overlap"),
                       ("leakage", "leakage")]:
            if e.get(m) is None:
                continue
            d[lab + "_pct_err"] = entry(e[m]["pct_err"], e[m]["ci95_pct_err"], "% of errors", "MEASURED", SRC_E4, f"{key}.{m}")
            d[lab + "_pct_shots"] = entry(e[m]["pct_shots"], e[m]["ci95_pct_shots"], "% of shots", "MEASURED", SRC_E4, f"{key}.{m}")
        res[f"{q}_{w}"] = d
OUT["results_per_qubit_window_cnn"] = res

# overlap per-shot fall 400 ns -> 1 us (ratio, delta-method CI on log ratio from the two CIs)
def ratio_ci(a, aci, b, bci):
    sa = (aci[1] - aci[0]) / (2 * Z) / a
    sb = (bci[1] - bci[0]) / (2 * Z) / b
    s = math.sqrt(sa * sa + sb * sb)
    rr = a / b
    return rr, [rr * math.exp(-Z * s), rr * math.exp(Z * s)]

falls = {}
for q in ["Q1", "Q3", "Q5"]:
    a = res[f"{q}_400ns"]["overlap_pct_shots"]
    b = res[f"{q}_1us"]["overlap_pct_shots"]
    rr, ci = ratio_ci(a["value"], a["ci95"], b["value"], b["ci95"])
    falls[q] = entry(rr, ci, "x (400 ns / 1 us, per-shot overlap)", "MEASURED", SRC_E4, "derived from results_per_qubit_window_cnn")
OUT["overlap_per_shot_fall_400ns_to_1us"] = falls
OUT["overlap_per_shot_fall_400ns_to_1us"]["paper_text"] = "overlap errors fall 2.5--14x"

# pooled Q1,Q3,Q5 window trend
trend = {}
for w, key in [("400ns", "400ns_cnn_soft400"), ("600ns", "600ns_cnn_soft600")]:
    pp = E4["windows"][key]["pooled"]["Q1Q3Q5"]
    trend[w] = {"state_change": entry(pp["state_change_total_T_plus_S"]["pct_err"], pp["state_change_total_T_plus_S"]["ci95_pct_err"],
                                      "% of errors", "MEASURED lower bound", SRC_E4, f"windows.{key}.pooled.Q1Q3Q5"),
                "overlap": entry(pp["overlap_near_boundary_upper"]["pct_err"], pp["overlap_near_boundary_upper"]["ci95_pct_err"],
                                 "% of errors", "MEASURED upper bound", SRC_E4, f"windows.{key}.pooled.Q1Q3Q5")}
trend["1us"] = {"state_change": tab["Q1Q3Q5"]["state_change"], "overlap": tab["Q1Q3Q5"]["overlap"]}
trend["paper_text"] = "share grows from 37% (400 ns) to 81% (1 us)"
OUT["pooled_Q1Q3Q5_window_trend"] = trend

# --------------------------------------------------------------------------------------
# C. Late-window (1-2 us) half-plane identity check (e5vstat #5; e5vrepro #5)
# --------------------------------------------------------------------------------------
idc = {}
num_s, num_sc, den = 0.0, 0.0, 0
for q in ["Q1", "Q3", "Q4", "Q5"]:
    iv = VS["identity_prep1"]["1us|cnn_canonical"][q]
    n_e = VC["per_qubit"][q]["models"]["1us|cnn_canonical"]["prep1"]["N"]
    n_c = 560000 - n_e
    fe, fc = iv["late_other_frac_err"], iv["late_other_frac_correct"]
    s = (fe - fc) / (1 - fc)
    # delta method
    vfe, vfc = fe * (1 - fe) / n_e, fc * (1 - fc) / n_c
    ds_dfe = 1 / (1 - fc)
    ds_dfc = (fe - 1) / (1 - fc) ** 2
    ss = math.sqrt(ds_dfe ** 2 * vfe + ds_dfc ** 2 * vfc)
    idc[q] = {"errors_on_0_side_pct": entry(100 * fe, wilson(round(fe * n_e), n_e), "% of prep-|1> errors", "MEASURED", SRC_VS,
                                            f"identity_prep1.1us|cnn_canonical.{q}.late_other_frac_err"),
              "correct_on_0_side_pct": entry(100 * fc, wilson(round(fc * n_c), n_c), "% of correctly read prep-|1>", "MEASURED", SRC_VS,
                                             f"identity_prep1.1us|cnn_canonical.{q}.late_other_frac_correct"),
              "identity_estimate_pct": entry(100 * s, [100 * (s - Z * ss), 100 * (s + Z * ss)], "% of prep-|1> errors",
                                             "MEASURED fractions; reading as state change INFERRED", SRC_VS, f"identity_prep1...{q}.s_identity"),
              "attributed_state_change_prep1_pct": r(iv["SC_prep1_e4ref"]), "n_err": n_e}
    if q != "Q4":
        num_s += s * n_e
        num_sc += iv["SC_prep1_e4ref"] / 100 * n_e
        den += n_e
idc["pooled_Q1Q3Q5"] = {"identity_estimate_pct": r(100 * num_s / den), "attributed_state_change_prep1_pct": r(100 * num_sc / den)}
idc["paper_text"] = ("in the unused 1-2 us, 91% (Q3) / 95% (Q5) of prepared-|1> errors sit on the |0> side, "
                     "against 33% / 15% of correctly read shots")
OUT["late_window_half_plane_check_1us"] = idc

# --------------------------------------------------------------------------------------
# D. Resolved mid-readout step T: detector bracket (e5vrepro #1, #2; e5vstat #6)
# --------------------------------------------------------------------------------------
emp = VR["attribution"]["1us"]["emp"]
tb = {}
for q in ["Q1", "Q3", "Q5"]:
    e4t = T1["per_qubit"][q]["transition_resolved"]
    tb[q] = {"e2seg_gaussian_null_alpha0.01": entry(e4t["pct_err"], e4t["ci95_pct_err"], "% of errors",
                                                    "MEASURED, conservative lower bound", SRC_E4,
                                                    f"table_1us_cnn_canonical.per_qubit.{q}.transition_resolved"),
             "e5vrepro_gaussian_null": entry(g[q]["transition_T"]["value"], g[q]["transition_T"]["ci95"], "% of errors",
                                             "MEASURED (own detector)", SRC_VR, f"attribution.1us.gauss.{q}.transition_T"),
             "e5vrepro_empirical_null": entry(emp[q]["transition_T"]["value"], emp[q]["transition_T"]["ci95"], "% of errors",
                                              "MEASURED (own detector, empirical overlap null)", SRC_VR, f"attribution.1us.emp.{q}.transition_T")}
tb["Q3"]["note"] = "excitation only (relaxation test n/m in both pipelines)"
tb["bracket_text"] = "Q1 4.3-7.8, Q5 5.0-7.4, Q3 (excitation only) 1.7-1.75 % of errors"
tb["paper_text"] = "only 2--8% of errors show a step resolvable inside the window"
sn = VS["step_null_misread_stationary"]
obs = sum(sn[q][k]["obs"] for q in ["Q1", "Q5"] for k in ["down_prep1_e4_lc_misread", "up_prep0_e4_lc_misread"])
exp = sum(sn[q][k]["exp"] for q in ["Q1", "Q5"] for k in ["down_prep1_e4_lc_misread", "up_prep0_e4_lc_misread"])
tb["fp_null_on_misread_stationary_Q1Q5"] = {"observed_flags": obs, "expected_flags": r(exp, 1), "ratio": r(obs / exp, 2),
                                             "status": "MEASURED", "source": f"{SRC_VS} step_null_misread_stationary",
                                             "meaning": "FP correction over-subtracts on misread shots -> T conservative"}
tb["sensitivity"] = {"p_flag_ge_0.5_ns": {"Q3_excitation": [225, 775], "Q5_relaxation": [225, 700], "Q5_excitation": [250, 675]},
                     "partial_100_150ns": "P(flag) 0.1-0.3 (Q3/Q5)", "Q1_max_p_flag": 0.44,
                     "efficiency_given_misread": {"e2seg": "16-28%", "e5vrepro": "36-56%"},
                     "source": f"{SRC_E4} step_detector_gating_and_sensitivity; e5vstat issue 6; {SRC_VR} step_test"}
tb["efficiency_corrected_INFERRED"] = {"prep1": "10-27% of prep-|1> errors (Q1 22-27, Q5 10-16)",
                                       "prep0": "19-51% of prep-|0> errors (Q5 19-48, Q3 19-51)",
                                       "source": f"{SRC_E4} transition_efficiency_corrected_INFERRED_pct_of_bit_errors"}
q3p = VP["1us_q2"]
tb["Q3_relaxation_null_by_partners"] = {
    "Q1=Q4=Q5=0": [q3p["Q1=0,Q2=0,Q4=0,Q5=0"]["ratio"], q3p["Q1=0,Q2=1,Q4=0,Q5=0"]["ratio"]],
    "all_three_excited": [q3p["Q1=1,Q2=0,Q4=1,Q5=1"]["ratio"], q3p["Q1=1,Q2=1,Q4=1,Q5=1"]["ratio"]],
    "status": "MEASURED (obs/exp flags on quasi-null shots); mechanism (neighbour exchange) INFERRED", "source": SRC_VP + " 1us_q2"}
OUT["resolved_step_T"] = tb

# --------------------------------------------------------------------------------------
# E. Leakage (hidden channel, window growth)
# --------------------------------------------------------------------------------------
lw = E4["leakage_E3_by_window"]
OUT["leakage"] = {
    "Q3_prep1_by_window_pct": {w: entry(lw[w]["Q3"]["prep1"]["E3_pct"], lw[w]["Q3"]["prep1"]["ci95"], "% of prep-|1> shots",
                                         "MEASURED (k=3 lower bound)", SRC_E4, f"leakage_E3_by_window.{w}.Q3.prep1")
                               for w in ["400ns", "600ns", "1us"]},
    "Q3_prep1_1us_frame_range": {"range": [11.3, 11.6], "multi_frame_central": 11.3, "frames": [10.7, 11.8],
                                 "e5vrepro_own_IF": 11.61, "source": f"{SRC_REP} 3.2; {SRC_VR} leak_E3_prep1_prep0_all"},
    "Q3_prep0_1us_pct": entry(lw["1us"]["Q3"]["prep0"]["E3_pct"], lw["1us"]["Q3"]["prep0"]["ci95"], "% of prep-|0> shots",
                              "MEASURED", SRC_E4, "leakage_E3_by_window.1us.Q3.prep0"),
    "Q5_prep1_by_window_pct": {w: entry(lw[w]["Q5"]["prep1"]["E3_pct"], lw[w]["Q5"]["prep1"]["ci95"], "% of prep-|1> shots",
                                         "MEASURED", SRC_E4, f"leakage_E3_by_window.{w}.Q5.prep1") for w in ["400ns", "600ns", "1us"]},
    "Q1_prep1_1us_pct": entry(lw["1us"]["Q1"]["prep1"]["E3_pct"], lw["1us"]["Q1"]["prep1"]["ci95"], "% of prep-|1> shots", "MEASURED", SRC_E4, "leakage_E3_by_window.1us.Q1.prep1"),
    "Q2_prep1_1us_pct": entry(lw["1us"]["Q2"]["prep1"]["E3_pct"], lw["1us"]["Q2"]["prep1"]["ci95"], "% of prep-|1> shots", "MEASURED (probable)", SRC_E4, "leakage_E3_by_window.1us.Q2.prep1"),
    "Q3_leaked_prep0_errors_share_pct": entry(13.39, [12.72, 14.05], "% of Q3 prep-|0> errors", "MEASURED", SRC_E4,
                                              "table_1us_cnn_canonical.per_qubit.Q3.per_bit.prep0.leakage"),
    "Q3_leakage_share_k_range": {"k5": 2.3, "k3": 3.1, "k2": 3.8, "k1": 4.7, "Q3_state_change_with_k1": 75.3,
                                 "source": "e5vstat issue 7 (em_e5vstat_check.json L1..L5)"},
    "identity": "non-computational; geometry disfavours |2> (pull ratio 4-7x vs 0.76x); never call it |2> (INFERRED)",
    "paper_text": "11.5% of Q3's prepared-|1> shots (6.1% of all its shots) and 4.7% of Q5's leak; 97-99% of leaked shots read correctly; 3.5/6.5/11.5% at 400/600/1000 ns",
    "decision_11.5_vs_11.3": ("3.5 and 6.5 are single line-frame values; 11.3 is the 1 us multi-frame central. For one consistent "
                              "frame use 11.5 (11.46 [11.38, 11.55]); both verifiers support it (e5vstat 11.46, e5vrepro 11.61).")}

# --------------------------------------------------------------------------------------
# F. HERQULES (per-trace) comparison
# --------------------------------------------------------------------------------------
H1 = E4["herqules"]["1us_herq_pertrace"]["per_qubit"]
cf = VS["cofailure"]
herq = {}
for q in ["Q3", "Q5"]:
    e = H1[q]
    herq[q] = {"err_rate_pct": entry(e["err_rate_pct"], e["err_rate_ci95_pct"], "% of shots", "MEASURED", SRC_E4, f"herqules.1us_herq_pertrace.per_qubit.{q}"),
               "state_change_pct_err": entry(e["state_change_total_T_plus_S"]["pct_err"], e["state_change_total_T_plus_S"]["ci95_pct_err"], "% of errors", "MEASURED lower bound", SRC_E4, f"herqules.1us_herq_pertrace.per_qubit.{q}.state_change_total_T_plus_S"),
               "overlap_pct_err": entry(e["overlap_near_boundary_upper"]["pct_err"], e["overlap_near_boundary_upper"]["ci95_pct_err"], "% of errors", "MEASURED upper bound", SRC_E4, f"herqules.1us_herq_pertrace.per_qubit.{q}.overlap_near_boundary_upper"),
               "state_change_pct_shots": entry(e["state_change_total_T_plus_S"]["pct_shots"], e["state_change_total_T_plus_S"]["ci95_pct_shots"], "% of shots", "MEASURED", SRC_E4, "..."),
               "overlap_pct_shots": entry(e["overlap_near_boundary_upper"]["pct_shots"], e["overlap_near_boundary_upper"]["ci95_pct_shots"], "% of shots", "MEASURED", SRC_E4, "..."),
               "cofailure_frac_herq_err_also_cnn_pct": entry(100 * cf[f"1us_{q}"]["frac_herq_err_also_cnn"], r([100 * x for x in cf[f"1us_{q}"]["ci"]], 3), "% of HERQULES errors", "MEASURED", SRC_VS, f"cofailure.1us_{q}")}
# CNN fewer Q3 errors than HERQULES: 1 - n_cnn/n_herq; CI via log ratio of Poisson counts (conservative, ignores overlap)
nc, nh = cf["1us_Q3"]["cnn_err"], cf["1us_Q3"]["herq_err"]
lr = math.log(nc / nh)
se = math.sqrt(1 / nc + 1 / nh)
herq["Q3_cnn_fewer_errors_pct"] = entry(100 * (1 - nc / nh), [100 * (1 - math.exp(lr + Z * se)), 100 * (1 - math.exp(lr - Z * se))],
                                        "% fewer errors than per-trace HERQULES", "MEASURED", SRC_VS, "cofailure.1us_Q3 (cnn_err, herq_err)")
herq["model"] = "per-trace HERQULES (tab:fidelity); scrambled/demux HERQULES excluded (features from different shots)"
herq["paper_text"] = ("per-trace HERQULES: Q3 69% state change / 27% overlap; 68% of its Q3 and 92% of its Q5 errors are also CNN errors; "
                      "CNN 23% fewer Q3 errors, in both modes (4.5 vs 5.3% / 1.2 vs 2.1% of shots)")
herq["dropped_claim"] = "'same proportions despite a 2x rate gap': no 2x gap between single-shot models (CNN/HERQ 0.77-0.98 at 1 us)"
OUT["herqules_pertrace_1us"] = herq

# --------------------------------------------------------------------------------------
# G. Robustness / method facts quoted in the appendix
# --------------------------------------------------------------------------------------
OUT["appendix_facts"] = {
    "order_swap": E4["robustness_notes"]["order_swap"],
    "noise_independence": "CNN error rate vs late displacement flat within CIs for Q1/Q3/Q5 (both bits); not flat for Q2 (33.7->15.2%) and Q4 prep |1> (e5vstat)",
    "repro_independent_pipeline": "e5vrepro (own demod, frames, detector, late core): L, O, T+S agree within combined CIs for Q3, Q5 (1 us, 400 ns) and the Q1,Q3-Q5 pool",
    "old_procedure_on_canonical_cnn": "97.41 / 2.44 / 0.16 (near / T1 / amp outlier), vs paper 97.4 / 2.4 / 0.2: the wrong method, not the wrong model, set the shares",
    "old_rule": "near-boundary = P(correct) < 0.6, true for every error; 'T1' = sign test on un-centred halves (cannot fire on Q1/Q3; flags 9.5% of all Q5 prep-|0> shots); 'leakage' = |z(mean|IQ|)|>3 among HybridCNN errors",
    "late_core_upper_bound": E4["robustness_notes"]["late_core_is_upper_bound"],
    "source": SRC_E4 + " robustness_notes, old_method; e5vstat/e5vrepro issues"}

# --------------------------------------------------------------------------------------
# H. Verdicts on verifier issues
# --------------------------------------------------------------------------------------
OUT["verifier_verdicts"] = {
    "e5vstat_1_300x": "ACCEPTED: dropped; LDA matches CNN on Q5 prep-|1> late-core shots, so the late-core rate is contamination, not CNN overlap.",
    "e5vstat_2_Q2_circular": "ACCEPTED: Q2 state change/overlap shown as n/m; SNR-bound argued from LDA Gaussian overlap (28% >= 25.4%); all-5 pool not headlined.",
    "e5vstat_3_Q4_in_headline": "ACCEPTED: headline pool = Q1, Q3, Q5 (81.3 / 16.9 / 1.8); Q4 row indicative with footnote.",
    "e5vstat_4_reference_systematic": "ACCEPTED: 79-81% band quoted next to the headline.",
    "e5vstat_5_late_window_understated": "ACCEPTED: half-plane numbers (89-95% vs 6-33%) used, with the identity estimate as cross-check.",
    "e5vstat_6_step_null": "ACCEPTED: T labelled conservative lower bound; sensitivity 225-775 ns with partial 0.1-0.3 at 100-150 ns; efficiency-corrected 10-27% (|1>) / 19-51% (|0>). Q3 excitation T kept at 1.7-1.75 (e5vrepro's own detector passes its errors-only null at 1.09 and gives 1.72), with e5vstat's 115-vs-46 excess noted.",
    "e5vstat_7_labels": "ACCEPTED: caption 'errors on shots that stayed in their state (SNR and model boundary; upper bound)'; T+S includes sub-threshold leakage (Q3 >= 75.3% with k=1).",
    "e5vrepro_1_T_detector_dependent": "ACCEPTED: T reported only as a bracket across two detectors (2-8% of errors); not in the table.",
    "e5vrepro_2_FP_oversubtraction": "ACCEPTED (both verifiers measured it: 0.44 and 0.34-0.58): T called conservative; bracket upper end = empirical-null value.",
    "e5vrepro_3_Q3_relax_cause": "ACCEPTED: cause = neighbour-driven 1->0->1 excursions (obs/exp 1.0 with Q1,Q4,Q5 in |0>, up to 5.4); mechanism INFERRED.",
    "e5vrepro_4_all5_pool": "ACCEPTED (same as e5vstat #2).",
    "e5vrepro_5_core_size": "ACCEPTED: half-plane definition stated in the appendix.",
    "e5vrepro_6_Q3_11.5": "ACCEPTED: 11.5% (11.3-11.6 by frame/IF)."}

# --------------------------------------------------------------------------------------
# I. Extra derived numbers used in the paper
# --------------------------------------------------------------------------------------
q3w = OUT["leakage"]["Q3_prep1_by_window_pct"]
rr, ci = ratio_ci(q3w["1us"]["value"], q3w["1us"]["ci95"], q3w["400ns"]["value"], q3w["400ns"]["ci95"])
OUT["leakage"]["Q3_growth_1us_over_400ns"] = entry(rr, ci, "x", "MEASURED", SRC_E4, "leakage_E3_by_window Q3 prep1 1us/400ns",
                                                   paper="leakage, by contrast, triples on Q3")
pq3 = VC["per_qubit"]["Q3"]["models"]["1us|cnn_canonical"]["prep1"]
nl, vnl, le, vle = pq3["leaked_shots3"], pq3["leaked_shots3_var"], pq3["L3"], pq3["L3_var"]
frac = 1 - le / nl
sf = math.sqrt(vle / nl ** 2 + le ** 2 * vnl / nl ** 4)
OUT["leakage"]["Q3_leaked_prep1_read_as_1_pct"] = entry(100 * frac, [100 * (frac - Z * sf), 100 * (frac + Z * sf)],
                                                        "% of leaked prep-|1> Q3 shots", "MEASURED", SRC_VC,
                                                        "per_qubit.Q3.models.1us|cnn_canonical.prep1 (1 - L3/leaked_shots3)",
                                                        paper="99.3% of leaked prepared-|1> Q3 shots are read as |1>",
                                                        note="LEAKAGE_REPORT 3.5 (f frame): 99.35%")
OUT["order_swap_Q3_L_to_T_pp"] = {"value": round(3.14 - 1.70, 2), "L_orderA": 3.14, "L_orderB": 1.70, "T_orderA": 1.75, "T_orderB": 3.18,
                                  "status": "MEASURED", "source": SRC_E4 + " robustness_notes.order_swap",
                                  "paper_text": "only on Q3 do 1.4 pp of errors (both leaked and step-flagged) move from L to T"}

# --------------------------------------------------------------------------------------
# J. Where every number appears in the paper
# --------------------------------------------------------------------------------------
OUT["paper_text_map"] = [
 {"where": "ml_discriminators.tex, Results", "text": "state change >=76% of MCMit-CNN's Q3 and >=90% of its Q5 errors, vs <=21% and <=9% overlap",
  "numbers": "Q3 SC 76.76 [76.13, 77.40] (written as CI lower edge, rounded down); Q5 SC 90.22 [89.76, 90.67]; Q3 O 20.09 [19.47, 20.72] (CI upper edge, rounded up); Q5 O 8.75 [8.31, 9.19]",
  "keys": "table_error_modes_1us.Q3/Q5"},
 {"where": "Results", "text": "91%/95% of Q3/Q5 prepared-|1> errors on the |0> side (33%/15% of correct shots)",
  "numbers": "Q3 90.97 [90.72, 91.21] vs 32.96 [32.83, 33.09]; Q5 94.66 [94.40, 94.92] vs 15.33 [15.23, 15.42]", "keys": "late_window_half_plane_check_1us"},
 {"where": "Results", "text": "only 2-8% show a step inside the window",
  "numbers": "Q1 4.33-7.76, Q5 5.02-7.36, Q3 (excitation only) 1.68-1.75 % of errors, two detectors (CIs in resolved_step_T)", "keys": "resolved_step_T"},
 {"where": "Results", "text": "At 400 ns overlap rises to 40% (Q3) and 56% (Q5)",
  "numbers": "Q3 40.14 [39.35, 40.93]; Q5 55.90 [55.04, 56.76]", "keys": "results_per_qubit_window_cnn.Q3_400ns/Q5_400ns.overlap_pct_err"},
 {"where": "Results", "text": "11.5% of Q3's and 4.7% of Q5's prepared-|1> shots leak; 97-99% of leaked shots read correctly; Q3 3.5/6.5/11.5% at 400/600/1000 ns",
  "numbers": "Q3 11.46 [11.38, 11.55]; Q5 4.72 [4.66, 4.78]; read correctly Q3 97.0 [96.9, 97.1], Q5 98.7 [98.6, 98.9]; Q3 3.51 [3.46, 3.56] / 6.45 [6.39, 6.52] / 11.46",
  "keys": "leakage.*; table_error_modes_1us.*.leaked_read_correctly_pct"},
 {"where": "Table tab:error_modes", "text": "rows Q1-Q5 and pooled Q1,Q3,Q5",
  "numbers": "see table_error_modes_1us (error rates Wilson; leakage, state change, overlap, leaked shots with CIs; all mode CIs <= +/-0.7 pp)", "keys": "table_error_modes_1us"},
 {"where": "Analysis", "text": "pooled Q1,Q3,Q5: state change 81%, overlap 17% (79-81% and 17-20% across references), leakage 1.8%",
  "numbers": "SC 81.27 [80.90, 81.64], band 78.6-81.4; O 16.90 [16.54, 17.27], band 16.8-19.6; L 1.83 [1.75, 1.91]", "keys": "table_error_modes_1us.Q1Q3Q5"},
 {"where": "Analysis", "text": "only Q2 is SNR-bound (its Gaussian overlap alone predicts 28% error)",
  "numbers": "Phi(-Delta/2) = 28.26% with Delta = 1.151 (TRAIN LDA); CNN error 25.43 [25.35, 25.51]", "keys": "Q2_snr_bound"},
 {"where": "Analysis", "text": "relaxation and excitation, rarely as a mid-window step (resolved_step_T), produce the hardest traces; share 37% at 400 ns, 81% at 1 us; overlap falls 2.5-14x; Q3 4.4->4.5%, Q5 2.1->2.7% of shots; leakage triples on Q3",
  "numbers": "SC share 37.35 [36.90, 37.80] -> 81.27; overlap fall Q3 2.52 [2.43, 2.61], Q5 10.5 [9.97, 11.07], Q1 13.9 [13.5, 14.4]; per-shot SC Q3 4.355 [4.297, 4.413] -> 4.484 [4.447, 4.521], Q5 2.147 [2.105, 2.188] -> 2.671 [2.658, 2.685]; Q3 leakage 1us/400ns 3.27x",
  "keys": "pooled_Q1Q3Q5_window_trend; overlap_per_shot_fall_400ns_to_1us; results_per_qubit_window_cnn; leakage.Q3_growth_1us_over_400ns"},
 {"where": "Analysis", "text": "per-trace HERQULES: 68%/92% of its Q3/Q5 errors also CNN errors; Q3 69% state change; CNN 23% fewer Q3 errors",
  "numbers": "68.23 [67.92, 68.55]; 92.12 [91.83, 92.40]; 69.38 [68.70, 70.06]; 23.1 [22.4, 23.9]", "keys": "herqules_pertrace_1us"},
 {"where": "Takeaway #5", "text": "~80% state change (Q1,Q3,Q5), <=20% overlap, 0.4-6% of shots silently leak",
  "numbers": "SC 81.27 (band 78.6-81.4); O <=16.9 (band <=19.6); leaked shots Q1 0.41 [0.39, 0.42] .. Q3 6.14 [6.10, 6.19]", "keys": "table_error_modes_1us"},
 {"where": "Fig. iq_taxonomy", "text": "legend 'leakage' (was 'leakage |2>'); caption: off-axis, non-computational, not necessarily |2>",
  "numbers": "none", "keys": "plots/iq_error_gallery_q5_v2.pdf (sims/leakage/em_e6edit_iq_gallery.py)"},
 {"where": "discussion.tex", "text": "(mostly qubit state changes and SNR overlap) replaces (~99.8% of residual errors are device-set)", "numbers": "qualitative", "keys": "-"},
 {"where": "system_bottleneck.tex RQ13 hypothesis", "text": "'at short readouts' added", "numbers": "supported by 400 ns overlap: pooled Q1,Q3,Q5 62.4 [62.0, 62.9]%, Q1 83.5, Q5 55.9, Q3 40.1",
  "keys": "windows.400ns_cnn_soft400.pooled.Q1Q3Q5 (em_e4recon_final.json)"},
 {"where": "appendix_herqules.tex, App. error-modes", "text": "procedure + validation + leakage + models",
  "numbers": "order swap: O,S <=0.03 pp, Q3 1.4 pp L->T; 0.5 sigma; 79-81%; 89-95% vs 6-33%, implied 89% vs attributed 90% (89.13 vs 90.29); T conservative 24 vs 54.1 flags; two detectors 4-8% (Q1/Q5), 1.7% (Q3); Q3 relax null obs/exp 1.035/0.958 (partners |0>) up to 5.39; sensitivity 225-775 ns, efficiency 16-28%; leakage Q3 11.46 / 0.82 (prep0), Q5 4.72, Q2 1.62, Q1 0.82; 13.4% of Q3 prep-|0> errors; 3.1% of Q3 errors; 99.32% read as |1>",
  "keys": "appendix_facts; resolved_step_T; leakage; late_window_half_plane_check_1us; order_swap_Q3_L_to_T_pp"},
]

json.dump(OUT, open(os.path.join(DL, "em_e6edit_final_numbers.json"), "w"), indent=1)
print("wrote", os.path.join(DL, "em_e6edit_final_numbers.json"))
print(json.dumps(OUT["leakage"]["Q3_growth_1us_over_400ns"]), json.dumps(OUT["leakage"]["Q3_leaked_prep1_read_as_1_pct"]))
