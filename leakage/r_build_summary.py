"""r_build_summary.py -- assemble data/leakage/r_leakage_summary.json (tag r).

Reconciled per-qubit leakage table for the Lienhard 5-qubit readout dataset. Best estimates are the mean of the
two independent E3 implementations (m1tail_v2 and r_spotcheck), CI = envelope of their statistical 95 % CIs;
method ranges span m2mix mixture weights and m3time HMM occupancies.  Reads the r_spotcheck / m1tail JSONs for the
E3 values so the numbers trace to their sources; reconciled judgements are written explicitly.

Usage: python r_build_summary.py /home/manosgior/Documents/GitHub/oraqle-ics/data/leakage
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else str(LP.RES)
R = json.load(open(os.path.join(D, "r_spotcheck.json")))
M1 = json.load(open(os.path.join(D, "m1tail_summary.json")))
PAPER = ["Q1", "Q2", "Q3", "Q4", "Q5"]


def pct(x):
    return None if x is None else round(100 * x, 4)


def combo(q, b, w="1us"):
    """mean of m1 and r E3, CI envelope (in %)."""
    Q = PAPER[q]
    r = R["A"][Q][w][f"prep{b}"]["E3"]
    m = M1["qubits"][Q]["windows"][w][f"prep{b}"]
    me, mlo, mhi = m["E3"], m["E3_ci95"][0], m["E3_ci95"][1]
    return {"best_pct": pct((r[0] + me) / 2), "ci95_pct": [pct(min(r[1], mlo)), pct(max(r[2], mhi))],
            "r_E3_pct": [pct(v) for v in r], "m1_E3_pct": [pct(me), pct(mlo), pct(mhi)]}


def overall(q, w="1us"):
    a, b = combo(q, 0, w), combo(q, 1, w)
    return {"best_pct": round((a["best_pct"] + b["best_pct"]) / 2, 4),
            "ci95_pct": [round((a["ci95_pct"][0] + b["ci95_pct"][0]) / 2, 4),
                         round((a["ci95_pct"][1] + b["ci95_pct"][1]) / 2, 4)]}


def rE3(q, w, b):
    v = R["A"][PAPER[q]][w][f"prep{b}"]["E3"]
    return [pct(x) for x in v]


def cnn(q):
    c = R["C"][PAPER[q]]
    out = {"cnn_error_rate_pct": [pct(v) for v in c["cnn_error_rate"]], "n_err": c["n_err"]}
    for b in ("prep0", "prep1", "prepall"):
        p = c[b]
        out[b] = {"N": p["N"], "n_err": p["n_err"], "err_rate_pct": [pct(v) for v in p["err_rate"]],
                  "leaked_share_of_errors_mirror3_pct": [pct(v) for v in p["share_of_errors_mirror3"]],
                  "leaked_share_of_errors_hmm_everL_pct": [pct(v) for v in p["hmm_everL_bins0_9"]["share_of_errors"]],
                  "cnn_err_on_z_gt3_pct": [pct(v) if v is not None else None for v in p["z_gt3"]["cnn_err_rate_on_flag"]],
                  "cnn_err_on_z_gt5_pct": [pct(v) if v is not None else None for v in p["z_gt5"]["cnn_err_rate_on_flag"]],
                  "cnn_err_on_hmm_everL_pct": [pct(v) if v is not None else None
                                               for v in p["hmm_everL_bins0_9"]["cnn_err_rate_on_flag"]]}
    return out


WINDOW = "paper '1 us' = raw samples 0-499 at 500 MS/s (feat1us bins 0-9, strictly causal FIR = CNN input)"
DEN = "TEST split: 560,000 shots per prepared bit per qubit (1,120,000 per qubit); train replicates within CI"

per_qubit = []
# ---------------------------------------------------------------- Q1
per_qubit.append({
    "qubit_paper": "Q1", "qubit_code": "q0", "window": WINDOW, "denominators": DEN,
    "verdict": "third-state-like population present on prepared |1>, mostly paired with Q3 (exchange); own part small",
    "evidence_level": "moderate (MEASURED one-sided excess; |2>/exchange identity INFERRED)",
    "prep0": {**combo(0, 0), "reported_as": "consistent with 0; 95% upper bound 0.017 %",
              "method_values_pct": {"m1_E3": 0.003, "r_E3": -0.000, "m2_M1f_G2": 0.0, "m4_BASE_5sigma_trainplustest": 0.0}},
    "prep1": {**combo(0, 1), "method_range_pct": [0.43, 0.77],
              "method_values_pct": {"m1_E3": 0.614, "r_E3": 0.768, "m2_M1f_G2": 0.429, "m2_ci": [0.186, 0.749],
                                    "m4_BASE_5sigma_floor": 0.0174,
                                    "m3_startL_unreliable": 4.77, "m3_null_false_positive_startL": 1.19}},
    "overall": overall(0),
    "conditional_prep1_1us_pct": {"Q3_prep0": [pct(v) for v in R["A"]["Q1"]["1us"]["prep1"]["E3_given_Q3prep0"]],
                                  "Q3_prep1": [pct(v) for v in R["A"]["Q1"]["1us"]["prep1"]["E3_given_Q3prep1"]],
                                  "m1_per_config_range_Q3_0": [0.08, 0.16], "m1_per_config_range_Q3_1": [0.98, 1.26]},
    "E3_by_window_r_pct": {w: {"prep0": rE3(0, w, 0), "prep1": rE3(0, w, 1)} for w in ("200ns", "400ns", "1us", "2us")},
    "dynamics": "HMM L 1.46 sigma/bin from |1> (unreliable rates); L entries coincide with Q3 |1>->|0> (lift 9.4), "
                "L->|1> with Q3 |0>->|1> (18.5); Q3 decay rate 2.07 -> 4.33 %/100 ns when Q1 prepared |1>",
    "location_sep": "(1.44-1.60 along, +0.5-0.9 perp) separations from c0, beyond |1>",
    "cnn": cnn(0), "paper_table_leakage_share_of_errors_pct": 0.0,
})
# ---------------------------------------------------------------- Q2
per_qubit.append({
    "qubit_paper": "Q2", "qubit_code": "q1", "window": WINDOW, "denominators": DEN,
    "verdict": "probable |2>-like off-axis population (sourced from |1>, decays to |1>); moderate confidence (low SNR, partly correlated with Q3)",
    "evidence_level": "moderate-low",
    "prep0": {**combo(1, 0), "method_range_pct": [0.047, 0.34],
              "method_values_pct": {"m1_E3": 0.168, "r_E3": 0.166, "m2": 0.110, "m3_everL_1us": 0.342,
                                    "m3_null_everL": 0.031, "m4_BASE_5sigma_floor": 0.0468}},
    "prep1": {**combo(1, 1), "method_range_pct": [0.46, 2.80],
              "method_values_pct": {"m1_E3": 1.539, "r_E3": 1.518, "m2": 1.503, "m2_ci": [1.401, 1.626],
                                    "m2_model_range": [0.46, 2.45], "m3_everL_1us": 2.797, "m3_L_at_bin9": 2.092,
                                    "m3_startL": 0.280, "m3_null_startL": 0.021, "m4_BASE_5sigma_floor": 0.661}},
    "overall": overall(1),
    "conditional_prep1_1us_pct": {"Q3_prep0": [pct(v) for v in R["A"]["Q2"]["1us"]["prep1"]["E3_given_Q3prep0"]],
                                  "Q3_prep1": [pct(v) for v in R["A"]["Q2"]["1us"]["prep1"]["E3_given_Q3prep1"]],
                                  "all_other_bits_0": [pct(v) for v in R["A"]["Q2"]["1us"]["prep1"]["E3_given_all_other_bits_0"]]},
    "E3_by_window_r_pct": {w: {"prep0": rE3(1, w, 0), "prep1": rE3(1, w, 1)} for w in ("200ns", "400ns", "1us", "2us")},
    "dynamics": "L decays to |1> (HMM 99.9 %, L-starters 90.3 % [88.3, 92.0]); tau_L 0.71-0.78 us vs Q2 |1>->|0> under readout "
                "0.64-1.04 us (ratio not diagnostic); 30 % of prep1 / 65 % of prep0 excess in shots where Q3 is off-axis (m1)",
    "location_sep": "plume at about (+2.5, +2.9) ADC from |0> (separation units inflated by the tiny separation)",
    "cnn": cnn(1), "paper_table_leakage_share_of_errors_pct": 0.2,
})
# ---------------------------------------------------------------- Q3
per_qubit.append({
    "qubit_paper": "Q3", "qubit_code": "q2", "window": WINDOW, "denominators": DEN,
    "verdict": "LEAKAGE PRESENT: distinct third state with |2> dynamics, mostly created during the readout",
    "evidence_level": "strong (all four methods + r spot checks; identity as |2> INFERRED from dynamics)",
    "prep0": {**combo(2, 0), "method_range_pct": [0.33, 1.25],
              "method_values_pct": {"m1_E3": 0.811, "r_E3": 0.649, "m2_M2f_far": 0.331, "m3_everL_1us": 1.248,
                                    "m3_L_at_bin9": 1.117, "m3_startL": 0.054, "m3_null_startL": 0.0,
                                    "m4_BASE_5sigma_floor": 0.382}},
    "prep1": {**combo(2, 1), "method_range_pct": [8.94, 14.53],
              "method_values_pct": {"m1_E3": 11.217, "r_E3": 11.163, "m2_M2f_far": 8.937, "m2_ci": [8.796, 9.041],
                                    "m2_model_range": [8.9, 14.3], "m3_everL_1us": 14.530, "m3_everL_ci": [14.438, 14.623],
                                    "m3_L_at_bin9": 12.593, "m3_startL_bins1_2": 1.971, "m3_hmm_P_L_bin0": 0.605,
                                    "m3_null_startL": 0.005, "m4_BASE_5sigma_floor": 4.663,
                                    "m4_candidates_surviving_screens": "95.7 %",
                                    "r_threshold_curve_E2_E3_E4_E5_E6_m4style": [15.55, 11.16, 9.14, 7.54, 6.07, 4.87]}},
    "overall": overall(2),
    "overall_method_range_pct": [4.63, 7.89],
    "E3_by_window_r_pct": {w: {"prep0": rE3(2, w, 0), "prep1": rE3(2, w, 1)} for w in ("200ns", "400ns", "1us", "2us")},
    "m1_E3_0_500ns_prep1_pct": 5.05, "lienhard_tabV_P2_pi_0p5us_MF": 0.052,
    "present_at_readout_start_prep1_pct": {"hmm_P_L_bin0": [0.605, 0.589, 0.622], "viterbi_L_bins1_2": [1.971, 1.934, 2.007]},
    "present_at_readout_start_prep0_pct": {"hmm_P_L_bin0": [0.0097, 0.0076, 0.0118], "viterbi_L_bins1_2": [0.054, 0.048, 0.060]},
    "dynamics": {
        "L_exit_branch_to_1": "HMM 99.91 % [99.90, 99.93]; L-starters 96.4 % [95.8, 97.0]; |0>-origin L episodes 96.8 % [96.2, 97.4] (r_spot2); model-light |1>:|0> destination 3:1 for both prepared bits",
        "tau_L_us": "1.79 [1.77, 1.80] (HMM), 1.63 (no FIR), 1.81 [1.75, 1.87] (survival)",
        "tau_1_to_0_under_readout_us": "2.47 [2.46, 2.48]; 4.78 [4.73, 4.84] with Q1 and Q5 prepared |0>",
        "tau_L_over_tau_10": "0.72 (0.37 vs exchange-free)",
        "tau_1_to_L_us": "3.79 (about 2.6 % per 100 ns)",
        "entry_prob_bins3_9_by_start_state": "start |1>: 13.0 % (prep1), 15.8 % (prep0); start |0>: 1.34 % (prep1), 1.05 % (prep0)",
    },
    "location_sep": "(1.73-1.90 along, +1.05-1.51 perp) separations from c0; same for both prepared bits and all 16 neighbour configurations",
    "cnn": cnn(2), "paper_table_leakage_share_of_errors_pct": 0.1,
})
# ---------------------------------------------------------------- Q4
per_qubit.append({
    "qubit_paper": "Q4", "qubit_code": "q3", "window": WINDOW, "denominators": DEN,
    "verdict": "INCONCLUSIVE: multimodal |1> response; no method isolates a |2> population",
    "evidence_level": "inconclusive",
    "prep0": {"best_pct": None, "ci95_pct": None, "reported_as": "not measurable"},
    "prep1": {"best_pct": None, "ci95_pct": None, "reported_as": "not measurable"},
    "overall": {"best_pct": None, "ci95_pct": None},
    "descriptive_measurements": {
        "sub_cloud_between_cores_m2_pct": {"prep1_1us": 20.80, "prep0_1us": 0.30, "prep1_2us": 14.35,
                                           "location_sep": [0.41, -0.22]},
        "main_cloud_in_prep0_m2_pct": 1.55,
        "ratio_subcloud_to_main_prep0_vs_prep1": [0.19, 0.30],
        "r_gmm_segment_crosstab_prep1": "sub-cloud (0.5-1 us) -> (1.2-2 us): stays 72 %, -> main 20 %, -> |0> 8 %; the two halves of the broad main cloud exchange 24-34 % (this is m3's |1><->L telegraph)",
        "m4_offaxis_5sigma_prep1": "2 / 800,000 (Gaussian expectation 0.46)",
        "m4_shot_to_shot_memory": "sub-cloud lag-1 autocorrelation 0.15, over-dispersed in 500-shot bins",
        "mirror_E3_prep1_1us_pct": {"m1": 2.511, "r": rE3(3, "1us", 1)[0], "note": "frame dependent, not interpretable"},
    },
    "lienhard_tabV_P2_pi": "0.734: most likely the split |1> response, not a |2> population",
    "cnn": cnn(3), "paper_table_leakage_share_of_errors_pct": 0.04,
})
# ---------------------------------------------------------------- Q5
per_qubit.append({
    "qubit_paper": "Q5", "qubit_code": "q4", "window": WINDOW, "denominators": DEN,
    "verdict": "|2>-like population of Q5 created mainly by a two-qubit exchange with Q3 (|1_Q3 1_Q5> -> |0_Q3 2_Q5>), weakly Q4-dependent; own baseline 0.3-0.5 % of prepared |1>",
    "evidence_level": "moderate-strong for the population and its Q3 pairing (MEASURED); mechanism INFERRED",
    "prep0": {**combo(4, 0), "method_range_pct": [0.037, 0.79],
              "method_values_pct": {"m1_E3": 0.106, "r_E3": 0.101, "m2_M2f_far": 0.037, "m3_everL_1us": 0.785,
                                    "m3_null_everL": 0.0625, "m4_BASE_5sigma_floor": 0.0086}},
    "prep1": {**combo(4, 1), "method_range_pct": [3.09, 13.26],
              "method_values_pct": {"m1_E3": 4.398, "r_E3": 4.538, "m2_M2f_far": 3.090, "m2_ci": [2.728, 3.368],
                                    "m2_model_range": [3.1, 8.5], "m3_everL_1us": 13.263, "m3_null_everL": 1.165,
                                    "m3_L_at_bin9": 11.252, "m3_startL": 3.331, "m3_null_startL": 0.359,
                                    "m4_BASE_5sigma_floor": 0.146,
                                    "r_threshold_curve_E2_E3_E4_E5_E6_m4style": [6.14, 4.54, 2.27, 0.72, 0.13, 0.21]}},
    "overall": overall(4),
    "conditional_prep1_1us_pct": {
        "Q3_prep0": [pct(v) for v in R["A"]["Q5"]["1us"]["prep1"]["E3_given_Q3prep0"]],
        "Q3_prep1": [pct(v) for v in R["A"]["Q5"]["1us"]["prep1"]["E3_given_Q3prep1"]],
        "m1_per_config": {"Q3=0,Q4=0": [0.33, 0.49], "Q3=1,Q4=0": [8.56, 8.89], "Q3=0,Q4=1": [1.35, 1.66], "Q3=1,Q4=1": [6.80, 7.11]},
        "m3_startL_Q3=0_Q4=0": [0.394, 0.362, 0.429], "m3_null_startL": 0.359},
    "E3_by_window_r_pct": {w: {"prep0": rE3(4, w, 0), "prep1": rE3(4, w, 1)} for w in ("200ns", "400ns", "1us", "2us")},
    "dynamics": "Q5 |1>->L coincides with Q3 |1>->|0> (lift 10.3 at lag 0), Q5 L->|1> with Q3 |0>->|1> (lift 20.4); Q3 decay 2.07 -> 3.92 %/100 ns when Q5 prepared |1>; Q3 flip precedes Q5 perp rise by about 200 ns (m4); L exits to |1> 87-94 % (HMM), model-light 13-14 % to |1> vs 1.4-2.0 % to |0>; Lienhard frequencies: Q3 f01 5.000 GHz vs Q5 f12 4.965 GHz (35 MHz)",
    "location_sep": "(1.02-1.05 along, +0.48-0.65 perp) separations: directly above |1>",
    "cnn": cnn(4), "paper_table_leakage_share_of_errors_pct": 0.0,
})

existing = [
    {"source": "oraqle-ics/chapters/ml_discriminators.tex Table tab:error_modes (HybridCNN, from herqules_error_breakdown_hybridcnn.csv)",
     "value": "Leakage share of errors: Q1 0.0, Q2 0.2, Q3 0.1, Q4 0.04, Q5 0.0 %; pooled 0.2 %",
     "assessment": "inconsistent (Q3, Q5) / not a leakage measurement",
     "reason": "misclassified shots with |z(time-mean |IQ|)|>3 only; radial test misses the off-axis state and leaked shots read as correct are never counted; measured leaked share of canonical MCMit-CNN errors: Q1 0.14 %, Q2 0.32 %, Q3 4.06 % [3.90, 4.22], Q5 1.16 % [1.02, 1.30] (HMM 3.9 %); model is HybridCNN, not MCMit-CNN"},
    {"source": "same, converted to share of shots (s1local)",
     "value": "Q1 0, Q2 0.060 %, Q3 0.0081 %, Q4 0.0023 %, Q5 0; pooled 0.014 %",
     "assessment": "inconsistent", "reason": "measured leaked share of shots at 1 us: Q1 0.35 %, Q2 0.85 %, Q3 5.96 %, Q5 2.29 %"},
    {"source": "paper text 'Excitation errors stay at 1e-3--1e-2 % of shots'", "value": "1e-3-1e-2 %",
     "assessment": "mislabelled and inconsistent", "reason": "same amplitude-outlier metric"},
    {"source": "three_level_labels.hdf5 (build_three_level_labels.py, perp > 3 sigma_pooled)",
     "value": "overall 0.58/1.07/5.08/1.00/1.92 %; prep1 0.93/1.77/9.43/1.94/3.65 %; prep0 0.23/0.38/0.74/0.06/0.19 %",
     "assessment": "circular calibration; roughly consistent in magnitude for Q2/Q3/Q5 prepared |1>, inconsistent for Q1, Q4 and prepared |0>",
     "reason": "k tuned to a target rate (and the target is not met); Q1 prep0 labels are pure Gaussian tail (0 % survive), Q4 prep1 labels are its broad core (0.013 % survive); per-shot Jaccard with screened candidates 0.008-0.52; perp-only rule misses along-axis displacement"},
    {"source": "Mude et al. 2025, arXiv 2405.08982", "value": "487 (Q1) to 17,642 (Q4) of 1.6 M = 0.030-1.10 %",
     "assessment": "not comparable / inconsistent", "reason": "smallest spectral cluster is method dependent; the largest measured population here is Q3 (5.96 %), Q4 is not measurable, Q1 is 0.35 % overall"},
    {"source": "herqules_error_modes_*.csv, herqules_error_breakdown_{hybridcnn,full}.csv",
     "value": "amp outliers among errors at 1 us: HybridCNN Q2 673, Q3 91, Q4 26; HERQULES Q2 641, Q3 162, Q4 18; full-data Q2 537, Q3 137",
     "assessment": "not a leakage measurement; inconsistent", "reason": "same radial rule; measured leaked shots among canonical CNN errors for Q3 about 2,660 of 65,425"},
    {"source": "synth_transfer_sweep_leakage_F0.98_*.csv", "value": "gmean 0.978->0.953 (HERQULES), 0.971->0.942 (CNN) for p=0-0.2",
     "assessment": "not comparable", "reason": "synthetic |2> on the MF boundary labelled 1; real third states (Q1, Q3, Q5) lie on/beyond the |1> side and cost little 2-level accuracy"},
    {"source": "HERQULES preclassifier 'ket2' (preclassifier_state.pkl)", "value": "41-88 % of |1>-prepared",
     "assessment": "not comparable (geometric artefact)", "reason": "tangent purity circles collect the Gaussian tail"},
    {"source": "iq_error_gallery_q5 (paper Fig. iq_taxonomy)", "value": ">=2500 shots; 1.20 % of a 4000-shot |1> subsample",
     "assessment": "consistent", "reason": "stricter subset of the measured Q5 population (4.47 %); contradicts the table's Q5 0.0 %"},
    {"source": "s2cluster Gaussian-null excess (prep1, k=3)", "value": "Q1 0.60, Q2 1.44, Q3 8.78, Q4 -0.72, Q5 3.14 %",
     "assessment": "consistent", "reason": "same one-sided idea, HERQULES frame"},
    {"source": "Lienhard et al. Tab. V P(2|pi)", "value": "Q1 0.001, Q2 0.019, Q3 0.052, Q4 0.734, Q5 <<0.001",
     "assessment": "Q3 consistent (m1 0-500 ns: 5.05 %); Q2 roughly consistent; Q1/Q5 not comparable (1-D MF blind to off-axis states); Q4 not a |2> population",
     "reason": "per-qubit MF boxcar (Q3 0.5 us)"},
    {"source": "s3lit thermal prior", "value": "<= about 2e-3 per |1>-prepared shot",
     "assessment": "inconsistent for Q3", "reason": "Q3 leakage is readout-induced (|1>->L about 2.6 %/100 ns), within the literature readout-induced range 0.1-8 %"},
]

out = {
    "tag": "r", "date": "2026-11-04",
    "question": "does the readout trace dataset contain actual leakage (|2>) errors, and how much?",
    "headline": ("Yes, qubit-dependent. Q3 (q2) has a genuine |2>-like third state: 11.2 % [11.1, 11.3] of prepared-|1> and "
                 "0.73 % [0.62, 0.84] of prepared-|0> shots at the paper's 1 us window (6.0 % of all Q3 shots), decaying to |1> "
                 "with tau 1.8 us < T1 and mostly created during the readout. Q5 (4.5 % / 0.10 %) and Q1 (0.69 % / <=0.02 %) carry "
                 "|2>-like populations produced mainly by exchange with Q3; Q2 1.5 % / 0.17 % (probable); Q4 inconclusive. "
                 "Most leaked shots are read as |1>, so leakage is a hidden channel beside, not inside, the 2-level error budget; "
                 "the paper's 0.0-0.2 %-of-errors column is not a leakage measurement."),
    "leakage_present": "qubit-dependent",
    "definitions": {
        "window": WINDOW, "denominators": DEN,
        "primary_statistic": "E3 = [N(z>3) - N(z<-3)]/N, z = (perp - centre of prepared 5-bit state)/(perp noise width of the prepared-|0> state with the same neighbours); one-sided mirror excess, MEASURED, lower bound on 'visited the third state within the window'",
        "best_estimate": "mean of m1tail_v2 and r_spotcheck E3; ci95 = envelope of their statistical 95 % CIs (includes the frame systematic)",
        "method_range": "span of m2mix far-component weights and m3time HMM occupancies (different definitions), not a CI",
        "overall": "(prep0 + prep1)/2 = share of all shots (equal counts)",
        "measured_vs_inferred": "existence, size, location, timing and correlations MEASURED; identification as |2> and the exchange mechanism INFERRED",
    },
    "per_qubit": per_qubit,
    "error_budget": {
        "cnn_source": R["C"]["source"],
        "pooled_Q1_Q2_Q3_Q5_leaked_share_of_errors_pct": [0.960, 0.926, 0.994],
        "pooled_Q1_Q3_Q5_leaked_share_of_errors_pct": 2.34,
        "statement": "Q3 leaked share of shots (5.96 %) equals its CNN error rate (5.84 %); Q5 2.29 % vs 2.96 %; Q1 0.35 % vs 2.98 %; Q2 0.85 % vs 25.43 %. About 98 % of leaked prepared-|1> shots are read as |1>.",
    },
    "existing_numbers": existing,
    "spot_checks": {
        "r_spotcheck.json": "A: unified E3 at 200/400 ns, 1/2 us, m4-style reproduction (Q3 prep1 4.869 % vs m4 4.872 %); B: GMM segment cross-tabs (Q4 telegraph = halves of broad |1> cloud); C: canonical MCMit-CNN errors vs leakage flags",
        "r_spot2.json": "third-state exit destination and origin (Q3 |0>-origin L exits to |1> 96.8 %)",
        "cluster_dir": "graham:~/oraqle-ics-sims/leakage/r/",
    },
}
with open(os.path.join(D, "r_leakage_summary.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote", os.path.join(D, "r_leakage_summary.json"))
for p in per_qubit:
    print(p["qubit_paper"], p["prep0"].get("best_pct"), p["prep0"].get("ci95_pct"), p["prep1"].get("best_pct"),
          p["prep1"].get("ci95_pct"), p["overall"])
