#!/usr/bin/env python3
"""em_e1hmm_summarize.py (tag e1hmm): assemble the compact results JSON and a markdown table digest from
  full_v6/em_e1hmm_v6_results.json   (em_e1hmm_attrib_v6.py; primary featbox / lab_p50 / null Ac)
  oldproc/em_e1hmm_oldproc.json      (em_e1hmm_oldproc.py; exact replica of the old paper procedure)
Usage: python em_e1hmm_summarize.py <v6_results.json> <oldproc.json> <out_results.json> <out_tables.md>
"""
import json
import sys

V6, OLD, OUTJ, OUTMD = sys.argv[1:5]
R = json.load(open(V6))
OP = json.load(open(OLD))
NAMES = ["Q1", "Q2", "Q3", "Q4", "Q5"]
PRIM_KIND, PRIM_VAR = "featbox", "lab_p50"
WINS = ["1us", "600ns", "400ns"]
OLDLEN = {"1us": "500", "600ns": "300", "400ns": "200"}


def job(kind, q):
    return next((r for r in R["per_qubit"] if r["kind"] == kind and r["q"] == q), None)


def pct(d):
    if isinstance(d, str) or d is None:
        return d
    return {"pct_of_errors": 100 * d["share"], "ci95": [100 * x for x in d["share_ci95"]],
            "pct_of_shots": 100 * d["rate_of_shots"], "ci95_shots": [100 * x for x in d["rate_of_shots_ci95"]],
            "count": d["count"]}


def leak(a):
    if a["leakage_share_mirror_ci95_analytic"] == "n/m":
        return "n/m (Q4: multimodal |1> response; line-frame flag not interpretable)"
    ne, ns = a["n_err"], a["n_shots"]
    d = a["order_L_S"]["leakage"]
    ci = a["leakage_share_mirror_ci95_analytic"]
    return {"pct_of_errors": 100 * d["share"], "ci95_mirror_analytic": [100 * x for x in ci],
            "ci95_bootstrap": [100 * x for x in d["share_ci95"]], "pct_of_shots": 100 * d["rate_of_shots"],
            "ci95_shots_mirror_analytic": [100 * x * ne / ns for x in ci], "count": d["count"]}


def block(E, var, x="all"):
    a = E["variants"][var][x]
    S, S2, o = a["order_L_S"], a["order_S_L"], a["order_L_T_W"]
    return {
        "n_err": a["n_err"], "n_shots": a["n_shots"],
        "leakage": leak(a),
        "state_change_detected": pct(S["state_change_detected"]),
        "of_which_in_window_transition": pct(S["of_which_in_window_transition"]),
        "of_which_wrong_from_start_or_early": pct(S["of_which_wrong_from_start_or_early"]),
        "no_detected_event": pct(S["no_detected_event"]),
        "order_swap_S_then_L": {k: pct(v) if not isinstance(v, str) else v for k, v in S2.items()},
        "overlap_leak_and_transition_call": pct(a["overlap_leak_and_transcall"])
        if not isinstance(a["overlap_leak_and_transcall"], str) else "n/m",
        "null_inflated_Aci": {"state_change_detected": pct(a["state_change_detected_nullAci"]),
                              "in_window_transition": pct(a["transition_nullAci"]),
                              "no_detected_event": pct(a["no_detected_event_nullAci"])},
        "raw_calls_nonleaked": {"transition": pct(a["transition_raw_calls_nonleaked"]),
                                "wrong": pct(a["wrong_raw_calls_nonleaked"])},
        "deconvolved_composition_INFERRED": {
            "overlap": pct(a["deconv_N_overlap"]), "wrong_throughout": pct(a["deconv_N_wrong_throughout"]),
            "transition_sensitivity_corrected": pct(a["deconv_N_transition_sens_corrected"]),
            "state_change_total": pct(a["deconv_N_state_change_total_INFERRED"])},
        "v5_style_split_NOT_ROBUST": {"wrong_from_start_W_det": pct(o["wrong_from_start"]),
                                      "residual_after_T_and_W": pct(o["no_detected_event"])},
        "deconv_chi2": a["deconv_chi2"], "deconv_chi2_Aci": a["deconv_chi2_nullAci"], "deconv_dof": a["deconv_dof"]}


def near(E):
    nb = E["near_boundary"]
    out = {}
    for bnd in ("cnn_surrogate_boundary", "optimal_lda_boundary"):
        out[bnd] = {g: {"weighted_n": d["weighted_n"], "frac_within_1sigma": d["frac_within_1sigma"],
                        "frac_within_2sigma": d["frac_within_2sigma"],
                        "frac_beyond_3sigma_wrong_side": d["frac_beyond_3sigma_wrong_side"],
                        "frac_boundary_correct_side": d["frac_boundary_correct_side"],
                        "median_sigma": d["quantiles"]["q50"], "hist": d["hist"], "hist_edges": d["hist_edges"]}
                    for g, d in nb[bnd].items() if d is not None}
    for k in ("gaussian_overlap_prediction_cnn_surrogate", "gaussian_overlap_prediction_optimal_lda"):
        out[k] = nb[k]
    out["observed_nonleaked_errors_mirror"] = nb["observed_nonleaked_errors_mirror"]
    return out


def inj_summary(rows, v):
    p = [r[f"P_call_{v}"][0] for r in rows]
    t = [r["tau_ns"] for r in rows]
    above = [i for i, x in enumerate(p) if x >= 0.5]
    return {"tau_first_P_ge_0.5_ns": t[above[0]] if above else None,
            "tau_last_P_ge_0.5_ns": t[above[-1]] if above else None,
            "P_at_tau0_wrong_throughout": p[0], "P_max": max(p),
            "curve": [[ti, round(pi, 4)] for ti, pi in zip(t, p)]}


res = {"tag": "e1hmm", "date": "2026-11-05", "method_version": "em_e1hmm_attrib_v6.py (supersedes v3 full_v3, v5)",
       "primary": {"features": "featbox (boxcar 100 ns bins, no look-ahead)", "detector_variant": PRIM_VAR,
                   "null": "Ac: per neighbour configuration and state, no transitions, white tied HMM noise; errors "
                           "selected by a per-config linear surrogate of the CNN",
                   "attribution_order": "leakage -> detected state change -> no detected event (swap reported)"},
       "conventions": {"qubits": "paper Q1..Q5 = code q0..q4",
                       "windows": {"1us": "samples 0-499 = featbox bins 0-9", "600ns": "samples 0-299 = bins 0-5",
                                   "400ns": "samples 0-199 = bins 0-3"},
                       "shares": "percent of that qubit's CNN errors; pct_of_shots: percent of its 1,120,000 TEST shots",
                       "CIs": "state change / transition / residual: 95% bootstrap (Poisson weights over errors + "
                              "multinomial resampling of the null call matrices); leakage: analytic mirror "
                              "+/-1.96 sqrt(N+ + N-)/N; error rates: Wilson. Statistical only: see deconv_chi2 and the "
                              "variant spread for systematics"},
       "meta": R["meta_shared_v3"], "per_qubit": {}, "pooled": {}, "detector": {}, "old_procedure": {}}

for q in range(5):
    P, F = job(PRIM_KIND, q), job("feat", q)
    Qd = {"hmm_A_canonical": P["hmm"]["A"], "hmm_pi_prep0": P["hmm"]["pi_prep0"], "hmm_pi_prep1": P["hmm"]["pi_prep1"],
          "cfg_mean_loglik_gain_per_shot": P["cfg_means"]["loglik_gain_per_shot_mean"]}
    for w in WINS:
        W = P["windows"][w]
        Wd = {"surrogate": {k: W["surrogate"][k] for k in ("agreement_heldout_mean", "agreement_heldout_min",
                                                            "surrogate_err_rate", "cnn_err_rate",
                                                            "frac_cnn_errors_also_surrogate_errors",
                                                            "weights_abs_per_bin_cfgmean")},
              "noise_inflation_Aci": W["noise_inflation_Aci"],
              "lda_error_rate_test": W["lda_error_rate_test"],
              "null_fp_per_shot_stationary_Ac": W["null_fp_per_shot_stationary"]["Ac"],
              "trans_calls_all_shots": W["trans_calls_all_shots"], "hidden_leakage": W["hidden_leakage"],
              "errsets": {}}
        for en, E in W["errsets"].items():
            ed = {"err_rate_pct": 100 * E["err_rate"], "err_rate_ci95_pct": [100 * x for x in E["err_rate_ci95"]],
                  "err_rate_prep0_pct": 100 * E["err_rate_prep0"], "err_rate_prep1_pct": 100 * E["err_rate_prep1"],
                  "err_rate_prep0_ci95_pct": [100 * x for x in E["err_rate_prep0_ci95"]],
                  "err_rate_prep1_ci95_pct": [100 * x for x in E["err_rate_prep1_ci95"]],
                  "n_err": E["n_err"], "n_err_prep0": E["n_err_prep0"], "n_err_prep1": E["n_err_prep1"],
                  "primary": {x: block(E, PRIM_VAR, x) for x in ("all", "prep0", "prep1")},
                  "variants": {v: {x: block(E, v, x) for x in ("all", "prep0", "prep1")}
                               for v in ("p50", "viterbi", "fut_p50", "p80")},
                  "posterior_mass_agnostic": {"transition": E["posterior_mass_transition_agnostic"],
                                              "wrong_from_start": E["posterior_mass_wrong_from_start_agnostic"]},
                  "near_boundary": near(E)}
            if F is not None and en in F["windows"][w]["errsets"]:
                ed["features_feat_FIR_lab_p50"] = {x: block(F["windows"][w]["errsets"][en], PRIM_VAR, x)
                                                   for x in ("all", "prep0", "prep1")}
            Wd["errsets"][en] = ed
        Qd[w] = Wd
        res["detector"].setdefault(w, {})[NAMES[q]] = {
            "injection_summary": {f"prep{x}": {v: inj_summary(W["injection"][f"prep{x}"], v)
                                               for v in ("lab_p50", "p50", "fut_p50")} for x in (0, 1)},
            "injection_P_trans_given_surrogate_error": {
                f"prep{x}": [[r["tau_ns"], r["P_surrogate_error"], r["P_trans_lab_p50_given_surrogate_error"],
                              r["P_trans_p50_given_surrogate_error"]] for r in W["injection"][f"prep{x}"]]
                for x in (0, 1)},
            "closure_test": W["closure_test"], "null_fp_per_shot_stationary": W["null_fp_per_shot_stationary"]}
    res["per_qubit"][NAMES[q]] = Qd

for key, p in R["pooled"].items():
    kind, w, en, nm, v = key.split("|")
    if kind != PRIM_KIND:
        continue
    S, S2 = p["order_L_S"], p["order_S_L"]
    res["pooled"].setdefault(w, {}).setdefault(en, {}).setdefault(nm, {})[v] = {
        "qubits": p["qubits"], "n_err": p["n_err"], "n_shots": p["n_shots"], "err_rate_pct": 100 * p["err_rate_pooled"],
        "err_rate_ci95_pct": [100 * x for x in p["err_rate_pooled_ci95"]],
        "leakage": {"pct_of_errors": 100 * S["leakage"]["share"],
                    "ci95_mirror_analytic": [100 * x for x in p["leakage_share_mirror_ci95_analytic"]],
                    "pct_of_shots": 100 * S["leakage"]["rate_of_shots"]},
        "state_change_detected": pct(S["state_change_detected"]),
        "of_which_in_window_transition": pct(S["of_which_in_window_transition"]),
        "of_which_wrong_from_start_or_early": pct(S["of_which_wrong_from_start_or_early"]),
        "no_detected_event": pct(S["no_detected_event"]),
        "order_swap_S_then_L": {k: pct(x) for k, x in S2.items()},
        "state_change_detected_nullAci": pct(p["state_change_detected_nullAci"]),
        "deconvolved_composition_INFERRED": {"overlap": pct(p["deconv_N_overlap"]),
                                             "state_change_total": pct(p["deconv_N_state_change_total_INFERRED"])},
        "note": p["note"]}

L = OP["lengths"]
for w, Lk in OLDLEN.items():
    d = {}
    for src in ("hybridcnn_csv_reference", "hybridcnn_reinferred", "mcmit_canonical_hard", "mcmit_soft"):
        if src not in L[Lk]:
            continue
        d[src] = {}
        for qn in NAMES + ["pooled_all5"]:
            x = L[Lk][src].get(qn)
            if x is None:
                continue
            ne = x["n_error"]
            d[src][qn] = {"n_error": ne,
                          "err_rate_pct": 100 * ne / (5600000 if qn == "pooled_all5" else 1120000),
                          "T1_pct_of_errors": 100 * x["excl_n_t1_pattern"] / ne,
                          "amp_outlier_pct_of_errors": 100 * x["excl_n_amp_outlier"] / ne,
                          "near_boundary_pct_of_errors": 100 * x["excl_n_near_boundary"] / ne,
                          "near_boundary_true_for_every_error": (x["any_n_near_boundary"] == ne)
                          if "any_n_near_boundary" in x else None}
    if "hybridcnn_replica_vs_csv" in L[Lk]:
        d["hybridcnn_replica_vs_csv"] = L[Lk]["hybridcnn_replica_vs_csv"]
    d["flag_diagnostics"] = L[Lk].get("flag_diagnostics")
    res["old_procedure"][w] = d
res["old_procedure"]["definition"] = OP["doc"]
json.dump(res, open(OUTJ, "w"), indent=1)


def f1(d, nd=1):
    if isinstance(d, str) or d is None:
        return "n/m"
    return f"{d['pct_of_errors']:.{nd}f} [{d['ci95'][0]:.{nd}f}, {d['ci95'][1]:.{nd}f}]"


def lk(d, nd=2):
    if isinstance(d, str):
        return "n/m"
    return f"{d['pct_of_errors']:.{nd}f} [{d['ci95_mirror_analytic'][0]:.{nd}f}, {d['ci95_mirror_analytic'][1]:.{nd}f}]"


lines = []
A = lines.append
for w in WINS:
    en = "canon" if w == "1us" else "soft"
    A(f"\n### {w} ({'canonical preds_MCMitCNN_pb' if en == 'canon' else 'soft model, threshold 0.5'}), shares of errors in %\n")
    A("| Qubit | Error rate % [CI] | Leakage | Detected state change | of which in-window transition | "
      "No detected event | resid. within 1 / 2 sigma of CNN boundary | Gauss-overlap expected / resid. count |")
    A("|---|---|---|---|---|---|---|---|")
    for q in range(5):
        E = res["per_qubit"][NAMES[q]][w]["errsets"][en]
        P_ = E["primary"]["all"]
        nb = E["near_boundary"]["cnn_surrogate_boundary"].get("no_event_calls_stay_or_other", {})
        g = E["near_boundary"]["gaussian_overlap_prediction_cnn_surrogate"]
        A(f"| {NAMES[q]} | {E['err_rate_pct']:.2f} [{E['err_rate_ci95_pct'][0]:.2f}, {E['err_rate_ci95_pct'][1]:.2f}] | "
          f"{lk(P_['leakage'])} | {f1(P_['state_change_detected'])} | {f1(P_['of_which_in_window_transition'])} | "
          f"{f1(P_['no_detected_event'])} | {100 * nb.get('frac_within_1sigma', float('nan')):.0f} / "
          f"{100 * nb.get('frac_within_2sigma', float('nan')):.0f} | {g['expected_overlap_errors']:.0f} / "
          f"{P_['no_detected_event']['count']:.0f} |")
    for nm in ("all5", "Q1Q3Q4Q5", "Q1Q2Q3Q5"):
        p = res["pooled"].get(w, {}).get(en, {}).get(nm, {}).get(PRIM_VAR)
        if p is None:
            continue
        A(f"| pooled {nm} | {p['err_rate_pct']:.2f} | {p['leakage']['pct_of_errors']:.2f} "
          f"[{p['leakage']['ci95_mirror_analytic'][0]:.2f}, {p['leakage']['ci95_mirror_analytic'][1]:.2f}] | "
          f"{f1(p['state_change_detected'])} | {f1(p['of_which_in_window_transition'])} | "
          f"{f1(p['no_detected_event'])} | | |")
open(OUTMD, "w").write("\n".join(lines) + "\n")
print("wrote", OUTJ, OUTMD)
