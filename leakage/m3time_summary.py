#!/usr/bin/env python3
"""m3time_summary.py -- collect the key temporal-signature numbers into one JSON (agent tag m3time).
Inputs: data/leakage/m3time_re_{feat_xt1,featbox_xt1,feat_xt0}_q{q}.json, m3time_xchk_feat_xt1.json
Output: data/leakage/m3time_summary.json
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import numpy as np
from scipy.stats import beta

D = str(LP.RES)
NAMES = ["Q1", "Q2", "Q3", "Q4", "Q5"]


def cp(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def r(x, n=5):
    return None if x is None else float(np.round(x, n))


def pick(d, run):
    H = d["hard_test"]["viterbi_filtered"]
    R = d["hmm_rates_test_onestep"]
    O = d["soft_occupancy_test"]
    g = d["geometry"]["1us"]
    o = {"run": run, "relabelled_vs_m3time_hmm": d["relabelled"],
         "L_location_bins3_9": {"along_in_sep": r(g["L_along_in_sep"], 3), "perp_in_sep": r(g["L_perp_in_sep"], 3),
                                "dist_to_1_sigma_bin": r(g["L_dist_to_1_in_sigma_bin"], 2),
                                "dist_to_0_sigma_bin": r(g["L_dist_to_0_in_sigma_bin"], 2),
                                "sep01_sigma_bin": r(g["sep_in_sigma_bin"], 2)},
         "hmm_test": {
             "tau_L_total_ns": [r(R["from2"]["tau_ns"], 0)] + [r(x, 0) for x in R["from2"]["tau_ns_ci95"]],
             "tau_L_to1_ns": [r(R["from2"]["tau_to1_ns"], 0)] + [r(x, 0) for x in R["from2"]["tau_to1_ns_ci95"]],
             "tau_L_to0_ns": [r(R["from2"]["tau_to0_ns"], 0)] + [r(x, 0) for x in R["from2"]["tau_to0_ns_ci95"]],
             "branch_L_to1": [r(R["from2"]["branch_to1"], 4)] + [r(x, 4) for x in R["from2"]["branch_to1_ci95"]],
             "tau_1_to0_ns(T1 under readout)": [r(R["from1"]["tau_to0_ns"], 0)] + [r(x, 0) for x in R["from1"]["tau_to0_ns_ci95"]],
             "tau_1_toL_ns": [r(R["from1"]["tau_to2_ns"], 0)] + [r(x, 0) for x in R["from1"]["tau_to2_ns_ci95"]],
             "tau_0_toL_ns": [r(R["from0"]["tau_to2_ns"], 0)] + [r(x, 0) for x in R["from0"]["tau_to2_ns_ci95"]],
             "tau_0_to1_ns": [r(R["from0"]["tau_to1_ns"], 0)] + [r(x, 0) for x in R["from0"]["tau_to1_ns_ci95"]],
             "ratio_tauL_over_tau10": r(R["from2"]["tau_ns"] / R["from1"]["tau_to0_ns"], 3)},
         "soft_pi_L_t0": {f"prep{b}": [r(O[f"prep{b}"]["pi_L_onestep"], 6)] + [r(x, 6) for x in O[f"prep{b}"]["pi_L_onestep_ci95"]]
                          for b in (0, 1)}}
    for w in ("1us", "2us"):
        W = H[w]
        pe = W["pooled_exit_from2"]
        o[f"viterbi_{w}"] = {
            "L_starters_pooled": {"n": pe["n_start"], "n_exit": pe["n_exit"], "exit_to": pe["n_exit_to"],
                                  "tau_ns": [r(pe["tau_ns"], 0)] + [r(x, 0) for x in pe["tau_ns_ci95"]],
                                  "branch_to1": [r(pe["branch_to1"]["frac"], 4)] + [r(x, 4) for x in pe["branch_to1"]["ci95"]]},
            "one_starters_prep1_tau_to0_ns": [r(W["prep1"]["exit_from1"]["tau_to0_ns"], 0)] +
                                             [r(x, 0) for x in W["prep1"]["exit_from1"]["tau_to0_ns_ci95"]],
            "cascade_after_L_to_1": {k: W["pooled_cascade_L_to_1"][k] for k in ("n_Lstart_exit_to1", "next_after_1")}}
        for b in (0, 1):
            P = W[f"prep{b}"]
            me = P["mid_entry_L"]
            o[f"viterbi_{w}"][f"prep{b}"] = {
                "start_L(bins1-2)": {k: r(v, 6) if isinstance(v, float) else v for k, v in P["start_L"].items()},
                "ever_L": {k: r(v, 6) if isinstance(v, float) else v for k, v in P["ever_L"].items()},
                "mid_entry_L": {"k": me["enter_L"]["k"], "n": me["enter_L"]["n"], "frac": r(me["enter_L"]["frac"], 6),
                                "ci95": [r(x, 6) for x in me["enter_L"]["ci95"]], "from0": me["from0"]["k"],
                                "from1": me["from1"]["k"]}}
            if w == "1us":
                nn = d["null_sim"]["1us"][f"prep{b}"]
                o[f"viterbi_{w}"][f"prep{b}"]["null_MA1_false_positive"] = {
                    "start_L": r(nn["start_L"]["frac"], 6), "start_L_ci95": [r(x, 6) for x in nn["start_L"]["ci95"]],
                    "ever_L": r(nn["ever_L"]["frac"], 6), "mid_entry_L": r(nn["mid_entry_L"]["frac"], 6),
                    "n": nn["start_L"]["n"]}
        k = W["prep0"]["start_L"]["k"] + W["prep1"]["start_L"]["k"]
        n = W["prep0"]["start_L"]["n"] + W["prep1"]["start_L"]["n"]
        o[f"viterbi_{w}"]["pooled_start_L"] = {"k": k, "n": n, "frac": r(k / n, 6), "ci95": [r(x, 6) for x in cp(k, n)]}
        k = W["prep0"]["ever_L"]["k"] + W["prep1"]["ever_L"]["k"]
        o[f"viterbi_{w}"]["pooled_ever_L"] = {"k": k, "n": n, "frac": r(k / n, 6), "ci95": [r(x, 6) for x in cp(k, n)]}
    return o


def main():
    X = json.load(open(f"{D}/m3time_xchk_feat_xt1.json"))
    out = {"definitions": {
        "window": "20 bins x 100 ns over the full stored 2 us trace (raw samples 0-999 at 500 MS/s); paper's 1 us = bins 0-9",
        "start_L": "filtered Viterbi state equals L in both bins 1 and 2 (100-300 ns); bin 0 is ring-up",
        "soft_pi_L_t0": "HMM posterior P(L) at bin 0 averaged over shots (one-step M-step estimate of the initial distribution)",
        "mid_entry_L": "shot started in |0> or |1> and enters L at bin >= 3 within the window (filtered Viterbi)",
        "tau": "HMM: per-100 ns exit probability from frozen-TRAIN posteriors on TEST, delta-method CI; Viterbi: censored-geometric MLE, Clopper-Pearson CI",
        "null_MA1_false_positive": "same pipeline on synthetic shots from the fitted model with L switched off and MA(1) noise matched to the measured lag-1 correlation",
        "denominators": "TEST: 560000 shots per prepared bit per qubit; pooled 1120000"},
        "per_qubit": {}}
    for q in range(5):
        e = {"qubit_code": f"q{q}", "qubit_paper": NAMES[q]}
        for run in ("feat_xt1", "featbox_xt1", "feat_xt0"):
            e[run] = pick(json.load(open(f"{D}/m3time_re_{run}_q{q}.json")), run)
        if q in (0, 4):
            s = X["startL_by_Q3_Q4_prep"][NAMES[q]]
            e["Q3_dependence"] = {k: {"start_L": v["start_L"], "ever_L_1us": v["ever_L_1us"]} for k, v in s.items()}
            e["lag0_lift_1toL_vs_Q3_1to0(prep both 1)"] = X["lags"][f"{NAMES[q]} 1->L vs Q3 1->0 | prepq1_prepr1"]["0"]
            e["lag0_lift_Lto1_vs_Q3_0to1(prep both 1)"] = X["lags"][f"{NAMES[q]} L->1 vs Q3 0->1 | prepq1_prepr1"]["0"]
        out["per_qubit"][NAMES[q]] = e
    out["Q3_decay_by_Q1_Q5_prep"] = X["Q3_decay_by_Q1_Q5_prep"]
    with open(f"{D}/m3time_summary.json", "w") as f:
        json.dump(out, f, indent=1)
    for q in range(5):
        e = out["per_qubit"][NAMES[q]]["feat_xt1"]
        v = e["viterbi_1us"]
        print(NAMES[q], "pooled startL", v["pooled_start_L"], "\n   pooled everL1us", v["pooled_ever_L"],
              "\n   hmm", e["hmm_test"], "\n   pi", e["soft_pi_L_t0"])


if __name__ == "__main__":
    main()
