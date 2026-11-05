"""em_e5vstat_summarize.py (tag e5vstat): compact numeric summary of the adversarial re-computation.
Inputs: data/leakage/em_e5vstat_check.json (v2), em_e5vstat_lda.json (fixed masks). Output: em_e5vstat_summary.json."""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import numpy as np

P = str(LP.RES) + "/"
C = json.load(open(P + "em_e5vstat_check.json"))
LD = json.load(open(P + "em_e5vstat_lda.json"))
QS = ["Q1", "Q2", "Q3", "Q4", "Q5"]
KEYS = ["1us|cnn_canonical", "600ns|cnn_soft600", "400ns|cnn_soft400", "1us|herq_pertrace", "400ns|herq_pertrace"]


def pool(key, qs, ref, bits=("prep0", "prep1"), k=3):
    N = L = Lv = O = Ov = n = 0
    for q in qs:
        for b in bits:
            x = C["per_qubit"][q]["models"][key][b]
            N += x["N"]; n += x["n"]
            Lq = x[f"L{k}"] if q != "Q4" else 0
            if q != "Q4":
                L += Lq; Lv += x[f"L{k}_var"]
            O += min(x[f"O_{ref}"], x["N"] - Lq); Ov += x[f"O_{ref}_var"]
    SC = N - L - O
    h = 1.96 * np.sqrt(Lv + Ov)
    return {"n_err": N, "err_pct": 100 * N / n, "L_pct_err": 100 * L / N, "O_pct_err": 100 * O / N,
            "SC_pct_err": 100 * SC / N, "SC_ci95": [100 * (SC - h) / N, 100 * (SC + h) / N],
            "SC_pct_shots": 100 * SC / n, "O_pct_shots": 100 * O / n}


out = {"note": __doc__, "pooled": {}, "identity_prep1": {}, "lda_vs_cnn_latecore": {}, "step_null_misread_stationary": {},
       "late_window_separation_sigma": {q: C["per_qubit"][q]["late_info"]["sep_over_s_median"] for q in QS},
       "noise_corr_early_late_prep0_gk": {q: C["per_qubit"][q]["noise_corr"]["prep0"] for q in QS},
       "cofailure": C["cofailure"], "checks": C["checks"]}
for key in KEYS:
    out["pooled"][key] = {}
    for name, qs in [("Q1Q3Q4Q5", ["Q1", "Q3", "Q4", "Q5"]), ("Q1Q3Q5", ["Q1", "Q3", "Q5"]), ("all5", QS)] + [(q, [q]) for q in QS]:
        out["pooled"][key][name] = {ref: pool(key, qs, ref) for ref in ["e4_lc", "r1.0", "r1.5", "r2.0", "r2.5", "hp"]}
    out["identity_prep1"][key] = {}
    for q in QS:
        x = C["per_qubit"][q]["models"][key]["prep1"]
        out["identity_prep1"][key][q] = {"late_other_frac_err": x["late_other_frac_err"], "late_other_frac_correct": x["late_other_frac_correct"],
                                         "s_identity": x["s_identity_vs_correct"],
                                         "SC_prep1_e4ref": pool(key, [q], "e4_lc", bits=("prep1",))["SC_pct_err"]}
for q in QS:
    out["lda_vs_cnn_latecore"][q] = {}
    for wn, w in LD["per_qubit"][q]["windows"].items():
        out["lda_vs_cnn_latecore"][q][wn] = {b: {k: w[b][k] for k in ("n_latecore", "lda_err_latecore_pct", "lda_err_latecore_ci",
                                                                      "cnn_err_latecore_pct", "cnn_err_latecore_ci", "lda_gauss_pred_pct",
                                                                      "cnn_latecore_err_lda_score_hist")} | {"delta": w["delta_median"]}
                                             for b in ("prep0", "prep1")}
    sn = C["per_qubit"][q].get("step_null", {})
    out["step_null_misread_stationary"][q] = {k: v for k, v in sn.items() if "e4_lc" in k or k.startswith("T_orderA")}
json.dump(out, open(P + "em_e5vstat_summary.json", "w"), indent=1, default=float)
print("written")
