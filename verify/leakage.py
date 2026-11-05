#!/usr/bin/env python3
"""verify/leakage.py -- check every number of the readout error-mode analysis against the paper.

Covers Sec. 5.3 (subsec:failures), Table tab:error_modes, Takeaway #5 and App. app:error-modes
(chapters/ml_discriminators.tex, chapters/appendix_herqules.tex). No dataset needed.

What it does
  1. RECOMPUTE: every paper number is recomputed from the upstream result files in results/leakage
     (em_e4recon_final.json counts, em_e5vstat_{summary,check}.json, em_e5vrepro_*.json, ...), not just read
     from the final-numbers file; where em_e8fix_final_numbers.json (the file the paper text was written from)
     stores the same number, the two are cross-checked.
  2. COMPARE: each value is compared with results/leakage/paper_values.json (printed value, chapter:line).
  3. TABLE: the regenerated LaTeX rows of tab:error_modes (paper_figures/leakage) must equal the paper rows.
  4. REGEN (default on, ~5 s): the dataset-free assembly scripts of the chain (e2seg/e4recon/e5vrepro/e5vstat
     assembly, e6edit and e8fix finalize, m1tail/m3time/m4conf/r summaries) are re-run on a scratch copy of
     results/leakage and must reproduce the committed JSON byte for byte.
  Optional: --figure re-renders fig:iq_taxonomy and compares it pixel by pixel with the paper PDF
  (needs matplotlib, seaborn, pdftoppm); --tex <oraqle-ics checkout> re-checks the paper line references.

Exit status: 0 if everything matches, 1 otherwise. Paper numbers that are off by rounding and are documented in
paper_values.json ("known_issue") are listed as DOCUMENTED DISCREPANCIES and fail only with --strict. Other
caveats are printed as NOTES.

Usage:  python verify/leakage.py [--no-regen] [--figure] [--tex /path/to/oraqle-ics] [--strict] [-v]
"""
import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LEAK = os.path.join(ROOT, "leakage")
PF = os.path.join(ROOT, "paper_figures", "leakage")
sys.path.insert(0, LEAK)
sys.path.insert(0, PF)

EPS = 1e-9
FAIL, NOTES, KNOWN = [], [], []


def load(res, name):
    with open(os.path.join(res, name)) as f:
        return json.load(f)


def v(e):
    return e["value"] if isinstance(e, dict) and "value" in e else e


def xcheck(tag, a, b, tol):
    """Upstream recomputation a vs stored final number b."""
    if a is None or b is None or isinstance(a, str) or isinstance(b, str):
        if a != b:
            FAIL.append(f"[xcheck] {tag}: {a!r} vs {b!r}")
        return
    if abs(a - b) > tol + EPS:
        FAIL.append(f"[xcheck] {tag}: recomputed {a:.6g} vs em_e8fix {b:.6g} (tol {tol})")


def code_const(fname, name):
    """Value of a module-level constant in a leakage script (parsed, not imported)."""
    tree = ast.parse(open(os.path.join(LEAK, fname)).read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(f"{name} not in {fname}")


# ======================================================================================== recomputation
class Calc:
    def __init__(self, res):
        self.E4 = load(res, "em_e4recon_final.json")
        self.E8 = load(res, "em_e8fix_final_numbers.json")
        self.VS = load(res, "em_e5vstat_summary.json")
        self.VC = load(res, "em_e5vstat_check.json")
        self.VR = load(res, "em_e5vrepro_results.json")
        self.VP = load(res, "em_e5vrepro_diag_partners.json")
        self.H3 = load(res, "em_e3herq_summary_v3.json")
        self.M1 = load(res, "m1tail_v2_percfg_test.json")
        self.T = self.E4["table_1us_cnn_canonical"]
        self.W = self.E4["windows"]
        self.LK = self.E4["leakage_E3_by_window"]
        self.t8 = self.E8["table_error_modes_1us"]

    # ---- attribution rows (counts / n_err)
    def row(self, q, window="1us", model="cnn"):
        if model == "herq":
            return self.E4["herqules"][f"{window}_herq_pertrace"]["per_qubit"][q]
        if window == "1us":
            return self.T["pooled"][q] if q.startswith("Q1Q") else self.T["per_qubit"][q]
        w = self.W[f"{window}_cnn_soft{window[:-2]}"]
        return w["pooled"][q] if q.startswith("Q1Q") else w["per_qubit"][q]

    def share(self, q, mode, window="1us", model="cnn", per="err"):
        r = self.row(q, window, model)
        if r.get(mode) is None or r[mode].get("count") is None:
            return "n/m"
        return 100.0 * r[mode]["count"] / (r["n_err"] if per == "err" else r["n_shots"])

    def err_rate(self, q):
        r = self.row(q)
        return 100.0 * r["n_err"] / r["n_shots"]

    def pooled_from_qubits(self, mode, qs=("Q1", "Q3", "Q5")):
        num = sum(self.T["per_qubit"][q][mode]["count"] for q in qs)
        den = sum(self.T["per_qubit"][q]["n_err"] for q in qs)
        return 100.0 * num / den

    # ---- leakage populations (E3, % of prepared-b shots)
    def e3(self, q, b, window="1us"):
        x = self.LK[window][q]
        return "n/m" if x is None else x[f"prep{b}"]["E3_pct"]

    def leaked_pct_shots(self, q):
        a, b = self.e3(q, 0), self.e3(q, 1)
        return "n/m" if a == "n/m" else (a + b) / 2.0          # 560k shots per prepared bit

    def leaked_read_correctly(self, q):
        h = self.E4["text_numbers"]["hidden_leakage_1us"].get(q)
        if h is None:
            return "n/m"
        return 100.0 * (1.0 - h["leaked_errors_pct_of_shots"]["value"] / h["leaked_pct_of_shots"]["value"])

    # ---- model-free half-plane check (late window)
    def halfplane(self, q):
        d = self.VS["identity_prep1"]["1us|cnn_canonical"][q]
        return 100 * d["late_other_frac_err"], 100 * d["late_other_frac_correct"]

    def halfplane_pooled(self):
        num_s = num_a = den = 0.0
        for q in ("Q1", "Q3", "Q5"):
            d = self.VS["identity_prep1"]["1us|cnn_canonical"][q]
            n = self.VC["per_qubit"][q]["models"]["1us|cnn_canonical"]["prep1"]["N"]
            fe, fc = d["late_other_frac_err"], d["late_other_frac_correct"]
            num_s += (fe - fc) / (1 - fc) * n
            num_a += d["SC_prep1_e4ref"] / 100 * n
            den += n
        return 100 * num_s / den, 100 * num_a / den

    # ---- resolved-step detector variants (% of errors)
    def step_T(self, q):
        vals = [self.share(q, "transition_resolved")]
        for null in ("gauss", "emp"):
            vals.append(self.VR["attribution"]["1us"][null][q]["transition_T"]["value"])
        return vals

    def gating(self):
        return self.E4["step_detector_gating_and_sensitivity"]["1us"]

    def sc_ref_band(self):
        p = self.VS["pooled"]["1us|cnn_canonical"]["Q1Q3Q5"]
        sc = [p[k]["SC_pct_err"] for k in ("e4_lc", "r1.0", "r1.5", "r2.0", "r2.5", "hp")]
        sc.append(self.VR["attribution"]["1us"]["gauss"]["pooled_Q1Q3Q5"]["state_change_TS_lower"]["value"]
                  if "pooled_Q1Q3Q5" in self.VR["attribution"]["1us"]["gauss"] else
                  v(self.t8["Q1Q3Q5"]["state_change_reference_band"]["values"]["e5vrepro_2sigma_disc"]))
        ov = [p[k]["O_pct_err"] for k in ("e4_lc", "r1.0", "r2.5", "hp")]
        return sc, ov

    def q3cond_share(self, qtarget):
        c = self.E8["e8fix_changes"]["Q3_conditioning_of_Q1_Q5_leakage_1us"][f"{qtarget}_prep1_pct"]
        a, b = v(c["Q3_prep0"]), v(c["Q3_prep1"])               # equal shot counts (280k each)
        assert c["Q3_prep0"]["n_shots"] == c["Q3_prep1"]["n_shots"]
        s = 100 * b / (a + b)
        xcheck(f"Q3-conditioning share {qtarget}", s, v(c["share_of_population_with_Q3_prep1_pct"]), 0.06)
        return s


def getters(C):
    E8, t8 = C.E8, C.t8
    G = {}

    # ---------------- table rows
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q1Q3Q5"):
        def er(q=q):
            x = C.err_rate(q)
            xcheck(f"err_rate {q}", x, v(t8[q]["err_rate_pct"]), 0.0006)
            return x

        def lk(q=q):
            if q == "Q4":
                st = C.T["per_qubit"]["Q4"].get("leakage_status", "")
                return "n/m" if (C.T["per_qubit"]["Q4"].get("leakage") is None or "n/m" in st) else C.share(q, "leakage")
            x = C.share(q, "leakage")
            xcheck(f"leakage {q}", x, v(t8[q]["leakage"]), 0.006)
            if q == "Q1Q3Q5":
                xcheck("pooled leakage from qubits", C.pooled_from_qubits("leakage"), x, 0.006)
            return x

        def sc(q=q):
            if v(t8[q]["state_change"]) == "n/m":
                # Q2: decided n/m because its late-window cores are only ~0.5 sigma apart (O ~ N by construction)
                assert C.VS["late_window_separation_sigma"][q] < 1.0
                return "n/m"
            x = C.share(q, "state_change_total_T_plus_S")
            xcheck(f"state_change {q}", x, v(t8[q]["state_change"]), 0.006)
            if q == "Q1Q3Q5":
                xcheck("pooled state change from qubits", C.pooled_from_qubits("state_change_total_T_plus_S"), x, 0.006)
            return x

        def ov(q=q):
            if v(t8[q]["overlap"]) == "n/m":
                assert C.VS["late_window_separation_sigma"][q] < 1.0
                return "n/m"
            x = C.share(q, "overlap_near_boundary_upper")
            xcheck(f"overlap {q}", x, v(t8[q]["overlap"]), 0.006)
            return x

        def ls(q=q):
            x = C.leaked_pct_shots(q)
            if x != "n/m":
                xcheck(f"leaked shots {q}", x, v(t8[q]["leaked_shots_pct_of_shots"]), 0.006)
            return x

        def rc(q=q):
            x = C.leaked_read_correctly(q)
            xcheck(f"read correctly {q}", x, v(t8[q]["leaked_read_correctly_pct"]), 0.06)
            return x
        G.update({f"tab.{q}.err_rate": er, f"tab.{q}.leakage": lk, f"tab.{q}.state_change": sc,
                  f"tab.{q}.overlap": ov, f"tab.{q}.leaked_shots": ls, f"tab.{q}.leaked_read_correctly": rc})
    G["tab.footnote.Q2_sep_sigma"] = lambda: C.VS["late_window_separation_sigma"]["Q2"]

    # ---------------- Sec. 5.3 scope + results
    def scope():
        ok = (all(q in C.T["per_qubit"] for q in ("Q1", "Q2", "Q3", "Q4", "Q5"))
              and all(q in C.W["400ns_cnn_soft400"]["per_qubit"] for q in ("Q1", "Q3", "Q5"))
              and all(q in C.E4["herqules"]["1us_herq_pertrace"]["per_qubit"] for q in ("Q3", "Q5")))
        return "ok" if ok else "missing"
    G["s53.scope"] = scope
    G["s53.res.Q3_state_change"] = lambda: C.share("Q3", "state_change_total_T_plus_S")
    G["s53.res.Q5_state_change"] = lambda: C.share("Q5", "state_change_total_T_plus_S")
    G["s53.res.Q3_overlap"] = lambda: C.share("Q3", "overlap_near_boundary_upper")
    G["s53.res.Q5_overlap"] = lambda: C.share("Q5", "overlap_near_boundary_upper")

    def hp(q, i):
        def f():
            x = C.halfplane(q)[i]
            key = "errors_on_0_side_pct" if i == 0 else "correct_on_0_side_pct"
            xcheck(f"half-plane {q} {key}", x, v(E8["late_window_half_plane_check_1us"][q][key]), 0.0006)
            return x
        return f
    G["s53.res.Q3_err_on_0_side"], G["s53.res.Q5_err_on_0_side"] = hp("Q3", 0), hp("Q5", 0)
    G["s53.res.Q3_corr_on_0_side"], G["s53.res.Q5_corr_on_0_side"] = hp("Q3", 1), hp("Q5", 1)

    def stepT():
        vals = C.step_T("Q1") + C.step_T("Q3") + C.step_T("Q5")
        return min(vals), max(vals)
    G["s53.res.step_T_range"] = stepT
    G["s53.res.Q3_overlap_400ns"] = lambda: C.share("Q3", "overlap_near_boundary_upper", "400ns")
    G["s53.res.Q5_overlap_400ns"] = lambda: C.share("Q5", "overlap_near_boundary_upper", "400ns")
    G["s53.res.Q3_leak_prep1"] = lambda: C.e3("Q3", 1)
    G["s53.res.Q5_leak_prep1"] = lambda: C.e3("Q5", 1)
    G["s53.res.Q5_leak_mostly_Q3_1"] = lambda: C.q3cond_share("Q5")
    G["s53.res.leak_read_correctly"] = lambda: (min(C.leaked_read_correctly("Q3"), C.leaked_read_correctly("Q5")),
                                                max(C.leaked_read_correctly("Q3"), C.leaked_read_correctly("Q5")))
    G["s53.res.Q3_leak_400ns"] = lambda: C.e3("Q3", 1, "400ns")
    G["s53.res.Q3_leak_600ns"] = lambda: C.e3("Q3", 1, "600ns")
    G["s53.res.Q3_leak_1us"] = lambda: C.e3("Q3", 1, "1us")

    # ---------------- Sec. 5.3 analysis
    G["s53.ana.pooled_state_change"] = lambda: C.share("Q1Q3Q5", "state_change_total_T_plus_S")
    G["s53.ana.pooled_overlap"] = lambda: C.share("Q1Q3Q5", "overlap_near_boundary_upper")

    def band(i):
        def f():
            sc, ovl = C.sc_ref_band()
            vals = sc if i == 0 else ovl
            stored = t8["Q1Q3Q5"]["state_change_reference_band" if i == 0 else "overlap_reference_band"]["range"]
            xcheck("reference band lo", min(vals), stored[0], 0.06)
            xcheck("reference band hi", max(vals), stored[1], 0.06)
            return min(vals), max(vals)
        return f
    G["s53.ana.pooled_state_change_band"], G["s53.ana.pooled_overlap_band"] = band(0), band(1)
    G["app.pooled_sc_reference_band"] = band(0)
    G["s53.ana.pooled_leakage"] = lambda: C.share("Q1Q3Q5", "leakage")

    def q2gauss():
        lda = C.VS["lda_vs_cnn_latecore"]["Q2"]["1us"]["prep0"]
        pred = lda["lda_gauss_pred_pct"]                  # per-configuration Phi(-Delta_cfg/2), averaged (em_e5vstat_lda.py)
        xcheck("Q2 LDA Gaussian error", pred, E8["Q2_snr_bound"]["lda_gaussian_predicted_error_pct"], 0.006)
        xcheck("Q2 Phi(-Delta_median/2) sanity", 100 * 0.5 * math.erfc(lda["delta"] / 2 / math.sqrt(2)), pred, 0.05)
        return pred
    G["s53.ana.Q2_gauss_pred_err"] = q2gauss
    G["s53.ana.Q2_gauss_ge_cnn"] = lambda: (q2gauss(), C.err_rate("Q2"))

    def growth(q):
        a = C.share(q, "state_change_total_T_plus_S", "400ns", per="shots")
        b = C.share(q, "state_change_total_T_plus_S", "1us", per="shots")
        return b / a
    G["s53.ana.Q1_sc_growth"] = lambda: growth("Q1")
    G["s53.ana.sc_growth_is_max_on_Q1"] = lambda: max(("Q1", "Q3", "Q5"), key=growth)
    G["s53.ana.Q3_sc_400ns"] = lambda: C.share("Q3", "state_change_total_T_plus_S", "400ns", per="shots")
    G["s53.ana.Q3_sc_1us"] = lambda: C.share("Q3", "state_change_total_T_plus_S", "1us", per="shots")
    G["s53.ana.pooled_sc_400ns"] = lambda: C.share("Q1Q3Q5", "state_change_total_T_plus_S", "400ns")
    G["s53.ana.pooled_sc_1us"] = lambda: C.share("Q1Q3Q5", "state_change_total_T_plus_S", "1us")

    def fall():
        r = [C.share(q, "overlap_near_boundary_upper", "400ns", per="shots")
             / C.share(q, "overlap_near_boundary_upper", "1us", per="shots") for q in ("Q1", "Q3", "Q5")]
        for q, x in zip(("Q1", "Q3", "Q5"), r):
            xcheck(f"overlap fall {q}", x, v(E8["overlap_per_shot_fall_400ns_to_1us"][q]), 0.01)
        return min(r), max(r)
    G["s53.ana.overlap_fall"] = fall
    G["s53.ana.Q3_leak_triples"] = lambda: C.e3("Q3", 1, "1us") / C.e3("Q3", 1, "400ns")

    def cof(q):
        def f():
            c = C.VS["cofailure"][f"1us_{q}"]
            x = 100 * c["both"] / c["herq_err"]
            xcheck(f"co-failure {q}", x, v(E8["herqules_pertrace_1us"][q]["cofailure_frac_herq_err_also_cnn_pct"]), 0.0006)
            assert c["herq_err"] == C.row(q, model="herq")["n_err"] and c["cnn_err"] == C.row(q)["n_err"]
            return x
        return f
    G["s53.ana.herq_cofail_Q3"], G["s53.ana.herq_cofail_Q5"] = cof("Q3"), cof("Q5")
    G["s53.ana.herq_Q3_state_change"] = lambda: C.share("Q3", "state_change_total_T_plus_S", model="herq")

    def fewer():
        x = 100 * (1 - C.row("Q3")["n_err"] / C.row("Q3", model="herq")["n_err"])
        xcheck("CNN fewer Q3 errors", x, v(E8["herqules_pertrace_1us"]["Q3_cnn_fewer_errors_pct"]), 0.0006)
        return x
    G["s53.ana.cnn_fewer_Q3_errors"] = fewer
    G["s53.ana.fewer_errors_span_both_modes"] = lambda: {
        m: (C.share("Q3", k, per="shots"), C.share("Q3", k, model="herq", per="shots"))
        for m, k in (("state_change", "state_change_total_T_plus_S"), ("overlap", "overlap_near_boundary_upper"))}

    # ---------------- Takeaway #5
    G["tk5.state_change_approx80"] = lambda: [C.share("Q1Q3Q5", "state_change_total_T_plus_S")] + list(band(0)())
    G["tk5.overlap_le20"] = lambda: ([C.share(q, "overlap_near_boundary_upper") for q in ("Q1", "Q3", "Q5", "Q1Q3Q5")]
                                     + [band(1)()[1]])
    G["tk5.leak_shots_range"] = lambda: (min(C.leaked_pct_shots(q) for q in ("Q1", "Q3", "Q5")),
                                         max(C.leaked_pct_shots(q) for q in ("Q1", "Q3", "Q5")))
    G["tk5.model_choice_le23"] = fewer

    # ---------------- appendix
    G["app.scope_windows"] = lambda: "ok" if all(q in C.W[w]["per_qubit"] for w in ("400ns_cnn_soft400", "600ns_cnn_soft600")
                                                 for q in ("Q1", "Q3", "Q5")) else "missing"

    def ncfg():
        ns = {len(C.M1[f"1us_q{q}"][f"prep{b}"]["rows"]) for q in range(5) for b in (0, 1)}
        assert len(ns) == 1
        return ns.pop()
    G["app.n_configs"] = ncfg

    def zcut():
        s = E8["e8fix_changes"]["attribution_definitions"]["z_sign"]
        m = re.search(r"leaked if z>(\d+)", s)
        return int(m.group(1))
    G["app.z_cut"] = zcut
    G["app.bins_ns"] = lambda: E8["e8fix_changes"]["attribution_definitions"]["bins_ns"]

    def alpha():
        a0 = code_const("em_e2seg_attr_v3.py", "ALPHA0")
        al, idx = code_const("em_e4recon_analyze.py", "ALPHA_IDX"), code_const("em_e2seg_attr_v3.py", "ALPHAS")
        assert idx[al] == a0, "em_e4recon_analyze.py ALPHA_IDX does not select ALPHA0"
        return a0
    G["app.alpha"] = alpha

    def used_tests():
        g = C.gating()
        return {f"{q}_prep{b}": g[q][f"prep{b}"] for q in g for b in (0, 1)}

    def pwin():
        u = [t["ts_window_p_flag_ge_0.5_ns"] for k, t in used_tests().items()
             if t["measurable"] and k[:2] in ("Q3", "Q5")]
        return min(w[0] for w in u), max(w[1] for w in u)
    G["app.p_flag_window"] = pwin
    G["app.Q1_p_flag_max"] = lambda: max(used_tests()["Q1_prep0"]["p_flag_max"], used_tests()["Q1_prep1"]["p_flag_max"])

    def effr():
        e = [100 * t["eff_given_cnn_proxy_misread"] for t in used_tests().values() if t["measurable"]]
        return min(e), max(e)
    G["app.efficiency_range"] = effr

    def gate_rule():
        # recompute the gate: measurable iff 0.6 <= obs/exp <= 1.3 and efficiency >= 5 %
        lo, hi, emin = 0.6, 1.3, 0.05
        bad = [k for k, t in used_tests().items()
               if t["measurable"] != (lo <= t["quasinull_obs_over_exp"] <= hi and t["eff_given_cnn_proxy_misread"] >= emin)]
        assert not bad, f"gate rule does not reproduce 'measurable' for {bad}"
        return lo, hi, emin
    G["app.gate_obs_exp"] = lambda: list(gate_rule()[:2])
    G["app.gate_efficiency"] = lambda: round(100 * gate_rule()[2])
    G["app.gate_excluded"] = lambda: sorted(k for k, t in used_tests().items() if not t["measurable"])

    def late_r():
        a = code_const("em_e2seg_attr_v3.py", "LATE_R")
        xcheck("LATE_R", a, E8["e8fix_changes"]["attribution_definitions"]["late_core_radius_LATE_R"], 0)
        return a
    G["app.late_core_radius"] = late_r

    def sk1():
        d = E8["e8fix_changes"]["S_absorbs_subthreshold_leakage_Q3"]
        x = d["state_change_k3"] - d["state_change_k1"]
        m = C.VC["per_qubit"]["Q3"]["models"]["1us|cnn_canonical"]
        n = m["prep0"]["N"] + m["prep1"]["N"]
        naive = 100 * sum(m[b]["L1"] - m[b]["L3"] for b in ("prep0", "prep1")) / n
        NOTES.append(f"app:384 'z>1 lowers state change by 1.5 pp': e5vstat value {x:.2f} pp; the naive count "
                     f"difference (L1-L3)/N_err from em_e5vstat_check.json is {naive:.2f} pp (both round to 1.5).")
        return x
    G["app.S_k1_shift"] = sk1

    def order_bound():
        s = C.E4["robustness_notes"]["order_swap"]
        m = re.search(r"move by <= ([0-9.]+) pp", s)
        return float(m.group(1))
    G["app.order_swap_O_S"] = order_bound

    def order_q3():
        s = C.E4["robustness_notes"]["order_swap"]
        m = re.search(r"Q3 L ([0-9.]+) -> ([0-9.]+) %, T ([0-9.]+) -> ([0-9.]+) %", s)
        la, lb, ta, tb = map(float, m.groups())
        xcheck("order swap L(A)=table Q3 leakage", la, round(C.share("Q3", "leakage"), 2), 0.005)
        xcheck("order swap moves L into T", la - lb, tb - ta, 0.011)
        return la - lb
    G["app.order_swap_Q3_L_to_T"] = order_q3
    G["app.Q2_late_sep"] = lambda: C.VS["late_window_separation_sigma"]["Q2"]
    G["app.halfplane_err_range"] = lambda: (min(C.halfplane(q)[0] for q in ("Q1", "Q3", "Q5")),
                                            max(C.halfplane(q)[0] for q in ("Q1", "Q3", "Q5")))
    G["app.halfplane_corr_range"] = lambda: (min(C.halfplane(q)[1] for q in ("Q1", "Q3", "Q5")),
                                             max(C.halfplane(q)[1] for q in ("Q1", "Q3", "Q5")))

    def hps(i):
        def f():
            x = C.halfplane_pooled()[i]
            k = "identity_estimate_pct" if i == 0 else "attributed_state_change_prep1_pct"
            xcheck(f"half-plane pooled {k}", x, E8["late_window_half_plane_check_1us"]["pooled_Q1Q3Q5"][k], 0.006)
            return x
        return f
    G["app.halfplane_s"], G["app.halfplane_attributed"] = hps(0), hps(1)

    def fpnull(i):
        def f():
            sn = C.VS["step_null_misread_stationary"]
            o = sum(t["obs"] for q in ("Q1", "Q5") for k, t in sn[q].items() if k.endswith("_misread"))
            e = sum(t["exp"] for q in ("Q1", "Q5") for k, t in sn[q].items() if k.endswith("_misread"))
            fp = E8["resolved_step_T"]["fp_null_on_misread_stationary_Q1Q5"]
            xcheck("FP null observed", o, fp["observed_flags"], 0)
            xcheck("FP null expected", e, fp["expected_flags"], 0.06)
            return o if i == 0 else e
        return f
    G["app.fp_null_observed"], G["app.fp_null_expected"] = fpnull(0), fpnull(1)
    G["app.step_T_Q1Q5_range"] = lambda: (min(C.step_T("Q1") + C.step_T("Q5")), max(C.step_T("Q1") + C.step_T("Q5")))
    G["app.step_T_Q3"] = lambda: C.step_T("Q3")

    def partners(sel):
        def f():
            d = C.VP["1us_q2"]
            pick = []
            for k, t in d.items():
                bits = dict(kv.split("=") for kv in k.split(","))
                if all(bits[x] == sel for x in ("Q1", "Q4", "Q5")):
                    pick.append(t["ratio"])
            if sel == "1":
                assert max(pick) == max(t["ratio"] for t in d.values()), "all-excited is not the maximum"
                return max(pick)
            return pick                                    # every value must round to 1.0
        return f
    G["app.Q3_relax_null_partners0"] = partners("0")
    G["app.Q3_relax_null_all_excited"] = partners("1")
    G["app.leak_Q3_prep1"] = lambda: C.e3("Q3", 1)
    G["app.leak_Q3_prep0"] = lambda: C.e3("Q3", 0)
    G["app.leak_Q5_prep1"] = lambda: C.e3("Q5", 1)
    G["app.leak_Q1_prep1"] = lambda: C.e3("Q1", 1)
    G["app.leak_Q1Q5_share_Q3_1"] = lambda: [C.q3cond_share("Q5"), C.q3cond_share("Q1")]
    G["app.leak_Q2_prep1"] = lambda: C.e3("Q2", 1)
    G["app.leak_Q4_nm"] = lambda: C.e3("Q4", 1)
    G["app.leak_share_Q3_prep0_err"] = lambda: (100 * C.T["per_qubit"]["Q3"]["per_bit"]["prep0"]["leakage"]["count"]
                                                / C.T["per_qubit"]["Q3"]["per_bit"]["prep0"]["n_err"])
    G["app.leak_share_Q3_err"] = lambda: C.share("Q3", "leakage")

    def readas1():
        m = C.VC["per_qubit"]["Q3"]["models"]["1us|cnn_canonical"]["prep1"]
        x = 100 * (1 - m["L3"] / m["leaked_shots3"])
        xcheck("Q3 leaked read as 1", x, v(E8["leakage"]["Q3_leaked_prep1_read_as_1_pct"]), 0.0006)
        return x
    G["app.Q3_leaked_read_as_1"] = readas1
    return G


# ======================================================================================== comparison
def rounds_to(x, printed, d):
    return abs(x - printed) <= 0.5 * 10 ** (-d) + EPS


def compare(it, val):
    k, p, d = it["kind"], it["printed"], it.get("decimals")
    if k == "num":
        return isinstance(val, (int, float)) and rounds_to(val, p, d)
    if k == "nm":
        return val == "n/m"
    if k == "range":
        ds = it.get("decimals_each", [d, d])
        return rounds_to(min(val), p[0], ds[0]) and rounds_to(max(val), p[1], ds[1])
    if k == "range_all":
        return all(rounds_to(x, p, d) for x in val)
    if k == "approx":
        vals = val if isinstance(val, (list, tuple)) else [val]
        return all(abs(x - p) <= it["tol"] + EPS for x in vals)
    if k == "ge":
        return round(val, d) >= p
    if k == "le":
        vals = val if isinstance(val, (list, tuple)) else [val]
        return all(round(x, d) <= p for x in vals)
    if k == "param":
        return val == p or (isinstance(val, (int, float)) and isinstance(p, (int, float)) and abs(val - p) < EPS)
    if k == "set":
        return sorted(val) == sorted(p)
    if k == "scope":
        return val == "ok"
    if k == "argmax":
        return val == p
    if k == "ge_pair":
        return val[0] >= val[1]
    if k == "both_less":
        return all(a < b for a, b in val.values())
    if k == "text_bound":
        return abs(val - p) < EPS
    raise ValueError(k)


def fmt(val):
    if isinstance(val, float):
        return f"{val:.4g}"
    if isinstance(val, (list, tuple)):
        return "[" + ", ".join(fmt(x) for x in val) + "]"
    if isinstance(val, dict):
        return "{" + ", ".join(f"{a}: {fmt(b)}" for a, b in val.items()) + "}"
    return str(val)


# ======================================================================================== extra checks
def check_table(res):
    import make_error_modes_table as T
    rows = T.table_rows(load(res, "em_e8fix_final_numbers.json"), load(res, "em_e4recon_final.json"))
    ref = open(os.path.join(PF, "reference", "tab_error_modes.tex")).read().split("\n")
    pv = {it["id"].split(".")[1]: it["snippet"] for it in load(res, "paper_values.json")["items"]
          if it["id"].startswith("tab.") and it["id"].endswith(".err_rate")}
    bad = []
    for q, row in rows.items():
        if row not in ref:
            bad.append(q)
        if not row.startswith(pv[q]) and row.replace(r" \\", "") != pv[q]:
            if pv[q] not in row:
                bad.append(q + "(snippet)")
    full = T.render(load(res, "em_e8fix_final_numbers.json"), load(res, "em_e4recon_final.json"))
    same_block = full.rstrip("\n").split("\n") == [l for l in ref if l != ""]
    if bad or not same_block:
        FAIL.append(f"[table] regenerated tab:error_modes differs from the paper (rows {bad}, block equal={same_block})")
    return not bad and same_block


REGEN = [  # (script, argv, outputs) -- dataset-free stages, in dependency order
    ("em_e2seg_assemble.py", [], ["em_e2seg_results.json"]),
    ("em_e4recon_assemble.py", [], ["em_e4recon_final.json"]),
    ("em_e5vrepro_assemble.py", [], ["em_e5vrepro_results.json"]),
    ("em_e5vstat_summarize.py", [], ["em_e5vstat_summary.json"]),
    ("em_e6edit_finalize.py", [], ["em_e6edit_final_numbers.json"]),
    ("em_e8fix_finalize.py", [], ["em_e8fix_final_numbers.json"]),
    ("m1tail_v2_summary.py", ["{RES}"], ["m1tail_summary.json"]),
    ("m3time_summary.py", [], ["m3time_summary.json"]),
    ("m4conf_summary.py", ["{RES}"], ["m4conf_summary.json"]),
    ("r_build_summary.py", ["{RES}"], ["r_leakage_summary.json"]),
]


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def regen(res, verbose):
    tmp = tempfile.mkdtemp(prefix="leak_regen_")
    try:
        work = os.path.join(tmp, "res")
        shutil.copytree(res, work)
        env = dict(os.environ, ORAQLE_LEAK_RESULTS=work, ORAQLE_WORK=os.path.join(tmp, "work"),
                   ORAQLE_OUT=os.path.join(tmp, "out"), MPLBACKEND="Agg")
        ok = True
        for script, argv, outs in REGEN:
            for o in outs:
                os.remove(os.path.join(work, o))
            r = subprocess.run([sys.executable, os.path.join(LEAK, script)] + [a.format(RES=work) for a in argv],
                               env=env, cwd=tmp, capture_output=True, text=True)
            for o in outs:
                p = os.path.join(work, o)
                same = r.returncode == 0 and os.path.exists(p) and md5(p) == md5(os.path.join(res, o))
                ok &= same
                print(f"  regen {script:28s} -> {o:32s} {'byte-identical' if same else 'DIFFERS'}")
                if not same:
                    FAIL.append(f"[regen] {script} did not reproduce {o} (rc={r.returncode}) {r.stderr[-400:]}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def figure_check():
    import make_iq_taxonomy as F  # noqa: F401  (checks importability)
    tmp = tempfile.mkdtemp(prefix="leak_fig_")
    try:
        out = os.path.join(tmp, "iq.pdf")
        r = subprocess.run([sys.executable, os.path.join(PF, "make_iq_taxonomy.py"), "--out", out],
                           capture_output=True, text=True, env=dict(os.environ, MPLBACKEND="Agg"))
        if r.returncode != 0:
            FAIL.append(f"[figure] make_iq_taxonomy.py failed: {r.stderr[-400:]}")
            return False
        ref = os.path.join(PF, "reference", "iq_error_gallery_q5_v2.pdf")
        for p, stem in ((out, "new"), (ref, "ref")):
            subprocess.run(["pdftoppm", "-r", "300", "-gray", "-png", p, os.path.join(tmp, stem)], check=True)
        import numpy as np
        import matplotlib.image as mi
        a = mi.imread(os.path.join(tmp, "new-1.png")).astype(float)
        b = mi.imread(os.path.join(tmp, "ref-1.png")).astype(float)
        d = float(np.abs(a - b).mean() * 255) if a.shape == b.shape else float("inf")
        print(f"  fig:iq_taxonomy re-rendered vs paper PDF at 300 dpi: mean |diff| = {d:.3g} grey levels")
        if d > 0.5:
            FAIL.append(f"[figure] fig:iq_taxonomy differs from the paper PDF (mean |diff| {d:.3g} grey levels)")
        return d <= 0.5
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def tex_check(tex, items):
    moved = []
    for it in items:
        f, ln = it["ref"].rsplit(":", 1)
        lines = open(os.path.join(tex, f)).read().split("\n")
        if not (int(ln) <= len(lines) and it["snippet"] in lines[int(ln) - 1]):
            hit = [i + 1 for i, l in enumerate(lines) if it["snippet"] in l and not l.lstrip().startswith("%")]
            moved.append((it["id"], it["ref"], hit))
    for m in moved:
        FAIL.append(f"[tex] {m[0]}: snippet no longer at {m[1]} (now at lines {m[2] or 'none'})")
    return not moved


def notes(C):
    h = C.H3["windows"]["1us"]["configs"]["pertrace_ckpt"]
    tab = {1000: [0.9686, 0.7398, 0.9237, 0.9437, 0.9696]}      # tab:fidelity HERQULES row (herqules_length_faithful_vs_demux.csv)
    d = [abs(a - b) * 100 for a, b in zip(h["fq"], tab[1000])]
    pp = max(abs(x) for x in C.H3["windows"]["1us"]["repro_pertrace_ckpt_vs_paper_tabfidelity"]["diff_pp"])
    NOTES.append("HERQULES in Sec. 5.3 is the per-trace checkpoint HERQULES_len500.pth (+ rmf_len500.pkl) of "
                 "_train_and_analyze_errors.py (2026-06-02), re-inferred by em_e3herq_regen_pertrace_ckpt.py; the "
                 "tab:fidelity HERQULES row comes from _herqules_length_both.py (retrained, no checkpoint). Per-qubit "
                 f"fidelity at 1 us differs by at most {max(d):.2f} pp from the unrounded table CSV (Q{1 + d.index(max(d))}; "
                 f"{pp:.2f} pp from the printed 3-decimal table, as the paper states since the 2026-11-05 [audit fix]); "
                 f"F5Q {h['F5Q']} vs 0.9046.")
    NOTES.append("The 1 us MCMit-CNN is preds_MCMitCNN_pb.npy (runners/_xtalk_cnn.py, a separate training run of the "
                 "tab:fidelity recipe; per-qubit fidelity within 1e-4 of the table); the 400/600 ns models are "
                 "cnn_soft_{400,600}ns.npz of sims/soft_readout_cnn.py (same recipe).")
    NOTES.append("app:372 'Q1 peaks at 0.44': injection P(flag) maxima are 0.445 (excitation) and 0.437 (relaxation); "
                 "0.445 sits exactly on the rounding boundary. app:402 '1.7--1.8% on Q3': detector variants 1.75/1.72/1.68.")
    NOTES.append("Leakage numbers are k=3 mirror-corrected lower bounds (E3); Q4 leakage and Q2 state change/overlap "
                 "are n/m by construction (multimodal |1> / late cores 0.52 sigma apart).")
    NOTES.append("app:387 '<= 0.03 pp' is quoted from em_e4recon_final.json robustness_notes (string), not recomputable "
                 "from the shipped JSON; the order-B per-shot arrays are cluster intermediates.")


# ======================================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=os.environ.get("ORAQLE_LEAK_RESULTS", os.path.join(ROOT, "results", "leakage")))
    ap.add_argument("--no-regen", action="store_true", help="skip re-running the assembly scripts")
    ap.add_argument("--figure", action="store_true", help="re-render fig:iq_taxonomy and compare with the paper PDF")
    ap.add_argument("--tex", default=None, help="oraqle-ics checkout: re-check the chapter:line references")
    ap.add_argument("--strict", action="store_true", help="also fail on the documented discrepancies")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    res = a.results
    pv = load(res, "paper_values.json")
    C = Calc(res)
    G = getters(C)
    n_ok = 0
    print(f"== paper numbers ({len(pv['items'])} items, results: {res})")
    for it in pv["items"]:
        g = G.get(it["id"])
        if g is None:
            FAIL.append(f"[missing] no recomputation for {it['id']}")
            continue
        try:
            val = g()
            ok = compare(it, val)
        except Exception as e:  # noqa: BLE001
            val, ok = f"ERROR {type(e).__name__}: {e}", False
        tag = "ok " if ok else "BAD"
        if not ok and "known_issue" in it:
            ki = it["known_issue"]
            alt = dict(it, **ki["accept"])
            if compare(alt, val):
                tag = "KNOWN"
                KNOWN.append(f"{it['id']} ({it['ref']}): paper prints {it['printed']!r}, results give {fmt(val)}. "
                             f"{ki['why']} Suggested: {ki.get('suggested_text', '')}")
                if a.strict:
                    FAIL.append(f"[known, --strict] {it['id']}")
        n_ok += ok
        if tag == "BAD":
            FAIL.append(f"[value] {it['id']} ({it['ref']}): paper {it['printed']!r}, results {fmt(val)}")
        if a.verbose or not ok:
            print(f"  {tag:5s} {it['id']:38s} {it['ref']:38s} paper {fmt(it['printed']):>14s}  results {fmt(val)}")
    print(f"  {n_ok}/{len(pv['items'])} paper numbers reproduced exactly at the printed precision; "
          f"{len(KNOWN)} documented discrepancies")
    print("== table tab:error_modes")
    print("  regenerated LaTeX block identical to the paper:", check_table(res))
    if not a.no_regen:
        print("== regeneration of the dataset-free assembly chain")
        regen(res, a.verbose)
    if a.figure:
        print("== figure fig:iq_taxonomy")
        figure_check()
    if a.tex:
        print("== paper line references")
        print("  all snippets on their lines:", tex_check(a.tex, pv["items"]))
    notes(C)
    if KNOWN:
        print("== documented discrepancies between the results and the paper text (fail only with --strict)")
        for k in KNOWN:
            print("  -", k)
    print("== notes (documented caveats, not failures)")
    for n in NOTES:
        print("  -", n)
    if FAIL:
        print(f"== FAILED ({len(FAIL)})")
        for f in FAIL:
            print("  ", f)
        return 1
    print("== ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
