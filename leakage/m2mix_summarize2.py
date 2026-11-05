"""m2mix_summarize2.py -- aggregate m2mix v2 results (fits, bootstrap, LR nulls, traces,
coincidences) into data/leakage/m2mix_summary.json and m2mix_summary.csv.
Usage: python m2mix_summarize2.py <dir with fit2 json + boot2 json + traj2 summary + coinc> <outdir>
"""
import sys, json, glob, csv, numpy as np

D, OUT = sys.argv[1], sys.argv[2]
WINS = ["early", "1us", "2us"]
WDESC = {"early": "0-200 ns (raw samples 0-99, feat bins 0-1)",
         "1us": "paper '1 us' (raw samples 0-499, strictly causal feat1us bins 0-9)",
         "2us": "full stored trace 2 us (raw samples 0-999, feat bins 0-19)"}


def pct(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) < 5:
        return None
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def nearest(rec, mu):
    """in a bootstrap M2f record, pick the extra component nearest to the full-data far mean."""
    best = min(("G2", "G3"), key=lambda g: np.linalg.norm(np.asarray(rec[g]["mu"]) - mu))
    return best


summary = {}
rows = []
coinc = json.load(open(f"{D}/m2mix_coinc2.json"))
traj = json.load(open(f"{D}/m2mix_traj2_summary.json"))
for q in range(5):
    for w in WINS:
        key = f"q{q}_{w}"
        E = {}
        for sp in ("test", "train"):
            J = json.load(open(f"{D}/m2mix_fit2_{sp}_q{q}_{w}.json"))
            R = J["results"]
            far = R["M2f"]["far"]
            oth = "G3" if far == "G2" else "G2"

            def comp(m, g):
                c = R[m][g]
                return {"w_prep0": R[m]["weights"][g][0], "w_prep1": R[m]["weights"][g][1],
                        "along_sep": c["along_anchor_sep"], "perp_sep": c["perp_anchor_sep"],
                        "along_sig0": c["along_anchor_sig0"], "perp_sig0": c["perp_anchor_sig0"],
                        "sigma_over_sig0": c["sigma_ratio_vs_sig0"], "BC_G0": c["BC_with_G0"], "BC_G1": c["BC_with_G1"],
                        "D_M_G0": c["D_M_from_G0"], "D_M_G1": c["D_M_from_G1"], "collapsed": c["collapsed"], "mu_adc": c["mu"]}
            e = {"sig0": J["sig0"], "anchors_adc": J["anchors"], "anchor_sep_adc": R["M0"]["anchor_sep"],
                 "N_per_prepared_bit": R["M0"]["N_per_bit"],
                 "M1_G2": comp("M1", "G2"), "M1f_G2": comp("M1f", "G2"),
                 "M2f_far": comp("M2f", far), "M2f_other": comp("M2f", oth), "M2f_far_name": far,
                 "M1fsplit_prep0": comp("M1fsplit", "G2a_prep0") if False else None,
                 "LR": {k: v["LR"] for k, v in J["compare"].items()},
                 "dBIC": {k: v["dBIC"] for k, v in J["compare"].items()},
                 "dk": {k: v["dk"] for k, v in J["compare"].items()},
                 "R_weights_M1f": R["M1f"]["weights"]["R"], "lam_M1f": R["M1f"]["lam"],
                 "G1_in_prep0_M1f": R["M1f"]["weights"]["G1"][0], "G0_in_prep1_M1f": R["M1f"]["weights"]["G0"][1]}
            s = R["M1fsplit"]
            e["M1fsplit"] = {"prep0": {"along_sep": s["G2a_prep0"]["along_anchor_sep"], "perp_sep": s["G2a_prep0"]["perp_anchor_sep"],
                                       "w": s["weights"]["G2a"][0]},
                             "prep1": {"along_sep": s["G2b_prep1"]["along_anchor_sep"], "perp_sep": s["G2b_prep1"]["perp_anchor_sep"],
                                       "w": s["weights"]["G2b"][1]}}
            s = R["M1split"]
            e["M1split"] = {"prep0": {"along_sep": s["G2a_prep0"]["along_anchor_sep"], "perp_sep": s["G2a_prep0"]["perp_anchor_sep"],
                                      "w": s["weights"]["G2a"][0]},
                            "prep1": {"along_sep": s["G2b_prep1"]["along_anchor_sep"], "perp_sep": s["G2b_prep1"]["perp_anchor_sep"],
                                      "w": s["weights"]["G2b"][1]}}
            e["M1fr12"] = {"G2": R["M1fr12"]["weights"]["G2"], "R12": R["M1fr12"]["weights"]["R12"]}
            e["M2f_noncollapsed_sum"] = [sum(R["M2f"]["weights"][g][bb] for g in ("G2", "G3") if not R["M2f"][g]["collapsed"])
                                         for bb in (0, 1)]
            E[sp] = e
        # bootstrap
        B = {}
        farmu = np.asarray(E["test"]["M2f_far"]["mu_adc"])
        recs = []
        for f in glob.glob(f"{D}/m2mix_boot2_np_test_q{q}_{w}_s*.json"):
            recs += json.load(open(f))["reps"]
        if recs:
            for m, g in (("M1", "G2"), ("M1f", "G2")):
                for bb in (0, 1):
                    B[f"{m}_G2_w_prep{bb}"] = pct([r[m]["weights"][g][bb] for r in recs if r[m]["LL"] is not None])
                B[f"{m}_G2_along"] = pct([r[m][g]["along_anchor_sep"] for r in recs if r[m]["LL"] is not None])
                B[f"{m}_G2_perp"] = pct([r[m][g]["perp_anchor_sep"] for r in recs if r[m]["LL"] is not None])
            fw = {0: [], 1: []}; fa, fp = [], []
            for r in recs:
                if r["M2f"]["LL"] is None:
                    continue
                g = nearest(r["M2f"], farmu)
                for bb in (0, 1):
                    fw[bb].append(r["M2f"]["weights"][g][bb])
                fa.append(r["M2f"][g]["along_anchor_sep"]); fp.append(r["M2f"][g]["perp_anchor_sep"])
            for bb in (0, 1):
                B[f"M2f_far_w_prep{bb}"] = pct(fw[bb])
            B["M2f_far_along"], B["M2f_far_perp"] = pct(fa), pct(fp)
            B["n_rep"] = len(recs)
        # LR nulls
        NL = {}
        for kind in ("nullA", "nullB"):
            lr = []
            for f in glob.glob(f"{D}/m2mix_boot2_{kind}_test_q{q}_{w}_s*.json"):
                lr += [r["LR"] for r in json.load(open(f))["reps"] if r.get("LR") is not None and np.isfinite(r["LR"])]
            if lr:
                obs = E["test"]["LR"]["M0f->M1f" if kind == "nullA" else "M1f_nofar->M2f"]
                NL[kind] = {"n": len(lr), "max": float(np.max(lr)), "median": float(np.median(lr)), "p95": float(np.percentile(lr, 95)),
                            "observed": obs, "p_value_upper": (1 + int(np.sum(np.asarray(lr) >= obs))) / (1 + len(lr))}
        summary[key] = {"window": WDESC[w], "test": E["test"], "train": E["train"], "bootstrap": B, "LR_null": NL,
                        "traj_M2f": traj.get(key, {}).get("M2f")}
        t = E["test"]; tr = E["train"]
        rows.append({"qubit_paper": f"Q{q+1}", "qubit_code": f"q{q}", "window": w,
                     "anchor_sep_adc": round(t["anchor_sep_adc"], 3), "sig0_adc": round(t["sig0"], 3),
                     "M1_G2_w0_%": 100 * t["M1_G2"]["w_prep0"], "M1_G2_w1_%": 100 * t["M1_G2"]["w_prep1"],
                     "M1_G2_along": t["M1_G2"]["along_sep"], "M1_G2_perp": t["M1_G2"]["perp_sep"], "M1_G2_collapsed": t["M1_G2"]["collapsed"],
                     "M1f_G2_w0_%": 100 * t["M1f_G2"]["w_prep0"], "M1f_G2_w1_%": 100 * t["M1f_G2"]["w_prep1"],
                     "M1f_G2_along": t["M1f_G2"]["along_sep"], "M1f_G2_perp": t["M1f_G2"]["perp_sep"], "M1f_G2_collapsed": t["M1f_G2"]["collapsed"],
                     "M2f_far_w0_%": 100 * t["M2f_far"]["w_prep0"], "M2f_far_w1_%": 100 * t["M2f_far"]["w_prep1"],
                     "M2f_far_w0_CI": B.get("M2f_far_w_prep0"), "M2f_far_w1_CI": B.get("M2f_far_w_prep1"),
                     "M2f_far_along": t["M2f_far"]["along_sep"], "M2f_far_perp": t["M2f_far"]["perp_sep"],
                     "M2f_far_sigma_sig0": t["M2f_far"]["sigma_over_sig0"], "M2f_far_collapsed": t["M2f_far"]["collapsed"],
                     "train_M2f_far_w0_%": 100 * tr["M2f_far"]["w_prep0"], "train_M2f_far_w1_%": 100 * tr["M2f_far"]["w_prep1"],
                     "train_M2f_far_along": tr["M2f_far"]["along_sep"], "train_M2f_far_perp": tr["M2f_far"]["perp_sep"],
                     "LR_M0_M1": t["LR"]["M0->M1"], "LR_M0f_M1f": t["LR"]["M0f->M1f"], "LR_far_test": t["LR"]["M1f_nofar->M2f"],
                     "dBIC_M0_M1": t["dBIC"]["M0->M1"], "dBIC_M0f_M1f": t["dBIC"]["M0f->M1f"], "dBIC_far_test": t["dBIC"]["M1f_nofar->M2f"],
                     "nullA_max": NL.get("nullA", {}).get("max"), "nullB_max": NL.get("nullB", {}).get("max")})
json.dump({"summary": summary, "coincidence": coinc}, open(f"{OUT}/m2mix_summary.json", "w"), indent=1)
with open(f"{OUT}/m2mix_summary.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    for r in rows:
        wr.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
for r in rows:
    print(r["qubit_paper"], r["window"], "M2f far w0/w1 %.4f/%.4f%%" % (r["M2f_far_w0_%"], r["M2f_far_w1_%"]), r["M2f_far_w0_CI"], r["M2f_far_w1_CI"],
          "loc (%.2f,%.2f)" % (r["M2f_far_along"], r["M2f_far_perp"]), "LRfar=%.0f nullB_max=%s LRflex=%.0f nullA_max=%s" % (
              r["LR_far_test"], r["nullB_max"], r["LR_M0f_M1f"], r["nullA_max"]))
