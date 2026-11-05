"""m2mix_feature2.py -- per-qubit 'feature' table: the extra component identified at 1 us,
followed across windows (component nearest the 1 us location, in anchor-separation units,
among M1f G2 (if not collapsed) and M2f G2/G3), with paired bootstrap 95% CIs (TEST) for
w(prep0), w(prep1), overall = (w0+w1)/2 (equal shots per prepared bit), location; TRAIN replicate;
model range over M1 / M1f / M2f; LR-null details. Writes m2mix_feature_table.json/.csv.
Usage: python m2mix_feature2.py <dir> <outdir>
"""
import sys, json, glob, csv, numpy as np
D, OUT = sys.argv[1], sys.argv[2]
REF = {0: "M1f", 1: "M1f", 2: "M2f_far", 3: "M1f", 4: "M2f_far"}   # 1 us definition of the feature


RAW = {1}   # Q2/q1: anchor separation shrinks 0.86 -> 0.80 -> 0.57 ADC with window (|1> decays), so match in ADC units
QCUR = [0]


def loc(c):
    if QCUR[0] in RAW:
        return np.asarray(c["mu"], float)
    return np.array([c["along_anchor_sep"], c["perp_anchor_sep"]])


def cands(R):
    out = []
    if not R["M1f"]["G2"]["collapsed"]:
        out.append(("M1f", "G2"))
    out += [("M2f", g) for g in ("G2", "G3") if not R["M2f"][g]["collapsed"]]
    return out


def pick(R, ref):
    cs = cands(R)
    if not cs:
        cs = [("M1f", "G2"), ("M2f", "G2"), ("M2f", "G3")]
    d = [np.linalg.norm(loc(R[m][g]) - ref) for m, g in cs]
    i = int(np.argmin(d))
    # prefer M1f when it is within 0.25 separation of the best M2f match
    if cs[i][0] == "M2f" and ("M1f", "G2") in cs and d[cs.index(("M1f", "G2"))] < d[i] + 0.25:
        i = cs.index(("M1f", "G2"))
    return cs[i], d[i]


def ci(v):
    v = np.asarray(v, float)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) >= 5 else None


rows, out = [], {}
for q in range(5):
    QCUR[0] = q
    J1 = json.load(open(f"{D}/m2mix_fit2_test_q{q}_1us.json"))["results"]
    m = REF[q]
    g1 = J1["M2f"]["far"] if m == "M2f_far" else "G2"
    m1 = "M2f" if m == "M2f_far" else "M1f"
    ref = loc(J1[m1][g1])
    for w in ("early", "1us", "2us"):
        rec = {}
        for sp in ("test", "train"):
            R = json.load(open(f"{D}/m2mix_fit2_{sp}_q{q}_{w}.json"))["results"]
            (mm, gg), dist = pick(R, ref)
            c = R[mm][gg]
            rec[sp] = {"model": mm, "comp": gg, "dist_from_1us_loc_sep": float(dist),
                       "w_prep0": R[mm]["weights"][gg][0], "w_prep1": R[mm]["weights"][gg][1],
                       "along_sep": c["along_anchor_sep"], "perp_sep": c["perp_anchor_sep"],
                       "along_sig0": c["along_anchor_sig0"], "perp_sig0": c["perp_anchor_sig0"],
                       "mu_adc": c["mu"], "sigma_over_sig0": c["sigma_ratio_vs_sig0"], "collapsed": c["collapsed"],
                       "BC_G0": c["BC_with_G0"], "BC_G1": c["BC_with_G1"],
                       "M1_G2": [R["M1"]["weights"]["G2"][0], R["M1"]["weights"]["G2"][1], loc(R["M1"]["G2"]).tolist(), R["M1"]["G2"]["collapsed"]],
                       "M1f_G2": [R["M1f"]["weights"]["G2"][0], R["M1f"]["weights"]["G2"][1], loc(R["M1f"]["G2"]).tolist(), R["M1f"]["G2"]["collapsed"]],
                       "M2f": {g: [R["M2f"]["weights"][g][0], R["M2f"]["weights"][g][1], loc(R["M2f"][g]).tolist(), R["M2f"][g]["collapsed"]] for g in ("G2", "G3")}}
        t = rec["test"]
        reps = []
        for f in glob.glob(f"{D}/m2mix_boot2_np_test_q{q}_{w}_s*.json"):
            reps += json.load(open(f))["reps"]
        W0, W1, A, P = [], [], [], []
        mu_full = np.asarray(t["mu_adc"])
        for r in reps:
            if t["model"] == "M1f":
                x = r["M1f"]; g = "G2"
            else:
                x = r["M2f"]; g = min(("G2", "G3"), key=lambda gg: np.linalg.norm(np.asarray(x[gg]["mu"]) - mu_full))
            if x["LL"] is None:
                continue
            W0.append(x["weights"][g][0]); W1.append(x["weights"][g][1])
            A.append(x[g]["along_anchor_sep"]); P.append(x[g]["perp_anchor_sep"])
        W0, W1 = np.array(W0), np.array(W1)
        rec["boot"] = {"n": len(W0), "w_prep0": ci(W0), "w_prep1": ci(W1), "overall": ci(0.5 * (W0 + W1)),
                       "along_sep": ci(A), "perp_sep": ci(P)}
        allw1 = [t["M1_G2"][1], t["M1f_G2"][1]] + [t["M2f"][g][1] for g in ("G2", "G3") if not t["M2f"][g][3]]
        rec["null"] = {}
        for kind in ("nullA", "nullB"):
            lr = []
            for f in glob.glob(f"{D}/m2mix_boot2_{kind}_test_q{q}_{w}_s*.json"):
                lr += [rr["LR"] for rr in json.load(open(f))["reps"] if rr.get("LR") is not None]
            if lr:
                rec["null"][kind] = {"n": len(lr), "median": float(np.median(lr)), "max": float(np.max(lr)), "values": [float(v) for v in lr]}
        out[f"q{q}_{w}"] = rec
        b = rec["boot"]
        f = lambda v: "NA" if v is None else f"[{100*v[0]:.3f}, {100*v[1]:.3f}]"
        print(f"Q{q+1}/q{q} {w:5s} {t['model']}:{t['comp']} d={t['dist_from_1us_loc_sep']:.2f} w0={100*t['w_prep0']:.3f}% {f(b['w_prep0'])} "
              f"w1={100*t['w_prep1']:.3f}% {f(b['w_prep1'])} overall={50*(t['w_prep0']+t['w_prep1']):.3f}% {f(b['overall'])} "
              f"loc=({t['along_sep']:+.2f},{t['perp_sep']:+.2f}) sig={t['sigma_over_sig0']:.2f} coll={t['collapsed']} | train {rec['train']['model']}:{rec['train']['comp']} "
              f"w0={100*rec['train']['w_prep0']:.3f}% w1={100*rec['train']['w_prep1']:.3f}% loc=({rec['train']['along_sep']:+.2f},{rec['train']['perp_sep']:+.2f})"
              + "".join(f" | {k} med={v['median']:.0f} max={v['max']:.0f}" for k, v in rec['null'].items()))
        rows.append({"qubit_paper": f"Q{q+1}", "qubit_code": f"q{q}", "window": w, "model": t["model"] + ":" + t["comp"],
                     "w_prep0_pct": 100 * t["w_prep0"], "w_prep0_ci": f(b["w_prep0"]), "w_prep1_pct": 100 * t["w_prep1"], "w_prep1_ci": f(b["w_prep1"]),
                     "overall_pct": 50 * (t["w_prep0"] + t["w_prep1"]), "overall_ci": f(b["overall"]),
                     "along_sep": t["along_sep"], "perp_sep": t["perp_sep"], "sigma_over_sig0": t["sigma_over_sig0"], "collapsed": t["collapsed"],
                     "train_w_prep0_pct": 100 * rec["train"]["w_prep0"], "train_w_prep1_pct": 100 * rec["train"]["w_prep1"],
                     "train_along_sep": rec["train"]["along_sep"], "train_perp_sep": rec["train"]["perp_sep"],
                     "w_prep1_model_range_pct": f"{100*min(allw1):.3f}-{100*max(allw1):.3f}"})
json.dump(out, open(f"{OUT}/m2mix_feature_table.json", "w"), indent=1)
with open(f"{OUT}/m2mix_feature_table.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader()
    for r in rows:
        wr.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
