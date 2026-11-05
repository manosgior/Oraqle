"""m4conf_summary.py -- assemble the per-qubit leakage-estimate table from the m4conf JSONs (agent tag m4conf).

Reads m4conf_results.json (candidate sets, B) and m4conf_final.json (funnel) from <dir>; writes m4conf_summary.json.
Because A (|z_perp| > 5, 1 us) is a strict subset of B (prior |2> labels) for every qubit and prepared bit, the B shots
that pass the frame screen (|z_perp| > 5 in the calibrated 1 us frame) are exactly the A shots, so B survivors = A
survivors and only the denominators differ.
Usage: python m4conf_summary.py /home/manosgior/Documents/GitHub/oraqle-ics/data/leakage
"""
import json, sys
import numpy as np

D = sys.argv[1]
R = json.load(open(f"{D}/m4conf_results.json"))
F = json.load(open(f"{D}/m4conf_final.json"))["funnel"]


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0, c - h)), float(min(1, c + h))]


def rate(k, n):
    return {"k": int(k), "n": int(n), "rate": (k / n if n else None), "ci95": wilson(k, n)}


def katz(k1, n1, k0, n0):
    if not (k1 and k0):
        return [None, None]
    rr = (k1 / n1) / (k0 / n0); se = np.sqrt(1 / k1 - 1 / n1 + 1 / k0 - 1 / n0)
    return [float(rr * np.exp(-1.96 * se)), float(rr * np.exp(1.96 * se))]


S = {"note": __doc__, "window_primary": "1us = raw samples 0-499 (feat1us bins 0-9, strictly causal FIR)", "per_qubit": {}}
for wn in ("1us", "2us"):
    for q in range(5):
        e = {}
        kb_tot = nb_tot = 0
        for b in (0, 1):
            f = F[wn][f"q{q}_prep{b}"]
            n = f["n_prepared"]; kc = f["candidates"]["k"]; kb = f["BASE_survivor_rate"]["k"]
            ks = f["SINGLE_QUBIT_rate"]["k"]; nq = f["SINGLE_QUBIT_rate"]["n"]
            d = {"n_prepared": n, "A_candidates": rate(kc, n), "BASE_survivors": rate(kb, n),
                 "BASE_surviving_fraction_of_A": rate(kb, kc) if kc else None,
                 "SINGLE_qubit_rate_quiet_neighbours": rate(ks, nq),
                 "SINGLE_over_A_rate": ((ks / nq) / (kc / n) if kc else None), "SINGLE_over_A_rate_ci95": katz(ks, nq, kc, n)}
            if wn == "1us":
                B = R["d_prepdep"]["B"][f"q{q}"][f"prep{b}"]
                d["B_prior_labels"] = rate(B["k"], B["n"])
                d["B_surviving_fraction_BASE"] = rate(kb, B["k"]) if B["k"] else None
                d["B_surviving_fraction_SINGLE"] = ((ks / nq) / (B["k"] / B["n"]) if B["k"] else None)
                d["B_surviving_fraction_SINGLE_ci95"] = katz(ks, nq, B["k"], B["n"])
                d["B_failing_frame_screen_fraction"] = rate(B["k"] - kc, B["k"]) if B["k"] else None
            e[f"prep{b}"] = d
            kb_tot += kb; nb_tot += n
        e["overall_BASE"] = rate(kb_tot, nb_tot)
        S["per_qubit"].setdefault(wn, {})[f"Q{q + 1}/q{q}"] = e
json.dump(S, open(f"{D}/m4conf_summary.json", "w"), indent=1)
fmt = lambda r: (f"{r['k']}/{r['n']} = {100 * r['rate']:.4f}% [{100 * r['ci95'][0]:.4f}, {100 * r['ci95'][1]:.4f}]" if r and r["rate"] is not None else "n/a")
for wn in ("1us", "2us"):
    print("=====", wn)
    for qn, e in S["per_qubit"][wn].items():
        for b in (0, 1):
            d = e[f"prep{b}"]
            print(qn, f"p{b}", "A", fmt(d["A_candidates"]), "| BASE", fmt(d["BASE_survivors"]), "| frac", fmt(d["BASE_surviving_fraction_of_A"]),
                  "| SINGLE", fmt(d["SINGLE_qubit_rate_quiet_neighbours"]))
            if wn == "1us":
                print("      B", fmt(d["B_prior_labels"]), "| B surv frac BASE", fmt(d["B_surviving_fraction_BASE"]),
                      "| B surv frac SINGLE %s %s" % (d["B_surviving_fraction_SINGLE"] and round(d["B_surviving_fraction_SINGLE"], 4), d["B_surviving_fraction_SINGLE_ci95"]))
        print(qn, "overall BASE", fmt(e["overall_BASE"]))
