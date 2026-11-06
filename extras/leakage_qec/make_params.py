#!/usr/bin/env python3
"""make_params.py -- simulation parameters from the measured readout outputs.

Per readout length T and qubit: P(leaves the readout in L) per measurement, created in this readout (end-of-window occupancy
minus the occupancy at the start, for half |0> and half |1> outcomes); lifetime under readout from the frozen HMM; the flag's
recall and false-flag rate from the linear detector on the CNN input (readout_outputs.py, 1% false-flag point).

Scenarios: q3_like   every ancilla behaves like Q3, the worst qubit measured
           pooled    the mean over Q1, Q3, Q4, Q5 (the qubits sweep_soft.py pools), Q4 entering with zero because its
                     "L" fails the label-free off-axis check (it is a split |1> response, see LEAKAGE_REPORT.md)
Lifetimes outside the readout are unknown; IDLE_TAU_US is an assumption, and the idle part of a round is short.
"""
import json, math, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
IDLE_TAU_US = 10.0
GATES_NS = 340
QS = ["Q1", "Q3", "Q5"]
CODE = {"Q1": 0, "Q3": 2, "Q5": 4}


def tau_l_us(q):
    d = json.load(open(ROOT / "results" / "leakage" / f"m3time_re_featbox_xt1_q{CODE[q]}.json"))
    return d["params_train_canonical"]["tau_state2_ns"] / 1000


def main(src, out):
    r = json.load(open(src))["per_T"]
    res = {"note": __doc__, "idle_tau_us": IDLE_TAU_US, "tau_L_us": {q: tau_l_us(q) for q in QS}, "T": {}}
    for T, row in r.items():
        Tn = int(T)
        per = {}
        for q in QS:
            x = row["per_qubit"][q]
            p1 = max(0.0, x["P_L_end"]["prep1"] - x["P_L_start"]["prep1"])
            p0 = max(0.0, x["P_L_end"]["prep0"] - x["P_L_start"]["prep0"])
            pc = 0.5 * (p0 + p1)
            op = x["detectors"]["input"][2]
            tau = res["tau_L_us"][q]
            s = math.exp(-Tn / 1000 / tau - GATES_NS / 1000 / IDLE_TAU_US)
            ops = x["detectors"]["input"]
            per[q] = dict(p_create=pc, frac_from_0=p0 / max(1e-9, p0 + p1), survive=s, tpr=op["tpr"], fpr=op["fpr"],
                          future_offaxis=op.get("future_offaxis"),
                          roc={str(o["fpr_target"]): dict(tpr=o["tpr"], fpr=o["fpr"]) for o in ops})
        tot = sum(per[q]["p_create"] for q in QS)
        pooled = dict(p_create=tot / 4,
                      frac_from_0=sum(per[q]["p_create"] * per[q]["frac_from_0"] for q in QS) / tot,
                      survive=sum(per[q]["p_create"] * per[q]["survive"] for q in QS) / tot,
                      tpr=sum(per[q]["p_create"] * per[q]["tpr"] for q in QS) / tot,
                      fpr=sum(per[q]["fpr"] for q in QS) / 3)
        pooled["roc"] = {t: dict(tpr=sum(per[q]["p_create"] * per[q]["roc"][t]["tpr"] for q in QS) / tot,
                                 fpr=sum(per[q]["roc"][t]["fpr"] for q in QS) / 3) for t in per["Q3"]["roc"]}
        res["T"][T] = dict(per_qubit=per, q3_like={k: v for k, v in per["Q3"].items() if k != "future_offaxis"}, pooled=pooled,
                           confidence_table=row["confidence_table"])
    json.dump(res, open(out, "w"), indent=1)
    for T, v in res["T"].items():
        print(T, "q3_like", {k: round(x, 4) for k, x in v["q3_like"].items() if k != "roc"}, "pooled", {k: round(x, 4) for k, x in v["pooled"].items() if k != "roc"})
    print("tau_L_us", res["tau_L_us"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE / "readout_outputs.json", sys.argv[2] if len(sys.argv) > 2 else HERE / "leak_params.json")
