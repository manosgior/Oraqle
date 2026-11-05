"""Part A and C numbers from the sweeps: Lambda, required distance, qubits, logical clock and spacetime
per readout operating point, and the decoder's per-round cost against the round.

Definitions (one lattice-sim experiment = R = 2(d+1) rounds of two patches):
  per-round error   eps = (1 - (1 - 2 P_L)^(1/R)) / 2
  per-op error      P_op(d) = (1 - (1 - 2 eps)^(d+1)) / 2      (a lattice-surgery op lasts d+1 rounds)
  fit               ln eps(d) = a + b d over distances with >= MIN_FAILS failures and P_L < 0.3;
                    Lambda = exp(-2 b)
  required d        smallest odd d with P_op(d) <= target, from the fit
  round             t_r = T + 340 ns (4 CNOTs of 70 ns and 2 Hadamards; checked against circuit_ns)
  qubits            2 d^2 - 1 per logical qubit (rotated surface code)
  op time           (d + 1) t_r;  spacetime per op = qubits * op time

Usage: python system/analyze.py [--data DIR]   (reads DIR/ler_*.jsonl and decode_timing_d*.jsonl, writes DIR/summary.json;
       DIR defaults to $ORAQLE_WORK/system, where system/run_paper_sweeps.sh writes; to re-derive the committed
       summary: cp -r results/system work/system first, or run verify/qec_system.py)
"""
import argparse
import collections
import glob
import json
import math
import os

import sys

import numpy as np

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oraqle_env import WORK  # noqa: E402

MIN_FAILS = 30
TARGETS = [1e-6, 1e-9, 1e-12]
GATES_NS = 340


def load(path):
    acc = collections.defaultdict(lambda: dict(shots=0, fails=0))
    meta = {}
    for line in open(path):
        r = json.loads(line)
        k = (r["label"], r["T"], round(r["p"], 9), r["d"])
        acc[k]["shots"] += r["shots"]
        acc[k]["fails"] += r["fails"]
        meta[k] = r
    out = {}
    for k, v in acc.items():
        r = meta[k]
        P = v["fails"] / v["shots"]
        R = r["rounds"]
        eps = (1 - (1 - 2 * P) ** (1 / R)) / 2 if P < 0.5 else float("nan")
        out[k] = dict(v, P=P, eps=eps, R=R, circuit_ns=r["circuit_ns"])
    return out


def fit(points):
    """points: {d: row}. Returns (a, b, Lambda, used distances)."""
    ok = [(d, r["eps"]) for d, r in sorted(points.items()) if r["fails"] >= MIN_FAILS and r["P"] < 0.3]
    if len(ok) < 2:
        return None
    b, a = np.polyfit([d for d, _ in ok], np.log([e for _, e in ok]), 1)
    return a, b, math.exp(-2 * b), [d for d, _ in ok]


def required_d(a, b, target, dmax=401):
    if b >= 0:
        return None
    for d in range(3, dmax, 2):
        eps = math.exp(a + b * d)
        if (1 - (1 - 2 * eps) ** (d + 1)) / 2 <= target:
            return d
    return None


def summarize(rows):
    groups = collections.defaultdict(dict)
    for (label, T, p, d), r in rows.items():
        groups[(label, T, p)][d] = r
    out = []
    for (label, T, p), pts in sorted(groups.items()):
        f = fit(pts)
        t_r = (T + GATES_NS) / 1000
        s = dict(label=label, T=T, p=p, t_r_us=t_r, readout_share=T / (T + GATES_NS),
                 eps={d: pts[d]["eps"] for d in sorted(pts)}, fails={d: pts[d]["fails"] for d in sorted(pts)})
        if f:
            a, b, lam, used = f
            s.update(Lambda=lam, fit_ds=used, a=a, b=b)
            for t in TARGETS:
                d = required_d(a, b, t)
                if d:
                    q = 2 * d * d - 1
                    s[f"d@{t:g}"] = d
                    s[f"qubits@{t:g}"] = q
                    s[f"op_us@{t:g}"] = (d + 1) * t_r
                    s[f"spacetime@{t:g}"] = q * (d + 1) * t_r
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(WORK / "system"))  # was <oraqle-ics>/data/sims
    a = ap.parse_args()
    res = {}
    for name in ("disc", "disc_hi", "grid"):
        path = os.path.join(a.data, f"ler_{name}.jsonl")
        if os.path.exists(path):
            res[name] = summarize(load(path))
    dec = [json.loads(l) for f in sorted(glob.glob(os.path.join(a.data, "decode_timing_d*.jsonl"))) for l in open(f)]
    res["decode"] = dec
    json.dump(res, open(os.path.join(a.data, "summary.json"), "w"), indent=1, default=float)

    disc = "disc_hi" if "disc_hi" in res else "disc"
    if disc in res:
        base = next((s for s in res[disc] if s["label"] == "MCMit-CNN" and s["T"] == 1000), None)
        for t in TARGETS:
            key = f"spacetime@{t:g}"
            print(f"\n== {disc}: target {t:g} per op  (spacetime relative to MCMit-CNN @ 1000 ns)")
            print(f"{'model':<13}{'T':>5}{'p':>8}{'Lambda':>8}{'d':>5}{'qubits':>8}{'op us':>8}{'ST':>7}  fit ds")
            for s in res[disc]:
                st = s.get(key)
                rel = st / base[key] if st and base and base.get(key) else float("nan")
                print(f"{s['label']:<13}{s['T']:>5}{s['p']:>8.4f}{s.get('Lambda', float('nan')):>8.2f}"
                      f"{s.get(f'd@{t:g}', '-'):>5}{s.get(f'qubits@{t:g}', '-'):>8}"
                      f"{s.get(f'op_us@{t:g}', float('nan')):>8.1f}{rel:>7.2f}  {s.get('fit_ds', '')}")
    if dec:
        print("\n== decode (PyMatching, one pinned core): per round vs the round t_r")
        print(f"{'d':>3}{'T':>6}{'ev/rnd':>8}{'batch/rnd':>11}{'med/rnd':>10}{'p99 shot':>10}{'t_r':>7}{'cores/patch':>12}")
        for r in sorted(dec, key=lambda r: (r["d"], r["T"])):
            t_r = r["T"] + GATES_NS
            print(f"{r['d']:>3}{r['T']:>6}{r['events_per_round']:>8.2f}{r['batch_ns_per_round'] / 1e3:>10.2f}u"
                  f"{r['single_median_ns_per_round'] / 1e3:>9.2f}u{r['single_p99_ns'] / 1e3:>9.0f}u{t_r / 1e3:>6.2f}u"
                  f"{r['batch_ns_per_round'] / 2 / t_r:>12.1f}")


if __name__ == "__main__":
    main()
